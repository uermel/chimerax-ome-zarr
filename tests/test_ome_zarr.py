from copy import deepcopy

import fsspec
import numpy as np
import pytest
import zarr

from src.map_data.ome_metadata import (
    OMEZarrFormatError,
    parse_ome_zarr_metadata,
    spatial_transform_angstrom,
)
from src.map_data.zarr_grid import WrappedZarrGrid, ZarrGridSlice, ZarrModel
from src.open import open_ome_zarr_from_fs, open_ome_zarr_from_store

AXIS_CASES = {
    "yx": (["space", "space"], (6, 8)),
    "zyx": (["space", "space", "space"], (4, 6, 8)),
    "tyx": (["time", "space", "space"], (2, 6, 8)),
    "cyx": (["channel", "space", "space"], (3, 6, 8)),
    "tcyx": (["time", "channel", "space", "space"], (2, 3, 6, 8)),
    "tzyx": (["time", "space", "space", "space"], (2, 4, 6, 8)),
    "czyx": (["channel", "space", "space", "space"], (3, 4, 6, 8)),
    "tczyx": (["time", "channel", "space", "space", "space"], (2, 3, 4, 6, 8)),
}


def _axes(axis_types):
    spatial_names = iter(("z", "y", "x") if axis_types.count("space") == 3 else ("y", "x"))
    axes = []
    for axis_type in axis_types:
        if axis_type == "time":
            axes.append({"name": "t", "type": "time", "unit": "second"})
        elif axis_type == "channel":
            axes.append({"name": "c", "type": "channel"})
        else:
            axes.append({"name": next(spatial_names), "type": "space", "unit": "nanometer"})
    return axes


def _create_array(group, zarr_format, name, data, chunks, dimension_names=None, shards=None):
    kwargs = {"name": name, "shape": data.shape, "dtype": data.dtype, "chunks": chunks}
    if zarr_format == 3:
        kwargs["dimension_names"] = dimension_names
        if shards is not None:
            kwargs["shards"] = shards
    array = group.create_array(**kwargs)
    array[:] = data
    return array


def _make_image(
    zarr_format,
    axis_types,
    shape,
    *,
    transforms=None,
    group_transforms=None,
    omero=None,
    shards=None,
    store=None,
):
    if store is None:
        store = zarr.storage.MemoryStore()
    group = zarr.create_group(store=store, zarr_format=zarr_format)
    axes = _axes(axis_types)
    names = tuple(axis["name"] for axis in axes)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    _create_array(
        group,
        zarr_format,
        "0",
        data,
        tuple(max(1, min(4, size)) for size in shape),
        dimension_names=names,
        shards=shards,
    )

    scale = [1.0] * len(shape)
    raw_multiscale = {
        "name": "image",
        "axes": axes,
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": transforms or [{"type": "scale", "scale": scale}],
            },
        ],
    }
    if group_transforms is not None:
        raw_multiscale["coordinateTransformations"] = group_transforms
    if zarr_format == 2:
        raw_multiscale["version"] = "0.4"
        attrs = {"multiscales": [raw_multiscale]}
        if omero is not None:
            attrs["omero"] = omero
        group.attrs.update(attrs)
    else:
        ome = {"version": "0.5", "multiscales": [raw_multiscale]}
        if omero is not None:
            ome["omero"] = omero
        group.attrs["ome"] = ome
    return group, data


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(("case", "axis_types_shape"), AXIS_CASES.items())
def test_supported_axis_combinations(zarr_format, case, axis_types_shape):
    del case
    axis_types, shape = axis_types_shape
    group, _ = _make_image(zarr_format, axis_types, shape)

    metadata = parse_ome_zarr_metadata(group)

    assert metadata.zarr_format == zarr_format
    assert metadata.ome_version == ("0.4" if zarr_format == 2 else "0.5")
    assert [axis.type for axis in metadata.multiscales.axes] == axis_types
    assert metadata.multiscales.spatial_ndim == axis_types.count("space")
    assert metadata.multiscales.datasets[0].path == "0"


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_omero_channel_metadata_is_normalized(zarr_format):
    omero = {
        "channels": [
            {
                "label": "DNA",
                "color": "3366FF",
                "active": False,
                "window": {"start": 1, "end": 10, "min": 0, "max": 255},
            },
        ],
    }
    group, _ = _make_image(zarr_format, ["channel", "space", "space"], (1, 6, 8), omero=omero)

    metadata = parse_ome_zarr_metadata(group)

    channel = metadata.omero.channels[0]
    assert channel.label == "DNA"
    assert channel.color == "3366FF"
    assert channel.active is False
    assert channel.window.end == 10


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_dataset_and_group_transforms_are_composed_and_converted(zarr_format):
    dataset_transforms = [
        {"type": "scale", "scale": [1, 2, 3]},
        {"type": "translation", "translation": [5, 6, 7]},
    ]
    group_transforms = [
        {"type": "scale", "scale": [1, 10, 10]},
        {"type": "translation", "translation": [0, 1, 2]},
    ]
    group, _ = _make_image(
        zarr_format,
        ["time", "space", "space"],
        (2, 6, 8),
        transforms=dataset_transforms,
        group_transforms=group_transforms,
    )

    metadata = parse_ome_zarr_metadata(group)
    dataset = metadata.multiscales.datasets[0]
    step, origin = spatial_transform_angstrom(metadata.multiscales, dataset)

    # Spatial units are nanometers, so values are multiplied by 10 Angstrom/nm.
    assert step == pytest.approx((200.0, 300.0))
    assert origin == pytest.approx((610.0, 720.0))


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_path_backed_transform_vectors(zarr_format):
    store = zarr.storage.MemoryStore()
    group = zarr.create_group(store=store, zarr_format=zarr_format)
    axes = _axes(["space", "space", "space"])
    _create_array(
        group,
        zarr_format,
        "0",
        np.zeros((4, 6, 8), dtype=np.uint8),
        (2, 3, 4),
        dimension_names=("z", "y", "x"),
    )
    _create_array(
        group,
        zarr_format,
        "vectors/scale",
        np.asarray([2.0, 3.0, 4.0]),
        (3,),
        dimension_names=("axis",),
    )
    multiscale = {
        "axes": axes,
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "path": "vectors/scale"}]}],
    }
    if zarr_format == 2:
        multiscale["version"] = "0.4"
        group.attrs["multiscales"] = [multiscale]
    else:
        group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale]}

    metadata = parse_ome_zarr_metadata(group)

    assert metadata.multiscales.datasets[0].scale == (2.0, 3.0, 4.0)


def test_2d_time_channel_slice_becomes_singleton_z_grid():
    group, data = _make_image(3, ["time", "channel", "space", "space"], (2, 3, 6, 8))
    array = group["0"]
    grid = ZarrGridSlice(
        array,
        fixed_indices=(1, 2),
        spatial_ndim=2,
        origin=(20.0, 30.0),
        step=(2.0, 3.0),
        time_index=1,
        channel_index=2,
    )

    matrix = grid.read_matrix((0, 0, 0), grid.size, (1, 1, 1))

    assert grid.size == (8, 6, 1)
    assert grid.origin == (30.0, 20.0, 0.0)
    assert grid.step == (3.0, 2.0, 1.0)
    assert grid.time == 1
    assert grid.channel == 2
    np.testing.assert_array_equal(matrix, data[1, 2][np.newaxis, :, :])


def test_3d_time_channel_slice_reads_requested_values():
    group, data = _make_image(3, ["time", "channel", "space", "space", "space"], (2, 3, 4, 6, 8))
    grid = ZarrGridSlice(group["0"], fixed_indices=(1, 2), spatial_ndim=3)

    matrix = grid.read_matrix((1, 1, 1), (4, 3, 2), (2, 1, 1))

    np.testing.assert_array_equal(matrix, data[1, 2, 1:3, 1:4, 1:5:2])


def test_v3_sharded_array_reads_through_grid():
    group, data = _make_image(
        3,
        ["space", "space", "space"],
        (8, 8, 8),
        shards=(8, 8, 8),
    )
    grid = ZarrGridSlice(group["0"], spatial_ndim=3)

    matrix = grid.read_matrix((1, 1, 1), (4, 3, 2), (2, 1, 1))

    np.testing.assert_array_equal(matrix, data[1:3, 1:4, 1:5:2])


def test_wrapped_grid_uses_aligned_coarse_level():
    coarse_group, _ = _make_image(3, ["space", "space", "space"], (2, 3, 4))
    fine_group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    coarse_group["0"][:] = 7
    fine_group["0"][:] = 9
    coarse = ZarrGridSlice(coarse_group["0"], spatial_ndim=3, step=(2.0, 2.0, 2.0))
    fine = ZarrGridSlice(fine_group["0"], spatial_ndim=3, step=(1.0, 1.0, 1.0))
    wrapped = WrappedZarrGrid(grids=[coarse, fine])

    coarse_matrix = wrapped.read_matrix((0, 0, 0), (8, 6, 4), (2, 2, 2))
    fine_matrix = wrapped.read_matrix((1, 1, 1), (4, 4, 2), (2, 2, 2))

    assert np.all(coarse_matrix == 7)
    assert np.all(fine_matrix == 9)


def test_wrapped_grid_rejects_noninteger_scale_and_misaligned_translation():
    group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    fine = ZarrGridSlice(group["0"], spatial_ndim=3)
    noninteger = ZarrGridSlice(group["0"], spatial_ndim=3, step=(1.5, 2.0, 2.0))
    translated = ZarrGridSlice(group["0"], spatial_ndim=3, step=(2.0, 2.0, 2.0), origin=(0.5, 0, 0))

    with pytest.raises(OMEZarrFormatError, match="Non-integer"):
        WrappedZarrGrid(grids=[noninteger, fine])
    with pytest.raises(OMEZarrFormatError, match="align"):
        WrappedZarrGrid(grids=[translated, fine])


def test_translated_levels_can_be_opened_explicitly_but_not_wrapped():
    from chimerax.core.session import Session

    group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    _create_array(
        group,
        3,
        "1",
        np.zeros((2, 3, 4), dtype=np.uint16),
        (2, 3, 4),
        dimension_names=("z", "y", "x"),
    )
    ome = deepcopy(group.attrs["ome"])
    ome["multiscales"][0]["datasets"].append(
        {
            "path": "1",
            "coordinateTransformations": [
                {"type": "scale", "scale": [2, 2, 2]},
                {"type": "translation", "translation": [0.5, 0, 0]},
            ],
        },
    )
    group.attrs["ome"] = ome
    session = Session("OME-Zarr translated scales test", offscreen_rendering=True)

    model = ZarrModel("explicit", session, group.store, scales=["0", "1"])
    session.models.add([model])
    volumes = list(model.child_models())

    assert [volume.data.scale_path for volume in volumes] == ["0", "1"]
    assert volumes[1].data.origin == pytest.approx((0.0, 0.0, 5.0))
    session.models.close([model])


def test_v3_dimension_names_must_match_axes():
    group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    ome = deepcopy(group.attrs["ome"])
    ome["multiscales"][0]["axes"][0]["name"] = "depth"
    group.attrs["ome"] = ome

    with pytest.raises(OMEZarrFormatError, match="dimension_names"):
        parse_ome_zarr_metadata(group)


@pytest.mark.parametrize("unsupported", ["labels", "multiple"])
def test_unsupported_image_structures_fail_clearly(unsupported):
    group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    if unsupported == "labels":
        group.create_group("labels")
        match = "label"
    else:
        ome = deepcopy(group.attrs["ome"])
        ome["multiscales"].append(deepcopy(ome["multiscales"][0]))
        group.attrs["ome"] = ome
        match = "Exactly one"

    with pytest.raises(OMEZarrFormatError, match=match):
        parse_ome_zarr_metadata(group)


def test_ome_and_zarr_versions_must_match():
    group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    ome = deepcopy(group.attrs["ome"])
    ome["version"] = "0.4"
    group.attrs["ome"] = ome

    with pytest.raises(OMEZarrFormatError, match="expected OME-Zarr 0.5"):
        parse_ome_zarr_metadata(group)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_zarr_model_builds_2d_time_channel_volumes_in_chimerax(zarr_format):
    from chimerax.core.session import Session

    group, _ = _make_image(zarr_format, ["time", "channel", "space", "space"], (2, 3, 6, 8))
    session = Session("OME-Zarr test", offscreen_rendering=True)

    model = ZarrModel("image", session, group.store)
    session.models.add([model])

    volumes = list(model.child_models())
    assert len(volumes) == 6
    assert all(volume.data.size == (8, 6, 1) for volume in volumes)
    assert sorted({volume.data.time for volume in volumes}) == [0, 1]
    assert sorted({volume.data.channel for volume in volumes}) == [0, 1, 2]
    session.models.close([model])


def test_explicit_scales_preserve_independent_time_channel_hierarchies():
    from chimerax.core.session import Session
    from chimerax.map.volume import MultiChannelSeries

    group, _ = _make_image(3, ["time", "channel", "space", "space"], (2, 3, 6, 8))
    _create_array(
        group,
        3,
        "1",
        np.zeros((2, 3, 3, 4), dtype=np.uint16),
        (2, 3, 3, 4),
        dimension_names=("t", "c", "y", "x"),
    )
    ome = deepcopy(group.attrs["ome"])
    ome["multiscales"][0]["datasets"].append(
        {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 2, 2]}]},
    )
    group.attrs["ome"] = ome
    session = Session("OME-Zarr explicit scales test", offscreen_rendering=True)

    models, _ = open_ome_zarr_from_store(session, group.store, "image", scales=["0", "1"])
    session.models.add(models)
    scale_models = list(models[0].child_models())

    assert len(scale_models) == 2
    assert all(isinstance(model, MultiChannelSeries) for model in scale_models)
    assert [model.name for model in scale_models] == ["image - 0", "image - 1"]
    assert all(len(model.map_series) == 3 for model in scale_models)
    assert all(len(series.maps) == 2 for model in scale_models for series in model.map_series)
    session.models.close(models)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_fsspec_entry_point_opens_both_zarr_formats(zarr_format):
    from chimerax.core.session import Session

    filesystem = fsspec.filesystem("memory")
    path = f"/image-{zarr_format}.zarr"
    write_store = zarr.storage.FsspecStore.from_mapper(filesystem.get_mapper(path))
    _make_image(zarr_format, ["space", "space", "space"], (4, 6, 8), store=write_store)
    session = Session("OME-Zarr fsspec test", offscreen_rendering=True)

    models, message = open_ome_zarr_from_fs(session, filesystem, path, log=False)
    session.models.add(models)

    assert len(models) == 1
    assert message == f"Opened {path}."
    assert list(models[0].child_models())[0].data.size == (8, 6, 4)
    session.models.close(models)
