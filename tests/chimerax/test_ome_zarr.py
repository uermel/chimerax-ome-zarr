"""Native ChimeraX integration and reader tests."""

from copy import deepcopy

import fsspec
import numpy as np
import pytest
import zarr

from src.info import NGFFFetcherInfo, OMEZarrOpenerInfo
from src.map_data.ome_metadata import (
    OMEZarrFormatError,
    bioformats2raw_series_paths,
    ome_zarr_group_kind,
    parse_labels_metadata,
    parse_ome_zarr_metadata,
    spatial_transform_angstrom,
)
from src.map_data.temporal_cache import TemporalReadAheadManager
from src.map_data.zarr_grid import WrappedZarrGrid, ZarrGridSlice, ZarrModel
from src.open import open_ome_zarr_from_fs, open_ome_zarr_from_store

pytestmark = pytest.mark.chimerax

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
    path=None,
):
    if store is None:
        store = zarr.storage.MemoryStore()
    group = zarr.create_group(store=store, path=path, zarr_format=zarr_format)
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


def _make_bioformats2raw_collection(zarr_format, series_paths, *, explicit_series):
    store = zarr.storage.MemoryStore()
    root = zarr.create_group(store=store, zarr_format=zarr_format)
    if zarr_format == 2:
        root.attrs["bioformats2raw.layout"] = 3
    else:
        root.attrs["ome"] = {"version": "0.5", "bioformats2raw.layout": 3}

    if explicit_series:
        ome_group = root.create_group("OME")
        if zarr_format == 2:
            ome_group.attrs["series"] = series_paths
        else:
            ome_group.attrs["ome"] = {"version": "0.5", "series": series_paths}

    for path in series_paths:
        _make_image(zarr_format, ["space", "space", "space"], (4, 6, 8), store=store, path=path)
    return root


def _add_label_image(source_group, zarr_format, axis_types, data, *, colors=None, properties=None, name="cells"):
    labels_group = source_group.create_group("labels")
    if zarr_format == 2:
        labels_group.attrs["labels"] = [name]
    else:
        labels_group.attrs["ome"] = {"version": "0.5", "labels": [name]}

    label_group = labels_group.create_group(name)
    axes = _axes(axis_types)
    dimension_names = tuple(axis["name"] for axis in axes)
    _create_array(
        label_group,
        zarr_format,
        "0",
        data,
        tuple(max(1, min(4, size)) for size in data.shape),
        dimension_names=dimension_names,
    )
    multiscale = {
        "name": name,
        "axes": axes,
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1] * data.ndim}]}],
    }
    image_label = {
        "version": "0.4" if zarr_format == 2 else "0.5",
        "source": {"image": "../../"},
        "colors": colors or [],
        "properties": properties or [],
    }
    if zarr_format == 2:
        multiscale["version"] = "0.4"
        label_group.attrs.update({"multiscales": [multiscale], "image-label": image_label})
    else:
        label_group.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [multiscale],
            "image-label": image_label,
        }
    return label_group


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
def test_identity_transform_is_noop_at_group_and_dataset_levels(zarr_format):
    group_identity, _ = _make_image(
        zarr_format,
        ["time", "space", "space"],
        (2, 6, 8),
        transforms=[{"type": "scale", "scale": [1, 2, 3]}],
        group_transforms=[{"type": "identity"}],
    )
    identity_dataset, _ = _make_image(
        zarr_format,
        ["time", "space", "space"],
        (2, 6, 8),
        transforms=[{"type": "identity"}],
    )

    scaled = parse_ome_zarr_metadata(group_identity).multiscales.datasets[0]
    unchanged = parse_ome_zarr_metadata(identity_dataset).multiscales.datasets[0]

    assert scaled.scale == pytest.approx((1.0, 2.0, 3.0))
    assert scaled.translation == pytest.approx((0.0, 0.0, 0.0))
    assert unchanged.scale == pytest.approx((1.0, 1.0, 1.0))
    assert unchanged.translation == pytest.approx((0.0, 0.0, 0.0))


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


def test_wrapped_grid_keeps_chimerax_caches_separate_by_scale():
    from chimerax.map_data.datacache import Data_Cache

    coarse_group, _ = _make_image(3, ["space", "space", "space"], (2, 3, 4))
    fine_group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    coarse_group["0"][:] = 7
    fine_group["0"][:] = 9
    coarse = ZarrGridSlice(coarse_group["0"], spatial_ndim=3, step=(2.0, 2.0, 2.0))
    fine = ZarrGridSlice(fine_group["0"], spatial_ndim=3, step=(1.0, 1.0, 1.0))
    wrapped = WrappedZarrGrid(grids=[coarse, fine])
    cache = Data_Cache(1024**2)
    wrapped.data_cache = cache

    fine_matrix = wrapped.matrix((0, 0, 0), (8, 6, 4), (1, 1, 1))
    coarse_matrix = wrapped.matrix((0, 0, 0), (8, 6, 4), (2, 2, 2))

    assert coarse.data_cache is fine.data_cache is cache
    assert np.all(fine_matrix == 9)
    assert np.all(coarse_matrix == 7)


def test_plane_reads_reuse_chunk_aligned_decoded_slab():
    from chimerax.map_data.datacache import Data_Cache

    group, data = _make_image(3, ["space", "space", "space"], (8, 8, 8))

    class CountingGrid(ZarrGridSlice):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reads = []

        def read_matrix(self, ijk_origin=(0, 0, 0), ijk_size=None, ijk_step=(1, 1, 1), progress=None):
            self.reads.append((ijk_origin, ijk_size, ijk_step))
            return super().read_matrix(ijk_origin, ijk_size, ijk_step, progress)

    grid = CountingGrid(group["0"], spatial_ndim=3)
    grid.data_cache = Data_Cache(1024**2)

    first = grid.matrix((0, 0, 1), (8, 8, 1), (1, 1, 1))
    adjacent = grid.matrix((0, 0, 2), (8, 8, 1), (1, 1, 1))
    resampled = grid.matrix((0, 0, 3), (8, 8, 1), (2, 2, 2))

    np.testing.assert_array_equal(first, data[1:2])
    np.testing.assert_array_equal(adjacent, data[2:3])
    np.testing.assert_array_equal(resampled, data[3:4:2, ::2, ::2])
    assert grid.reads == [((0, 0, 0), (8, 8, 4), (1, 1, 1))]

    crossing = grid.matrix((0, 0, 4), (8, 8, 1), (1, 1, 1))
    np.testing.assert_array_equal(crossing, data[4:5])
    assert grid.reads[-1] == ((0, 0, 4), (8, 8, 4), (1, 1, 1))
    assert len(grid.reads) == 2


def test_plane_read_ahead_respects_decoded_cache_limit():
    from chimerax.map_data.datacache import Data_Cache

    group, data = _make_image(3, ["space", "space", "space"], (8, 8, 8))

    class CountingGrid(ZarrGridSlice):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reads = []

        def read_matrix(self, ijk_origin=(0, 0, 0), ijk_size=None, ijk_step=(1, 1, 1), progress=None):
            self.reads.append((ijk_origin, ijk_size, ijk_step))
            return super().read_matrix(ijk_origin, ijk_size, ijk_step, progress)

    grid = CountingGrid(group["0"], spatial_ndim=3)
    grid.data_cache = Data_Cache(32)

    matrix = grid.matrix((0, 0, 1), (8, 8, 1), (1, 1, 1))

    np.testing.assert_array_equal(matrix, data[1:2])
    assert grid.reads == [((0, 0, 1), (8, 8, 1), (1, 1, 1))]


def test_plane_read_ahead_clamps_step_aligned_plane_past_upper_boundary():
    """ChimeraX can align a one-plane region to size when size is not step-aligned."""

    from chimerax.map_data.datacache import Data_Cache

    coarse_array = zarr.create_array(
        store=zarr.storage.MemoryStore(),
        shape=(125, 232, 232),
        chunks=(256, 256, 256),
        dtype=np.float32,
        zarr_format=2,
    )
    fine_array = zarr.create_array(
        store=zarr.storage.MemoryStore(),
        shape=(500, 928, 928),
        chunks=(256, 256, 256),
        dtype=np.float32,
        zarr_format=2,
    )
    coarse = ZarrGridSlice(coarse_array, spatial_ndim=3, step=(4.0, 4.0, 4.0))
    fine = ZarrGridSlice(fine_array, spatial_ndim=3)
    wrapped = WrappedZarrGrid(grids=[coarse, fine])
    wrapped.data_cache = Data_Cache(1024**3)

    # For size 500 at step 4, ChimeraX's _step_aligned_region maps z=499 to
    # z=500. The coarse-level request is consequently z=125, one past its end.
    matrix = wrapped.matrix((0, 0, 500), (928, 928, 1), (4, 4, 4))

    assert matrix.shape == (1, 232, 232)


def test_temporal_read_ahead_promotes_decoded_matrix_to_chimerax_cache():
    from chimerax.map_data.datacache import Data_Cache

    group, data = _make_image(3, ["time", "space", "space", "space"], (3, 4, 6, 8))

    class CountingGrid(ZarrGridSlice):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.read_count = 0

        def read_matrix(self, ijk_origin=(0, 0, 0), ijk_size=None, ijk_step=(1, 1, 1), progress=None):
            self.read_count += 1
            return super().read_matrix(ijk_origin, ijk_size, ijk_step, progress)

    grids = [CountingGrid(group["0"], fixed_indices=(time,), spatial_ndim=3, time_index=time) for time in range(3)]
    cache = Data_Cache(1024**2)
    for grid in grids:
        grid.data_cache = cache
    manager = TemporalReadAheadManager(1024**2)
    sequence = manager.create_sequence(grids)
    for index, grid in enumerate(grids):
        grid.set_temporal_read_ahead(sequence, index)

    current = grids[0].matrix((0, 0, 0), grids[0].size, (1, 1, 1))
    manager.flush()

    assert manager.wait_for_idle()
    assert grids[1].read_count == 1
    assert grids[1].cached_data((0, 0, 0), grids[1].size, (1, 1, 1)) is None
    prefetched = grids[1].matrix((0, 0, 0), grids[1].size, (1, 1, 1))
    np.testing.assert_array_equal(current, data[0])
    np.testing.assert_array_equal(prefetched, data[1])
    assert grids[1].read_count == 1
    assert grids[1].cached_data((0, 0, 0), grids[1].size, (1, 1, 1)) is prefetched
    manager.close(wait=True)


def test_temporal_read_ahead_ignores_planes_and_cache_only_probes():
    from chimerax.map_data.datacache import Data_Cache

    group, _ = _make_image(3, ["time", "space", "space", "space"], (2, 4, 6, 8))

    class CountingGrid(ZarrGridSlice):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.read_count = 0

        def read_matrix(self, ijk_origin=(0, 0, 0), ijk_size=None, ijk_step=(1, 1, 1), progress=None):
            self.read_count += 1
            return super().read_matrix(ijk_origin, ijk_size, ijk_step, progress)

    grids = [CountingGrid(group["0"], fixed_indices=(time,), spatial_ndim=3) for time in range(2)]
    manager = TemporalReadAheadManager(1024**2)
    sequence = manager.create_sequence(grids)
    for index, grid in enumerate(grids):
        grid.data_cache = Data_Cache(1024**2)
        grid.set_temporal_read_ahead(sequence, index)

    grids[0].matrix((0, 0, 1), (8, 6, 1), (1, 1, 1))
    assert grids[0].matrix((0, 0, 0), grids[0].size, (1, 1, 1), from_cache_only=True) is not None
    manager.flush()

    assert manager.wait_for_idle()
    assert grids[1].read_count == 0
    assert manager.reserved_bytes == 0
    manager.close(wait=True)


def test_zarr_model_configures_adaptive_or_disabled_temporal_read_ahead():
    from chimerax.core.session import Session

    group, _ = _make_image(3, ["time", "channel", "space", "space", "space"], (3, 2, 4, 6, 8))
    adaptive_session = Session("OME-Zarr adaptive read-ahead test", offscreen_rendering=True)
    adaptive = ZarrModel("adaptive", adaptive_session, group.store)
    adaptive_levels = [level for volume in adaptive.child_models() for level in volume.data.grids]

    assert len(adaptive_levels) == 6
    assert all(level._temporal_read_ahead is not None for level in adaptive_levels)
    assert hasattr(adaptive_session, "_ome_zarr_temporal_read_ahead")
    current = next(level for level in adaptive_levels if level.time == 0 and level.channel == 0)
    following = next(level for level in adaptive_levels if level.time == 1 and level.channel == 0)
    current.matrix((0, 0, 0), current.size, (1, 1, 1))
    adaptive_session.triggers.activate_trigger("frame drawn", None)
    assert adaptive_session._ome_zarr_temporal_read_ahead.wait_for_idle()
    assert adaptive_session._ome_zarr_temporal_read_ahead.ready_count == 2
    assert following.matrix((0, 0, 0), following.size, (1, 1, 1)) is not None
    assert adaptive_session._ome_zarr_temporal_read_ahead.ready_count == 1

    disabled_session = Session("OME-Zarr disabled read-ahead test", offscreen_rendering=True)
    disabled = ZarrModel("disabled", disabled_session, group.store, read_ahead=0)
    disabled_levels = [level for volume in disabled.child_models() for level in volume.data.grids]

    assert all(level._temporal_read_ahead is None for level in disabled_levels)
    assert not hasattr(disabled_session, "_ome_zarr_temporal_read_ahead")
    with pytest.raises(OMEZarrFormatError, match="nonnegative"):
        ZarrModel("invalid", disabled_session, group.store, read_ahead=-1)

    adaptive_session._ome_zarr_temporal_read_ahead.close(wait=True)


def test_open_and_fetch_providers_expose_nonnegative_read_ahead_option():
    from chimerax.core.commands import NonNegativeIntArg

    assert OMEZarrOpenerInfo().open_args["read_ahead"] is NonNegativeIntArg
    assert NGFFFetcherInfo().fetch_args["read_ahead"] is NonNegativeIntArg


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


def test_image_group_with_labels_child_remains_a_supported_image():
    group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    group.create_group("labels")

    metadata = parse_ome_zarr_metadata(group)

    assert metadata.image_label is None
    assert ome_zarr_group_kind(group) == "image"


def test_multiple_multiscales_entries_still_fail_clearly():
    group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    ome = deepcopy(group.attrs["ome"])
    ome["multiscales"].append(deepcopy(ome["multiscales"][0]))
    group.attrs["ome"] = ome

    with pytest.raises(OMEZarrFormatError, match="Exactly one"):
        parse_ome_zarr_metadata(group)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_label_collection_and_image_label_metadata_are_normalized(zarr_format):
    group, _ = _make_image(zarr_format, ["space", "space", "space"], (4, 6, 8))
    label_group = _add_label_image(
        group,
        zarr_format,
        ["space", "space", "space"],
        np.zeros((4, 6, 8), dtype=np.uint16),
        colors=[{"label-value": 3, "rgba": [10, 20, 30, 128]}],
        properties=[{"label-value": 3, "class": "nucleus", "score": 7}],
    )

    labels = parse_labels_metadata(group["labels"])
    metadata = parse_ome_zarr_metadata(label_group)

    assert labels.paths == ("cells",)
    assert ome_zarr_group_kind(label_group) == "image-label"
    assert metadata.image_label.source_image == "../../"
    assert metadata.image_label.values[0].value == 3
    assert metadata.image_label.values[0].rgba == (10, 20, 30, 128)
    assert metadata.image_label.values[0].properties == {"class": "nucleus", "score": 7}


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_label_arrays_must_be_integer(zarr_format):
    group, _ = _make_image(zarr_format, ["space", "space", "space"], (4, 6, 8))
    label_group = _add_label_image(
        group,
        zarr_format,
        ["space", "space", "space"],
        np.zeros((4, 6, 8), dtype=np.float32),
        colors=[{"label-value": 1, "rgba": [255, 0, 0, 255]}],
    )

    with pytest.raises(OMEZarrFormatError, match="integer"):
        parse_ome_zarr_metadata(label_group)


def test_ome_and_zarr_versions_must_match():
    group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    ome = deepcopy(group.attrs["ome"])
    ome["version"] = "0.4"
    group.attrs["ome"] = ome

    with pytest.raises(OMEZarrFormatError, match="expected OME-Zarr 0.5"):
        parse_ome_zarr_metadata(group)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("explicit_series", [False, True])
def test_bioformats2raw_collection_discovers_and_opens_every_series(zarr_format, explicit_series):
    from chimerax.core.session import Session
    from chimerax.map.volume import Volume

    series_paths = ["images/first", "images/second"] if explicit_series else ["0", "1"]
    root = _make_bioformats2raw_collection(zarr_format, series_paths, explicit_series=explicit_series)
    session = Session("OME-Zarr bioformats2raw collection test", offscreen_rendering=True)

    assert ome_zarr_group_kind(root) == "bioformats2raw"
    assert bioformats2raw_series_paths(root) == tuple(series_paths)

    models, message = open_ome_zarr_from_store(session, root, "collection")
    session.models.add(models)
    collection = models[0]
    volumes = [model for model in collection.all_models() if isinstance(model, Volume)]

    assert message == "Opened collection."
    assert [model.name for model in collection.child_models()] == [f"collection - {path}" for path in series_paths]
    assert len(volumes) == 2
    assert all(volume.data.size == (8, 6, 4) for volume in volumes)
    session.models.close(models)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_bioformats2raw_collection_rejects_missing_declared_series(zarr_format):
    root = _make_bioformats2raw_collection(zarr_format, ["0"], explicit_series=True)
    ome_group = root["OME"]
    if zarr_format == 2:
        ome_group.attrs["series"] = ["missing"]
    else:
        ome_group.attrs["ome"] = {"version": "0.5", "series": ["missing"]}

    with pytest.raises(OMEZarrFormatError, match="series path 'missing' does not exist"):
        bioformats2raw_series_paths(root)


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
    assert all(volume.data.data_cache is not None for volume in volumes)
    assert all(level.data_cache is volume.data.data_cache for volume in volumes for level in volume.data.grids)
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
def test_associated_labels_open_as_index_map_and_native_segmentations(zarr_format):
    from chimerax.core.session import Session
    from chimerax.map.volume import Volume
    from chimerax.segmentations import Segmentation
    from chimerax.segmentations.segmentation_tracker import get_tracker, register_model_trigger_handlers

    group, _ = _make_image(zarr_format, ["space", "space", "space"], (4, 6, 8))
    label_data = np.zeros((4, 6, 8), dtype=np.uint16)
    label_data[0, 0, 0] = 1
    label_data[0, 0, 1] = 2
    label_group = _add_label_image(
        group,
        zarr_format,
        ["space", "space", "space"],
        label_data,
        colors=[
            {"label-value": 1, "rgba": [255, 0, 0, 255]},
            {"label-value": 2, "rgba": [0, 255, 0, 128]},
        ],
        properties=[{"label-value": 1, "class": "nucleus"}, {"label-value": 2, "name": "cell"}],
    )
    session = Session("OME-Zarr labels integration test", offscreen_rendering=True)
    register_model_trigger_handlers(session)

    models, _ = open_ome_zarr_from_store(session, group.store, "image", labels=True)
    session.models.add(models)
    all_models = models[0].all_models()
    segmentations = [model for model in all_models if isinstance(model, Segmentation)]
    index_maps = [
        model
        for model in all_models
        if isinstance(model, Volume) and getattr(model.data, "file_type", None) == "ome-zarr-label"
    ]

    assert len(index_maps) == 1
    assert index_maps[0].display is False
    assert index_maps[0].data.data_cache is not None
    assert sorted(segmentation.ome_label_value for segmentation in segmentations) == [1, 2]
    assert all(segmentation.data.data_cache is not None for segmentation in segmentations)
    assert all(segmentation.reference_volume is not None for segmentation in segmentations)
    reference = segmentations[0].reference_volume
    assert set(segmentations) == get_tracker().segmentations_for_volume(reference)
    assert all(segmentation.display is False for segmentation in segmentations)
    assert next(segmentation for segmentation in segmentations if segmentation.ome_label_value == 2).default_rgba == (
        0.0,
        1.0,
        0.0,
        128 / 255,
    )

    class AddVoxel:
        def execute(self, grid, reference_grid):
            del reference_grid
            grid.array[0, 0, 0] = 1

    label_two = next(segmentation for segmentation in segmentations if segmentation.ome_label_value == 2)
    label_one = next(segmentation for segmentation in segmentations if segmentation.ome_label_value == 1)
    _ = index_maps[0].data.matrix((0, 0, 0), (2, 1, 1), (1, 1, 1))
    _ = label_one.data.matrix((0, 0, 0), (2, 1, 1), (1, 1, 1))
    label_two.segment(AddVoxel())

    assert index_maps[0].data.matrix((0, 0, 0), (2, 1, 1), (1, 1, 1)).tolist() == [[[2, 2]]]
    assert label_one.data.matrix((0, 0, 0), (2, 1, 1), (1, 1, 1)).tolist() == [[[0, 0]]]
    np.testing.assert_array_equal(label_group["0"][:], label_data)
    session.models.close(models)


def test_singleton_channel_labels_broadcast_and_share_edits():
    from chimerax.core.session import Session
    from chimerax.segmentations import Segmentation

    group, _ = _make_image(3, ["channel", "space", "space", "space"], (2, 4, 6, 8))
    label_data = np.zeros((1, 4, 6, 8), dtype=np.uint8)
    label_data[0, 0, 0, 0] = 5
    _add_label_image(
        group,
        3,
        ["channel", "space", "space", "space"],
        label_data,
        colors=[{"label-value": 5, "rgba": [10, 20, 30, 255]}],
    )
    session = Session("OME-Zarr label broadcasting test", offscreen_rendering=True)

    models, _ = open_ome_zarr_from_store(session, group.store, "image", labels=True)
    session.models.add(models)
    segmentations = [model for model in models[0].all_models() if isinstance(model, Segmentation)]

    assert len(segmentations) == 2
    assert sorted(segmentation.reference_volume.data.channel for segmentation in segmentations) == [0, 1]

    class RemoveVoxel:
        def execute(self, grid, reference_grid):
            del reference_grid
            grid.array[0, 0, 0] = 0

    segmentations[0].segment(RemoveVoxel())
    other = segmentations[1]
    assert other.data.read_matrix((0, 0, 0), (1, 1, 1), (1, 1, 1)).item() == 0
    session.models.close(models)


def test_label_pyramid_stays_lazy_until_first_edit():
    from chimerax.core.session import Session
    from chimerax.map.volume import Volume
    from chimerax.segmentations import Segmentation

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
        {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [2, 2, 2]}]},
    )
    group.attrs["ome"] = ome
    label_group = _add_label_image(
        group,
        3,
        ["space", "space", "space"],
        np.full((4, 6, 8), 7, dtype=np.uint8),
        colors=[{"label-value": 7, "rgba": [255, 0, 0, 255]}],
    )
    _create_array(
        label_group,
        3,
        "1",
        np.full((2, 3, 4), 7, dtype=np.uint8),
        (2, 3, 4),
        dimension_names=("z", "y", "x"),
    )
    label_ome = deepcopy(label_group.attrs["ome"])
    label_ome["multiscales"][0]["datasets"].append(
        {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [2, 2, 2]}]},
    )
    label_group.attrs["ome"] = label_ome
    session = Session("OME-Zarr lazy label pyramid test", offscreen_rendering=True)

    models, _ = open_ome_zarr_from_store(session, group.store, "image", labels=True)
    session.models.add(models)
    index_map = next(
        model
        for model in models[0].all_models()
        if isinstance(model, Volume) and getattr(model.data, "file_type", None) == "ome-zarr-label"
    )
    segmentation = next(model for model in models[0].all_models() if isinstance(model, Segmentation))

    assert index_map.data.state.materialized is False
    assert np.all(index_map.data.read_matrix((0, 0, 0), index_map.data.size, (2, 2, 2)) == 7)
    assert index_map.data.state.materialized is False
    _ = segmentation.data.array
    assert index_map.data.state.materialized is True
    session.models.close(models)


def test_undeclared_label_values_open_index_map_without_scanning():
    from chimerax.core.session import Session
    from chimerax.map.volume import Volume
    from chimerax.segmentations import Segmentation

    group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    _add_label_image(
        group,
        3,
        ["space", "space", "space"],
        np.full((4, 6, 8), 9, dtype=np.uint8),
    )
    session = Session("OME-Zarr undeclared labels test", offscreen_rendering=True)

    models, _ = open_ome_zarr_from_store(session, group.store, "image", labels=True)
    session.models.add(models)
    all_models = models[0].all_models()
    index_map = next(
        model
        for model in all_models
        if isinstance(model, Volume) and getattr(model.data, "file_type", None) == "ome-zarr-label"
    )

    assert not any(isinstance(model, Segmentation) for model in all_models)
    assert index_map.data.state.materialized is False
    session.models.close(models)


def test_associated_labels_are_opt_in():
    from chimerax.core.session import Session
    from chimerax.map.volume import Volume
    from chimerax.segmentations import Segmentation

    group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8))
    _add_label_image(
        group,
        3,
        ["space", "space", "space"],
        np.ones((4, 6, 8), dtype=np.uint8),
        colors=[{"label-value": 1, "rgba": [255, 0, 0, 255]}],
    )
    session = Session("OME-Zarr labels opt-in test", offscreen_rendering=True)

    models, _ = open_ome_zarr_from_store(session, group.store, "image")
    session.models.add(models)

    all_models = models[0].all_models()
    assert not any(isinstance(model, Segmentation) for model in all_models)
    assert not any(
        isinstance(model, Volume) and getattr(model.data, "file_type", None) == "ome-zarr-label" for model in all_models
    )
    session.models.close(models)


def test_direct_image_label_open_resolves_its_source():
    from chimerax.core.session import Session
    from chimerax.segmentations import Segmentation

    filesystem = fsspec.filesystem("memory")
    source_path = "/direct-source.zarr"
    write_store = zarr.storage.FsspecStore.from_mapper(filesystem.get_mapper(source_path))
    group, _ = _make_image(3, ["space", "space", "space"], (4, 6, 8), store=write_store)
    _add_label_image(
        group,
        3,
        ["space", "space", "space"],
        np.ones((4, 6, 8), dtype=np.uint8),
        colors=[{"label-value": 1, "rgba": [255, 0, 0, 255]}],
    )
    session = Session("OME-Zarr direct label test", offscreen_rendering=True)

    models, _ = open_ome_zarr_from_fs(
        session,
        filesystem,
        f"{source_path}/labels/cells",
        log=False,
    )
    session.models.add(models)

    segmentations = [model for model in models[0].all_models() if isinstance(model, Segmentation)]
    assert len(segmentations) == 1
    assert segmentations[0].reference_volume in models[0].all_models()
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
