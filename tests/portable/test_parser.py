"""Portable coverage of the supported OME-Zarr parser surface."""

import numpy as np
import pytest
import zarr

from src.map_data.ome_metadata import (
    bioformats2raw_series_paths,
    ome_zarr_group_kind,
    parse_ome_zarr_metadata,
)


def _make_tczyx_image(zarr_format, *, store=None, path=None):
    if store is None:
        store = zarr.storage.MemoryStore()
    group = zarr.create_group(store=store, path=path, zarr_format=zarr_format)
    axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "nanometer"},
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    kwargs = {
        "name": "0",
        "shape": (2, 3, 4, 5, 6),
        "chunks": (1, 1, 2, 3, 3),
        "dtype": np.uint16,
    }
    if zarr_format == 3:
        kwargs["dimension_names"] = ("t", "c", "z", "y", "x")
    group.create_array(**kwargs)
    multiscale = {
        "axes": axes,
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "identity"}]}],
    }
    if zarr_format == 2:
        multiscale["version"] = "0.4"
        group.attrs["multiscales"] = [multiscale]
    else:
        group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale]}
    return group


@pytest.mark.parametrize(("zarr_format", "ome_version"), [(2, "0.4"), (3, "0.5")])
def test_parser_supports_time_channel_3d_and_identity(zarr_format, ome_version):
    metadata = parse_ome_zarr_metadata(_make_tczyx_image(zarr_format))

    assert metadata.zarr_format == zarr_format
    assert metadata.ome_version == ome_version
    assert [axis.name for axis in metadata.multiscales.axes] == ["t", "c", "z", "y", "x"]
    assert [axis.type for axis in metadata.multiscales.axes] == ["time", "channel", "space", "space", "space"]
    assert metadata.multiscales.datasets[0].scale == (1.0,) * 5
    assert metadata.multiscales.datasets[0].translation == (0.0,) * 5


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("explicit_series", [False, True])
def test_bioformats2raw_series_discovery(zarr_format, explicit_series):
    store = zarr.storage.MemoryStore()
    root = zarr.create_group(store=store, zarr_format=zarr_format)
    series_paths = ["images/first", "images/second"] if explicit_series else ["0", "1"]
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
        _make_tczyx_image(zarr_format, store=store, path=path)

    assert ome_zarr_group_kind(root) == "bioformats2raw"
    assert bioformats2raw_series_paths(root) == tuple(series_paths)
