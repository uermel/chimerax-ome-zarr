"""Portable coverage of the supported OME-Zarr parser surface."""

import numpy as np
import pytest
import zarr

from src.map_data.ome_metadata import parse_ome_zarr_metadata


def _make_tczyx_image(zarr_format):
    group = zarr.create_group(store=zarr.storage.MemoryStore(), zarr_format=zarr_format)
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
