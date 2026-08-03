"""Open the pinned, complete stores from ome-zarr-examples."""

from pathlib import Path

import pytest
import zarr

from src.map_data.ome_metadata import OMEZarrFormatError, parse_ome_zarr_metadata


@pytest.mark.upstream
@pytest.mark.parametrize(
    ("name", "levels"),
    [("image-01.zarr", 1), ("image-02.zarr", 1), ("image-03.zarr", 1), ("image-04.zarr", 2)],
)
def test_valid_complete_image_stores(examples_root, name, levels):
    group = zarr.open_group(str(Path(examples_root) / "data" / "valid" / name), mode="r")

    metadata = parse_ome_zarr_metadata(group)

    assert metadata.zarr_format == 2
    assert metadata.ome_version == "0.4"
    assert len(metadata.multiscales.datasets) == levels
    coarsest = group[metadata.multiscales.datasets[-1].path]
    assert coarsest[tuple(0 for _ in range(coarsest.ndim))] is not None


@pytest.mark.upstream
@pytest.mark.parametrize("name", [f"image-{index:02}.zarr" for index in range(1, 5)])
def test_invalid_complete_image_stores_are_rejected(examples_root, name):
    group = zarr.open_group(str(Path(examples_root) / "data" / "invalid" / name), mode="r")

    with pytest.raises(OMEZarrFormatError):
        parse_ome_zarr_metadata(group)


@pytest.mark.upstream
def test_warning_store_remains_readable(examples_root):
    group = zarr.open_group(str(Path(examples_root) / "data" / "warning" / "image-01.zarr"), mode="r")

    metadata = parse_ome_zarr_metadata(group)

    assert len(metadata.multiscales.datasets) == 2


@pytest.mark.upstream
def test_plate_store_reports_the_supported_boundary(examples_root):
    group = zarr.open_group(str(Path(examples_root) / "data" / "valid" / "plate-01.zarr"), mode="r")

    with pytest.raises(OMEZarrFormatError, match="plate"):
        parse_ome_zarr_metadata(group)
