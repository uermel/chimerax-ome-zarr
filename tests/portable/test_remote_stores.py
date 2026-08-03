"""Required smoke tests for the public EBI OME-Zarr reference stores."""

import fsspec
import pytest
import zarr

from src.map_data.ome_metadata import (
    bioformats2raw_series_paths,
    ome_zarr_group_kind,
    parse_labels_metadata,
    parse_ome_zarr_metadata,
)

V04_IMAGE = "https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"
V05_IMAGE_WITH_LABELS = "https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr"
V05_BIOFORMATS2RAW = (
    "https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0051/"
    "180712_H2B_22ss_Courtney1_20180712-163837_p00_c00_preview.zarr"
)

pytestmark = pytest.mark.remote


def _open_http_group(url):
    # HTTPFileSystem verifies TLS certificates by default. FsspecStore lets Zarr
    # fetch only metadata and the chunks explicitly indexed by these tests.
    filesystem = fsspec.filesystem("http")
    mapper = filesystem.get_mapper(url)
    store = zarr.storage.FsspecStore.from_mapper(mapper, read_only=True)
    return zarr.open_group(store=store, mode="r")


def _assert_reference_image(group, zarr_format, ome_version):
    metadata = parse_ome_zarr_metadata(group)

    assert metadata.zarr_format == zarr_format
    assert metadata.ome_version == ome_version
    assert [axis.name for axis in metadata.multiscales.axes] == ["c", "z", "y", "x"]
    assert [axis.type for axis in metadata.multiscales.axes] == ["channel", "space", "space", "space"]
    assert [group[dataset.path].shape for dataset in metadata.multiscales.datasets] == [
        (2, 236, 275, 271),
        (2, 236, 137, 135),
        (2, 236, 68, 67),
    ]

    coarsest = group[metadata.multiscales.datasets[-1].path]
    assert int(coarsest[0, 0, 0, 0]) == 28


def test_remote_v04_image_metadata_and_tiny_read():
    _assert_reference_image(_open_http_group(V04_IMAGE), zarr_format=2, ome_version="0.4")


def test_remote_v05_image_and_labels_metadata_and_tiny_reads():
    root = _open_http_group(V05_IMAGE_WITH_LABELS)
    _assert_reference_image(root, zarr_format=3, ome_version="0.5")

    labels = parse_labels_metadata(root["labels"])
    label_group = root[f"labels/{labels.paths[0]}"]
    label_metadata = parse_ome_zarr_metadata(label_group)

    assert labels.paths == ("0",)
    assert label_metadata.image_label.source_image == "../.."
    assert [value.value for value in label_metadata.image_label.values] == list(range(1, 62))
    assert [label_group[dataset.path].shape for dataset in label_metadata.multiscales.datasets] == [
        (1, 236, 275, 271),
        (1, 236, 137, 135),
        (1, 236, 68, 67),
    ]

    coarsest = label_group[label_metadata.multiscales.datasets[-1].path]
    assert int(coarsest[0, 0, 0, 0]) == 0


def test_remote_v05_bioformats2raw_collection_discovers_image():
    root = _open_http_group(V05_BIOFORMATS2RAW)

    assert ome_zarr_group_kind(root) == "bioformats2raw"
    assert bioformats2raw_series_paths(root) == ("0",)

    image = root["0"]
    metadata = parse_ome_zarr_metadata(image)
    assert [axis.name for axis in metadata.multiscales.axes] == ["t", "c", "z", "y", "x"]
    assert [image[dataset.path].shape for dataset in metadata.multiscales.datasets] == [
        (79, 1, 201, 333, 333),
        (79, 1, 201, 166, 166),
    ]
