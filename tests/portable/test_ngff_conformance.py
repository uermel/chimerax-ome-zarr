"""Exercise the reader against the pinned OME-NGFF JSON test suites."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import zarr

from src.map_data.ome_metadata import OMEZarrFormatError, parse_ome_zarr_metadata, spatial_transform_angstrom

SUITES = ("image", "strict_image", "label", "strict_label")
EXPECTED_COUNTS = {
    ("0.4", "image"): 30,
    ("0.4", "strict_image"): 5,
    ("0.4", "label"): 9,
    ("0.4", "strict_label"): 2,
    ("0.5", "image"): 28,
    ("0.5", "strict_image"): 5,
    ("0.5", "label"): 9,
    ("0.5", "strict_label"): 1,
}
EXPECTED_SUITE_SHA256 = {
    ("0.4", "image"): "58b5198b378a0aa54bea6ec98cd1b899adf5ea58cc4c74435d41161501da6bd7",
    ("0.4", "strict_image"): "7fd7fa567587f63c621b6eb1707ddaa08eb5dac028c55776855926399b184c67",
    ("0.4", "label"): "bdd1fb7065b85da254b3dc9743dff43e8330b2d820abb849d32151ebd37927ce",
    ("0.4", "strict_label"): "effdde7b03dc6baa7d9c3f38e7ffe26b006095ecf2470226c39168ccbe1f80c0",
    ("0.5", "image"): "6bc52b4ec57cb83442fdd242c5513cff276dd464cbed90485d3118290f67cf2c",
    ("0.5", "strict_image"): "7f3f11c343df6f2f5aefe3547b2dc514ec53c2a39b5960d8d7efc6b650a817d4",
    ("0.5", "label"): "b84beae7458afcefe52211db3984d046ba6cd356fc7b905bfad48fac78989cfa",
    ("0.5", "strict_label"): "a80a7fa0baeb63b6364b2adc44c6068881d8caf9a1222e6a45d9a203b0b536aa",
}

# These schema-valid documents describe dimensions or units outside the image
# subset supported by ChimeraX. They must continue to fail explicitly.
UNSUPPORTED = {
    f"{version}/image/{name}"
    for version in ("0.4", "0.5")
    for name in (
        "valid/custom_type_axes.json",
        "valid/invalid_axis_units.json",
        "valid/mismatch_axes_units.json",
    )
}

# The reader deliberately keeps useful display metadata permissive. OME-Zarr
# 0.4 also recommends retaining the last duplicate label color.
TOLERATED = (
    {
        f"{version}/image/invalid/{name}.json"
        for version in ("0.4", "0.5")
        for name in ("invalid_channels_color", "invalid_channels_window")
    }
    | {
        f"{version}/label/image-label/{name}"
        for version in ("0.4", "0.5")
        for name in ("empty_colors", "empty_properties", "colors_duplicate")
    }
    | {
        "0.4/strict_label/image-label/no_version",
        "0.4/strict_label/image-label/no_colors",
        "0.5/strict_label/image-label/no_colors",
    }
)


def _load_cases(spec_root):
    cases = []
    seen = set()
    for version in ("0.4", "0.5"):
        for suite in SUITES:
            suite_path = spec_root / version / "tests" / f"{suite}_suite.json"
            suite_bytes = suite_path.read_bytes()
            assert (
                sha256(suite_bytes).hexdigest() == EXPECTED_SUITE_SHA256[(version, suite)]
            ), f"unexpected contents in pinned suite {version}/{suite}"
            document = json.loads(suite_bytes)
            suite_cases = document["tests"]
            assert len(suite_cases) == EXPECTED_COUNTS[(version, suite)]
            for case in suite_cases:
                case_id = f"{version}/{suite}/{case['formerly']}"
                assert case_id not in seen
                seen.add(case_id)
                cases.append((case_id, version, suite, case))

    assert seen >= UNSUPPORTED
    assert seen >= TOLERATED
    assert len(cases) == sum(EXPECTED_COUNTS.values()) == 89
    return cases


def _namespace(attrs, version):
    if version == "0.5":
        return attrs.get("ome", {})
    return attrs


def _array_description(attrs, version):
    namespace = _namespace(attrs, version)
    multiscales = namespace.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales or not isinstance(multiscales[0], dict):
        return 2, ("dim_0", "dim_1"), []

    multiscale = multiscales[0]
    axes = multiscale.get("axes")
    ndim = len(axes) if isinstance(axes, list) and axes else 2

    valid_names = (
        isinstance(axes, list)
        and len(axes) == ndim
        and all(isinstance(axis, dict) and isinstance(axis.get("name"), str) for axis in axes)
        and len({axis["name"] for axis in axes}) == ndim
    )
    names = tuple(axis["name"] for axis in axes) if valid_names else tuple(f"dim_{index}" for index in range(ndim))
    datasets = multiscale.get("datasets")
    return ndim, names, datasets if isinstance(datasets, list) else []


def _is_normalized_child_path(path):
    return (
        isinstance(path, str)
        and bool(path)
        and not path.startswith("/")
        and all(part not in {"", ".", ".."} for part in path.split("/"))
    )


def _materialize_case(version, suite, raw_data):
    zarr_format = 2 if version == "0.4" else 3
    attrs = deepcopy(raw_data)
    namespace = _namespace(attrs, version)

    if suite in {"label", "strict_label"}:
        baseline = {
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]},
            ],
        }
        if version == "0.4":
            baseline["version"] = "0.4"
        namespace["multiscales"] = [baseline]

    group = zarr.create_group(store=zarr.storage.MemoryStore(), zarr_format=zarr_format)
    ndim, dimension_names, datasets = _array_description(attrs, version)
    for dataset in datasets:
        if not isinstance(dataset, dict) or not _is_normalized_child_path(dataset.get("path")):
            continue
        kwargs = {
            "name": dataset["path"],
            "shape": (2,) * ndim,
            "chunks": (1,) * ndim,
            "dtype": np.uint16 if suite in {"label", "strict_label"} else np.uint8,
        }
        if zarr_format == 3:
            kwargs["dimension_names"] = dimension_names
        group.create_array(**kwargs)
    group.attrs.update(attrs)
    return group


def _parse_supported_image(group):
    metadata = parse_ome_zarr_metadata(group)
    for dataset in metadata.multiscales.datasets:
        spatial_transform_angstrom(metadata.multiscales, dataset)
    return metadata


@pytest.mark.upstream
def test_pinned_ngff_image_and_label_suites(ngff_spec_root):
    outcomes = {}
    for case_id, version, suite, case in _load_cases(Path(ngff_spec_root)):
        group = _materialize_case(version, suite, case["data"])
        try:
            _parse_supported_image(group)
        except (OMEZarrFormatError, KeyError, TypeError, ValueError):
            accepted = False
        else:
            accepted = True

        if case_id in UNSUPPORTED:
            expected = False
            category = "unsupported"
        elif case_id in TOLERATED:
            expected = True
            category = "tolerated"
        else:
            expected = bool(case["valid"])
            category = "valid" if expected else "invalid"
        outcomes[case_id] = category
        assert accepted is expected, f"{case_id}: expected {category}, accepted={accepted}"

    assert len(outcomes) == 89
