"""Shared paths for portable tests that consume external fixture repositories."""

import os
from pathlib import Path

import pytest


def _fixture_root(environment_name):
    configured = os.environ.get(environment_name)
    if not configured:
        if os.environ.get("REQUIRE_UPSTREAM_FIXTURES") == "1":
            pytest.fail(f"{environment_name} must point at the pinned fixture checkout")
        pytest.skip(f"set {environment_name} to run upstream fixture tests")

    root = Path(configured)
    if not root.is_dir():
        pytest.fail(f"{environment_name} does not exist or is not a directory: {root}")
    return root


@pytest.fixture(scope="session")
def ngff_spec_root():
    return _fixture_root("OME_NGFF_SPEC_ROOT")


@pytest.fixture(scope="session")
def examples_root():
    return _fixture_root("OME_ZARR_EXAMPLES_ROOT")
