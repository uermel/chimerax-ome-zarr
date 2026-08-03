# Contributing

Pull requests run portable parser tests, pinned upstream OME-NGFF fixtures, complete example stores, linting, and required live EBI smoke tests. A full ChimeraX installation cannot be redistributed to GitHub-hosted runners, so native ChimeraX tests and bundle compilation are the local release gate.

## Pull request titles

PR titles must follow Conventional Commits, matching the Copick repository policy:

```text
feat: support another OME-Zarr metadata feature
fix: preserve channel indices when opening labels
test: add a conformance fixture
ci: update the portable Python matrix
```

Accepted types are `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, and `chore`. An optional scope is allowed, for example `fix(labels): preserve label colors`.

## Portable tests

Install only the portable dependencies; installing the project itself would also request ChimeraX packages that are unavailable from a normal Python package index.

```bash
python -m pip install -r tests/requirements-portable.txt
PYTHONPATH="tests/stubs:$PWD" python -m pytest -m "not remote" tests/portable
```

The small `tests/stubs` package only permits importing the metadata parser. It must not be placed on `PYTHONPATH` for native ChimeraX tests.

The conformance and complete-store tests use immutable upstream revisions:

- `ome/ngff` at `5cdf83b6806b5e500477c7af69dae9ed2adcf77d` (OME-NGFF 0.5.2)
- `BioImageTools/ome-zarr-examples` at `8c10c88fbb77c3dcc206d9b234431f243beee576`

GitHub Actions checks these repositories out under `.upstream`. To reproduce the job locally, check out those exact revisions there and run:

```bash
OME_NGFF_SPEC_ROOT="$PWD/.upstream/ngff" \
OME_ZARR_EXAMPLES_ROOT="$PWD/.upstream/ome-zarr-examples" \
REQUIRE_UPSTREAM_FIXTURES=1 \
PYTHONPATH="tests/stubs:$PWD" \
python -m pytest -v -m "not remote" tests/portable
```

The required remote check uses TLS verification, two retries, and small reads from the public EBI v0.4 image,
combined v0.5 image/label, and v0.5 bioformats2raw collection stores:

```bash
PYTHONPATH="tests/stubs:$PWD" \
python -m pytest -vv --reruns 2 --reruns-delay 5 tests/portable/test_remote_stores.py
```

## ChimeraX 1.12 release gate

The bundle retains `ChimeraX-Core>=1.7` as its declared compatibility floor. Development, native integration testing, and bundle compilation are performed with ChimeraX 1.12. Point `CHIMERAX_PYTHON` at the Python executable inside that installation; on macOS, for example:

```bash
export CHIMERAX_PYTHON=/Applications/ChimeraX-1.12.app/Contents/bin/python3.11
```

Install the bundle's test dependencies into ChimeraX as needed, then run the real suite without the portable stub:

```bash
PYTHONPATH="$PWD" "$CHIMERAX_PYTHON" -m pytest -v tests/chimerax
PYTHONPATH="$PWD" "$CHIMERAX_PYTHON" -m chimerax.core --nogui --exit --cmd "devel build ."
```

This exercises the volume hierarchy, time/channel slicing, multiscale grids, labels, and ChimeraX Segmentations integration against the 1.12 APIs.

## Required branch checks

Configure branch protection for `main` to require these jobs:

- `Conventional Commit PR title`
- `pre-commit checks`
- `portable py3.11`
- `portable py3.12`
- `portable py3.13`
- `required EBI remote stores`

Branch-protection settings live in GitHub rather than the repository, so a repository administrator must enable these checks after the workflows have run once.
