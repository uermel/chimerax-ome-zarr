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

## ChimeraX Daily 1.13 release gate

The Copick-compatible alpha requires `ChimeraX-Core>=1.13.dev202608172052`, Python 3.14, and NumPy 2. Development, native integration testing, and bundle compilation are performed with ChimeraX Daily 1.13. ChimeraX 1.12 is not a supported host because its compiled volume modules use the NumPy 1 ABI and can crash after NumPy 2 is installed.

Point `CHIMERAX_PYTHON` at the Python executable inside the Daily installation; on macOS, for example:

```bash
export CHIMERAX_PYTHON=/Applications/ChimeraX_Daily.app/Contents/bin/python3.14
```

Install the bundle's test dependencies into ChimeraX as needed, then run the real suite without the portable stub at both supported Zarr endpoints:

```bash
"$CHIMERAX_PYTHON" -m pip install -r tests/requirements-portable.txt "zarr==3.1.6"
PYTHONPATH="$PWD" "$CHIMERAX_PYTHON" -m pytest -v tests/chimerax
"$CHIMERAX_PYTHON" -m pip install --upgrade -r tests/requirements-portable.txt "zarr>=3.1.6,<4"
PYTHONPATH="$PWD" "$CHIMERAX_PYTHON" -m pytest -v tests/chimerax
PYTHONPATH="$PWD" "$CHIMERAX_PYTHON" -m chimerax.core --nogui --exit --cmd "devel build ."
```

This exercises the volume hierarchy, time/channel slicing, multiscale grids, labels, and ChimeraX Segmentations integration against the Daily 1.13 APIs. Publish this line as an alpha until stable ChimeraX 1.13 passes the same gate.

## Interactive Lodstone streaming

Install the released Lodstone alpha and this development bundle into ChimeraX
Daily:

```bash
export CHIMERAX_PYTHON=/Applications/ChimeraX_Daily.app/Contents/bin/python3.14
"$CHIMERAX_PYTHON" -m pip install --no-deps --force-reinstall \
  lodstone==0.1.0a2
PYTHONPATH="$PWD" "$CHIMERAX_PYTHON" -m chimerax.core --nogui --exit \
  --cmd "devel build ."
"$CHIMERAX_PYTHON" -m chimerax.core --nogui --exit --cmd \
  "toolshed install $PWD/dist/chimerax_ome_zarr-1.0.0a1-py3-none-any.whl noDeps true reinstall true"
```

Launch the graphical application:

```bash
open -na /Applications/ChimeraX_Daily.app
```

Then enter this in ChimeraX's command line, after its main window is ready:

```chimerax
open ngff:https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr streaming true
```

Rotate and zoom the volume while watching the ChimeraX log. Each camera change
starts a new Lodstone generation; coarse data should remain visible while fine
resident chunks replace it. Closing the top-level image model must stop all
streams without a later callback traceback.

## Required branch checks

Configure branch protection for `main` to require these jobs:

- `Conventional Commit PR title`
- `pre-commit checks`
- `portable py3.14 zarr-minimum`
- `portable py3.14 zarr-latest`
- `required EBI remote stores`

Branch-protection settings live in GitHub rather than the repository, so a repository administrator must enable these checks after the workflows have run once.
