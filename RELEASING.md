# Releasing GhostFold

This checklist documents the release gate used before publishing a version tag.

## Prerequisites

- Python 3.11 available.
- Network access available for dependency resolution during isolated builds.
- Linux runner/environment for fold-extra resolver checks.

## Local release checklist

1. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,test]"
python -m pip install build twine
```

2. Run quality gates:

```bash
python -m ruff check src tests
python -m pytest -q
```

3. Build and validate artifacts:

```bash
python -m build
python -m twine check dist/*
```

4. Validate extras resolution on Linux:

```bash
python -m pip install --dry-run ".[test]"
python -m pip install --dry-run ".[fold]"
```

5. Run packaging smoke tests:

```bash
GHOSTFOLD_RUN_PACKAGING_SMOKE=1 python -m pytest -q tests/test_packaging_smoke.py
```

## Build isolation and network requirements

`python -m build` creates an isolated build environment and installs build backend requirements (for example `hatchling`).
In offline or restricted environments this can fail before packaging starts.

When running without network access, preinstall the backend and use:

```bash
python -m build --no-isolation
```

This bypasses isolated dependency bootstrapping and uses the current interpreter environment.

## Tag workflow

Pushing a `v*` tag triggers `.github/workflows/release.yml`, which enforces:

- lint (`ruff`)
- unit/integration tests (`pytest`)
- build (`python -m build`)
- artifact validation (`twine check`)
- packaging smoke tests

Artifacts are uploaded as `ghostfold-dist` only when all gates pass.
