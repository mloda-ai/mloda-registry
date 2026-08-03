# Releasing

Manual release workflow: semantic-release computes the version, PyPI gets the wheels.

## Trigger

Manual only, via GitHub Actions -> Release -> Run workflow. Nothing releases on push or merge.

## Flow

```
workflow_dispatch → semantic-release → PyPI publish
```

1. **Version bump**: semantic-release analyzes commits, updates `config/shared.toml` and `tox.ini`.
2. **Regenerate**: `scripts/generate_pyproject.py` updates all `pyproject.toml` files.
3. **Commit**: version changes committed to `main`.
4. **GitHub release**: tag created (e.g. `0.4.0`).
5. **PyPI publish**: wheels built and uploaded.
6. **Verify**: `tox -e verify-published` and `tox -e verify-extras` confirm the packages install.

## Published packages

The build list lives in `.github/workflows/release.yaml`; that workflow is the source
of truth, not this page. Not every package in `config/packages.toml` ships standalone:
the demo and example packages reach users inside the `mloda-community` /
`mloda-enterprise` bundle wheels instead. The header comment in `config/packages.toml`
states the policy, and `tests/test_end2end/test_py_typed_markers.py` asserts that every
package built by the workflow is declared in the config.

When adding a package to the release list, see
[Add a new package](packaging.md#add-a-new-package).

## Commit messages

Conventional commits determine the bump. This project deviates from the standard:
only `minor:` bumps the minor version, everything else (`feat:`, `fix:`, `docs:`,
`chore:`, `ci:`) is a patch. See `.releaserc.yaml`.

## Required secrets

| Secret | Purpose |
|--------|---------|
| `SEMANTIC_RELEASE_TOKEN` | GitHub PAT with `repo` scope |
| `PYPI_API_TOKEN` | PyPI token (account-wide or project-scoped) |

## Build flags

`--wheel --no-build-isolation` is required because the monorepo uses
`package-dir = {"" = "../.."}`, which needs access to parent directories during build.

## Files

- `.releaserc.yaml` - semantic-release config
- `.github/workflows/release.yaml` - release workflow
