# Releasing

Manual release workflow: semantic-release computes the version, PyPI gets the wheels.

## Trigger

Manual only, via GitHub Actions -> Release -> Run workflow. Nothing releases on push or merge.

## Flow

```text
workflow_dispatch → semantic-release → PyPI publish
```

1. **Version bump**: semantic-release analyzes commits and updates `config/shared.toml`.
2. **Regenerate**: `scripts/generate_pyproject.py` updates all `pyproject.toml` files.
3. **Commit**: version changes committed to `main`.
4. **GitHub release**: tag created (e.g. `0.4.0`).
5. **PyPI publish**: wheels built and uploaded.

The `prepareCmd` in `.releaserc.yaml` also seds a `MLODA_REGISTRY_VERSION:<version>}`
default into `tox.ini`. No such default remains there, so that half of the command is
a no-op; the version now comes from the workflow env.

## Post-release verification

Verification is not part of the release. It runs in a separate workflow,
`.github/workflows/verify-published.yaml` ("Weekly Package Verification"), on a
Monday cron plus manual dispatch, so a fresh release stays unverified until the next
run. To check a release immediately, dispatch that workflow.

It sets `MLODA_REGISTRY_VERSION` and runs three tox envs:

| Env | Checks |
|-----|--------|
| `verify-published` | The released set installs together and imports |
| `verify-published-independent` | Each package installs and imports on its own |
| `verify-extras` | The `[all]` extras resolve and pull in their variants |

## Published packages

The build list lives in `.github/workflows/release.yaml`; that workflow is the source
of truth, not this page. Not every package in `config/packages.toml` ships standalone:
most demo and example packages reach users inside the `mloda-community` /
`mloda-enterprise` bundle wheels instead. `mloda-community-example` and
`mloda-community-example-a` are the exceptions, published to keep end-to-end PyPI
dependency resolution covered. The header comment in `config/packages.toml` states
the policy, and `tests/test_end2end/test_py_typed_markers.py` asserts that every
package built by the workflow is declared in the config.

The released set is also hardcoded in the `verify-published` and `security` envs of
`tox.ini`, and nothing cross-checks those against the workflow. When adding a package
to the release list, see [Add a new package](packaging.md#add-a-new-package).

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
The build gate binds each wheel to its package by exact distribution name, because all
packages share one out-dir and prefix siblings (`mloda-community` vs
`mloda-community-offset`) would otherwise collide.

## Files

- `.releaserc.yaml` - semantic-release config
- `.github/workflows/release.yaml` - release workflow
- `.github/workflows/verify-published.yaml` - weekly post-release verification
