# Release & PyPI Publishing

## Overview

Manual release workflow using semantic-release for versioning and PyPI for distribution.

## Trigger

- **Manual only**: GitHub Actions → Release → Run workflow
- No automatic releases on push/merge

## Flow

```
workflow_dispatch → semantic-release → PyPI publish
```

1. **Version bump**: semantic-release analyzes commits, updates `config/shared.toml`
2. **Regenerate**: `generate_pyproject.py` updates all pyproject.toml files
3. **Commit**: Version changes committed to main
4. **GitHub Release**: Tag created (e.g., `0.3.0`)
5. **PyPI Publish**: 6 packages built with `--wheel --no-build-isolation` and uploaded
6. **Verify**: Run `tox -e verify-published` to confirm packages install correctly

## Packages Published

| Package | Type |
|---------|------|
| `mloda-registry` | Core |
| `mloda-testing` | Core |
| `mloda-community` | Bundle (all community plugins) |
| `mloda-enterprise` | Bundle (all enterprise plugins) |
| `mloda-community-example` | Example (regression testing) |
| `mloda-community-example-a` | Example (regression testing) |
| `mloda-community-example-b` | Example (regression testing) |
| `mloda-enterprise-example` | Example (regression testing) |
| `mloda-community-compute-frameworks-example` | Example (regression testing) |
| `mloda-community-extenders-example` | Example (regression testing) |
| `mloda-enterprise-compute-frameworks-example` | Example (regression testing) |
| `mloda-enterprise-extenders-example` | Example (regression testing) |

## Commit Message Format

Uses conventional commits to determine version bump:

- `feat:` → patch
- `fix:` → patch
- `minor:` → minor
- `docs:`, `chore:`, `ci:`, etc. → patch

## Required Secrets

| Secret | Purpose |
|--------|---------|
| `SEMANTIC_RELEASE_TOKEN` | GitHub PAT with `repo` scope |
| `PYPI_API_TOKEN` | PyPI token (account-wide or project-scoped) |

## Build Flags

`--wheel --no-build-isolation` required because monorepo uses `package-dir = {"" = "../.."}` which needs access to parent directories during build.

## Files

- `.releaserc.yaml` - semantic-release config
- `.github/workflows/release.yaml` - release workflow
