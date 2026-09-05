# Releasing

Manual release workflow: semantic-release computes the version, PyPI gets the wheels.

## Trigger

Manual only, via GitHub Actions -> Release -> Run workflow. Nothing releases on push or merge.

## Flow

```text
workflow_dispatch → semantic-release → PyPI publish
```

1. **Version bump**: semantic-release analyzes commits and updates `config/shared.toml`.
2. **Regenerate**: `scripts/generate_pyproject.py` updates all `pyproject.toml` files,
   then `uv lock` re-locks `uv.lock` against the bumped versions.
3. **Commit**: version changes, including `uv.lock`, committed to `main`.
4. **GitHub release**: tag created (e.g. `0.4.0`); its commit SHA is captured as a job
   output.
5. **PyPI publish**: the `publish` job checks out that exact SHA (not whatever `main`
   is when the job runs), then builds and uploads wheels with `twine --skip-existing`,
   so a rerun after a partial upload does not fail on the files that already made it.

The `prepareCmd` in `.releaserc.yaml` also seds a `MLODA_REGISTRY_VERSION:<version>}`
default into `tox.ini`. No such default remains there, so that half of the command is
a no-op; the version now comes from the workflow env.

## Post-release verification

Verification is not part of the release. It runs in a separate workflow,
`.github/workflows/verify-published.yaml` ("Weekly Package Verification"), on a
Monday cron plus manual dispatch, so a fresh release stays unverified until the next
run. To check a release immediately, dispatch that workflow.

It sets `MLODA_REGISTRY_VERSION` and runs these tox envs:

| Env | Checks |
|-----|--------|
| `verify-published` | The released set installs together and imports |
| `verify-published-independent` | Every published distribution installs and imports on its own |
| `verify-extras` | The `[all]` extras resolve and pull in their variants |
| `verify-typed-install` | A standalone leaf install is typed under mypy --strict |

`verify-typed-install` follows the fails-until-release pattern `verify-published` has: it
goes red until the base's `py.typed` marker ships. Sibling dependency floors need no
dedicated verification env: `verify-published-independent` already covers each leaf
installing alone at its generator-derived floor, and `verify-extras` covers each base
resolving together with its children (see
[packaging.md](packaging.md#sibling-dependency-floors)).

## Published packages

The released set is the `published = true` flag in `config/packages.toml`.
`scripts/published_packages.py` prints it, plain or pinned; the build array in
`.github/workflows/release.yaml` and the install lists of the `verify-published` and
`security` tox envs are all filled from that one command, so they cannot drift apart.
`verify-published-independent` derives its installed set from that same `published` flag,
and `verify-extras` derives its internal extras from `config/packages.toml`'s
`optional_dependencies`, so neither script names a package by hand.

Flagging a package does not publish it: it ships with the next release run, and
`tox -e verify-published` fails for it until then.

A release that introduces packages new to PyPI can be rejected with
`429 Too many new projects created`. PyPI throttles the creation of *new* project names,
independently of uploads to projects that already exist, and in practice the threshold is
low (around four) and the lockout lasts many hours. It is not something the release
workflow can pace or retry around. See
[pypi/support#10572](https://github.com/pypi/support/issues/10572) and
[this monorepo release thread](https://discuss.python.org/t/request-temporary-new-project-rate-limit-lift-on-pypi-for-a-coordinated-monorepo-release-user-pace/108030).
Because every leaf requires its base at the same version, a rejected, partial upload also
leaves the already-uploaded leaves uninstallable until the rerun completes.

Not every package ships standalone. Most demo and example packages reach users inside the
`mloda-community` / `mloda-enterprise` bundle wheels instead; `mloda-community-example`
and `mloda-community-example-a` are the exceptions, published to keep end-to-end PyPI
dependency resolution covered. When adding a package to the set, see
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
The build gate binds each wheel to its package by exact distribution name, because all
packages share one out-dir and prefix siblings (`mloda-community` vs
`mloda-community-offset`) would otherwise collide.

## Files

- `.releaserc.yaml` - semantic-release config
- `config/packages.toml` - the `published` flag, single source of the released set
- `scripts/published_packages.py` - prints that set for the workflow and the tox envs
- `.github/workflows/release.yaml` - release workflow
- `.github/workflows/verify-published.yaml` - weekly post-release verification
