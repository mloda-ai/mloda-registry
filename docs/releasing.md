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
   Upload order is the published order (`scripts/published_packages.py`, which rejects a
   config that is not dependency-first), bundles after everything they pin (only
   `mloda-testing` follows them, for its `binary-model` extra's `mloda-community` pin), so a
   new bundle version appears on PyPI only after everything it pins is already there.

The `prepareCmd` in `.releaserc.yaml` also seds a `MLODA_REGISTRY_VERSION:<version>}`
default into `tox.ini`. No such default remains there, so that half of the command is
a no-op; the version now comes from the workflow env.

## Post-release verification

Verification is not part of the release. It runs in a separate workflow,
`.github/workflows/verify-published.yaml` ("Weekly Package Verification"), on a
Monday cron plus manual dispatch, so a fresh release stays unverified until the next
run. To check a release immediately, dispatch that workflow.

The workflow resolves the latest release first and checks out its tag, so the published
set, tox envs and verify scripts are the ones that release shipped, not main's. It sets
`MLODA_REGISTRY_VERSION` and runs these tox envs:

| Env | Checks |
|-----|--------|
| `verify-published` | The released set installs together and imports |
| `verify-published-independent` | Every published distribution installs and imports on its own |
| `verify-extras` | The `[all]` extras resolve and pull in their variants |
| `verify-typed-install` | A standalone leaf install is typed under mypy --strict |

`verify-published` installs with the `exclude-newer` window lifted for the released set; see
[Published packages](#published-packages) below.

A change on main (a newly published package, the base's `py.typed` marker, a newly owned
bundle extra) is therefore verified only once a release ships it, so the weekly job does not
go red for it before then. The flip side: a fix to a verify script or tox env on main takes
effect in the weekly job only from the next release. Sibling dependency floors need no
dedicated verification env: `verify-published-independent` already covers each leaf
installing alone at its generator-derived floor, and `verify-extras` covers each base
resolving together with its children (see [packaging.md](packaging.md#sibling-dependency-floors)).

## Published packages

The released set is the `published = true` flag in `config/packages.toml`.
`scripts/published_packages.py` prints it, plain or pinned; the build array in
`.github/workflows/release.yaml` and the install list of the `verify-published` tox env
are both filled from that one command, so they cannot drift apart.
`verify-published-independent` derives its installed set from that same `published` flag,
and `verify-extras` derives its internal extras from `config/packages.toml`'s
`optional_dependencies`, so neither script names a package by hand.

Flagging a package does not publish it: it ships with the next release run, and the weekly
verification picks it up from that release's tag, so it does not fail before then. A local
`tox -e verify-published` run from main still pins the latest release and fails for it until
the next release.

The root `pyproject.toml`'s `exclude-newer` window would hide a release younger than
seven days, so `verify-published` passes `--exclude-newer-exempt`, which
lifts the cutoff for the released distributions only; third-party dependencies stay
behind the window. The script-driven verify envs install from a temporary directory,
where no project config applies.

A release that introduces packages new to PyPI can be rejected with
`429 Too many new projects created`. PyPI throttles the creation of *new* project names,
independently of uploads to projects that already exist, and in practice the threshold is
low (around four) and the lockout lasts many hours. It is not something the release
workflow can pace or retry around. See
[pypi/support#10572](https://github.com/pypi/support/issues/10572) and
[this monorepo release thread](https://discuss.python.org/t/request-temporary-new-project-rate-limit-lift-on-pypi-for-a-coordinated-monorepo-release-user-pace/108030).
With dependency-first upload order (see [Flow](#flow)), every uploaded package's same-release
requirements are already on PyPI, so a rejected, partial upload leaves nothing already
uploaded uninstallable; a rerun just uploads the rest. `mloda-enterprise` cannot resolve at a
newer version than `mloda-community`: it needs `mloda-community` and
`mloda-community-extenders-shared` at its own version or later and, through its own `[openlineage]`
extra, `mloda-community-openlineage` at its own version or a later patch of that minor, and
`mloda-community` pins each of the packages it owns exactly.
Install both bundles at the same version and upgrade them together: upgrading one alone, or a
single owned package, leaves a conflict that pip reports without stopping the install.

Yanking a broken package also needs its bundles yanked at the same version: the bundle pins an
owned package exactly (`==`), and PyPI still resolves an exact pin to a yanked file, unlike a
floor.

`pip install -U` from a `mloda-community` that still shipped these packages' files deletes
the newly installed packages' files when it removes the old bundle, without `pip check`
noticing (`uv pip install -U` is not affected); see the [README](../README.md#upgrading) for the
fix.

Not every package ships standalone: packages without `published = true` in
`config/packages.toml` reach users only inside the bundle wheels. `mloda-community-example`
and `mloda-community-example-a` are published to keep end-to-end PyPI dependency resolution
covered. When adding a package to the set, see
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
- `.github/workflows/commit-lint.yaml` - gates PRs on Conventional Commits, protects commit-analyzer's input
