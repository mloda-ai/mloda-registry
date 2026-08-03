# Packaging

All `pyproject.toml` files are auto-generated from `config/`. Never edit them directly.
For how those packages reach PyPI, see [Releasing](releasing.md).

```bash
python scripts/generate_pyproject.py          # Generate all
python scripts/generate_pyproject.py --check  # CI validation (tox -e check-generated)
```

## Architecture

```text
config/
├── shared.toml       # version, authors, urls, defaults
└── packages.toml     # per-package: description, deps, path
         │
         ▼
scripts/generate_pyproject.py
         │
         ├──► mloda/*/pyproject.toml
         └──► pyproject.toml (workspace members + mloda core dependency)
```

`--check` fails if any generated file has drifted, including the root
`pyproject.toml`'s `mloda` entry, which the generator rewrites from
`core_dependency`.

## Config files

### shared.toml

Single source for values every package shares. `core_dependency` is substituted
into any `{core_dependency}` placeholder in `packages.toml`, so the mloda floor
is declared once.

```toml
[project]
version = "0.4.0"
requires-python = ">=3.10,<3.15"
authors = [{ name = "Tom Kaltofen", email = "info@mloda.ai" }]

[defaults]
license = "Apache-2.0"
core_dependency = "mloda>=0.10.0,<0.11.0"
optional_dependencies = { dev = ["mloda-testing", "pytest>=9.0.3"] }
```

### packages.toml

| Field | Required | Description |
|-------|----------|-------------|
| `description` | Yes | PyPI description |
| `path` | Yes | Package directory |
| `dependencies` | By convention | Runtime deps; use `"{core_dependency}"` for the mloda floor. The generator defaults it to empty rather than failing, but every package declares it |
| `optional_dependencies` | No | Merged with defaults |
| `has_readme` | No | `true` points the package at its own `README.md` |
| `workspace_deps` | No | Marks a meta-package whose deps are workspace siblings. Mutually exclusive with `py_typed`; unused today |
| `entry_point_groups` | No | List of mloda entry-point groups the package's `manifest.py` populates (`mloda.feature_groups`, `mloda.compute_frameworks`, `mloda.extenders`) |
| `entry_point_bundle` | No | `true` on bundle packages (`mloda-community`, `mloda-enterprise`); aggregates the entry points of every nested plugin package under its path. Mutually exclusive with `entry_point_groups` |
| `py_typed` | No | `true` adds the dotted path to `packages` (what ships the marker) and emits `[tool.setuptools.package-data]` for it. Requires a committed `<path>/py.typed`. Mutually exclusive with `workspace_deps` |

A marker declares its whole subtree typed, including third-party distributions installed into it: on a namespace portion (`mloda/community`, `mloda/enterprise`) that is the entire namespace, on a shared base package (`mloda/community/feature_groups/data_operations`, `mloda/community/feature_groups/example`) it is everything published from below that base. mypy returns at the first `py.typed` on the module path, so those leaf packages need no flag of their own. The leaf's typing then depends on the marker-shipping base being installed, so its dependency floor has to be at or above the release that first shipped the marker. Raise that floor only in a follow-up change, after the marker-bearing release is published: a workspace member cannot require a sibling version above the workspace's own version in `config/shared.toml`, so bumping it in the same change makes `uv sync --all-extras` unsatisfiable.

**Generator infers:**

- `license` from path (`mloda/enterprise/*` → proprietary, else default)
- `packages` from filesystem (scans for `__init__.py`, excludes `tests/`, `build/`, etc.)

**Default dev deps skipped for:** `mloda-testing`, `mloda-community`, `mloda-enterprise`

## Package hierarchy

### Bundled packages

`mloda-community` and `mloda-enterprise` include all sub-package code directly, so
one install gets every plugin and nothing depends on an unpublished sub-package.
Many sub-packages can also be published separately for granular installs, but not
all are; see [Releasing](releasing.md).

```text
mloda-community (bundled)
  └── includes: mloda.community.*
        ├── feature_groups/*
        ├── compute_frameworks/*
        └── extenders/*
```

### Individual packages

Aggregation uses optional dependencies to avoid a circular dependency: the base
does not require its children, the children require the base.

```toml
[packages.mloda-community-example]
description = "Example community FeatureGroup plugin for mloda"
dependencies = ["{core_dependency}"]
path = "mloda/community/feature_groups/example"
optional_dependencies = { all = ["mloda-community-example-a", "mloda-community-example-b"] }
entry_point_groups = ["mloda.feature_groups"]
py_typed = true
```

| Command | Result |
|---------|--------|
| `pip install mloda-community` | All community plugins (bundled) |
| `pip install mloda-community-example` | Base example only |
| `pip install mloda-community-example[all]` | Base + all variants |
| `pip install mloda-community-example-a` | Variant A + base |

Entries in `optional_dependencies.all` are emitted unpinned, so a variant only has
to exist on PyPI at some version for the extra to resolve. A variant that is
dropped from the release list therefore keeps resolving at its last published
version, which is why `mloda-community-example-b` can be absent from
`.github/workflows/release.yaml` without breaking `[all]`.

## Entry points

mloda discovers installed plugins through the entry-point groups
`mloda.feature_groups`, `mloda.compute_frameworks`, and `mloda.extenders`. Each
plugin package ships a `manifest.py` listing the package's concrete plugin classes
under a per-group attribute:

| Group | Attribute | Base type |
|-------|-----------|-----------|
| `mloda.feature_groups` | `FEATURE_GROUPS` | `FeatureGroup` |
| `mloda.compute_frameworks` | `COMPUTE_FRAMEWORKS` | `ComputeFramework` |
| `mloda.extenders` | `EXTENDERS` | `Extender` |

Conventions:

- One `manifest.py` per plugin package. It lists concrete classes only, never the
  shared base class in `base.py` / `*_base.py` (those are non-abstract and would
  wrongly register).
- The generator emits `[project.entry-points."<group>"]` tables whose entry name is
  the distribution label and whose value is the canonical
  `<dotted.package.path>.manifest:<ATTR>` target, e.g.
  `mloda-community-ffill = "mloda.community.feature_groups.data_operations.row_preserving.ffill.manifest:FEATURE_GROUPS"`.
- Bundle packages set `entry_point_bundle = true` and aggregate the entry points of
  every nested plugin package under their path.

## UV workspace sources

The generator adds `mloda-testing = { workspace = true }` only for top-level packages
(depth <= 2) that receive default dev deps. Nested packages cannot use workspace
sources due to uv resolution limits; they get dev deps but rely on root workspace
resolution.

## Common workflows

### Bump version

```bash
vim config/shared.toml                  # Change version
python scripts/generate_pyproject.py    # Regenerate
```

### Add a new package

1. Add to `config/packages.toml` (description, dependencies, path; for a plugin
   package also `entry_point_groups = ["mloda.feature_groups" | ...]`).
2. For a plugin package, create `<path>/manifest.py` listing the concrete classes.
3. If it should ship standalone on PyPI, add it to all three hardcoded lists:
   the build array in `.github/workflows/release.yaml`, and the install lines of
   the `verify-published` and `security` envs in `tox.ini`. Nothing cross-checks
   them, so a package added to one and not the others either goes unverified or
   is verified at a version that was never published.
4. Regenerate and sync:

```bash
python scripts/generate_pyproject.py
uv sync --all-extras
```

### Add a variant to an existing plugin

Same as above, plus add the variant to the parent's `optional_dependencies.all`.
