# pyproject.toml Generation System

All `pyproject.toml` files are auto-generated from config.

## Quick Reference

```bash
python scripts/generate_pyproject.py          # Generate all
python scripts/generate_pyproject.py --check  # CI validation
```

## Architecture

```
config/
├── shared.toml       # version, authors, urls, defaults
└── packages.toml     # per-package: description, deps, path
         │
         ▼
scripts/generate_pyproject.py
         │
         ├──► mloda/*/pyproject.toml
         └──► pyproject.toml (workspace members)
```

## Config Files

### shared.toml

```toml
[project]
version = "0.2.0"
requires-python = ">=3.10"
authors = [{ name = "Tom Kaltofen", email = "info@mloda.ai" }]

[project.urls]
Homepage = "https://mloda.ai"
Repository = "https://github.com/mloda-ai/mloda-registry"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[defaults]
license = "Apache-2.0"
optional_dependencies = { dev = ["mloda-testing", "pytest"] }
```

### packages.toml

| Field | Required | Description |
|-------|----------|-------------|
| `description` | Yes | PyPI description |
| `dependencies` | Yes | Runtime deps |
| `path` | Yes | Package directory |
| `optional_dependencies` | No | Merged with defaults |
| `entry_point_groups` | No | List of mloda entry-point groups the package's `manifest.py` populates (`mloda.feature_groups`, `mloda.compute_frameworks`, `mloda.extenders`) |
| `entry_point_bundle` | No | `true` on bundle packages (`mloda-community`, `mloda-enterprise`); aggregates the entry points of every nested plugin package under its path. Mutually exclusive with `entry_point_groups` |

**Generator infers:**
- `license` from path (`mloda/enterprise/*` → proprietary, else default)
- `packages` from filesystem (scans for `__init__.py`, excludes `tests/`, `build/`, etc.)

**Default dev deps skipped for:** `mloda-testing`, `mloda-community`, `mloda-enterprise`

**Example - Bundled package:**
```toml
[packages.mloda-community]
description = "All community plugins for mloda"
dependencies = ["mloda>=X.Y.Z"]
path = "mloda/community"
# Generates: package-dir = {"" = "../.."}, packages = ["mloda.community.*"]
```

**Example - Regular package:**
```toml
[packages.mloda-registry]
description = "Plugin discovery for mloda"
dependencies = ["mloda>=X.Y.Z"]
path = "mloda/registry"
```

**Example - With extra optional deps:**
```toml
[packages.mloda-community-example]
description = "Example community plugin"
dependencies = ["mloda>=X.Y.Z"]
path = "mloda/community/feature_groups/example"
optional_dependencies = { all = ["mloda-community-example-a", "mloda-community-example-b"] }
# dev = ["mloda-testing", "pytest"] added from defaults
```

## Entry points

mloda 0.9.0 discovers installed plugins through the entry-point groups
`mloda.feature_groups`, `mloda.compute_frameworks`, and `mloda.extenders`
(issue #271). Each plugin package ships a `manifest.py` module that lists the
package's concrete plugin classes under a per-group attribute:

| Group | Attribute | Base type |
|-------|-----------|-----------|
| `mloda.feature_groups` | `FEATURE_GROUPS` | `FeatureGroup` |
| `mloda.compute_frameworks` | `COMPUTE_FRAMEWORKS` | `ComputeFramework` |
| `mloda.extenders` | `EXTENDERS` | `Extender` |

Conventions:
- One `manifest.py` per plugin package. It lists concrete classes only, never the
  shared base class in `base.py` / `*_base.py` (those are non-abstract and would
  wrongly register).
- The generator emits `[project.entry-points."<group>"]` tables whose entry name
  is the distribution label and whose value is the canonical
  `<dotted.package.path>.manifest:<ATTR>` target, e.g.
  `mloda-community-ffill = "mloda.community.feature_groups.data_operations.row_preserving.ffill.manifest:FEATURE_GROUPS"`.
- Bundle packages (`mloda-community`, `mloda-enterprise`) set
  `entry_point_bundle = true` and aggregate the entry points of every nested
  plugin package under their path.

Plugin authors adding a new plugin package must:
1. Add `entry_point_groups = ["<group>"]` to the package in `config/packages.toml`.
2. Create `<package path>/manifest.py` listing the concrete plugin classes.
3. Run `python scripts/generate_pyproject.py` to regenerate the pyproject files.

## UV Workspace Sources

The generator adds `mloda-testing = { workspace = true }` only for top-level packages (depth ≤ 2) that receive default dev deps. Nested packages can't use workspace sources due to uv resolution limitations.

## Common Workflows

### Bump version
```bash
vim config/shared.toml                  # Change version
python scripts/generate_pyproject.py    # Regenerate
```

### Add new package
```bash
# 1. Add to config/packages.toml (description, deps, path; for a plugin package
#    also add entry_point_groups = ["mloda.feature_groups" | ...])
# 2. For a plugin package, create <path>/manifest.py listing the concrete classes
# 3. Generate
python scripts/generate_pyproject.py
uv sync --all-extras
```
