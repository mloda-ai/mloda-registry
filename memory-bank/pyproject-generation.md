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
| `workspace_deps` | No | For meta-packages only |
| `optional_dependencies` | No | Merged with defaults |

**Generator infers:**
- `license` from path (`mloda/enterprise/*` → proprietary, else default)
- `include` from path (`mloda/foo` → `["mloda.foo", "mloda.foo.*"]`)
- Meta-package status from `workspace_deps` presence

**Default dev deps skipped for:** `mloda-testing`, `mloda-community`, `mloda-enterprise`

**Example - Regular package:**
```toml
[packages.mloda-registry]
description = "Plugin discovery for mloda"
dependencies = ["mloda>=X.Y.Z"]
path = "mloda/registry"
```

**Example - Meta-package:**
```toml
[packages.mloda-community]
description = "All community plugins (meta-package)"
dependencies = ["mloda-community-example[all]>=0.2.0"]
path = "mloda/community"
workspace_deps = ["mloda-community-example"]
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
# 1. Add to config/packages.toml (just description, deps, path)
# 2. Generate
python scripts/generate_pyproject.py
uv sync --all-extras
```