# pyproject.toml Generation System

## Overview

This monorepo uses a **config-driven generation system** to manage `pyproject.toml` files across all packages. Instead of manually maintaining duplicate metadata in each package, shared fields are defined once and pyproject.toml files are generated automatically.

### Why This Exists

With 100+ packages, duplicating `version`, `authors`, `urls`, `requires-python`, and `build-system` in each pyproject.toml is:
- Error-prone (easy to miss a file during version bumps)
- Time-consuming (manual updates across many files)
- Inconsistent (drift between packages over time)

### Key Benefits

- **Single source of truth**: Version bump = edit 1 file, regenerate all
- **Consistency**: All packages share identical metadata where appropriate
- **Scalability**: Adding new packages is a 5-line config entry
- **CI validation**: Automated checks ensure files stay in sync

---

## Architecture

```
config/
├── shared.toml           # Metadata shared by ALL packages
└── packages.toml         # Per-package configuration

scripts/
├── generate_pyproject.py # Generates pyproject.toml from config
└── verify_builds.py      # Builds all packages, verifies versions

Generated files (DO NOT EDIT DIRECTLY):
├── mloda/registry/pyproject.toml
├── mloda/testing/pyproject.toml
├── mloda/community/pyproject.toml
├── mloda/enterprise/pyproject.toml
├── mloda/community/feature_groups/example/pyproject.toml
└── mloda/enterprise/feature_groups/example/pyproject.toml
```

### Data Flow

```
config/shared.toml  ─┐
                     ├─► generate_pyproject.py ─► pyproject.toml (per package)
config/packages.toml ┘
```

---

## Config File Reference

### config/shared.toml

Contains fields applied to **all packages**:

```toml
[project]
version = "0.2.0"                                        # Version for all packages
requires-python = ">=3.10"                               # Python version constraint
authors = [{ name = "Tom Kaltofen", email = "info@mloda.ai" }]

[project.urls]
Homepage = "https://mloda.ai"
Repository = "https://github.com/mloda-ai/mloda-registry"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

### config/packages.toml

Contains **per-package configuration**:

```toml
[packages.mloda-registry]
description = "Plugin discovery and search for mloda"   # Required
license = "Apache-2.0"                                  # Required
dependencies = ["mloda>=0.4.2"]                         # Required (can be empty [])
path = "mloda/registry"                                 # Required: location of package
include = ["mloda.registry", "mloda.registry.*"]        # For namespace packages
has_readme = true                                       # Optional: include readme field
```

#### Package Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `description` | Yes | Package description for PyPI |
| `license` | Yes | SPDX license identifier |
| `dependencies` | Yes | Runtime dependencies list |
| `path` | Yes | Directory containing the package |
| `include` | No | Namespace package patterns (setuptools.packages.find) |
| `has_readme` | No | Whether to include `readme = "README.md"` |
| `is_meta_package` | No | If true, sets `packages = []` (no code, just deps) |
| `workspace_deps` | No | List of workspace deps for `[tool.uv.sources]` |
| `optional_dependencies` | No | Dict of optional dep groups, e.g., `{ dev = ["pytest"] }` |

#### Example: Regular Package

```toml
[packages.mloda-registry]
description = "Plugin discovery and search for mloda"
license = "Apache-2.0"
dependencies = ["mloda>=0.4.2"]
path = "mloda/registry"
include = ["mloda.registry", "mloda.registry.*"]
has_readme = true
```

#### Example: Meta-Package (aggregates other packages)

```toml
[packages.mloda-community]
description = "All community plugins for mloda (meta-package)"
license = "Apache-2.0"
dependencies = ["mloda-community-example>=0.2.0"]
path = "mloda/community"
is_meta_package = true
workspace_deps = ["mloda-community-example"]
```

---

## Common Workflows

### Bump Version Across All Packages

```bash
# 1. Edit the single source of truth
vim config/shared.toml   # Change: version = "0.3.0"

# 2. Regenerate all pyproject.toml files
python scripts/generate_pyproject.py

# 3. Verify builds work
python scripts/verify_builds.py

# 4. Commit changes
git add -A && git commit -m "chore: bump version to 0.3.0"
```

### Add a New Package

```bash
# 1. Add entry to packages.toml
cat >> config/packages.toml << 'EOF'

[packages.mloda-my-new-package]
description = "My new feature group"
license = "Apache-2.0"
dependencies = ["mloda>=0.4.2"]
path = "mloda/community/feature_groups/my_new"
include = ["mloda.community.feature_groups.my_new", "mloda.community.feature_groups.my_new.*"]
has_readme = true
optional_dependencies = { dev = ["mloda-testing", "pytest"] }
EOF

# 2. Generate the pyproject.toml
python scripts/generate_pyproject.py

# 3. Verify it builds
python scripts/verify_builds.py
```

### Check Files Are Up-to-Date (CI)

```bash
python scripts/generate_pyproject.py --check
```

Returns exit code 0 if all files match config, exit code 1 if any are out of date.

### Verify All Packages Build Correctly

```bash
python scripts/verify_builds.py
```

This script:
1. Checks version consistency across all pyproject.toml files
2. Builds each package with `uv build`
3. Verifies wheel metadata contains correct version
4. Cleans up egg-info directories on success

---

## CI/Tox Integration

Two tox environments are available for CI:

```bash
# Check generated files are up-to-date
tox -e check-generated

# Verify all packages build with correct versions
tox -e verify-builds
```

### tox.ini Configuration

```ini
[testenv:verify-builds]
description = Verify all packages build with correct versions
usedevelop = true
allowlist_externals = uv
commands =
    python scripts/verify_builds.py

[testenv:check-generated]
description = Check pyproject.toml files are up-to-date with config
usedevelop = true
commands =
    python scripts/generate_pyproject.py --check
```

---

## Troubleshooting / FAQ

### Q: Generated files are out of sync - what do I do?

```bash
python scripts/generate_pyproject.py   # Regenerate all
git diff                                # Review changes
```

### Q: How do I add a meta-package vs a regular package?

**Regular package** (contains code):
- Set `include = [...]` with namespace patterns
- Set `has_readme = true` if README exists

**Meta-package** (no code, just aggregates dependencies):
- Set `is_meta_package = true`
- Set `workspace_deps = [...]` if dependencies are workspace packages
- Don't set `include` (no code to include)

### Q: Why setuptools instead of hatchling?

Hatchling doesn't allow paths outside the package directory (e.g., `../..`). Since namespace packages need to reference the repo root, setuptools is required.

### Q: Why are pyproject.toml files in the repo if they're generated?

The generated files are committed to:
1. Allow `pip install` from GitHub without running the generator
2. Enable IDE tooling to work immediately after clone
3. Maintain compatibility with standard Python packaging workflows

The `--check` flag in CI ensures they stay in sync with config.

### Q: How does the version consistency check work?

`verify_builds.py` reads the version from all pyproject.toml files and fails if they differ. This catches cases where someone manually edited a file instead of using the generator.

---

## Files Reference

| File | Purpose |
|------|---------|
| `config/shared.toml` | Shared metadata (version, authors, urls) |
| `config/packages.toml` | Per-package configuration |
| `scripts/generate_pyproject.py` | Generates pyproject.toml files |
| `scripts/verify_builds.py` | Builds packages, verifies versions, cleans artifacts |
| `tox.ini` (verify-builds) | CI environment for build verification |
| `tox.ini` (check-generated) | CI environment for generation check |
