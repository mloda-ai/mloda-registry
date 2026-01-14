# pyproject.toml Deduplication via Generation

## Solution

All pyproject.toml files are now **generated** from two config files:

```
config/
├── shared.toml     # version, authors, urls, requires-python, build-system
└── packages.toml   # per-package: name, description, dependencies, path
```

## Workflow

**Bump version (edit ONE file):**
```bash
vim config/shared.toml  # change version = "0.3.0"
python scripts/generate_pyproject.py
python scripts/verify_builds.py
```

**Add new package:**
```bash
# Add entry to config/packages.toml
python scripts/generate_pyproject.py
```

**Check if files are up-to-date (CI):**
```bash
python scripts/generate_pyproject.py --check
```

## Files

| File | Purpose |
|------|---------|
| `config/shared.toml` | Shared metadata (version, authors, urls) |
| `config/packages.toml` | Per-package config (6 packages defined) |
| `scripts/generate_pyproject.py` | Generator script |
| `scripts/verify_builds.py` | Build verification + cleanup |

## Generated Files

All these are AUTO-GENERATED (do not edit directly):
- `mloda/registry/pyproject.toml`
- `mloda/testing/pyproject.toml`
- `mloda/community/pyproject.toml`
- `mloda/enterprise/pyproject.toml`
- `mloda/community/feature_groups/example/pyproject.toml`
- `mloda/enterprise/feature_groups/example/pyproject.toml`

## Verification

```bash
# Check generation is up-to-date
python scripts/generate_pyproject.py --check

# Build all packages and verify versions
python scripts/verify_builds.py

# Run tests
pytest -v
```
