# Package Hierarchy Pattern

## Package Types

### Bundled Packages
`mloda-community` and `mloda-enterprise` are **bundled packages** that include all sub-package code directly.

```
mloda-community (bundled)
  └── includes: mloda.community.*
        ├── feature_groups/example/*
        ├── compute_frameworks/example/*
        └── extenders/example/*
```

**Benefits:**
- Single `pip install mloda-community` gets all community plugins
- No dependency on unpublished sub-packages
- Sub-packages can still be published separately for granular installs

### Individual Packages with Optional Dependencies
Uses **optional dependencies** for aggregation without circular dependencies.

```
mloda-community-example (base)
  └── optional: [all]
        ├── mloda-community-example-a → depends on base
        └── mloda-community-example-b → depends on base
```

## How It Works

```toml
# Base package has optional aggregation
[project]
name = "mloda-community-example"
dependencies = ["mloda>=X.Y.Z"]

[project.optional-dependencies]
dev = ["mloda-testing", "pytest"]  # from defaults
all = ["mloda-community-example-a", "mloda-community-example-b"]
```

- Base doesn't require children (optional)
- Children depend on base
- No circular dependency

## Install Combinations

| Command | Result |
|---------|--------|
| `pip install mloda-community` | All community plugins (bundled) |
| `pip install mloda-community-example` | Base example only |
| `pip install mloda-community-example[all]` | Base + all variants |
| `pip install mloda-community-example-a` | Variant A + base |

## Adding a New Plugin

1. Add to `config/packages.toml` (description, deps, path)
2. Run `python scripts/generate_pyproject.py`
3. Code is automatically included in `mloda-community` bundle

## Adding a New Variant to Existing Plugin

1. Add to `config/packages.toml` (description, deps, path)
2. Add to parent's `optional_dependencies.all`
3. Run `python scripts/generate_pyproject.py`

## UV Limitation

Nested packages can't use `tool.uv.sources` for `mloda-testing` due to workspace resolution. They get dev deps but rely on root workspace resolution.
