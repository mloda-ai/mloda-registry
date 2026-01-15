# Package Hierarchy Pattern

Uses **optional dependencies** to create "meta of meta" without circular dependencies.

## Structure

```
mloda-community (meta)
  └── mloda-community-example[all]
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
| `pip install mloda-community-example` | Base only |
| `pip install mloda-community-example[all]` | Base + variants |
| `pip install mloda-community-example-a` | Variant A + base |
| `pip install mloda-community` | Everything |

## Adding a New Variant

1. Add to `config/packages.toml` (description, deps, path)
2. Add to parent's `optional_dependencies.all`
3. Run `python scripts/generate_pyproject.py`

## UV Limitation

Nested packages can't use `tool.uv.sources` for `mloda-testing` due to workspace resolution. They get dev deps but rely on root workspace resolution.
