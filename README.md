[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# mloda-registry

Community plugins, registry tooling, and development guides for [mloda](https://github.com/mloda-ai/mloda).

> **New to mloda?** Visit [mloda.ai](https://mloda.ai) for business context or the [core repository](https://github.com/mloda-ai/mloda) for technical details.

- Ready-to-use FeatureGroups (`pip install mloda-community`)
- Enterprise FeatureGroups (`pip install mloda-enterprise`)
- Plugin discovery tools (`pip install mloda-registry`)
- Test utilities for plugin development (`pip install mloda-testing`)
- Step-by-step [plugin development guides](docs/guides/index.md)

## Related Repositories

- **[mloda](https://github.com/mloda-ai/mloda)**: The core library for open data access. Declaratively define what data you need, not how to get it. mloda handles feature resolution, dependency management, and compute framework abstraction automatically.

- **[mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template)**: A GitHub template for creating standalone mloda plugins. Use this to quickly start building your own FeatureGroups, ComputeFrameworks, and Extenders.

## Documentation

See `docs/guides/` for the complete plugin development journey:
- Using and discovering existing plugins
- Creating FeatureGroups (in-project or as packages)
- Sharing plugins with teams or the community
- Advanced: ComputeFrameworks and Extenders

Start here: [Plugin Development Guides](docs/guides/index.md)

## PyPI Packages

This repository publishes the following packages to PyPI:

| Package | Description | License | Install |
|---------|-------------|---------|---------|
| `mloda-registry` | Plugin discovery and search | Apache 2.0 | `pip install mloda-registry` |
| `mloda-testing` | Test utilities for plugin development | Apache 2.0 | `pip install mloda-testing` |
| `mloda-community` | All community plugins (bundle) | Apache 2.0 | `pip install mloda-community` |
| `mloda-enterprise` | All enterprise plugins (bundle) | [Source-available](mloda/enterprise/LICENSE) ([Get license](https://mloda.ai/enterprise)) | `pip install mloda-enterprise` |

> **Note:** Only `mloda/enterprise/` and its PyPI package require a license. Everything else in this repository is Apache 2.0.

### Installing Individual Packages from Git

Packages not published to PyPI can be installed directly from the repository:

```bash
pip install "git+https://github.com/mloda-ai/mloda-registry.git#subdirectory=mloda/community/feature_groups/example/example_b"
```

Replace the subdirectory path with the package location (see `config/packages.toml` for paths).

## Development Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv && source .venv/bin/activate && uv sync --all-extras

# Run all checks via tox
uv run tox
```
