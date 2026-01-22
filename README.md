[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# mloda-registry

> **The central hub for discovering and sharing mloda plugins.** Part of the [mloda](https://github.com/mloda-ai/mloda) ecosystem for open data access. Visit [mloda.ai](https://mloda.ai) for an overview and business context, the [GitHub repository](https://github.com/mloda-ai/mloda) for technical context, or the [documentation](https://mloda-ai.github.io/mloda/) for detailed guides.

Browse community-contributed FeatureGroups, find integration guides, and publish your own plugins for others to use.

## Related Repositories

- **[mloda](https://github.com/mloda-ai/mloda)**: The core library for open data access. Declaratively define what data you need, not how to get it. mloda handles feature resolution, dependency management, and compute framework abstraction automatically.

- **[mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template)**: A GitHub template for creating standalone mloda plugins. Use this to quickly start building your own FeatureGroups, ComputeFrameworks, and Extenders.

## Documentation

Guides for plugin development: [Plugin Creation Guides](docs/guides/index.md)

## PyPI Packages

This repository publishes the following packages to PyPI:

| Package | Description | Install |
|---------|-------------|---------|
| `mloda-registry` | Plugin discovery and search | `pip install mloda-registry` |
| `mloda-testing` | Test utilities for plugin development | `pip install mloda-testing` |
| `mloda-community` | All community plugins (bundle) | `pip install mloda-community` |
| `mloda-enterprise` | All enterprise plugins (bundle) | `pip install mloda-enterprise` |

Example packages (for regression testing):

| Package | Install |
|---------|---------|
| `mloda-community-example` | `pip install mloda-community-example` |
| `mloda-community-example-a` | `pip install mloda-community-example-a` |

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
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Run all checks via tox
uv run tox
```
