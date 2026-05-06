# Contributing to mloda-registry

The `mloda-registry` repo is the home for community and enterprise plugin packages and the [40+ plugin development guides](docs/guides/index.md). Whether you are fixing bugs in registry tooling, improving the guides, or adding a new community plugin, your input is welcome.

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- Python 3.10 or higher (tested on 3.10, 3.11, 3.12, 3.13)
- [uv](https://docs.astral.sh/uv/) for dependency management
- [tox](https://tox.wiki/) as the test runner (installed via uv)

### Local Development Setup

1. Clone the repository:

```bash
git clone https://github.com/mloda-ai/mloda-registry.git
cd mloda-registry
```

2. Install uv (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync --all-extras
```

4. Verify your setup by running the full check suite:

```bash
uv run tox
```

## Code Style

All code must pass the automated checks enforced by tox. The toolchain includes:

- **ruff format** for code formatting (line length: 120 characters)
- **ruff check** for linting (modern type-hint enforcement via `UP006`/`UP007`)
- **mypy --strict --ignore-missing-imports** for static type checking
- **bandit** for security scanning

### Conventions

- No code in `__init__.py` files.
- Avoid `try/except` blocks unless absolutely necessary.
- Keep documentation to the necessary minimum.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (`fix:`, `feat:`, `chore:`, `docs:`, etc.). This project deviates from the standard: only `minor:` commits bump the minor version; `feat:` is treated as a patch bump along with everything else (see `.releaserc.yaml`).
- **Never edit `pyproject.toml` files directly.** They are auto-generated from `config/shared.toml` and `config/packages.toml` via `python scripts/generate_pyproject.py`. Edit the config files and regenerate. See `memory-bank/pyproject-generation.md` for details.

### Running Checks Locally

Run the full suite (linting, formatting, type checking, security, and tests). Always run tox before submitting a pull request:

```bash
uv run tox
```

For quick iteration during development, you can run only the tests. Note that this skips linting, type checking, and security checks, so it is not a substitute for tox:

```bash
pytest -n 2
```

## Ways to Contribute

### Improve a Community Plugin

The community plugin packages live under `mloda/community/`:

- **Feature groups**: `mloda/community/feature_groups/`
- **Compute frameworks**: `mloda/community/compute_frameworks/`
- **Extenders**: `mloda/community/extenders/`

Bug fixes, performance improvements, and new functionality for existing plugins are always welcome.

### Add a New Community Plugin

If you want to add a new plugin to the community registry:

1. Read [`docs/guides/04-create-plugin-package.md`](docs/guides/04-create-plugin-package.md) for packaging conventions.
2. Read [`docs/guides/06-publish-to-community.md`](docs/guides/06-publish-to-community.md) for the publication path.
3. For promotion to **official** plugin status, see [`docs/guides/07-contribute-to-official.md`](docs/guides/07-contribute-to-official.md).

### Improve the Plugin Development Guides

The guides under [`docs/guides/`](docs/guides/index.md) are a primary on-ramp for new mloda contributors. Clarifications, missing patterns, and new pattern catalogs are all valuable contributions. Start with the [Plugin Journey overview](docs/guides/index.md) to find the right entry point.

### Fix Registry Tooling

The non-plugin packages support discovery and testing:

- `mloda.registry` — plugin discovery and search
- `mloda.testing` — test utilities (e.g. `FeatureGroupTestBase`) for plugin development

### Report Issues

Found a bug or have a feature request? [Open an issue](https://github.com/mloda-ai/mloda-registry/issues/). The issue template will prompt you for a summary, reproduction or motivation, optional code pointers, and an optional definition of done.

## Pull Request Workflow

1. Fork the repository and clone your fork:

```bash
git clone https://github.com/<your-username>/mloda-registry.git
cd mloda-registry
```

2. Create a feature branch from `main`:

```bash
git checkout -b fix/short-description
```

3. Make your changes and ensure `uv run tox` passes locally.
4. Commit using [Conventional Commits](https://www.conventionalcommits.org/) format.
5. Push your branch to your fork and open a pull request targeting `main`.
6. CI runs the full tox suite. All checks must pass before merge.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License, Version 2.0](LICENSE), except for contributions to `mloda/enterprise/`, which are covered by the source-available [enterprise license](mloda/enterprise/LICENSE).
