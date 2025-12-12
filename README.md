# mloda-registry

Registry for mloda.

## Development Setup with uv

### Install uv

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Virtual Environment and Install Dependencies

```bash
# Create a virtual environment and install the project with dev dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Run Tests and Checks

```bash
# Run pytest
uv run pytest

# Run ruff formatter
uv run ruff format --line-length 120 .

# Run mypy type checking
uv run mypy --strict --ignore-missing-imports .

# Run bandit security checks
uv run bandit -c pyproject.toml -r -q .
```

### Using tox with uv backend

The project is configured to use `tox-uv` which uses uv as the backend for tox:

```bash
# Run all checks via tox (uses uv under the hood)
uv run tox
```

### Quick Commands

```bash
# Install everything and run tox in one go
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]" && uv run tox
```
