# mloda-registry

Plugin discovery and search for mloda.

## Installation

```bash
pip install mloda-registry
```

## Usage

```python
from mloda.registry import discover, search

# List available plugins
plugins = discover()

# Search plugins by tags
results = search(tags=["timeseries"])
```
