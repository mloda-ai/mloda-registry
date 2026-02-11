# Use an Existing Plugin

Install and use a community plugin in your project.

## Install

```bash
pip install mloda-community
```

## Use

```python
from mloda.user import PluginLoader, mloda, Feature

# Auto-discover installed plugins
PluginLoader.all()

# Use features from the plugin
result = mloda.run_all([Feature("example_feature")])
```

## Column Ordering

Control result column arrangement with the `column_ordering` parameter:

```python
# Preserve feature request order
result = mloda.run_all(
    [Feature("price"), Feature("quantity"), Feature("total")],
    column_ordering="request_order"
)
# Columns: price, quantity, total (in request order)

# Sort columns alphabetically
result = mloda.run_all(
    [Feature("price"), Feature("quantity"), Feature("total")],
    column_ordering="alphabetical"
)
# Columns: price, quantity, total (A-Z sorted)

# Default: no guaranteed ordering
result = mloda.run_all([Feature("price"), Feature("quantity")])
```

| Option | Behavior |
|--------|----------|
| `"request_order"` | Columns match feature request sequence |
| `"alphabetical"` | Columns sorted A-Z |
| `None` (default) | No guaranteed ordering |

## Direct Import

You can also import plugin classes directly:

```python
from mloda.community.feature_groups.example import CommunityExampleFeatureGroup
```
