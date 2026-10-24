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

# The installed community feature groups ship no data of their own, so this
# example feeds the data through api_data
api_data = {"MyKey": {"example_feature": [1, 2]}}
result = mloda.run_all([Feature("example_feature")], api_data=api_data)
```

## Compute Frameworks

Pass `compute_frameworks` as an ordered list; a set is rejected. mloda picks the cheapest plan, and among equal plans the first listed framework is preferred:

```python
result = mloda.run_all(
    [Feature("example_feature")], compute_frameworks=["PolarsLazyDataFrame", "PandasDataFrame"], api_data=api_data
)
```

## Column Ordering

Control result column arrangement with the `column_ordering` parameter:

```python
prices = {"MyKey": {"price": [1, 2], "quantity": [3, 4], "total": [3, 8]}}

# Preserve feature request order
result = mloda.run_all(
    [Feature("price"), Feature("quantity"), Feature("total")], api_data=prices, column_ordering="request_order"
)
# Columns: price, quantity, total (in request order)

# Sort columns alphabetically
result = mloda.run_all(
    [Feature("price"), Feature("quantity"), Feature("total")], api_data=prices, column_ordering="alphabetical"
)
# Columns: price, quantity, total (A-Z sorted)

# Default: no guaranteed ordering
result = mloda.run_all([Feature("price"), Feature("quantity")], api_data=prices)
```

| Option | Behavior |
|--------|----------|
| `"request_order"` | Columns match feature request sequence |
| `"alphabetical"` | Columns sorted A-Z |
| `None` (default) | No guaranteed ordering |

## Streaming

Process results incrementally as each feature group completes:

```python
from mloda.user import PluginLoader, stream_all, Feature

PluginLoader.all()

api_data = {"MyKey": {"feature_a": [1, 2], "feature_b": [3, 4]}}

for result in stream_all([Feature("feature_a"), Feature("feature_b")], api_data=api_data):
    print(result)
```

See [Streaming Results](feature-group-patterns/23-streaming.md) for details.

## Realtime

Build the execution plan once and reuse it for repeated calls with fresh data:

```python
from mloda.user import PluginLoader, mloda, Feature

PluginLoader.all()

session = mloda.prepare([Feature("col")], compute_frameworks=["PandasDataFrame"], api_data={"MyKey": {"col": [0]}})
result = session.run(api_data={"MyKey": {"col": [1, 2]}})
```

See [Realtime Execution](feature-group-patterns/24-realtime.md) for details.

## Direct Import

You can also import plugin classes directly:

```python
from mloda.community.feature_groups.example import CommunityExampleFeatureGroup
```

### Packages Without an Entry Point

A package that declares no entry point registers its feature groups only when imported; `PluginLoader.all()` does not load it. The experimental [german_ledger](../../mloda/community/feature_groups/experimental/german_ledger/README.md) package is one until [mloda#1745](https://github.com/mloda-ai/mloda/issues/1745): after `PluginLoader.all()` the stock read feature groups also claim its intermediate names (`revenue`, `gdpdu_journal`). Its totals (`revenue__sources`) still resolve, because the package scopes its own inputs. To request an intermediate name directly, scope it with `feature_group=` (see [Input-Feature Forwarding](feature-group-patterns/26-input-feature-forwarding.md)):

```py
from mloda.user import DataAccessCollection, Feature, PluginLoader, mloda

from mloda.community.feature_groups.experimental.german_ledger import SkrAccountFeatureGroup

PluginLoader.all()
# With your AdmissibilityPolicyGroup subclass defined, as in the package README.
result = mloda.run_all(
    [Feature("revenue", feature_group=SkrAccountFeatureGroup)],
    compute_frameworks=["PyArrowTable"],
    data_access_collection=DataAccessCollection(folders={"path/to/dossier"}),
)
```
