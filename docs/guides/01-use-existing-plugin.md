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

## Streaming and Realtime

For incremental results use [`stream_all`](feature-group-patterns/23-streaming.md); to build the plan once and rerun it with fresh data use [`prepare` + `run`](feature-group-patterns/24-realtime.md).

## Names Claimed by an Installed Package

The experimental [german_ledger](../../mloda/community/feature_groups/experimental/german_ledger/README.md) package registers through its entry point, so once installed, `PluginLoader.all()` registers its groups in every process. `SkrAccountFeatureGroup` answers `revenue` and `receivables`, and `SourcesFeatureGroup` any `<name>__sources`, only when a ledger source (a GDPdU dossier or DATEV batch folder) is in the data access collection or a reader option points at one, so a plain CSV `revenue` column resolves as usual. A request needs `feature_group=` only when the same collection also holds another source that answers the name; the example scopes the ledger's own group (see [Input-Feature Forwarding](feature-group-patterns/26-input-feature-forwarding.md)):

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
