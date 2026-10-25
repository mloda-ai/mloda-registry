# Pattern 1: Root Features (Data Sources)

Root features have no input dependencies - they are pipeline entry points.

**What**: Features with no input dependencies that provide data to the pipeline.
**When**: You need data from external sources OR generate data in-place.
**Why**: Every feature pipeline starts with data; root features are the entry points.
**Where**: CSV/Parquet files, database connectors, API clients, synthetic data, test data.
**How**: Subclass `DataCreatorFeatureGroup` to generate data in place, or subclass a source FeatureGroup (`CsvFG`, `ParquetFG`, `SqliteFG`, `ApiDataFG`, your own) that claims features itself.

## Key Characteristic

| Method | Behavior |
|--------|----------|
| `feature_names_supported()` | Names a `DataCreatorFeatureGroup` generates (omit when the only name is the class name); a source FeatureGroup claims features through `CLAIM_ROUTES` instead |
| `input_features()` | Returns `None` (no dependencies) |

## Complete Example

A `DataCreatorFeatureGroup` generates data in place, without an external source (test data, synthetic data). It lists the feature names the group creates; the class name always matches too.

```python
from typing import Any

from mloda.provider import DataCreatorFeatureGroup, FeatureSet


class OrderSyntheticData(DataCreatorFeatureGroup):
    """Generate synthetic order data."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"order_id", "product_id", "quantity"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"order_id": [1, 2, 3], "product_id": [101, 102, 103], "quantity": [5, 3, 7]}
```

The return shape is a contract, not a style choice. On `PythonDictFramework` it is columnar `dict[str, list]`, and a root feature group must return its columns even when it has no rows: `{"my_column": []}`, never `[]` or `{}`. An empty source directory or a query with no hits is a zero-row result, and a result with zero columns raises `EmptyResultError`. See [calculate_feature](12-calculate-feature.md#return-contract).

Files, documents and databases are read by source FeatureGroups, which load the data themselves. See [Your Own Root FeatureGroup](27-input-data-readers.md#your-own-root-featuregroup).

## Test

```python
def test_root_feature():
    # Verify it's a root feature (no input_features)
    assert issubclass(OrderSyntheticData, DataCreatorFeatureGroup)

    # Verify calculation
    result = OrderSyntheticData.calculate_feature(None, None)
    assert result["order_id"] == [1, 2, 3]
```

## CsvFG (File-Based Data)

`CsvFG` is a stock source FeatureGroup: it claims a feature when a CSV file or folder entry in the `DataAccessCollection` has that column. No root group of your own is needed.

```python
from mloda.user import DataAccessCollection, Feature

data_access = DataAccessCollection(folders={"/path/to/data/"})
features = [Feature("customer_id"), Feature("name")]
```

The set form above auto-names each resource. When a collection holds multiple
resources of the same kind, name them with a `dict[handle, value]` and pick one per
feature via `Options(context={"data_access_handle": ...})`. See
[Data Connection Matching](17-data-connection-matching.md#named-handles-and-multi-source-disambiguation).

Database credentials live in the same collection. Wrap each slot in the typed
`Credential` class (for example `DataAccessCollection(credentials=Credential(sqlite="/data/app.db"))`)
so a single credential dict is never mistaken for a `{handle: value}` registry. See
[Credentials](17-data-connection-matching.md#credentials).

## Several Sources, One Group Each

There is one source group per format (`CsvFG`, `ParquetFG`, `SqliteFG`, ...), and a feature points at one with an option keyed by the group's class name (`Feature("x", options={"CsvFG": "data/a.csv"})`) or scopes with `feature_group=CsvFG`. See
[Source FeatureGroups](27-input-data-readers.md).

## Real Implementations

| File | Description |
|------|-------------|
| [read_file_fg.py](https://github.com/mloda-ai/mloda/blob/0.17.0/mloda/core/abstract_plugins/components/input_data/read_file_fg.py) | `ReadFileFG` base file source group |
| [csv_fg.py](https://github.com/mloda-ai/mloda/blob/0.17.0/mloda_plugins/feature_group/input_data/file_formats/csv_fg.py) | `CsvFG` |
| [data_creator_feature_group.py](https://github.com/mloda-ai/mloda/blob/0.18.0/mloda/core/abstract_plugins/data_creator_feature_group.py) | `DataCreatorFeatureGroup` base |
| [create_synthetic_data.py](https://github.com/mloda-ai/mloda/blob/main/docs/docs/examples/mloda_basics/create_synthetic_data.py) | Synthetic data example |

## Combines With

- **Pattern 6 (Artifact)**: Cache expensive API responses
- **Pattern 5 (Multi-output)**: Return multiple columns
