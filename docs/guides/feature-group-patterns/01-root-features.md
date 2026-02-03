# Pattern 1: Root Features (Data Sources)

Root features have no input dependencies - they are pipeline entry points.

**What**: Features with no input dependencies that provide data to the pipeline.
**When**: You need data from external sources OR generate data in-place.
**Why**: Every feature pipeline starts with data; root features are the entry points.
**Where**: CSV/Parquet readers, database connectors, API clients, synthetic data, test data.
**How**: Return a `BaseInputData` subclass from `input_data()`.

## Key Characteristic

| Method | Behavior |
|--------|----------|
| `input_data()` | Returns `BaseInputData` subclass |
| `input_features()` | Returns `None` (no dependencies) |

## Complete Example

```python
from typing import Any, Optional
from mloda.provider import FeatureGroup
from mloda.provider import BaseInputData, FeatureSet


class MyInputData(BaseInputData):
    """Configuration for data source."""
    pass


class MyRootFeature(FeatureGroup):
    """Load data from external source."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return MyInputData()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"my_column": [1, 2, 3]}
```

## Test

```python
def test_root_feature():
    # Verify it's a root feature (has input_data, no input_features)
    assert isinstance(MyRootFeature.input_data(), MyInputData)

    # Verify calculation
    result = MyRootFeature.calculate_feature(None, None)
    assert "my_column" in result
```

## DataCreator (In-Place Generation)

Use `DataCreator` to generate data without external sources (test data, synthetic data).

```python
from mloda.provider import DataCreator

class OrderSyntheticData(FeatureGroup):
    """Generate synthetic order data."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"order_id", "product_id", "quantity"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"order_id": [1, 2, 3], "product_id": [101, 102, 103], "quantity": [5, 3, 7]}
```

## Real Implementations

| File | Description |
|------|-------------|
| [read_file.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/input_data/read_file.py) | Base file reader |
| [csv.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/input_data/read_files/csv.py) | CSV reader |
| [data_creator.py](https://github.com/mloda-ai/mloda/blob/main/mloda/core/abstract_plugins/components/input_data/creator/data_creator.py) | DataCreator base |
| [create_synthetic_data.py](https://github.com/mloda-ai/mloda/blob/main/docs/docs/examples/mloda_basics/create_synthetic_data.py) | Synthetic data example |

## Combines With

- **Pattern 6 (Artifact)**: Cache expensive API responses
- **Pattern 5 (Multi-output)**: Return multiple columns
