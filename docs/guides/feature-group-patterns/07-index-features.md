# Pattern 7: Index Features (Index Columns)

Index features require specific columns for ordering, grouping, or joining.

**What**: Features that require specific columns for ordering, grouping, or joining.
**When**: Time series operations, group-by aggregations, window functions, join operations.
**Why**: Many computations need data ordered or grouped by specific columns.
**Where**: Rolling aggregations, lag/lead features, time windows, grouped calculations.
**How**: Return `List[Index]` from `index_columns()` method.

## Key Characteristic

| Aspect | Value |
|--------|-------|
| Method | `index_columns()` returns `List[Index]` |
| Class | `Index` from components |

## Complete Example

```python
from typing import Any, Optional, Set, List
from mloda.provider import FeatureGroup
from mloda.provider import FeatureChainParserMixin, FeatureSet
from mloda.user import Feature, Options, FeatureName, Index
import re


class LagFeature(FeatureChainParserMixin, FeatureGroup):
    """Lag feature: input__lag_N → previous N rows."""

    PREFIX_PATTERN = r"^.+__lag_(\d+)$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    @classmethod
    def index_columns(cls) -> Optional[List[Index]]:
        return [Index(("timestamp",))]

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        source = feature_name.name.rsplit("__lag_", 1)[0]
        return {Feature.not_typed(source), Feature.not_typed("timestamp")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature_name = features.name_of_one_feature.name
        match = re.match(cls.PREFIX_PATTERN, feature_name)
        lag_n = int(match.group(1))
        source = feature_name.rsplit("__lag_", 1)[0]
        return data[source].shift(lag_n)
```

## Test

```python
import pandas as pd
from mloda.user import Index

def test_lag_feature():
    assert LagFeature.index_columns() is not None
    assert Index(("timestamp",)) in LagFeature.index_columns()

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5),
        "price": [10, 20, 30, 40, 50]
    })
    class MockFeatures:
        class name_of_one_feature:
            name = "price__lag_1"
    result = LagFeature.calculate_feature(df, MockFeatures())
    assert list(result[1:]) == [10.0, 20.0, 30.0, 40.0]
```

## Index Types

**Single column**
```python
Index(("user_id",))
```

**Composite key**
```python
Index(("user_id", "date"))
```

**Multiple indexes (primary + foreign keys)**
```python
@classmethod
def index_columns(cls) -> List[Index]:
    return [
        Index(("order_id",)),       # Primary key
        Index(("customer_id",)),    # Foreign key
    ]
```

## Real Implementations

| File | Description |
|------|-------------|
| [time_window/base.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/time_window/base.py) | Time window aggregations |
| [aggregated_feature_group/base.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/aggregated_feature_group/base.py) | Group-by aggregations |

## Combines With

- **Chained** (Pattern 3): `price__rolling_mean_7d`
- **Artifact** (Pattern 6): Time-based model checkpoints
- **Framework-specific** (Pattern 9): Pandas-only window functions
- **Links and Joins** (Pattern 8): For joining feature groups using indexes, see [Pattern 8: Links and Joins](08-links-joins.md)
