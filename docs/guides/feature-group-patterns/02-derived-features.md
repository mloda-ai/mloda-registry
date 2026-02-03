# Pattern 2: Simple Derived Features

Derived features transform existing features with a static name (no naming pattern).

**What**: Features that transform one or more input features into a single output.
**When**: You have a fixed transformation with a static output name (e.g., `TotalRevenue`).
**Why**: Most business logic features are derived: combining, transforming, or enriching data.
**Where**: Calculated fields, aggregations, business rules, data enrichment.
**How**: Return dependencies from `input_features()`, compute in `calculate_feature()`.

## Key Characteristic

| Method | Behavior |
|--------|----------|
| `input_features()` | Returns `Set[Feature]` of dependencies |
| Matching | By class name (default) - see [Feature Naming](13-feature-naming.md) |

## Complete Example

```python
from typing import Any, Optional, Set
from mloda.provider import FeatureGroup
from mloda.user import Feature, Options, FeatureName
from mloda.provider import FeatureSet


class DoubledValue(FeatureGroup):
    """Double the source_column value."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.not_typed("source_column")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data["source_column"] * 2
```

## Test

```python
import pandas as pd

def test_doubled_value():
    assert DoubledValue.match_feature_group_criteria("DoubledValue", None)

    df = pd.DataFrame({"source_column": [1, 2, 3]})
    result = DoubledValue.calculate_feature(df, None)
    assert list(result) == [2, 4, 6]
```

## Real Implementations

Most derived features in practice use [Pattern 3: Chained Features](03-chained-features.md) with naming patterns like `input__operation`. This enables reusability across different inputs. However, static-name derived features are useful for unique, one-off business logic.

| File | Description |
|------|-------------|
| [text_cleaning/base.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/text_cleaning/base.py) | Chained derived feature |

## Combines With

- **Pattern 6 (Artifact)**: Cache expensive computations
- **Pattern 7 (Index)**: Add grouping/ordering requirements
