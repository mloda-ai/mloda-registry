# Pattern 4: Multi-Input Features (& Separator)

Multi-input features combine two or more inputs using the `&` separator in the feature name.

**What**: Features that combine multiple inputs into one output using `&` separator.
**When**: Computing relationships between features (differences, ratios, distances).
**Why**: Express multi-input operations clearly: `lat1&lon1&lat2&lon2__distance`.
**Where**: Geographic distance, price differences, ratios, similarity scores.
**How**: Parse `&`-separated inputs, return all as dependencies, combine in calculation.

## Key Characteristic

| Aspect | Behavior |
|--------|----------|
| Separator | `&` between input features |
| `input_features()` | Parses both sides of `&` as dependencies |
| Matching | By class name or custom `match_feature_group_criteria` |

## Complete Example

```python
from typing import Any, Optional, Set
from mloda.provider import FeatureGroup
from mloda.user import Feature, Options, FeatureName
from mloda.provider import FeatureSet


class DiffFeature(FeatureGroup):
    """Difference between two features: a&b -> a - b."""

    @classmethod
    def match_feature_group_criteria(cls, feature_name: str, options: Options) -> bool:
        return "&" in feature_name and feature_name.endswith("__diff")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        base = feature_name.name.replace("__diff", "")
        a, b = base.split("&")
        return {Feature.not_typed(a), Feature.not_typed(b)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature_name = features.name_of_one_feature.name
        base = feature_name.replace("__diff", "")
        a, b = base.split("&")
        return data[a] - data[b]
```

## Test

```python
import pandas as pd

def test_diff_feature():
    assert DiffFeature.match_feature_group_criteria("revenue&cost__diff", None)
    assert not DiffFeature.match_feature_group_criteria("revenue__diff", None)

    df = pd.DataFrame({"revenue": [100, 200], "cost": [30, 50]})
    class MockFeatures:
        class name_of_one_feature:
            name = "revenue&cost__diff"
    result = DiffFeature.calculate_feature(df, MockFeatures())
    assert list(result) == [70, 150]
```

## Real Implementations

| File | Description |
|------|-------------|
| [geo_distance/base.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/geo_distance/base.py) | Haversine distance (combines with Pattern 3) |

## Combines With

- **Pattern 3 (Chained)**: Add `FeatureChainParserMixin` for reusable pattern matching
- **Pattern 7 (Index)**: For time-aligned calculations
