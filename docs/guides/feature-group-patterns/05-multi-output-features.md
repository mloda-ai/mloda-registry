# Pattern 5: Multi-Output Features (~ Separator)

Multi-output features produce multiple columns using the `~` separator (e.g., `embedding~0`, `embedding~1`).

**What**: Features that produce multiple output columns from a single computation.
**When**: Operations naturally produce multiple values (embeddings, stats, one-hot encoding).
**Why**: Keep related outputs together; compute once, output many columns.
**Where**: Embeddings, PCA components, one-hot encoding, statistical summaries.
**How**: Return dict with `~` suffixed keys; use `apply_naming_convention()` for arrays.

## Key Characteristic

| Aspect | Behavior |
|--------|----------|
| Separator | `~` between feature name and index/suffix |
| `calculate_feature()` | Returns dict with `~` suffixed keys |
| Helper | `apply_naming_convention()` for 2D arrays |

## Complete Example

```python
from typing import Any, Optional, Set, Dict
from mloda.provider import FeatureGroup
from mloda.user import Feature, Options, FeatureName
from mloda.provider import FeatureSet


class StatsFeature(FeatureGroup):
    """Compute stats: source__stats -> source__stats~mean, source__stats~std."""

    @classmethod
    def match_feature_group_criteria(cls, feature_name: str, options: Any) -> bool:
        return feature_name.endswith("__stats")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        source = feature_name.name.replace("__stats", "")
        return {Feature.not_typed(source)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Dict[str, Any]:
        feature_name = features.name_of_one_feature.name
        source = feature_name.replace("__stats", "")
        col = data[source]
        return {
            f"{feature_name}~mean": col.mean(),
            f"{feature_name}~std": col.std(),
        }
```

## Test

```python
import pandas as pd

def test_stats_feature():
    assert StatsFeature.match_feature_group_criteria("value__stats", None)

    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    class MockFeatures:
        class name_of_one_feature:
            name = "value__stats"
    result = StatsFeature.calculate_feature(df, MockFeatures())
    assert "value__stats~mean" in result
    assert "value__stats~std" in result
```

## Helper Methods

For 2D array outputs (embeddings, PCA):

```python
# Converts 2D array to dict with ~N column names
embedding = [[0.1, 0.2], [0.3, 0.4]]
result = cls.apply_naming_convention(embedding, "emb")
# Returns: {"emb~0": [0.1, 0.3], "emb~1": [0.2, 0.4]}
```

## Real Implementations

| File | Description |
|------|-------------|
| [dimensionality_reduction/base.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/dimensionality_reduction/base.py) | PCA output |
| [encoding/base.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/sklearn/encoding/base.py) | One-hot encoding |

## Consuming Sub-Columns

Consumers can depend on all columns or specific sub-columns:

| Dependency | Columns Available |
|------------|-------------------|
| `Feature("embedding")` | All: `embedding~0`, `embedding~1`, etc. |
| `Feature("embedding~1")` | Only: `embedding~1` |

### Example

```python
class SpecificSubColumnConsumer(FeatureGroup):
    """Consume only embedding~1 from a multi-column feature."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature("embedding~1")}  # Only this sub-column

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        embedding_values = data["embedding~1"]
        feature_name = features.get_name_of_one_feature().name
        return {feature_name: embedding_values * 2}
```

### How It Works

1. Framework strips the `~N` suffix to find the base feature
2. Locates and computes the parent FeatureGroup
3. Extracts only the requested sub-column for the consumer
4. Parent is computed once even if multiple consumers request different sub-columns

### Best Practices

| Use | When |
|-----|------|
| `Feature("base")` | Need all sub-columns |
| `Feature("base~N")` | Need one specific sub-column |
| `resolve_multi_column_feature()` | Need to discover sub-columns at runtime |

Prefer specific sub-column dependencies - they make dependencies explicit and reduce data transfer.

## Combines With

- **Pattern 3 (Chained)**: `text__embedded~0`
- **Pattern 6 (Artifact)**: Fitted encoders/models need storage
