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
from typing import Any
from mloda.provider import FeatureGroup
from mloda.user import Feature, Options, FeatureName
from mloda.provider import FeatureSet


class StatsFeature(FeatureGroup):
    """Compute stats: source__stats -> source__stats~mean, source__stats~std."""

    @classmethod
    def match_feature_group_criteria(cls, feature_name: str, options: Any) -> bool:
        return feature_name.endswith("__stats")

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        source = str(feature_name).replace("__stats", "")
        return {Feature.not_typed(source)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> dict[str, Any]:
        feature_name = str(features.name_of_one_feature)
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
from mloda.user import FeatureName


def test_stats_feature():
    assert StatsFeature.match_feature_group_criteria("value__stats", None)

    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

    class MockFeatures:
        name_of_one_feature = FeatureName("value__stats")

    result = StatsFeature.calculate_feature(df, MockFeatures())
    assert "value__stats~mean" in result
    assert "value__stats~std" in result
```

## Helper Methods

For 2D array outputs (embeddings, PCA):

```python
# Converts 2D array to dict with ~N column names.
# Requires an object with a .shape (e.g. numpy); a plain list of lists returns {}.
import numpy as np

embedding = np.array([[0.1, 0.2], [0.3, 0.4]])
result = cls.apply_naming_convention(embedding, "emb")
# Returns: {"emb~0": array([0.1, 0.3]), "emb~1": array([0.2, 0.4])}
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

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("embedding~1")}  # Only this sub-column

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        embedding_values = data["embedding~1"]
        feature_name = str(features.get_name_of_one_feature())
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

## Requesting Multiple Parts Together

Requesting two or more parts of the *same* multi-output feature directly, e.g. `Feature("category__onehot_encoded~0")` and `Feature("category__onehot_encoded~1")` in one run, differs from [Consuming Sub-Columns](#consuming-sub-columns): there a downstream *consumer* depends on one sub-column and the framework dedupes the parent across consumers. Here the caller requests parts of the producer's own output directly, so every requested part reaches the producer's own `calculate_feature()` as a separate `Feature` in one `FeatureSet`, and the group must handle them itself.

The following must hold for this to happen:

| Precondition | Why |
|---|---|
| The group's matcher accepts the `~part` suffix | e.g. `EncodingFeatureGroup.PREFIX_PATTERN = r".*__(onehot\|label\|ordinal)_encoded(~\d+)?$"`. A pattern that doesn't tolerate a trailing `~` (like `DimensionalityReductionFeatureGroup`'s) never receives a `~N` name as a top-level request. |
| `feature_names_supported()` doesn't list the base name | `FeatureGroup.set_feature_name()` rewrites `base~N` back to `base` when `base` is in `feature_names_supported()`, collapsing the parts into one Feature before `calculate_feature()` runs. |
| The parts share options, compute framework and data type | These form the grouping hash that batches features into one `FeatureSet`; the name itself is not part of it. |

`PandasEncodingFeatureGroup._add_result_to_data` ([encoding/pandas.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/sklearn/encoding/pandas.py)) is the tested example ([`test_onehot_encoding_specific_column_access`](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/feature_group/experimental/sklearn/test_encoding_feature_group/test_encoding_feature_group_integration.py)): each requested `~N` part writes only its own column,

```python
column_match = re.match(r"^(.+)~(\d+)$", feature_name)
if column_match:
    # (real code also bounds-checks the index; elided here)
    data[feature_name] = result[:, int(column_match.group(2))]
```

never the whole multi-column dict, so co-requested parts never collide on the same key. The cost is that `calculate_feature()` recomputes the full result once per requested part: that group refits its encoder per part, since the artifact cache (see [Pattern 6: Artifact](06-artifact-features.md)) only helps across separate runs, not across sibling parts within the same call.

If recomputing per part is too expensive, cache the shared result once per output base, inside the same `calculate_feature()` call:

```python
@classmethod
def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
    computed: dict[str, Any] = {}
    for feature in features.get_sorted_features():
        base = ...  # this group's own base-name extraction, see below
        if base not in computed:
            computed[base] = cls._expensive_compute(data, base)
        data = cls._add_result_to_data(data, feature.name, computed[base])
    return data
```

**Base-name extraction**: neither `FeatureGroup.get_column_base_feature()` (`column_name.split(COLUMN_SEPARATOR)[0]`, splits at the *first* `~`) nor an unconditional `rsplit("~", 1)[0]` is safe for a chained name. An upstream part can sit inside the source portion of the name: `"x__op~a__next~b".split("~")[0]` gives `"x__op"`, discarding the embedded `~a` reference; `.rsplit("~", 1)[0]` gives the correct `"x__op~a__next"`. But `rsplit` fails too when a co-requested sibling has no trailing part of its own: `"x__op~a__next".rsplit("~", 1)[0]` gives `"x__op"`, when the whole name is already the base. Strip only the trailing token your own group emits, matched against your own value space (e.g. `\d+` for `EncodingFeatureGroup`), not just "whatever follows the last `~`".

## Combines With

- **Pattern 3 (Chained)**: `text__embedded~0`
- **Pattern 6 (Artifact)**: Fitted encoders/models need storage
