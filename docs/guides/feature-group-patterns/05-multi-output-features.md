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
| `OUTPUT_PARTS` / `output_parts()` | Declares the `~part` names the group returns, see [Declaring Output Parts](#declaring-output-parts) |

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
import numpy as np
from mloda.provider import FeatureGroup

# Converts 2D array to dict with ~N column names.
# Requires an object with a .shape (e.g. numpy); a plain list of lists returns {}.
embedding = np.array([[0.1, 0.2], [0.3, 0.4]])
result = FeatureGroup.apply_naming_convention(embedding, "emb")
# Returns: {"emb~0": array([0.1, 0.3]), "emb~1": array([0.2, 0.4])}
assert set(result) == {"emb~0", "emb~1"}
```

## Real Implementations

| File | Description |
|------|-------------|
| [sklearn/encoding/base.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/sklearn/encoding/base.py) | One-hot encoding, `OUTPUT_PARTS = (ANY_OUTPUT_PART,)` |
| [dimensionality_reduction/base.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/dimensionality_reduction/base.py) | PCA output, overrides `output_parts()` from the name |
| [forecasting/base.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/forecasting/base.py) | Forecast output, `OUTPUT_PARTS = (ANY_OUTPUT_PART,)` |

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

1. Framework strips the `~part` suffix to find the base feature (any `~N`, or a part the group declares)
2. Locates and computes the parent FeatureGroup
3. Extracts only the requested sub-column for the consumer
4. Parent is computed once even if multiple consumers request different sub-columns

### Best Practices

| Use | When |
|-----|------|
| `Feature("base")` | Need all sub-columns |
| `Feature("base~N")` | Need one specific sub-column |
| `Feature("base~part")` | Need one declared named part |
| `resolve_multi_column_feature()` | Need to discover sub-columns at runtime |

Prefer specific sub-column dependencies - they make dependencies explicit and reduce data transfer.

## Declaring Output Parts

Declare the `~part` names with `OUTPUT_PARTS`. Core then computes the base feature once and returns only the requested part columns.

```python
from mloda.user import mloda
from mloda.provider import DataCreator, FeatureGroup, FeatureSet
from mloda.user import PluginCollector


class Guide05Measure(FeatureGroup):
    """Returns a value and a unit for the feature guide05_measure."""

    OUTPUT_PARTS = ("value", "unit")

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"guide05_measure"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        base = str(features.get_name_of_one_feature())  # the base name, not "guide05_measure~value"
        return {f"{base}~value": [1.5, 2.5], f"{base}~unit": ["kg", "kg"]}


result = mloda.run_all(
    ["guide05_measure~value"],
    compute_frameworks=["PandasDataFrame"],
    plugin_collector=PluginCollector.enabled_feature_groups({Guide05Measure}),
)
assert list(result[0].columns) == ["guide05_measure~value"]
```

When the part names are only known at run time, declare `OUTPUT_PARTS = (ANY_OUTPUT_PART,)`:

```python
import pandas as pd
from mloda.provider import ANY_OUTPUT_PART


class Guide05Labels(FeatureGroup):
    """Source of a guide05_label column."""

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"guide05_label"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"guide05_label": ["a", "b", "a"]}


class Guide05OneHot(FeatureGroup):
    """guide05_label__hot -> one column per label found in the data."""

    OUTPUT_PARTS = (ANY_OUTPUT_PART,)

    @classmethod
    def match_feature_group_criteria(cls, feature_name: str, options: Any, data_access_collection: Any = None) -> bool:
        return str(feature_name).endswith("__hot")

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.not_typed(str(feature_name).removesuffix("__hot"))}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        base = str(features.get_name_of_one_feature())
        source = base.removesuffix("__hot")
        dummies = pd.get_dummies(data[source]).astype(int)
        return {f"{base}~{label}": dummies[label].tolist() for label in dummies.columns}


result = mloda.run_all(
    ["guide05_label__hot~a"],
    compute_frameworks=["PandasDataFrame"],
    plugin_collector=PluginCollector.enabled_feature_groups({Guide05Labels, Guide05OneHot}),
)
assert list(result[0].columns) == ["guide05_label__hot~a"]
```

To derive the parts from the feature name or options, override the classmethod `output_parts(feature_name, options)`; it receives the base name (dimensionality reduction does this).

- The base is computed once and `calculate_feature()` sees the base name
- Only the requested parts are returned
- An undeclared part fails at resolution, listing the declared parts
- With `ANY_OUTPUT_PART`, a requested part that was not produced fails after the computation
- A nested `f~0~lower` is declared by its first level
- A trailing `~N` is read against the declared parts too: `measure~0` fails unless `0` is declared
- `match_feature_group_criteria()` sees only the base name
- Artifacts are keyed by the base name

See [Declaring Output Parts](https://github.com/mloda-ai/mloda/blob/0.17.0/docs/docs/in_depth/multiple_result_columns.md#declaring-output-parts) in the core docs.

## Requesting Multiple Parts Together

With declared parts, requesting several in one run, e.g. `Feature("guide05_measure~value")` and `Feature("guide05_measure~unit")`, needs no per-part handling.

### Groups that declare no parts

Without `OUTPUT_PARTS`, every requested part reaches `calculate_feature()` as its own `Feature` in one `FeatureSet`, provided:

| Precondition | Why |
|---|---|
| The group's matcher accepts a trailing `~N` | A matcher that rejects `~` never receives a `name~N` top-level request. A group that refuses an unknown part should say why: see [Reporting Why a Match Declined](14-feature-matching.md#reporting-why-a-match-declined). |
| `feature_names_supported()` doesn't list the base name | `FeatureGroup.set_feature_name()` rewrites `base~N` back to `base` when `base` is in `feature_names_supported()`, collapsing the parts into one Feature before `calculate_feature()` runs. |
| The parts share options, compute framework and data type | These form the grouping hash that batches features into one `FeatureSet`; the name itself is not part of it. |

The group must write only the requested part's column in each, so it recomputes per part.

`FeatureGroup.get_column_base_feature()` strips only a trailing numeric `~N`; named parts are declared instead.

## Combines With

- **Pattern 3 (Chained)**: `text__embedded~0`
- **Pattern 6 (Artifact)**: Fitted encoders/models need storage
