# Pattern 3: Chained Features (FeatureChainParserMixin)

Chained features use naming patterns like `price__scaled` for reusable transformations. They support two creation methods: **string-based** (parameters in name) and **configuration-based** (parameters in Options). String-based is built on top of configuration-based as a convenience layer. Prefer it for readability when feature complexity is low.

**What**: Reusable transformations that work on any input via naming patterns (`input__operation`).
**When**: The same operation applies to many different inputs (scaling, encoding, cleaning).
**Why**: Avoid duplicating code for each column; one class handles `price__scaled`, `age__scaled`, etc.
**Where**: Normalization, encoding, text cleaning, mathematical transforms.
**How**: Use `FeatureChainParserMixin`, define `PREFIX_PATTERN` regex, parse input from name.

## Key Characteristic

| Aspect | Value |
|--------|-------|
| Separator | `__` (double underscore) |
| Mixin | `FeatureChainParserMixin` |
| Required | `PREFIX_PATTERN` |
| Optional | `PROPERTY_MAPPING` (for configuration-based creation) |

## Complete Example

```python
from typing import Any, Optional, Set
from mloda.provider import DefaultOptionKeys, FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda.user import Feature, Options, FeatureName


class MeanImputedFeature(FeatureChainParserMixin, FeatureGroup):
    """
    Impute missing values with mean/median.

    String-based: `income__mean_imputed`
    Config-based: `Options(context={"imputation_method": "mean", "in_features": "income"})`
    """

    PREFIX_PATTERN = r".*__([\w]+)_imputed$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    # Optional: enables configuration-based creation
    PROPERTY_MAPPING = {
        "imputation_method": {
            "mean": "Impute with mean",
            "median": "Impute with median",
            DefaultOptionKeys.context: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature to impute",
            DefaultOptionKeys.context: True,
        },
    }

    # input_features() and match_feature_group_criteria() inherited from mixin

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            name = feature.get_name()

            # Resolve operation from name or config (handles both paths)
            method = cls._resolve_operation(feature, "imputation_method")
            source = next(iter(feature.options.get_in_features())).get_name()

            col = data[source]
            data[name] = col.fillna(col.mean() if method == "mean" else col.median())
        return data
```

> **Manual alternative**: Before `_resolve_operation()`, plugins called `FeatureChainParser.parse_feature_name()` directly and handled the options fallback themselves. The helper handles this dual-path lookup automatically, so prefer `_resolve_operation()` in new code.

## Usage

```python
# String-based (traditional)
Feature("income__mean_imputed")

# Configuration-based (modern) - enables complex types, dynamic creation
Feature("imputed_income", Options(context={"imputation_method": "mean", "in_features": "income"}))
```

> **Note on Context**: Context options are local by default and do not flow to input features. If you need context to propagate through the chain (e.g., a trace ID), use `propagate_context_keys`. See [Options](11-options.md#context-propagation) for details.

### Discriminator Keys

When your feature group supports multiple method variants (e.g., different scaling algorithms), use a **unique discriminator key** in `PROPERTY_MAPPING`. The mixin handles matching automatically for both string-based and config-based creation.

| Feature Group | Discriminator Key | Example Values |
|--------------|------------------|----------------|
| `ScalingFeatureGroup` | `scaler_type` | standard, minmax, robust |
| `EncodingFeatureGroup` | `encoder_type` | onehot, label, ordinal |
| `AggregatedFeatureGroup` | `aggregation_type` | sum, avg, min, max |

Use specific keys (not generic `operation_type`) to avoid collisions between feature group types. See [Feature Matching](14-feature-matching.md#discriminator-keys-for-configuration-based-matching) for the full pattern.

### Advanced PROPERTY_MAPPING Features

PROPERTY_MAPPING supports additional capabilities that reduce boilerplate in `match_feature_group_criteria` overrides:

- **`required_when`**: Declare options that are only required under certain conditions via a predicate callable.
- **`type_validator`**: Validate raw option values with a callable (e.g., check that a value is a list of strings). Does not require `strict_validation`.
- **`validation_function`**: Validate individual parsed values when `strict_validation` is enabled (e.g., check that a value is a positive integer).

See [Feature Matching: Conditional Requirements](14-feature-matching.md#conditional-requirements-with-required_when) for full details, examples, and a comparison table.

## Test

```python
import pandas as pd

def test_mean_imputed():
    assert MeanImputedFeature.match_feature_group_criteria("price__mean_imputed", None)
    assert not MeanImputedFeature.match_feature_group_criteria("price", None)
```

## Real Implementations

| File | Description |
|------|-------------|
| [missing_value/base.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/data_quality/missing_value/base.py) | Full PROPERTY_MAPPING example |
| [aggregated_feature_group/base.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/aggregated_feature_group/base.py) | Window aggregations |

## Combines With

- **Pattern 5 (Multi-output)**: `text__embedded~0`, `text__embedded~1`
- **Pattern 6 (Artifact)**: Fitted scalers need storage
- **Pattern 7 (Index)**: Window functions need ordering
