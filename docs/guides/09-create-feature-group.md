# Create a Feature Group

A detailed guide to creating feature groups - the core building blocks of mloda plugins.

## What is a Feature Group?

A Feature Group defines:
- How to calculate one or more features
- What input features it depends on (if any)
- Which compute frameworks it supports
- How to match feature names to this group

## Basic Structure

```python
from mloda.provider import FeatureGroup

class MyFeatureGroup(FeatureGroup):
    """Description of what this feature group does."""

    @classmethod
    def calculate_feature(cls, data, features):
        """Calculate the feature value."""
        # Your calculation logic here
        return result
```

## Key Methods

### Required Methods

#### `calculate_feature(data, features)`

The core calculation logic. Receives input data and returns computed features.

```python
@classmethod
def calculate_feature(cls, data, features):
    # Access input data
    input_col = data["input_column"]

    # Calculate
    result = input_col * 2

    # Return result
    return {"my_feature": result}
```

### Optional Methods

#### `description()`

Human-readable description. Defaults to class docstring or class name.

```python
@classmethod
def description(cls) -> str:
    return "Calculates customer risk scores based on transaction history"
```

#### `input_features(options, feature_name)`

Define dependencies on other features. Return `None` for root features (no dependencies).

```python
from mloda.user import Feature

def input_features(self, options, feature_name):
    # This feature depends on "amount" and "timestamp"
    return {Feature("amount"), Feature("timestamp")}
```

#### `match_feature_group_criteria(feature_name, options, data_access_collection)`

Control which feature names this group handles. Default matching:
- Class name equals feature name
- Feature name starts with class prefix (e.g., `MyFeatureGroup_something`)
- Feature name in `feature_names_supported()`

```python
@classmethod
def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):
    # Only handle features starting with "risk_"
    if isinstance(feature_name, str):
        return feature_name.startswith("risk_")
    return feature_name.name.startswith("risk_")
```

#### `feature_names_supported()`

Explicitly list supported feature names.

```python
@classmethod
def feature_names_supported(cls):
    return {"customer_score", "risk_level", "fraud_probability"}
```

#### `compute_framework_rule()`

Limit which compute frameworks this group supports.

```python
from mloda.provider import ComputeFramework, get_all_subclasses

@classmethod
def compute_framework_rule(cls):
    # Support all frameworks (default)
    return True

    # Or limit to specific frameworks
    # from mloda_plugins.compute_framework.pandas import PandasDataframe
    # return {PandasDataframe}
```

#### `index_columns()`

Define index columns for data merging.

```python
from mloda.user import Index

@classmethod
def index_columns(cls):
    return [Index("customer_id"), Index("date")]
```

## Feature Chaining Pattern

For features that transform other features, use `FeatureChainParserMixin`:

```python
from mloda.provider import FeatureGroup, FeatureChainParserMixin

class ScaledFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Scale features using min-max normalization."""

    # Pattern: {input_feature}__scaled
    PREFIX_PATTERN = r".*__scaled$"

    # Expect exactly 1 input feature
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    @classmethod
    def calculate_feature(cls, data, features):
        # Input feature name parsed automatically
        input_feature = features.get_options().get("in_features")
        col = data[input_feature]

        # Scale to 0-1
        scaled = (col - col.min()) / (col.max() - col.min())
        return {features.get_feature_name(): scaled}
```

Usage:
```python
Feature("price__scaled")  # Scales the "price" feature
Feature("amount__scaled")  # Scales the "amount" feature
```

## Testing

Use `FeatureGroupTestBase` for consistent testing:

```python
from mloda.testing.base import FeatureGroupTestBase
from my_plugin import MyFeatureGroup

class TestMyFeatureGroup(FeatureGroupTestBase):
    feature_group_class = MyFeatureGroup

    def test_calculation(self):
        # Test your feature calculation
        result = self.feature_group_class.calculate_feature(
            {"input": [1, 2, 3]},
            mock_features
        )
        assert "my_feature" in result
```

## Complete Example

```python
"""Customer scoring feature group."""

from typing import Any, Set
from mloda.provider import FeatureGroup
from mloda.user import Feature, Options, FeatureName

class CustomerScoring(FeatureGroup):
    """Calculate customer risk scores based on transaction patterns."""

    @classmethod
    def description(cls) -> str:
        return "Customer risk scoring based on transaction history"

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"customer_risk_score", "customer_fraud_score"}

    def input_features(self, options: Options, feature_name: FeatureName) -> Set[Feature]:
        return {
            Feature("transaction_count"),
            Feature("avg_transaction_amount"),
            Feature("days_since_last_transaction"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        # Simple risk calculation
        tx_count = data["transaction_count"]
        avg_amount = data["avg_transaction_amount"]
        days_inactive = data["days_since_last_transaction"]

        risk_score = (avg_amount / 1000) + (days_inactive / 30) - (tx_count / 10)

        return {"customer_risk_score": risk_score}
```

## Next Steps

- [Create a plugin package](04-create-plugin-package.md) - Package your feature group
- [Share with your team](05-share-with-team.md) - Distribute via private git repo
- [Create a compute framework](10-create-compute-framework.md) - Custom data processing
- [Create an extender](11-create-extender.md) - Add cross-cutting behavior
