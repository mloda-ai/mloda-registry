# Experimental Shortcuts

Convenience utilities for advanced use cases in `mloda_plugins.feature_group.experimental`.

**What**: Helper classes that reduce boilerplate for specific patterns.
**When**: You need dynamic creation or complex input source handling.
**Why**: Avoid writing full class definitions for one-off or generated feature groups.
**Where**: `mloda_plugins.feature_group.experimental`

## DynamicFeatureGroupFactory

Create FeatureGroup classes at runtime without explicit class definitions.

**Use when**:
- Generating feature groups programmatically (e.g., one per file/table)
- Creating temporary feature groups for joins or transformations
- Building feature groups with computed matching criteria

```python
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator
)

properties = {
    "calculate_feature": lambda cls, data, features: data * 2,
    "match_feature_group_criteria": lambda cls, name, opts, dac: name == "double_value",
}

DoubleValueFG = DynamicFeatureGroupCreator.create(
    properties=properties,
    class_name="DoubleValueFeatureGroup"
)
```

See [dynamic_feature_group_factory.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/dynamic_feature_group_factory/dynamic_feature_group_factory.py)

## SourceInputFeature

Mixin that simplifies defining complex input sources via Options.

**Use when**:
- Input features come from multiple origins (API, DataCreator, other features)
- You need joins/merges between sources
- Defining input features declaratively via Options rather than code

```python
from mloda_plugins.feature_group.experimental.source_input_feature import SourceInputFeature, SourceTuple

Feature(name="target_feature", options={
    "in_features": frozenset(["source_1", SourceTuple(feature_name="source_2", source_class=MyFG)])
})
```

See [source_input_feature.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/source_input_feature.py)
