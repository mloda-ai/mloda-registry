# Options

How to pass configuration to features and feature groups.

**What**: Configuration container with group/context separation.
**When**: Passing parameters to features (data sources, algorithms, thresholds).
**Why**: Group options affect feature group resolution; context options are metadata only.
**Where**: Feature creation, `input_features()`, `calculate_feature()`.

## Group vs Context

| Category | Purpose | Affects Hashing |
|----------|---------|-----------------|
| `group` | Parameters affecting resolution/splitting | Yes |
| `context` | Metadata that doesn't affect splitting | No |

**Group** options determine how features are batched and which FeatureGroup handles them. Two features with identical group options are considered equal and processed together. Use group for parameters that change the output: algorithm choice, data source, model version.

**Context** options carry metadata that doesn't affect grouping. Features with different context but same group are still batched together. Use context for input feature references (`in_features`), debug flags, logging levels, or runtime hints.

**Default**: When you pass a dict without specifying `group=` or `context=`, it goes to **group**. This means `Options({"algo": "kmeans"})` is equivalent to `Options(group={"algo": "kmeans"})`. Be explicit when you need context-only parameters.

## Example

```python
from mloda.user import Feature, Options

# Configuration-based feature creation
feature = Feature("imputed_income", Options(
    group={"algorithm": "mean"},
    context={"in_features": "income"}
))
```

## Context Propagation

By default, context options stay local to each feature and are **not** propagated to input features in a dependency chain. This prevents unintended side effects when chaining features.

To selectively propagate specific context keys to dependent features, use `propagate_context_keys`:

```python
from mloda.user import Feature, Options

# Propagate specific context keys through the dependency chain
feature = Feature("price__scaled", Options(
    context={
        "debug": True,
        "trace_id": "abc123"
    },
    propagate_context_keys=frozenset({"trace_id"})  # Only trace_id flows to input features
))
```

| Behavior | Description |
|----------|-------------|
| Default (no propagation) | Context stays local to the feature where it's defined |
| With `propagate_context_keys` | Listed keys are passed to all input features in the chain |

Use propagation sparingly—most context should remain local. Common use cases include trace IDs for debugging, tenant identifiers, or configuration that genuinely needs to flow through the entire pipeline.

## Full Documentation

See [Options API](https://mloda-ai.github.io/mloda/in_depth/options/) for detailed patterns.
