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

## PROPERTY_MAPPING value space

A `PROPERTY_MAPPING` entry declares a parameter's accepted values plus metadata flags. Historically the values were flattened in among the flags:

```python
from mloda.provider import DefaultOptionKeys

# Flattened form (still fully supported)
PROPERTY_MAPPING = {
    "operation_type": {
        "add": "Addition",
        "sub": "Subtraction",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: True,
    },
}
```

The parser recovers the value space by subtracting the reserved metadata keys (`RESERVED_PROPERTY_KEYS`: `explanation`, `allowed_values`, `default`, `context`, `group`, `strict_validation`, `validation_function`, `required_when`, `type_validator`). A value that collides with one of these names therefore cannot be expressed in the flattened form (it is read as metadata); use the explicit `allowed_values` field below for such a value.

### Recommended: explicit `allowed_values`

Declare the value space under `DefaultOptionKeys.allowed_values` so it stays separate from the flags and a doc-only key can never widen the accepted set:

```python
from mloda.provider import DefaultOptionKeys

PROPERTY_MAPPING = {
    "operation_type": {
        DefaultOptionKeys.allowed_values: {"add": "Addition", "sub": "Subtraction"},
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: True,
    },
}
```

When `allowed_values` is present the parser uses it directly and ignores the other keys for value-space purposes. It may be a mapping of value to one-line docstring, or a re-iterable collection (list, tuple, set). Do not pass a one-shot iterator (e.g. a generator) in a hand-written spec: the parser iterates the value space more than once, so an exhausted iterator behaves like an empty set. The value space here is taken verbatim, so it can hold any value (including one whose name matches a reserved key like `explanation`), unlike the flattened form.

### Builder: `property_spec`

`property_spec` builds the same dict and validates its invariants at construction (strict needs a non-empty `allowed_values` or a `validation_function`; `allowed_values` without strict is rejected as a no-op; a strict non-`None` `default` must be in the accepted set). It also materializes iterables, so you never hit the one-shot caveat. The contract stays a plain `dict[str, Any]`, so this is optional sugar:

```python
from mloda.provider import property_spec

PROPERTY_MAPPING = {
    "operation_type": property_spec(
        "Arithmetic operation",
        strict=True,
        allowed_values={"add": "Addition", "sub": "Subtraction"},
        default="add",
    ),
}
```

`property_spec` also accepts `context`, `validation_function`, `required_when`, and `type_validator`, matching the flattened keys of the same name.

### Strict defaults are checked at import time

Since mloda 0.9.0, defining a `FeatureGroup` whose `PROPERTY_MAPPING` declares a `strict_validation: True` default outside the accepted set (or one that fails the key's `validation_function`) raises `ValueError` at class definition, naming the class, key, default, and accepted values. Previously such a spec imported silently and only misbehaved at runtime. A `default` of `None` is exempt (the conventional "unset" sentinel), and the check is a no-op under `strict_validation: False`.

## Validation and Conditional Requirements

When using `PROPERTY_MAPPING` with `FeatureChainParserMixin`, you can declare validation rules and conditional requirements directly on option entries:

- **`validation_function`**: Validate individual option values with a callable (requires `strict_validation: True`).
- **`type_validator`**: Validate raw option values with a callable (no `strict_validation` needed). Useful for composite types like lists or dicts.
- **`required_when`**: Make an option conditionally required based on a predicate callable.

See [Feature Matching: Conditional Requirements](14-feature-matching.md#conditional-requirements-with-required_when) for full details, examples, and a comparison table.

## Full Documentation

See [Options API](https://mloda-ai.github.io/mloda/in_depth/options/) for detailed patterns and [PROPERTY_MAPPING Configuration](https://mloda-ai.github.io/mloda/in_depth/property-mapping/) for the full spec reference.
