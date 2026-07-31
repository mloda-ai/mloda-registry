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

Use propagation sparingly; most context should remain local. Common use cases include trace IDs for debugging, tenant identifiers, or configuration that genuinely needs to flow through the entire pipeline.

## PROPERTY_MAPPING value space

A `PROPERTY_MAPPING` entry declares a parameter's accepted values plus metadata flags. A value is a `PropertySpec`, built with `property_spec(...)`; from mloda 0.11.0 a raw dict raises `ValueError` at class definition. The old flattened form, where accepted values sat among the flags as extra dict keys, is gone.

### Accepted values: `allowed_values`

Declare the value space under `allowed_values` so it stays separate from the flags and a doc-only key can never widen the accepted set:

```python
from mloda.provider import property_spec

PROPERTY_MAPPING = {
    "operation_type": property_spec(
        "Arithmetic operation",
        strict=True,
        allowed_values={"add": "Addition", "sub": "Subtraction"},
    ),
}
```

`allowed_values` may be a mapping of value to one-line docstring, or any iterable of accepted values (list, tuple, set); it is materialized to a tuple. A bare `str` is rejected, since membership would silently become a substring test.

### Builder: `property_spec`

`property_spec` validates the spec invariants at construction (strict needs a non-empty `allowed_values` or an `element_validator`; an `element_validator` without strict is rejected as a no-op; a strict non-`None` `default` must be in the accepted set). Its `strict=` keyword sets the field named `strict_validation`:

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

`property_spec` also accepts `context`, `element_validator`, `required_when`, and `match_guard`, plus `allow_explicit_none` and `deferred_binding` from mloda 0.11.0. `allow_explicit_none` makes an explicitly passed `None` count as present. `deferred_binding` exempts a key without a default from the name-path required-presence check, for values bound outside match-time name capture; it does not change config-path requiredness.

Version note on `default=None`: mloda 0.10.0 drops it, so the key stays required. From mloda 0.11.0, omitting `default` declares no default (the key is required) and `default=None` declares a default of `None`, which makes the key optional.

### Strict defaults are checked at import time

Since mloda 0.9.0, defining a `FeatureGroup` whose `PROPERTY_MAPPING` declares a `strict=True` default outside the accepted set (or one that fails the key's `element_validator`) raises `ValueError` at class definition, pointing at the offending spec, its default, and the accepted values. Previously such a spec imported silently and only misbehaved at runtime. A `default` of `None` is exempt (the conventional "unset" sentinel), and the check is a no-op under `strict=False`.

## Validation and Conditional Requirements

When using `PROPERTY_MAPPING` with `FeatureChainParserMixin`, you can declare validation rules and conditional requirements directly on option entries:

- **`element_validator`**: Validate each parsed element with a callable (requires `strict=True`). A falsy return raises `ValueError`, which the mixin turns into a non-match plus a rejection reason in the resolution error.
- **`match_guard`**: Check the raw option value with a callable (no `strict_validation` needed). Useful for composite types like lists or dicts. A falsy return is a plain non-match, with no reason reported.
- **`required_when`**: Make an option conditionally required based on a predicate callable.

For `element_validator` and membership, the spec declares the arity: `list`, `tuple`, `set` and `frozenset` unpack element-wise and identically, a `str` stays a scalar, and a `dict` is one composite value. `match_guard` still sees the raw value with its original container type.

See [Feature Matching: Key Differences](14-feature-matching.md#key-differences-from-element_validator) for the comparison table and [Conditional Requirements](14-feature-matching.md#conditional-requirements-with-required_when) for `required_when`.

## Full Documentation

See [Feature Configuration](https://mloda-ai.github.io/mloda/in_depth/feature-config/) for detailed patterns and [PROPERTY_MAPPING Configuration](https://mloda-ai.github.io/mloda/in_depth/property-mapping/) for the full spec reference.
