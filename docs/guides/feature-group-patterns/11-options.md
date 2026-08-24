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
feature = Feature("imputed_income", Options(group={"algorithm": "mean"}, context={"in_features": "income"}))
```

## Context Propagation

By default, context options stay local to each feature and are **not** propagated to input features in a dependency chain. This prevents unintended side effects when chaining features.

To selectively propagate specific context keys to dependent features, use `propagate_context_keys`:

```python
from mloda.user import Feature, Options

# Propagate specific context keys through the dependency chain
feature = Feature(
    "price__scaled",
    Options(
        context={"debug": True, "trace_id": "abc123"},
        propagate_context_keys=frozenset({"trace_id"}),  # Only trace_id flows to input features
    ),
)
```

| Behavior | Description |
|----------|-------------|
| Default (no propagation) | Context stays local to the feature where it's defined |
| With `propagate_context_keys` | Listed keys are passed to all input features in the chain |

Use propagation sparingly; most context should remain local. Common use cases include trace IDs for debugging, tenant identifiers, or configuration that genuinely needs to flow through the entire pipeline.

## PROPERTY_MAPPING value space

A `PROPERTY_MAPPING` entry declares one parameter: its accepted values plus its metadata. Build every value with `property_spec(...)`, which returns a typed `PropertySpec`; a raw dict raises `ValueError` at class definition. Throughout this page the builder's `strict=` kwarg sets the spec field named `strict_validation`.

### Accepted values: `allowed_values`

`allowed_values` is the kwarg that declares a key's accepted value space:

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

Pass a mapping of value to one-line docstring, or a `list`, `tuple`, `set` or `frozenset` of accepted values (materialized to a tuple). Those shapes are the whole declared type, which keeps a forgotten comma a typing error rather than a silent substring value space, so avoid a generator even though the runtime materializes one. A bare `str` is rejected outright, since membership would otherwise become a substring test.

### Builder: `property_spec`

`property_spec` validates the spec invariants at construction (strict needs a non-empty `allowed_values` or an `element_validator`; an `element_validator` without strict is rejected as a no-op; a strict non-`None` `default` must be in the accepted set):

```python
from mloda.provider import property_spec

PROPERTY_MAPPING = {
    "operation_type": property_spec(
        "Arithmetic operation",
        strict=True,
        allowed_values={"add": "Addition", "sub": "Subtraction"},
        default="add",
    ),
    "model_version": property_spec("Model version", context=False),
}
```

`context` defaults to `True`, so `context=False` is how you declare a group-scoped key, one that lands in group options and therefore affects resolution and splitting. A caller who passes the key explicitly under `group=` or `context=` overrides that declaration.

`property_spec` also accepts `element_validator`, `required_when`, `match_guard`, `allow_explicit_none`, and `deferred_binding`. `allow_explicit_none` makes an explicitly passed `None` count as present. `deferred_binding` exempts a key without a default from the name-path required-presence check, for values bound outside match-time name capture; it does not change config-path requiredness.

Omitting `default` declares no default, so the key is required; `default=None` declares a default of `None`, which makes the key optional.

### Strict defaults are checked at construction

A spec validates itself when it is built, so a `strict=True` `default` outside the accepted set (or one the key's `element_validator` rejects) raises `ValueError` from the `property_spec(...)` call, not from the `FeatureGroup` class definition that reads the mapping. The message names the spec's explanation and the rejected default. A `default` of `None` is exempt (the conventional "unset" sentinel), and the check is a no-op under `strict=False`.

### Annotate with ClassVar

`FeatureGroup` declares `PROPERTY_MAPPING: ClassVar[...]`, but ruff's `RUF012` (mutable class defaults) does not follow inheritance across files, so a plain `PROPERTY_MAPPING = {...}` in a subclass still trips it once that rule is enabled. Annotate the assignment:

```python
from typing import ClassVar

from mloda.provider import FeatureGroup, property_spec


class ArithmeticFeature(FeatureGroup):
    PROPERTY_MAPPING: ClassVar = {
        "operation_type": property_spec("Arithmetic operation", context=False),
    }
```

A bare `ClassVar` is enough: `mypy --strict` infers `dict[str, PropertySpec]` from the initializer, so no `PropertySpec` import is needed just for the annotation. Apply the same to any other mutable class-level default on a plugin class, such as a backend registry dict or a set of supported methods.

## Validation and Conditional Requirements

When using `PROPERTY_MAPPING` with `FeatureChainParserMixin`, you can declare validation rules and conditional requirements directly on option entries:

- **`element_validator`**: Validate each parsed element with a callable (requires `strict=True`). A falsy return raises `ValueError`, which the mixin turns into a non-match plus a rejection reason in the resolution error.
- **`match_guard`**: Check the raw option value with a callable (no `strict_validation` needed). Useful for composite types like lists or dicts. A falsy return is a plain non-match, with no reason reported unless the feature group extends the rejection-reason hook, as the data operation families do.
- **`required_when`**: Make an option conditionally required based on a predicate callable.

For `element_validator` and membership, the spec declares the arity: `list`, `tuple`, `set` and `frozenset` unpack element-wise and identically, a `str` stays a scalar, and a `dict` is one composite value. `match_guard` still sees the raw value with its original container type.

See [Feature Matching: Key Differences](14-feature-matching.md#key-differences-from-element_validator) for the comparison table and [Conditional Requirements](14-feature-matching.md#conditional-requirements-with-required_when) for `required_when`.

## Full Documentation

See [Feature Configuration](https://mloda-ai.github.io/mloda/in_depth/feature-config/) for detailed patterns and [PROPERTY_MAPPING Configuration](https://mloda-ai.github.io/mloda/in_depth/property-mapping/) for the full spec reference.
