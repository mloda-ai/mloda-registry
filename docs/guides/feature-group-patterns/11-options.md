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

**Context** options carry metadata that doesn't affect grouping. Features with different context but same group are still batched together, except on context keys passed down from a consumer (see [Which Features Share a Call](12-calculate-feature.md#which-features-share-a-call)). Use context for input feature references (`in_features`), debug flags, logging levels, runtime hints, or the path of a store the group keeps beside its features (or the name of an environment variable holding it; see [Side Stores](20-versioning.md#side-stores)). Context does not split calls, so put the path in group options if one run may name different stores.

**Default**: When you pass a dict without specifying `group=` or `context=`, it goes to **group**. This means `Options({"algo": "kmeans"})` is equivalent to `Options(group={"algo": "kmeans"})`. Be explicit when you need context-only parameters.

## Example

```python
from mloda.user import Feature, Options

# Configuration-based feature creation
feature = Feature("imputed_income", Options(group={"algorithm": "mean"}, context={"in_features": "income"}))
```

## Context Propagation

By default, context options stay local to each feature and are **not** propagated to input features in a dependency chain. This prevents unintended side effects when chaining features.

To push specific context keys to the feature's direct input features, use `propagate_context_keys`:

```python
from mloda.user import Feature, Options

# Push specific context keys to the direct input features
feature = Feature(
    "price__scaled",
    Options(
        context={"debug": True, "trace_id": "abc123"},
        propagate_context_keys=frozenset({"trace_id"}),  # Only trace_id reaches the direct input features
    ),
)
```

The push is one hop: an input feature does not push the keys on to its own inputs. For a name-chained feature such as `a__b__c`, a key pushed from the requested feature reaches `a__b`, not the root `a`. To reach every level, list the key in `cascade_context_keys` instead (also in the JSON feature config); see core's [Context propagation](https://github.com/mloda-ai/mloda/blob/0.18.0/docs/docs/in_depth/property-mapping.md#context-propagation) for its limits. Group options also forward down the whole chain, but they are part of feature identity and resolution at every level. A feature group author can instead pull a key level by level with `inherit_context_keys` (see [Input-Feature Option Forwarding](26-input-feature-forwarding.md)).

Use the push sparingly; most context should remain local. Typical use: a key the direct input features need, such as a trace ID or a tenant identifier.

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

`property_spec` validates the spec invariants at construction (strict needs a non-empty `allowed_values` or an `element_validator`; an `element_validator` without strict is rejected as a no-op; a strict non-`None` `default` must be in the accepted set and pass `element_validator`):

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

`property_spec` also accepts `element_validator`, `required_when`, `match_guard`, `name_converter` (see [Converting a captured value](#converting-a-captured-value-name_converter)), `allow_explicit_none`, and `deferred_binding`. `allow_explicit_none` makes an explicitly passed `None` count as present. `deferred_binding` exempts a key without a default from the name-path required-presence check, for values bound outside match-time name capture; it does not change config-path requiredness.

Omitting `default` declares no default, so the key is required; `default=None` declares a default of `None`, which makes the key optional.

### Converting a captured value: name_converter

A `PREFIX_PATTERN` capture is text. `name_converter` turns it into the typed value before validation:

```python
from typing import Any, ClassVar

from mloda.provider import (
    DataCreatorFeatureGroup,
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    is_positive_int,
    property_spec,
)
from mloda.user import PluginCollector, PluginLoader, mloda

PluginLoader.all()


class Guide11Prices(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"guide11_price"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"guide11_price": [1, 2, 3]}


class Guide11Shift(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__shift_(?P<periods>\d+)$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1
    PROPERTY_MAPPING: ClassVar = {
        "periods": property_spec(
            "Rows to shift by",
            strict=True,
            element_validator=is_positive_int,
            name_converter=int,
        ),
        DefaultOptionKeys.in_features: property_spec("Source feature"),
    }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        result = {}
        for feature in features.get_sorted_features():
            periods = feature.options.get("periods")  # an int, converted from the captured text
            source = cls._extract_source_features(feature)[0]
            result[feature.name] = data[source].shift(periods).tolist()
        return result


result = mloda.run_all(
    ["guide11_price__shift_1"],
    compute_frameworks=["PandasDataFrame"],
    plugin_collector=PluginCollector.enabled_feature_groups({Guide11Prices, Guide11Shift}),
)
assert result[0]["guide11_price__shift_1"].tolist()[1:] == [1, 2]
```

- The converter runs before validation, so `allowed_values`, `element_validator`, `match_guard` and `required_when` see the converted value, and `feature.options` holds it
- A converter that raises or returns `None` makes the value rejected, a recorded non-match; here `__shift_0` is rejected by the validator after conversion
- Explicit options and the legacy positional fallback are never converted: an explicit `"2"` string reaches validation as text
- `_resolve_operation` and `value_for` still return the captured text
- Keep the converter pure and cheap, as it runs on every resolution

See [How a name-parsed value binds to a key](https://github.com/mloda-ai/mloda/blob/0.18.0/docs/docs/in_depth/property-mapping.md#how-a-name-parsed-value-binds-to-a-key) in the core docs.

### Annotate with ClassVar

`FeatureGroup` declares `PROPERTY_MAPPING: ClassVar[...]`, but ruff's `RUF012` (mutable class defaults) does not follow inheritance across files, so a plain `PROPERTY_MAPPING = {...}` in a subclass still trips it once that rule is enabled. Annotate the assignment:

```python
from typing import ClassVar

from mloda.provider import FeatureGroup, property_spec


class ArithmeticFeature(FeatureGroup):
    PROPERTY_MAPPING: ClassVar = {
        "operation_type": property_spec("Arithmetic operation", context=False, default=None),
    }
```

A bare `ClassVar` is enough: `mypy --strict` infers `dict[str, PropertySpec]` from the initializer, so no `PropertySpec` import is needed just for the annotation. Apply the same to any other mutable class-level default on a plugin class, such as a backend registry dict or a set of supported methods.

### Captureless patterns: `RECOGNITION_ONLY_PATTERN`

A captureless `PREFIX_PATTERN`/`SUFFIX_PATTERN` (no capture group) logs a `WARNING` at class definition: it binds no key from the feature name. If a mapping key must come from the name, add a named capture group (`(?P<key>...)`); if the pattern is only a recognition predicate and every value is meant to come from `Options`, set `RECOGNITION_ONLY_PATTERN = True` to declare that intent and silence the warning:

```python
class FfillFeatureGroup(FeatureGroup):
    PREFIX_PATTERN = r".*__ffill$"
    RECOGNITION_ONLY_PATTERN = True  # no capture group; every value comes from Options

    PROPERTY_MAPPING: ClassVar = {
        "order_by": property_spec("Column to order by (ascending) within each partition"),
    }
```

See [`ffill/base.py`](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/community/feature_groups/data_operations/row_preserving/ffill/base.py) for the full production example.

## Validation and Conditional Requirements

A `PROPERTY_MAPPING` declares validation rules and conditional requirements directly on option entries. Every `FeatureGroup` enforces them, with or without `FeatureChainParserMixin`.

See [Feature Matching: Key Differences](14-feature-matching.md#key-differences-from-element_validator) for the comparison table and [Conditional Requirements](14-feature-matching.md#conditional-requirements-with-required_when) for `required_when`.

**Both match paths enforce this.** Required-key presence and strict-value validation (`strict_validation` / `allowed_values` / `element_validator`) both run on the string/name-path match now, not just the configuration path. A `PROPERTY_MAPPING` key declaring no default at all (`NO_DEFAULT`; a declared `default=None` already exempts the key), without `required_when`, and without `deferred_binding=True`, must be present in the merged options on either path. `DefaultOptionKeys.in_features` is excluded from this presence check; its count is enforced separately via `MIN_IN_FEATURES`/`MAX_IN_FEATURES`. `deferred_binding` (above) is the one path-specific exemption: it skips only the name-path presence check, for a value the group parses itself from the name rather than through a `PROPERTY_MAPPING`-bound capture.

On a plain `FeatureGroup`, the default matcher applies strict values and `match_guard` on every match. Required presence applies when the group declares no `PREFIX_PATTERN`/`SUFFIX_PATTERN`; a patterned group checks it only for names its pattern owns. A key without a default is required, so declare an optional key with `default=None`.

Required-key **presence** holds even when the class overrides `match_feature_group_criteria()`, since the check is installed as a class-definition-time guard. **Strict-value validation and `match_guard` are not guarded**: they run inside the matcher an override replaces, so an override keeps them by delegating to `super().match_feature_group_criteria()`. That call also applies the parent's name rules, so it suits an override that narrows them; a widening override that skips `super()` keeps them with `cls.passes_option_declarations(options)`, which checks `options` as passed, not values carried by the feature name. `cls.match_parser_criteria()` on a mixin group keeps strict values only, not `match_guard`.

## Full Documentation

See [Feature Configuration](https://mloda-ai.github.io/mloda/in_depth/feature-config/) for detailed patterns and [PROPERTY_MAPPING Configuration](https://mloda-ai.github.io/mloda/in_depth/property-mapping/) for the full spec reference.
