# `return_data_type_rule` failure handling

Why every `return_data_type_rule` in `data_operations` wraps op-extraction in a broad `except Exception: return None`, what that trades away, and the decision recorded for issue [#244](https://github.com/mloda-ai/mloda-registry/issues/244).

**What**: The failure-handling contract for `return_data_type_rule` on the ten `data_operations` base classes that declare a deterministic output type.
**When**: Read this before adding a type rule to a new operation, before narrowing the catch, or before moving the defensiveness into mloda core.
**Why**: The broad catch is a deliberate trade. It keeps planning crash-free at the cost of observability, and the same boilerplate is duplicated ten times and growing. The trade deserves to be recorded once rather than re-litigated per operation.
**Where**: `return_data_type_rule` on the base classes under `mloda/community/feature_groups/data_operations/`, the framework call site `engine.set_data_type` in mloda core, and the cross-cutting guard test `tests/test_return_data_type_rule_invariants.py`.
**How**: This page states the invariant, maps the trade onto the three data-governance roles it affects, weighs the options, and records what was adopted now versus deferred.

---

## The invariant: planning must never raise

mloda core calls the rule unguarded. In `engine.set_data_type`:

```python
def set_data_type(self, feature, feature_group_class):
    fg_data_type = feature_group_class.return_data_type_rule(feature)
    ...
```

There is no `try/except` around that call. If a rule raises, planning crashes for the whole graph, not just the one feature whose type could not be determined. `return_data_type_rule` is a planning-time hint: a feature group telling the engine "I always emit this type, validate downstream against it." A hint that cannot be computed must degrade to "type undeclared" (`None`), never abort the build. **This is the invariant: `return_data_type_rule` must return, never raise, for any feature the framework hands it, including a malformed one.** It is the reason the broad catch exists, and `tests/test_return_data_type_rule_invariants.py::test_invariant_never_raises` exists to keep it from regressing.

## The shape, and the three outcomes it collapses

Each of the ten rules follows the same shape (here, aggregation):

```python
@classmethod
def return_data_type_rule(cls, feature: Feature) -> DataType | None:
    try:
        agg_type = cls._extract_aggregation_type(feature)
    except Exception:  # best-effort during planning; failure leaves the type undeclared
        return None
    if agg_type in {"count", "nunique"}:
        return DataType.INT64
    return None
```

`None` is returned on three distinct paths that look identical from outside:

1. **Intended open type.** The op is genuinely input-dependent (e.g. `sum` over an unknown numeric column). `None` is correct.
2. **Acceptable degradation.** The feature is malformed and cannot be parsed. `None` is the safe answer; validation simply does not engage for that feature.
3. **Hidden bug.** A typo, a renamed helper, a refactor that broke `_extract_*` raises `AttributeError`/`NameError`. The catch swallows it, the rule returns `None`, and 0.7.0 output-type validation is silently disabled for that feature. No log, no warning, no test failure. It is indistinguishable from path 1.

Path 3 is the cost. A regression can ship and look exactly like normal operation.

## Why narrowing the catch is not trivial

The obvious fix, `except ValueError`, is unsafe. The leaf extractors do raise `ValueError`, but at least one rule routes through a shared helper that raises other types. `FrameAggregateFeatureGroup.return_data_type_rule` calls `_extract_params`, whose config-based branch does `source_features[0]` after `_extract_source_features` -> `options.get_in_features()`. On a malformed config feature, `get_in_features()` raises:

- `ValueError` when the `in_features` key is missing or empty (`if not val: raise ValueError(...)`), and
- `TypeError` when an `in_features` entry is neither a `Feature` nor a `str` (`raise TypeError(f"Cannot convert {type(item)} ...")`).

So a naive narrow-to-`ValueError` would let a `TypeError` from a malformed feature propagate and break the never-raises invariant. Narrowing safely requires either auditing every `_extract_*` and `get_in_features` path to guarantee a single domain error, or widening the catch to the full set of data-shape errors (`ValueError, IndexError, TypeError, KeyError`) and only then letting `AttributeError`/`NameError` through as the "this is a bug" signal. Narrowing and observability are therefore coupled: you cannot make the swallow informative without first deciding which exceptions mean "bad data" and which mean "bad code."

## Who this affects: data users, providers, stewards

The trade is easiest to reason about through the three roles that touch a declared output type.

| Role | Relies on the type rule for | What silent path-3 failure costs them |
|---|---|---|
| **Data user** (consumes features) | The declared output type as a contract: downstream schema checks, type-safe joins, validation catching a wrong-typed column before it reaches a model or a dashboard. | Validation silently stops engaging for the affected feature. A type regression flows downstream undetected. The contract erodes quietly, and erosion that is invisible is the kind users stop trusting. |
| **Data provider** (authors / registers feature groups) | Fast feedback when their extraction logic breaks. They write the rule and the extractor it calls. | Their own bug (renamed helper, typo) is swallowed. Local runs look green; the rule "works" by returning `None`. They get no signal until a data user notices wrong types in production. |
| **Data steward** (governs quality, lineage, contracts across the platform) | Being able to answer "which features have a declared, enforced output type, and which are intentionally open?" | The swallow makes intended-open (path 1) and accidentally-broken (path 3) indistinguishable. A coverage or lineage audit cannot tell a deliberate open type from a disabled one. Governance metrics degrade with no signal. |

The down-the-road consequence sits underneath all three: **the per-rule `try/except` proliferates.** It is on ten base classes today and grows by one with every new deterministic operation. Each copy is a place where a future author can narrow the exception set wrong, or forget to narrow at all. Ten hand-maintained copies of a safety-critical catch are a maintenance liability independent of any single bug, and it is the central argument for eventually centralizing the defensiveness rather than duplicating it.

## Options weighed

These are not mutually exclusive; the decision combines three of them.

1. **Tighten the extraction contract.** Guarantee that extractors and the helpers they call raise only a known domain error (`ValueError`, or a dedicated `ExtractionError`), then narrow the catch to that. *Cost:* audit of every `_extract_*` + `_extract_source_features`/`get_in_features` path, plus an enforcement test, before the narrow is safe. Highest correctness, highest up-front cost.
2. **Narrow to realistic data-shape failures** (`ValueError, IndexError, TypeError, KeyError`) and let `AttributeError`/`NameError` propagate so genuine bugs surface. *Cost:* the propagating bug now crashes planning, which trades one failure mode (silent) for another (loud but graph-wide). Acceptable only once core owns the never-raises wrapper (option 4), otherwise it reintroduces the very fragility the broad catch prevents.
3. **Make the swallow observable.** Keep the broad catch but log/warn when it fires so silent validation-disabling is detectable. *Cost:* the rule is on the per-feature planning hot path, so logging every malformed-but-acceptable feature (path 2) is noise. Observability is only useful if it distinguishes expected data-shape errors (silent or debug) from unexpected `AttributeError`/`NameError` (warning) which means it depends on option 1 or 2 having classified the exceptions first.
4. **Lift the defensiveness into mloda core.** The engine already calls `return_data_type_rule` at one site. Wrapping that single call defensively (catch + classify + log, return `None`) would remove the per-rule boilerplate from all ten registry classes and every other plugin's type rule at once. *Tradeoff to state honestly:* this changes a framework contract. Today an unguarded rule that raises crashes planning; a core wrapper makes **every** plugin's type rule best-effort, overriding crash-on-raise for plugins that might prefer to fail loud. That is defensible for a planning-time hint, but it is not a unilateral edit. It belongs in mloda core, coordinated, not folded into a registry PR.
5. **Guard with a completeness test instead of a runtime catch.** Assert that each supported deterministic op yields its expected non-`None` type, so an extraction regression (path 3) fails CI rather than silently returning `None` at runtime. *This is the reconciliation of the apparent conflict.* "Never crash at planning time" and "surface real bugs" cannot both hold for path 3 at runtime, but they can hold across the lifecycle: the broad catch keeps runtime crash-free, and the completeness test moves bug-detection to CI, where a crash is exactly what you want.

## Decision

Adopt **5 + 3 + the explicit invariant now**, in this registry. Recommend **4** as the long-term direction and scope it as a coordinated follow-up in mloda core. Defer **1/2** until **4** lands (narrowing is only safe once core owns the never-raises guarantee).

Concretely, in this PR:

- **Completeness guarantee (option 5).** `tests/test_return_data_type_rule_invariants.py::test_completeness` enumerates, for all ten base classes, a representative feature for each deterministic op and asserts the rule returns the expected non-`None` `DataType`. This is the spine: any extraction regression that would have silently returned `None` (path 3) now fails CI. It supersedes the per-operation `TestReturnDataTypeRule` cases by reframing them as a single completeness contract rather than ten independent spot checks.
- **Invariant test (restates the contract).** `test_invariant_never_raises` feeds malformed and garbage features to every rule and asserts each returns (`None` is acceptable) and never raises. This pins the reason the broad catch exists so a future narrow cannot quietly regress it.
- **This document (option 3, the recordable half).** The broad catch stays, but its cost is now written down and discoverable, mapped to the three roles, with the proliferation consequence named. Runtime logging is deferred to the core wrapper, where classification (option 1/2) makes it actionable rather than noisy.

Deferred, with a follow-up filed against mloda core:

- **Core wrapper (option 4).** Wrap `engine.set_data_type`'s call to `return_data_type_rule` in a defensive catch that classifies the exception (data-shape -> debug, unexpected -> warning with feature + FG class + exception), returns `None`, and centralizes the never-raises invariant as a framework property. Once it lands, the ten registry catches can be removed and the rules left to raise naturally, with the completeness test still guarding correctness in CI. Tracked cross-repo in [mloda-ai/mloda#485](https://github.com/mloda-ai/mloda/issues/485).

## Regression signals

- `test_completeness` fails if any deterministic op stops returning its declared type (catches path-3 bugs in CI).
- `test_invariant_never_raises` fails if any rule raises on a malformed feature (guards the never-raises invariant against a future narrow).
- If the core wrapper (option 4) lands and the registry catches are removed, both tests must still pass unchanged; that is the acceptance criterion for the removal.
