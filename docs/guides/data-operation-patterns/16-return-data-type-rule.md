# `return_data_type_rule` failure handling

Why the `data_operations` rules no longer wrap op-extraction in a broad `except Exception: return None`, what the fail-fast contract is, and how the decision moved from issue [#244](https://github.com/mloda-ai/mloda-registry/issues/244) to its resolution in [#265](https://github.com/mloda-ai/mloda-registry/issues/265) and mloda core [#485](https://github.com/mloda-ai/mloda/issues/485) / PR [#493](https://github.com/mloda-ai/mloda/pull/493).

**What**: The failure-handling contract for `return_data_type_rule` on the ten `data_operations` base classes that declare a deterministic output type.
**When**: Read this before adding a type rule to a new operation, before adding any defensive catch to one, or before touching the matching/validation of an operation whose rule extracts from the feature.
**Why**: The rule is a planning-time hint that runs only *after* a feature group is already selected, and mloda core calls it unguarded. The contract is therefore fail-fast: a rule that raises signals a real bug in a committed component and must surface, not be swallowed. This page records that decision and how it superseded the earlier "keep a broad catch" position.
**Where**: `return_data_type_rule` on the base classes under `mloda/community/feature_groups/data_operations/`, the framework call site `engine.set_data_type` in mloda core, and the cross-cutting guard test `tests/test_return_data_type_rule_invariants.py`.
**How**: This page states the contract, explains why a selected feature's extraction cannot raise (and how the three families that could were fixed), and records the decision trail.

---

## The contract: fail-fast, post-selection

mloda core calls the rule unguarded. In `engine.set_data_type`:

```python
def set_data_type(self, feature, feature_group_class):
    fg_data_type = feature_group_class.return_data_type_rule(feature)
    if feature.data_type and fg_data_type:
        if feature.data_type != fg_data_type:
            raise ValueError("... data type mismatch ...")
        return fg_data_type
    return fg_data_type or feature.data_type
```

There is no `try/except` around that call, by design. Crucially, `return_data_type_rule` runs **only after the feature group has already been selected** for the feature: selection is decided by `match_feature_group_criteria`, and the rule is used nowhere in selection. So a raise from the rule is not a speculative "this candidate does not apply." It is a failure of an **already-committed** component, and the right behavior is to surface it (fail planning) rather than hide it behind a degraded `None` type.

**This is the contract: for any feature the framework actually hands it (i.e. a feature this group matched), `return_data_type_rule` returns a `DataType` or `None`, and does not raise.** It is allowed to raise on inputs the group would never match, because the framework never routes those to it.

## The shape: pure extract, then map

Each of the ten rules is a pure extract-then-map, with no catch (here, aggregation):

```python
@classmethod
def return_data_type_rule(cls, feature: Feature) -> DataType | None:
    agg_type = cls._extract_aggregation_type(feature)
    if agg_type in {"count", "nunique"}:
        return DataType.INT64
    return None
```

`None` means "no fixed type" (the op is genuinely input-dependent, e.g. `sum` over an unknown numeric column). A concrete `DataType` means the op always emits that type. A raise means a bug: a renamed helper, a typo, a refactor that broke `_extract_*`. Under the previous design a broad `except Exception: return None` swallowed that third case into an indistinguishable `None`, silently disabling output-type validation for the feature with no log, no warning, no test failure. Removing the catch makes that bug surface at planning time, which is where you want it.

## Why a selected feature's extraction cannot raise

The catch was safe to remove because, for every op the framework can actually route to a rule, extraction has what it needs:

- `match_feature_group_criteria` (via `FeatureChainParserMixin`) wraps matching in `except ValueError: return False`, and `parse_feature_name` raises on "pattern-but-no-source." So any name that would make the name-based `parse_feature_name` throw can never be selected.
- On the config path, the discriminator key is a **required** `PROPERTY_MAPPING` entry, so a selected config feature always carries a valid op.

Three families were exceptions where a *selected* feature could still reach a raising extraction. All were fixed at the matching/validation layer (not by re-adding a catch), as part of [#265](https://github.com/mloda-ai/mloda-registry/issues/265):

- **binning.** `PREFIX_PATTERN` was `.*__(bin|qbin)_\d+$`, which matched `value__bin_0`; `_validate_n_bins` then raised on `n_bins < 1`, and config `n_bins` had no validator, so a non-positive or non-numeric value also matched and threw. Fixed by tightening the pattern to `.*__(bin|qbin)_[1-9]\d*$` and adding a positive-integer `type_validator` on the `n_bins` config option. A selected binning feature now always has `n_bins >= 1`.
- **resample.** `PREFIX_PATTERN` used `\d+`, so `value__resample_0_hour_mean` matched and `_parse_resample_op` then raised on `n <= 0`; `resample_op` was not a `PROPERTY_MAPPING` key, so config matching never validated it; and `_token_from_name` used a first-marker `find`, which diverged from the anchored `PREFIX_PATTERN` and mis-parsed a chained `..__resample_..__resample_..` name into an unparseable token. Fixed by tightening the bucket-size group to `[1-9]\d*` (the same rejection binning got), adding `resample_op` to `PROPERTY_MAPPING` with a token `type_validator`, and anchoring `_token_from_name` / `_source_from_name` to the last marker (`rfind`) so they agree with the end-anchored pattern. A selected resample feature now always has `n >= 1` and a parseable token.
- **frame_aggregate.** The config match branch validated the aggregation and frame options but not `in_features`, so a config feature with no source column was selected and the rule's `_extract_params` then raised from `get_in_features()`. Fixed by requiring `in_features` on the config match branch (name-based features are unaffected, since they encode the source in the name). A selected config frame_aggregate feature now always has a source column.

With those three fixes in place, removing all ten catches keeps the fail-fast contract intact: matched features do not raise, and genuinely broken code does.

## Guarding against silent regressions

Removing the runtime catch does not weaken observability of the "hidden bug" case; it strengthens it, and the CI guard is unchanged:

- `tests/test_return_data_type_rule_invariants.py::test_completeness` enumerates, for all ten base classes, a representative feature for each deterministic op and asserts the rule returns the expected non-`None` `DataType`. Any extraction regression that would have silently returned `None` now fails CI here.
- `test_matching_feature_never_raises` asserts the current contract: for every base class, a **matching** feature makes the rule return `None` or a `DataType` without raising. (This replaces the old `test_invariant_never_raises`, which fed arbitrary unselected garbage straight to the rule and demanded no-raise. That encoded a never-raises-for-anything invariant the framework does not actually need, since the rule only ever sees matched features.)
- `test_matched_numeric_boundary_never_raises` generalizes the numeric-boundary axis to **every** family via a registry-driven fuzzer. It reuses the `FAMILIES` registry from `test_prefix_pattern_collisions.py`, takes each family's generated valid feature names, mutates every numeric slot to a zero/non-positive count (`0` and `00`), and asserts that anything a family still matches still types without raising. This replaces the old hand-picked binning/resample boundary rows, so a future op with a loose `\d+` name pattern (one that selects a name its extractor cannot handle) is caught automatically rather than needing a bespoke test. A family's numeric-axis coverage depends on its generator (`generate_valid_names`) emitting a digit-bearing exemplar, so `offset` and `rank` emit their dynamic `lag_/lead_` and `ntile_/top_/bottom_` numeric forms explicitly (the `>= 1` guard lives in the extractor, not the pattern). The options-driven `frame_aggregate` config-without-`in_features` case, which cannot be generated from the name registry, stays explicit in `test_matched_config_boundary_never_raises`.

## Decision trail

- **#244** analyzed the duplicated broad catch and initially recorded "keep the catch, add completeness + invariant guards, and defer removal until mloda core owns a never-raises wrapper" (that wrapper was tracked as core issue #485).
- **mloda core #485 / PR #493.** While implementing the core wrapper, the review established that the rule runs post-selection, so a raise is a committed-component bug that should surface. Core therefore **kept its existing fail-fast behavior and added no wrapper**; #485 was resolved by that different approach, with no core change required. The docs for the rule's `DataType | None` contract were split into a separate merged docs PR.
- **#265 (this change).** With core confirmed fail-fast, the registry removed the ten per-rule `except Exception: return None` catches and fixed the binning/resample matching misalignments the audit surfaced, so no selected feature can make a rule raise. No core dependency.

## Regression signals

- `test_completeness` fails if any deterministic op stops returning its declared type (catches extraction bugs in CI).
- `test_matching_feature_never_raises` fails if a rule raises for a feature its group matches (the fail-fast contract only tolerates raises on inputs that would never be selected).
- The binning, resample, and frame_aggregate matching tests fail if the tightened patterns / added validators / config `in_features` guard regress, which is what keeps a selected feature from reaching a raising extraction.
