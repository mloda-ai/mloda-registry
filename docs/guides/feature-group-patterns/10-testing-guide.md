# Feature Group Testing Guide

**What**: A structured testing strategy with unit, framework, and integration levels.
**When**: Always test feature groups; use appropriate level for each concern.
**Why**: Catch bugs early (unit), verify computation (framework), ensure end-to-end works (integration).
**Where**: `tests/` directory alongside your feature group code.
**How**: the shipped `mloda.testing` kit first; pytest with mock FeatureSets for unit/framework; `mloda.run_all()` for integration.

## Start from the Shipped Kit

`pip install mloda-testing`, then start from the check that fits your plugin; each module lives under `mloda.testing`.

| Your plugin | Start from | Module | Covered in |
|-------------|------------|--------|------------|
| Plain FeatureGroup | `assert_feature_group_shape` | `feature_group_shape` | [Level 1 of this guide](#level-1-unit-test-example) |
| Subclass that takes over on an option | `assert_subclass_scope` | `subclass_scope` | [Subclasses That Take Over on an Option](#subclasses-that-take-over-on-an-option) |
| Source or reader group | `SourceFeatureGroupTestMixin`, `ReadFileFGTestMixin`, `ReadDBFGTestMixin`, `ReadDocumentFGTestMixin`, the loads-into-framework mixins, `ColumnDiscoveryContractTestMixin` | `source` | [Source FeatureGroups: Test](27-input-data-readers.md#test) |
| Operation group | `assert_operation_group_shape`; a data-operation family uses `OperationGroupTestBase` (it already runs that assert) and the data-operation mixins | `operation_group_shape`, `feature_groups.data_operations` | [Operation Groups](#operation-groups-conformance-to-the-reference-contract) and [adding an operation](../data-operation-patterns/10-adding-new-operation.md) |
| Binary-backed group | simulated binary, simulated license, `BinaryModelConformanceBase` (extra `mloda-testing[binary-model]`) | `binary_model` | [Binary-Backed Features: Test](29-binary-backed-features.md#test) |
| Extender package | `ExtenderContractTestMixin` (extra `mloda-testing[pyarrow]`) | `extenders` | [Create an Extender: Testing](../11-create-extender.md#testing) |
| Extender package with an optional dependency | `OptionalDependencyPackageTestMixin` | `optional_dependency` | [Create an Extender: Optional dependency packages](../11-create-extender.md#optional-dependency-packages) |
| Any package: what an import loads, a missing dependency | `block_root`, `evict_root`, `evict_package`, `evict_entry_points`, `fresh_import_report` | `import_isolation` | [Testing What an Import Loads](#testing-what-an-import-loads) |
| Compute framework | `assert_compute_framework_shape`, the suites, `TypedTestData`, `FrameworkAdapter` | `compute_framework_shape`, `compute_framework`, `typed_data`, `framework_adapter` | [Compute Framework Testing Guide](../compute-framework-patterns/09-testing-guide.md) |

The source suites (including column discovery and loads-into-framework), `ExtenderContractTestMixin` and `BinaryModelConformanceBase` are `ConformanceSuite` bases: name your host class `Test*`, set its hooks, and switch an inherited test off only through `switched_off` (see [ExtenderContractTestMixin](../11-create-extender.md#extendercontracttestmixin)). `FeatureGroupTestBase` (`mloda.testing.base`) only carries a `feature_group_class` attribute and adds no tests.

## Testing Levels

| Level | Scope | Speed | What to Test |
|-------|-------|-------|--------------|
| 1: Unit | Matching logic | Fast | `match_feature_group_criteria()`, patterns, config methods |
| 2: Framework | Calculation | Medium | `calculate_feature()` with real DataFrames |
| 3: Integration | Full pipeline | Slow | `mloda.run_all()` end-to-end |

Stand-ins for your own groups, used by the examples below:

```python
from typing import Any

from mloda.provider import DataCreatorFeatureGroup, FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class MyFeature(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"my_feature"}

    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {str(name): [1, 2, 3] for name in features.get_all_names()}


class ScaledFeature(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r"^.+__scaled$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        name = str(features.name_of_one_feature)
        col = data[name.replace("__scaled", "")]
        return (col - col.min()) / (col.max() - col.min())
```

## Level 1: Unit Test Example

`assert_feature_group_shape` from `mloda.testing.feature_group_shape` checks the match contract of a plain FeatureGroup: the declared names match, a foreign name does not, `input_features()` resolves, required and strict options are enforced, no option is secret-shaped, and `version()` and `compute_framework_rule()` are well formed. It ships in `mloda-testing` 0.14+; on mloda 0.20 without it, import from core's `mloda._testing.feature_group_shape`. A pattern-matching group passes the names to probe as `requests=`. `check_feature_group_shape` returns the `CheckResult` (failures and notes) instead of failing. Keep hand asserts for what is specific to the group:

```python
from mloda.testing.feature_group_shape import assert_feature_group_shape


def test_my_feature_shape():
    assert_feature_group_shape(MyFeature)


def test_scaled_feature_shape():
    assert_feature_group_shape(ScaledFeature, requests=["price__scaled"])


def test_scaled_feature_specifics():
    assert not ScaledFeature.match_feature_group_criteria("price", Options())
    assert PandasDataFrame in ScaledFeature.compute_framework_rule()
```

## Level 2: Framework Test Example

```python
import pandas as pd
from mloda.user import FeatureName


def test_calculate_feature():
    df = pd.DataFrame({"input": [1, 2, 3]})

    class MockFeatures:
        name_of_one_feature = FeatureName("input__scaled")

    result = ScaledFeature.calculate_feature(df, MockFeatures())
    assert len(result) == 3
    assert result.min() >= 0 and result.max() <= 1
```

With `input_data=` the shape check also runs the group through the adapter's compute framework (default `PythonDictAdapter`) and checks the output columns, their types, a zero-row input and determinism. A run the check cannot make, such as an adapter whose framework the group's rule excludes, is a failure unless `require_run=False`.

```python
from mloda.testing.framework_adapter import PandasAdapter
from mloda.testing.typed_data import TypedTestData
from mloda.user import DataType


def test_scaled_feature_runs():
    assert_feature_group_shape(
        ScaledFeature,
        requests=["price__scaled"],
        input_data=TypedTestData({"price": (DataType.DOUBLE, [1.0, 2.0, 3.0])}),
        adapter=PandasAdapter(),
    )
```

`TypedTestData` raises `ValueError` for a value that does not fit its declared `DataType` (a `bool` in an `INT64` column, an int beyond 2**53 in a `DOUBLE` one). `check_values_fit(columns, types)` from `mloda.testing.value_fit` applies the same rules to plain lists such as hand-written expected columns and returns a `CheckResult`.

A group that reads a credential calls `calculate_feature` inside `credentials_scope` and keeps one Level 3 test with `data_access_collection=` (on `DataOpsIntegrationTestBase`, the `data_access_collection()` hook; see [Secrets](11-options.md#secrets)).

## Level 3: Integration Test Example

```python
from mloda.user import Feature, PluginCollector, mloda


def test_full_pipeline():
    result = mloda.run_all(
        [Feature.not_typed("my_feature")],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({MyFeature}),
    )
    assert "my_feature" in result[0]
```

`run_all()` returns one frame per step that computed a requested feature. Features of one group that share a call (see [which features share a call](12-calculate-feature.md#which-features-share-a-call)) share a frame; each other requested chain link, group or compute framework adds one, so requesting only a chain's final feature returns one frame. Pick a frame with `result.frame_for("name")` (pass a `Feature` with its options when a name has several frames). `run_all` and prepared sessions have no limit on groups or features per run; only the CLI/MCP request contract caps a request at 50 features.

### Level 3: Timing a Step

`result.step_durations` holds one `StepDuration(feature_group, step_uuid, duration_seconds)` per completed feature group step, so a test can enforce a per-step budget without patching `calculate_feature`. A group can run as several steps (chain links, differing group options), so join on `step_uuid` to budget each one.

```python
BUDGET_SECONDS = {"my_feature": 5.0}


def test_my_feature_within_budget():
    result = mloda.run_all(
        [Feature.not_typed("my_feature")],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({MyFeature}),
    )
    steps = {step.step_uuid: step for step in result.plan}
    entries = [e for e in result.step_durations if e.feature_group == MyFeature.get_class_name()]
    assert entries
    for entry in entries:
        (name,) = steps[entry.step_uuid].feature_names
        assert entry.duration_seconds < BUDGET_SECONDS[name], name
```

The duration covers the `calculate_feature` call only, so on a lazy framework such as `PolarsLazyDataFrame` or `DuckDBFramework` it measures building the frame, not executing it; see [the mloda API doc](https://github.com/mloda-ai/mloda/blob/0.21.0/docs/docs/in_depth/mloda-api.md#explain-and-resolved_plan).

### Level 3 Only: The Empty-Result Contract

The rule that a FeatureGroup must return columns even at zero rows (see [calculate_feature](12-calculate-feature.md#empty-results)) is enforced during the run, not inside `calculate_feature()`. A Level 2 test that calls `calculate_feature()` directly passes on a group that returns `[]` or `{}` for "nothing found"; the same group raises `EmptyResultError` the first time that path is hit through a run. Cover every reachable empty path (no source files, a query with no hits, a filter that removes every row) at Level 3:

```python
class EmptyFeature(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"empty_feature"}

    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"empty_feature": []}  # nothing found, but the column is kept


def test_empty_source_keeps_schema():
    result = mloda.run_all(
        [Feature.not_typed("empty_feature")],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({EmptyFeature}),
    )
    assert result[0] == {"empty_feature": []}  # zero rows, schema intact
```

**Tip**: pass `column_ordering="request_order"` when asserting on column lists; the default order is not guaranteed ([Column Ordering](../01-use-existing-plugin.md#column-ordering)).

**Note**: You can also use `stream_all` for integration tests when testing streaming behavior. See [Streaming Results](23-streaming.md).

## Subclasses That Take Over on an Option

`assert_subclass_scope(Subclass, Parent, selecting)` from `mloda.testing.subclass_scope` checks that the parent resolves without the selecting options, the subclass resolves with them, and another value of a selecting key does not select the subclass (gating on the key's mere presence fails as a steal). With `input_data=` (a `TypedTestData`) it also runs both groups and compares the output columns; a changed column passes only when a divergence record the subclass declares covers it, and a comparison that cannot run fails unless `require_run=False`; details in [core's doc](https://github.com/mloda-ai/mloda/blob/0.21.0/docs/docs/in_depth/troubleshooting/feature-group-resolution-errors.md#a-loaded-subclass-replaces-its-parent). A pattern-matching parent needs `requests=`. A selecting key that configures the subclass without selecting it also goes in `free=`, which skips its other-value probe.

```python
from mloda.testing.subclass_scope import assert_subclass_scope
from mloda.user import Feature, Options


class Prices(FeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"price_eur"}

    def input_features(self, options: Options, feature_name: Any) -> set[Feature] | None:
        return {Feature.not_typed("price")}

    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"price_eur": list(data["price"])}


class RoundedPrices(Prices):
    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Any, options: Options, data_access_collection: Any = None
    ) -> bool:
        selected = options.get("rounding") == "cents"
        return selected and super().match_feature_group_criteria(feature_name, options, data_access_collection)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"price_eur": [round(v, 2) for v in data["price"]]}


def test_rounded_prices_scope():
    assert_subclass_scope(RoundedPrices, Prices, Options(group={"rounding": "cents"}))
```

## Mocking Feature Dependencies

Isolate derived features from expensive upstream computations by disabling dependencies and injecting mock data.

```python
from mloda.user import Feature, Options, PluginCollector, mloda


class ExpensiveUpstreamFeature(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"upstream_feature"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RuntimeError("expensive call")


class MyDerivedFeature(FeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Any, options: Options, data_access_collection: Any = None
    ) -> bool:
        return str(feature_name) == "my_derived_feature"

    def input_features(self, options: Options, feature_name: Any) -> set[Feature] | None:
        return {Feature.not_typed("upstream_feature")}

    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"my_derived_feature": [v * 2 for v in data["upstream_feature"]]}


def test_derived_feature_isolated():
    result = mloda.run_all(
        [Feature.not_typed("my_derived_feature")],
        api_data={"MockData": {"upstream_feature": [100, 200, 300]}},
        plugin_collector=PluginCollector.disabled_feature_groups({ExpensiveUpstreamFeature}),
    )

    assert len(result) > 0
```

Use when upstream features are slow (API calls, ML inference) or you need controlled test data.

## Groups That Wrap a Model the Gate Cannot Install

Some groups get their result from a model too large or slow for CI (for example an NER model). Test them in three parts:

- One corpus of texts with the result each must give, read by every check.
- A fake model built from that corpus, so the gate proves the group's own mechanism on every OS. For spaCy, use a blank pipeline with an entity ruler; the example uses a plain callable so it needs no model.
- The same `test_corpus` run on the real model, skipped where it is not installed and run in a release job (as in [Against the real wheel](29-binary-backed-features.md#against-the-real-wheel)).

Record `tagged` from the real model rather than guessing it; the real run's tagged assert keeps the fake honest. Run the fake only through `enabled_feature_groups`, since it matches the same names as the real group. A last test fails when the group's rules outgrow the corpus: every `KEEP_CLEAR` word needs a case the model tags, or the fake never produces it.

```python
from functools import cache
from importlib.util import find_spec
from typing import Any, NamedTuple

import pytest

from mloda.provider import ApiDataFG, FeatureGroup, FeatureSet
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

KEEP_CLEAR = {"Claude"}  # a word the model misreads as a name; the group drops it


class Case(NamedTuple):
    text: str
    tagged: tuple[tuple[str, str], ...]  # (text, label) the model tags, misreads included
    names: tuple[str, ...]  # what the group must return


CORPUS = [
    Case("Anna Schmidt works at Google.", (("Anna Schmidt", "PERSON"), ("Google", "ORG")), ("Anna Schmidt",)),
    Case("Ask Claude to run the tests.", (("Claude", "PERSON"),), ()),
    Case("The build finished.", (), ()),
]


class PersonNames(FeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Any, options: Options, data_access_collection: Any = None
    ) -> bool:
        return str(feature_name) == "person_names"

    def input_features(self, options: Options, feature_name: Any) -> set[Feature] | None:
        return {Feature.not_typed("text")}

    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PythonDictFramework}

    @staticmethod
    @cache
    def load_model() -> Any:
        import spacy  # imported here so the gate never needs the model

        nlp = spacy.load("en_core_web_md")
        return lambda text: [(e.text, e.label_) for e in nlp(text).ents]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        model = cls.load_model()
        names = [[n for n, label in model(t) if label == "PERSON" and n not in KEEP_CLEAR] for t in data["text"]]
        return {"person_names": names}


def fake_model() -> Any:
    tagged = {case.text: list(case.tagged) for case in CORPUS}
    return lambda text: tagged[text]


class FakePersonNames(PersonNames):
    load_model = staticmethod(fake_model)


GROUPS = {"fake": FakePersonNames, "real": PersonNames}
MISSING = find_spec("en_core_web_md") is None
REAL = pytest.mark.skipif(MISSING, reason="real model not installed")


@pytest.mark.parametrize("kind", ["fake", pytest.param("real", marks=REAL)])
def test_corpus(kind: str) -> None:
    model = GROUPS[kind].load_model()
    for case in CORPUS:
        assert tuple(tuple(pair) for pair in model(case.text)) == case.tagged, case.text  # keeps the fake honest
    result = mloda.run_all(
        [Feature.not_typed("person_names")],
        compute_frameworks=[PythonDictFramework],
        api_data={"CorpusData": {"text": [case.text for case in CORPUS]}},
        plugin_collector=PluginCollector.enabled_feature_groups({GROUPS[kind], ApiDataFG}),
    )
    assert result[0]["person_names"] == [list(case.names) for case in CORPUS]


def test_every_keep_clear_word_has_a_case() -> None:
    tagged = {word for case in CORPUS for word, label in case.tagged if label == "PERSON"}
    assert KEEP_CLEAR <= tagged, "the fake never produces these, so the gate proves nothing"
```

## Operation Groups: Conformance to the Reference Contract

Test that each result differs from the reference exactly where `divergences` declares it. The example uses the `ColumnTotal` classes from [09](09-framework-specific.md#reference-contract-and-divergence) (in the same namespace, not imported again).

```py
import math
from typing import Any

from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
import pytest

from mloda.testing.feature_groups.data_operations.base import divergent_subtypes
from mloda.testing.feature_groups.data_operations.helpers import is_null
from mloda.testing.operation_group_shape import assert_operation_group_shape

VALUES = [1.0, math.nan, 3.0]

EXPECTED = {
    PandasColumnTotal: {"sum"},
    PyArrowColumnTotal: set(),
}


class Guide10Prices(DataCreatorFeatureGroup):
    """Pandas-only source with a missing value."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"guide10_price"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"guide10_price": VALUES}

    @classmethod
    def compute_framework_rule(cls) -> set[type[Any]]:
        return {PandasDataFrame}


def reference(op: str, values: list[float]) -> float | None:
    """The documented reference contract: a missing value propagates in sum, max skips missing values."""
    present = [v for v in values if not is_null(v)]
    if op == "sum":
        return None if len(present) < len(values) else sum(present)
    return max(present)


def same(a: Any, b: Any) -> bool:
    return (is_null(a) and is_null(b)) or a == b


@pytest.mark.parametrize("implementation", [PandasColumnTotal, PyArrowColumnTotal])
@pytest.mark.parametrize("op", ["sum", "max"])
def test_result_differs_from_the_reference_exactly_when_declared(implementation, op):
    assert divergent_subtypes(implementation, named_only=True) == EXPECTED[implementation]
    name = f"guide10_price__{op}_total"
    result = mloda.run_all(
        [Feature(name, implementation=implementation)],
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups({Guide10Prices, ColumnTotal}),
    )[0][name]
    value = list(result.to_pylist() if hasattr(result, "to_pylist") else result)[0]
    differs = not same(value, reference(op, VALUES))
    assert differs == (op in divergent_subtypes(implementation, named_only=True))


def test_column_total_shape():
    assert_operation_group_shape(ColumnTotal)
```

`assert_operation_group_shape` (`mloda.testing.operation_group_shape`) checks that the group declares a `CONTRACT_VERSION`, that each Implementation's `supported` and `divergences` stay inside the subtype universe, that every divergence record gives a reason, and that unsupported subtypes are declined; pass `alias_pairs=` to check that two aliased subtypes declare the same divergences and `probe=` for a group that resolves subtypes without an option key. A data-operation family gets the shape check from `OperationGroupTestBase` and the reference comparison from its family test base (see [adding an operation](../data-operation-patterns/10-adding-new-operation.md)).

Assert the chosen implementation with `mloda.explain(...)` (`step.implementation`). See mloda's own [test_aggregated_subtype_declaration.py](https://github.com/mloda-ai/mloda/blob/0.21.0/tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group/test_aggregated_subtype_declaration.py).

## Locking a Shipped Plan

An integration-scope check that resolves a request without running it: for a plugin that ships request definitions (recipes, demos), lock the resolved plan so a planner, compute framework or source group change fails a test. `mloda.explain` takes `run_all`'s planning arguments as keywords; pass the ones the shipped request uses, an ordered `compute_frameworks` list, a pinned `plugin_collector` so other installed plugins cannot change the plan, and `parallelization_modes={ParallelizationMode.SYNC}`, which `run_all` defaults to and `explain` does not:

```py
import os
from pathlib import Path

from mloda.steward import check_plan_lock, write_plan_lock
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

LOCK = Path(__file__).with_name("my_recipe.plan.lock.json")


def test_recipe_plan_matches_lock() -> None:
    plan = mloda.explain(
        [Feature.not_typed("my_feature")],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({MyFeature, MySource}),
        parallelization_modes={ParallelizationMode.SYNC},
    )
    if os.environ.get("UPDATE_PLAN_LOCKS"):
        write_plan_lock(plan, LOCK)
    check_plan_lock(plan, LOCK)
```

`check_plan_lock` never writes: it raises `PlanLockMismatchError` with a diff when the plan drifts, or with the content to write when the lock is missing. Regenerate with `UPDATE_PLAN_LOCKS=1 pytest ...` (the test's own switch, not an mloda setting), review the diff and commit the lock file. The lock records each feature group, reader and framework by `module:qualname`, so moving a class or upgrading mloda can change it too (mloda 0.16.0 moved the lock format to 5, adding input wiring and join keys, and mloda 0.20.0 moved it to 9, so rewrite each lock once after upgrading); they must live in an importable module, since `write_plan_lock` refuses classes defined in `__main__`.

## Refusing a Consumer of a Row-Replacing Output

The refusal is a plan-time check, so `mloda.explain` catches it without running anything. A `Link` on the key column makes the same request run (see [Joining a replaces_rows() Output](08-links-joins.md#joining-a-replaces_rows-output)):

```python
from typing import Any

import pytest
from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, JoinSpec, Link, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class Guide10Sales(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"guide10_region", "guide10_amount"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"guide10_region": ["a", "a", "b"], "guide10_amount": [1, 3, 5]}


class Guide10RegionTotal(FeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"guide10_region_total"}

    @classmethod
    def replaces_rows(cls) -> bool:
        return True

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.not_typed("guide10_region"), Feature.not_typed("guide10_amount")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        totals: dict[str, int] = {}
        for region, amount in zip(data["guide10_region"], data["guide10_amount"]):
            totals[region] = totals.get(region, 0) + amount
        return {"guide10_region": list(totals), "guide10_region_total": list(totals.values())}


class Guide10Share(FeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"guide10_share"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.not_typed("guide10_region_total"), Feature.not_typed("guide10_amount")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"guide10_share": [a / t for a, t in zip(data["guide10_amount"], data["guide10_region_total"])]}


GUIDE10_GROUPS = {Guide10Sales, Guide10RegionTotal, Guide10Share}


def test_consumer_without_link_is_refused() -> None:
    with pytest.raises(ValueError, match="different frames"):
        mloda.explain(
            [Feature.not_typed("guide10_share")],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups(GUIDE10_GROUPS),
        )


def test_consumer_with_link_computes() -> None:
    link = Link.left(JoinSpec(Guide10Sales, "guide10_region"), JoinSpec(Guide10RegionTotal, "guide10_region"))
    result = mloda.run_all(
        [Feature.not_typed("guide10_share")],
        compute_frameworks=[PythonDictFramework],
        links={link},
        plugin_collector=PluginCollector.enabled_feature_groups(GUIDE10_GROUPS),
    )
    assert result[0]["guide10_share"] == [0.25, 0.75, 1.0]
```

## Testing What an Import Loads

A packaging check alongside Level 1: which plugin modules an import pulls in and which FeatureGroups it loads (see [Package Layout](../04-create-plugin-package.md#package-layout-and-import-time-discovery)). `mloda.testing.import_isolation` holds the helpers. These four take a `pytest.MonkeyPatch` first and are undone with it:

| Helper | Effect |
|--------|--------|
| `block_root(monkeypatch, root)` | every import of `root` raises `ModuleNotFoundError` |
| `evict_root(monkeypatch, root)` | drops `root` and its submodules, so the next import runs them again |
| `evict_package(monkeypatch, dotted)` | the same for one package and its `manifest`, and removes what the test imports under it for the first time |
| `evict_entry_points(monkeypatch, group)` | removes the entry-point modules of `group` that the test loads for the first time |

Evict the helper's package and block `mloda`, then import it: the import fails if the helper needs mloda.

```python
import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from mloda.testing.import_isolation import block_root, evict_root


@pytest.fixture
def acme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stand-in for your installed package, imported as a real test module would have."""
    core = tmp_path / "acme" / "core"
    core.mkdir(parents=True)
    (core / "parsers.py").write_text("def parse(line):\n    return line.split(',')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.import_module("acme.core.parsers")
    yield
    for name in [m for m in sys.modules if m.split(".")[0] == "acme"]:
        del sys.modules[name]


def test_helper_import_loads_no_mloda(acme: None) -> None:
    with pytest.MonkeyPatch.context() as patch:
        evict_root(patch, "acme")  # the import below runs the modules again
        block_root(patch, "mloda")  # any mloda import now raises ModuleNotFoundError
        importlib.import_module("acme.core.parsers")
        loaded = sorted(m for m in sys.modules if m.startswith("acme."))
    assert loaded == ["acme.core", "acme.core.parsers"]
```

The `with` block keeps mloda blocked only while the helper imports. `evict_root` restores only modules that were imported before, so a real test imports the helper at module level (the fixture does that for the stand-in). `block_root` does not catch an import the helper guards with `try` / `except ImportError`.

A missing optional backend must make the loader skip that manifest. This example runs against your installed package and its entry point. Block the backend and load the entry points cold:

```py
from mloda.testing.import_isolation import block_root, evict_entry_points, evict_package
from mloda.user import PluginLoader


def test_loader_skips_the_rdf_manifest_without_rdflib(monkeypatch: pytest.MonkeyPatch) -> None:
    block_root(monkeypatch, "rdflib")
    evict_package(monkeypatch, "acme.feature_groups.rdf")  # the package of the optional manifest and its groups
    evict_entry_points(monkeypatch, "mloda.feature_groups")
    keys = PluginLoader().load_entry_points(group="mloda.feature_groups")
    assert not any(key.startswith("acme.feature_groups.rdf.") for key in keys)
```

Evict only the package of the optional manifest, since a cold import of a module whose FeatureGroups are already loaded defines them a second time.

Which FeatureGroups an import loads needs a fresh interpreter: the test process has already imported every FeatureGroup, and a cold in-process import of a plugin module would define each one a second time. `fresh_import_report(module, root)` imports `module` in a fresh interpreter that uses the caller's `sys.path` and returns an `ImportReport`: the modules and FeatureGroups (qualified names) under `root`, and whether mloda core was loaded. In-process patches such as `block_root` do not reach the child.

```python
from mloda.testing.import_isolation import ImportReport, fresh_import_report


@pytest.fixture
def acme_plugin(acme: None, tmp_path: Path) -> None:
    plugin = tmp_path / "acme" / "feature_groups" / "my_plugin"
    plugin.mkdir(parents=True)
    (tmp_path / "acme" / "feature_groups" / "__init__.py").write_text("")
    (plugin / "__init__.py").write_text(
        "from mloda.provider import FeatureGroup\n\n\nclass MyFeatureGroup(FeatureGroup):\n    pass\n"
    )


def test_imports_load_only_their_groups(acme_plugin: None) -> None:
    parsers = fresh_import_report("acme.core.parsers", "acme")
    assert parsers == ImportReport(mloda_core=False, modules=("acme", "acme.core", "acme.core.parsers"), groups=())
    assert fresh_import_report("acme.feature_groups.my_plugin", "acme").groups == (
        "acme.feature_groups.my_plugin.MyFeatureGroup",
    )
```

The report lists abstract bases too, so include them in the expected groups.

## Testing by Pattern

| Pattern | Level 1 Focus | Level 2 Focus |
|---------|---------------|---------------|
| Root | `feature_names_supported()` (a source group: `SourceFeatureGroupTestMixin`, its claim routes and `find_sources`) | Data loading works |
| Derived | `input_features()` correct | Transformation logic |
| Chained | Pattern regex matching | Suffix extraction |
| Multi-input | `&` parsing | Multiple inputs combined |
| Multi-output | Column count | `~` naming correct |
| Artifact | `artifact()` returns class | Save/load cycle |
| Index | `index_columns()` correct | Ordering respected |
| Framework | `compute_framework_rule()` | Framework-specific ops |
| Operation group | `assert_operation_group_shape` | Conformance to the reference contract |
| Model-backed | Every `KEEP_CLEAR` word has a corpus case | Corpus on the fake in the gate (Level 3); the real model in a release job |

## Real Test Examples

| Directory | Pattern |
|-----------|---------|
| [test_base_aggregated_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group) | Chained + Index |
| [test_clustering_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_clustering_feature_group) | Artifact |
| [test_geo_distance_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_geo_distance_feature_group) | Multi-input |
| [test_dimensionality_reduction_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_dimensionality_reduction_feature_group) | Multi-output + Artifact |
