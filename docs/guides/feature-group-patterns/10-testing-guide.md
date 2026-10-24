# Feature Group Testing Guide

A 3-level approach to testing feature groups comprehensively.

**What**: A structured testing strategy with unit, framework, and integration levels.
**When**: Always test feature groups; use appropriate level for each concern.
**Why**: Catch bugs early (unit), verify computation (framework), ensure end-to-end works (integration).
**Where**: `tests/` directory alongside your feature group code.
**How**: pytest with mock FeatureSets for unit/framework; `mloda.run_all()` for integration.

## Testing Levels

| Level | Scope | Speed | What to Test |
|-------|-------|-------|--------------|
| 1: Unit | Matching logic | Fast | `match_feature_group_criteria()`, patterns, config methods |
| 2: Framework | Calculation | Medium | `calculate_feature()` with real DataFrames |
| 3: Integration | Full pipeline | Slow | `mloda.run_all()` end-to-end |

Stand-ins for your own groups, used by the examples below:

```python
from typing import Any

from mloda.provider import DataCreator, FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class MyFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> DataCreator | None:
        return DataCreator({"my_feature", "feature_a", "feature_b"})

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

```python
def test_matching():
    # Class name matching
    assert MyFeature.match_feature_group_criteria("MyFeature", Options())
    assert not MyFeature.match_feature_group_criteria("Wrong", Options())

    # Chained pattern matching
    assert ScaledFeature.match_feature_group_criteria("price__scaled", Options())
    assert not ScaledFeature.match_feature_group_criteria("price", Options())

    # Framework restriction
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

`run_all()` returns a list of result frames, one per compute framework and feature set, so index into it before asserting on columns.

### Level 3 Only: The Empty-Result Contract

The rule that a FeatureGroup must return columns even at zero rows (see [calculate_feature](12-calculate-feature.md#empty-results)) is enforced during the run, not inside `calculate_feature()`. A Level 2 test that calls `calculate_feature()` directly passes on a group that returns `[]` or `{}` for "nothing found"; the same group raises `EmptyResultError` the first time that path is hit through a run. Cover every reachable empty path (no source files, a query with no hits, a filter that removes every row) at Level 3:

```python
class EmptyFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> DataCreator | None:
        return DataCreator({"empty_feature"})

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

**Tip**: Use `column_ordering="request_order"` for predictable test assertions:

```python
def test_multiple_features():
    result = mloda.run_all([Feature("feature_a"), Feature("feature_b")], column_ordering="request_order")
    # Columns guaranteed to be in request order
    assert list(result[0]) == ["feature_a", "feature_b"]
```

Without `column_ordering`, column order is non-deterministic and tests comparing column lists may be flaky.

**Note**: You can also use `stream_all` for integration tests when testing streaming behavior. See [Streaming Results](23-streaming.md).

## Mocking Feature Dependencies

Isolate derived features from expensive upstream computations by disabling dependencies and injecting mock data.

```python
from mloda.user import Feature, Options, PluginCollector, mloda


class ExpensiveUpstreamFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> DataCreator | None:
        return DataCreator({"upstream_feature"})

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

## Operation Groups: Conformance to the Reference Contract

Test that each result differs from the reference exactly where `divergent` declares it. The example uses the `ColumnTotal` classes from [09](09-framework-specific.md#reference-contract-and-divergence) (in the same namespace, not imported again).

```py
import math
from typing import Any

from mloda.provider import DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
import pytest

VALUES = [1.0, math.nan, 3.0]

EXPECTED = {
    PandasColumnTotal: {"sum": "skips NaN in sum"},
    PyArrowColumnTotal: {},
}


class Guide10Prices(FeatureGroup):
    """Pandas-only source with a missing value."""

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"guide10_price"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"guide10_price": VALUES}

    @classmethod
    def compute_framework_rule(cls) -> set[type[Any]]:
        return {PandasDataFrame}


def is_missing(value: Any) -> bool:
    return value is None or value != value


def reference(op: str, values: list[float]) -> float | None:
    """The documented reference contract: a missing value propagates in sum, max skips missing values."""
    present = [v for v in values if not is_missing(v)]
    if op == "sum":
        return None if len(present) < len(values) else sum(present)
    return max(present)


def same(a: Any, b: Any) -> bool:
    return (is_missing(a) and is_missing(b)) or a == b


@pytest.mark.parametrize("implementation", [PandasColumnTotal, PyArrowColumnTotal])
@pytest.mark.parametrize("op", ["sum", "max"])
def test_result_differs_from_the_reference_exactly_when_declared(implementation, op):
    assert dict(implementation.divergent) == EXPECTED[implementation]
    name = f"guide10_price__{op}_total"
    result = mloda.run_all(
        [Feature(name, implementation=implementation)],
        compute_frameworks=[PandasDataFrame, PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups({Guide10Prices, ColumnTotal}),
    )[0][name]
    value = list(result.to_pylist() if hasattr(result, "to_pylist") else result)[0]
    differs = not same(value, reference(op, VALUES))
    assert differs == (op in implementation.divergent)
```

Assert the chosen implementation with `mloda.explain(...)` (`step.implementation`). See mloda's own [test_aggregated_subtype_declaration.py](https://github.com/mloda-ai/mloda/blob/0.17.0/tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group/test_aggregated_subtype_declaration.py).

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

`check_plan_lock` never writes: it raises `PlanLockMismatchError` with a diff when the plan drifts, or with the content to write when the lock is missing. Regenerate with `UPDATE_PLAN_LOCKS=1 pytest ...` (the test's own switch, not an mloda setting), review the diff and commit the lock file. The lock records each feature group, reader and framework by `module:qualname`, so moving a class or upgrading mloda can change it too (mloda 0.16.0 moved the lock format to 5, adding input wiring and join keys, so rewrite each lock once after upgrading); they must live in an importable module, since `write_plan_lock` refuses classes defined in `__main__`.

## Testing What an Import Loads

A packaging check alongside Level 1: which plugin modules an import pulls in and which FeatureGroups it loads (see [Package Layout](../04-create-plugin-package.md#package-layout-and-import-time-discovery)). Probe a fresh interpreter, because the test process has already imported every FeatureGroup:

```py
import json
import subprocess
import sys
from typing import Any

_PROBE = """
import json, sys
import {module}
report = {{"mloda": "mloda" in sys.modules, "modules": sorted(m for m in sys.modules if m.startswith("acme."))}}
from mloda.provider import FeatureGroup, get_all_subclasses
report["groups"] = sorted(fg.__name__ for fg in get_all_subclasses(FeatureGroup) if fg.__module__.startswith("acme."))
print(json.dumps(report))
"""


def _import_report(module: str) -> dict[str, Any]:
    probe = [sys.executable, "-c", _PROBE.format(module=module)]
    return json.loads(subprocess.run(probe, check=True, capture_output=True, text=True).stdout)


def test_helper_import_loads_no_mloda() -> None:
    assert _import_report("acme.core.parsers") == {
        "mloda": False,
        "modules": ["acme.core", "acme.core.parsers"],
        "groups": [],
    }


def test_plugin_import_loads_only_its_groups() -> None:
    assert _import_report("acme.feature_groups.my_plugin")["groups"] == ["MyFeatureGroup"]
```

`get_all_subclasses` also returns abstract bases, so list those in the expected groups too.

## Testing by Pattern

| Pattern | Level 1 Focus | Level 2 Focus |
|---------|---------------|---------------|
| Root | `input_data()` returns instance (a source group: its claim routes and `find_sources`) | Data loading works |
| Derived | `input_features()` correct | Transformation logic |
| Chained | Pattern regex matching | Suffix extraction |
| Multi-input | `&` parsing | Multiple inputs combined |
| Multi-output | Column count | `~` naming correct |
| Artifact | `artifact()` returns class | Save/load cycle |
| Index | `index_columns()` correct | Ordering respected |
| Framework | `compute_framework_rule()` | Framework-specific ops |
| Operation group | Implementations load; `supported` / `accepts` | Conformance to the reference contract |

## Real Test Examples

| Directory | Pattern |
|-----------|---------|
| [test_base_aggregated_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group) | Chained + Index |
| [test_clustering_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_clustering_feature_group) | Artifact |
| [test_geo_distance_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_geo_distance_feature_group) | Multi-input |
| [test_dimensionality_reduction_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_dimensionality_reduction_feature_group) | Multi-output + Artifact |
