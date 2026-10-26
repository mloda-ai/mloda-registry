# Pattern 9: Framework-Specific Features

An operation that runs on several frameworks is one `OperationFeatureGroup` plus one `Implementation` per framework.

**What**: One operation group (matching, inputs, shared loop) and one `Implementation` class per compute framework.
**When**: The same operation must run natively on Pandas, PyArrow, Polars, DuckDB, SQLite or others.
**Why**: One feature name and one set of options for every framework; the framework is a deployment choice, not a new feature group.
**Where**: Pandas groupby, Polars lazy expressions, PyArrow compute, DuckDB SQL, SQLite SQL.
**How**: Subclass `OperationFeatureGroup` for the operation. Subclass `Implementation[Group]` per framework, set `feature_group` and `compute_framework`, and make `calculate` a one-liner through the serving group.

## Key Characteristic

| Piece | Behavior |
|-------|----------|
| `OperationFeatureGroup` | Frameworks are the ones its loaded Implementations serve; core intersects `compute_framework_rule()` with them, so leave the rule unset |
| `Implementation.feature_group` / `compute_framework` | The pair it serves; required, checked at class definition |
| `Implementation.calculate(group, data, features, framework)` | Runs the feature; call the group's shared loop |
| `Implementation.supported` | `frozenset` of subtypes it can run; `None` (default) = all |
| `Implementation.divergent` | `{subtype: reason}` where it differs from the reference result |
| `Feature(..., implementation=...)` | Pin by class, qualified `module:Class` string, bare name (only if unique among loaded classes) or list of these |

## Complete Example

The shared loop lives on the group as a classmethod that takes the framework step as a keyword-only callable. Each `calculate` passes its own step.

```python
from collections.abc import Callable
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import (
    ComputeFramework,
    DataCreatorFeatureGroup,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    Implementation,
    OperationFeatureGroup,
)
from mloda.user import Feature, PluginLoader, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class DoubledFeatureGroup(FeatureChainParserMixin, OperationFeatureGroup):
    """One operation: `<source>__doubled`. Frameworks are the Implementations below."""

    PREFIX_PATTERN = r"^.+__doubled$"  # the mixin matches names and derives input_features from it
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    @classmethod
    def double_all(cls, data: Any, features: FeatureSet, *, double: Callable[[Any, str, str], Any]) -> Any:
        """Shared loop: framework-agnostic, the framework step is a keyword-only callable."""
        for feature in features.features:
            data = double(data, feature.name, feature.name.removesuffix("__doubled"))
        return data


class PandasDoubled(Implementation[DoubledFeatureGroup]):
    feature_group = DoubledFeatureGroup
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(
        cls, group: type[DoubledFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.double_all(data, features, double=cls._double)

    @staticmethod
    def _double(data: pd.DataFrame, name: str, source: str) -> pd.DataFrame:
        data[name] = data[source] * 2
        return data


class PyArrowDoubled(Implementation[DoubledFeatureGroup]):
    feature_group = DoubledFeatureGroup
    compute_framework = PyArrowTable

    @classmethod
    def calculate(
        cls, group: type[DoubledFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.double_all(data, features, double=cls._double)

    @staticmethod
    def _double(data: pa.Table, name: str, source: str) -> pa.Table:
        return data.append_column(name, pc.multiply(data[source], 2))


class ValueSource(DataCreatorFeatureGroup):
    """Root data for the run below."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"value"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"value": [1, 2]}
```

`FeatureChainParserMixin` (Pattern 3) is what makes the group match `value__doubled`; `OperationFeatureGroup` alone does not.

Capability and differences are declared on the Implementation, next to its code. Both are keyed by the group's canonical subtype, so they need a group that declares `SUBTYPES` (a `SubtypeDeclaration`, see core's [`AggregatedFeatureGroup`](https://github.com/mloda-ai/mloda/blob/0.18.0/mloda_plugins/feature_group/experimental/aggregated_feature_group/base.py)):

```py
from mloda.provider import Implementation


class LimitedImplementation(Implementation[MyOperationGroup]):
    feature_group = MyOperationGroup
    compute_framework = PyArrowTable
    supported = frozenset({"sum", "min", "max"})  # other subtypes: this framework is not offered
    divergent = {"sum": "skips NaN"}  # runs, but differs from the reference for sum
```

- `supported`: a feature whose subtype is outside the set is not offered on that framework. Reject up front; do not compute the result in Python inside the backend.
- `divergent`: a declared, accepted difference from the reference. When several frameworks are enabled, non-divergent implementations rank before cost.

## Pin an Implementation

With several frameworks enabled, mloda picks one. Pin it per feature:

Pin by class or by the qualified string `"module:Class"`. A bare name (`"PandasDoubled"`) works only when it is unique among loaded classes; two loaded classes with the same name raise `ImplementationPinError`.

```python
PluginLoader.all()

by_class = Feature("value__doubled", implementation=PandasDoubled)
by_qualified = Feature("value__doubled", implementation=f"{PyArrowDoubled.__module__}:{PyArrowDoubled.__qualname__}")
by_list = Feature("value__doubled", implementation=[PandasDoubled])

frameworks = ["PandasDataFrame", "PyArrowTable"]
pandas_out = mloda.run_all([by_class], compute_frameworks=frameworks)[0]
assert list(pandas_out["value__doubled"]) == [2, 4]
arrow_out = mloda.run_all([by_qualified], compute_frameworks=frameworks)[0]
assert arrow_out["value__doubled"].to_pylist() == [2, 4]
```

## Reference Contract and Divergence

`divergent` is measured against a reference contract documented on the group, keyed by the subtypes it declares in `SUBTYPES`. Implementations that follow the contract declare nothing:

```python
from collections.abc import Callable
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import (
    ComputeFramework,
    FeatureChainParserMixin,
    FeatureSet,
    Implementation,
    OperationFeatureGroup,
    SubtypeDeclaration,
    property_spec,
)
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class ColumnTotal(FeatureChainParserMixin, OperationFeatureGroup):
    """price__sum_total, price__max_total. Reference contract: a missing value (NaN or null) propagates in sum; max skips missing values."""

    OP = "total_op"
    SUBTYPES = SubtypeDeclaration(key=OP)
    PREFIX_PATTERN = r".*__(?P<total_op>sum|max)_total$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1
    PROPERTY_MAPPING: ClassVar = {
        OP: property_spec("Total to compute", strict=True, allowed_values={"sum": "Sum", "max": "Maximum"}),
    }

    @classmethod
    def total(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        perform: Callable[[Any, str, str], Any],
        add_column: Callable[[Any, str, Any], Any],
    ) -> Any:
        """Shared loop: the keyword-only callables are the per-framework contract."""
        for feature in features.get_sorted_features():
            op = cls._resolve_operation(feature, cls.OP)
            source = cls._extract_source_features(feature)[0]
            data = add_column(data, feature.name, perform(data, str(op), source))
        return data


class PandasColumnTotal(Implementation[ColumnTotal]):
    feature_group = ColumnTotal
    compute_framework = PandasDataFrame
    divergent = {"sum": "skips NaN in sum"}

    @classmethod
    def calculate(
        cls, group: type[ColumnTotal], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.total(data, features, perform=cls._perform, add_column=cls._add_column)

    @classmethod
    def _perform(cls, data: Any, op: str, source: str) -> Any:
        return getattr(data[source], op)()

    @classmethod
    def _add_column(cls, data: Any, name: str, result: Any) -> Any:
        data[name] = result
        return data


class PyArrowColumnTotal(Implementation[ColumnTotal]):
    feature_group = ColumnTotal
    compute_framework = PyArrowTable

    @classmethod
    def calculate(
        cls, group: type[ColumnTotal], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.total(data, features, perform=cls._perform, add_column=cls._add_column)

    @classmethod
    def _perform(cls, data: pa.Table, op: str, source: str) -> Any:
        column = data[source]
        if op == "sum":
            return pc.sum(column, skip_nulls=False, min_count=0).as_py()
        return pc.max(pc.if_else(pc.is_nan(column), None, column)).as_py()

    @classmethod
    def _add_column(cls, data: pa.Table, name: str, result: Any) -> pa.Table:
        return data.append_column(name, pa.array([result] * data.num_rows))
```

- NaN can arrive as null after a framework transfer, so state the contract over missing values, not NaN alone
- Divergence is declared per operation, not detected from the data
- Besides `supported`, the classmethod `accepts(feature_name, options, *, part=None)` (default `True`; `part` is the requested output part) can decline a feature; a framework whose implementation declines it is not a candidate
- The same `Implementation[G]` serves source groups: `data` is then the `SourceMatch`, see [27-input-data-readers.md](27-input-data-readers.md)

With Pandas and PyArrow both enabled, a sum runs on PyArrow with the reason `prefers a matching version`. A pin overrides that choice:

```python
Feature("price__sum_total", implementation="PandasColumnTotal")
```

The pinned step records `pinned; runs a version that differs: skips NaN in sum`. The pin is part of the feature's identity and is not passed to input features. An unknown or ambiguous name, a disabled implementation, a conflict with `compute_framework=` or a wrong entry count raises `ImplementationPinError` (a `ValueError`). A steward disables one implementation with `PluginCollector.add_disabled_implementation_classes`. See core's [Operation Groups](https://github.com/mloda-ai/mloda/blob/0.18.0/docs/docs/in_depth/compute-framework-integration.md#operation-groups-one-group-one-implementation-per-framework).

## Test

Set `features.implementation`, then call the group's `calculate_feature`. The data-operation test kit wraps this as `calculate_with` (`mloda.testing.feature_groups.data_operations.helpers`), which also accepts a plain FeatureGroup class.

```python
def _feature_set(name: str) -> FeatureSet:
    features = FeatureSet()
    features.add(Feature(name))
    return features


def test_each_implementation_computes_the_operation():
    pandas_fs = _feature_set("value__doubled")
    pandas_fs.implementation = PandasDoubled
    out = DoubledFeatureGroup.calculate_feature(pd.DataFrame({"value": [1, 2]}), pandas_fs)
    assert list(out["value__doubled"]) == [2, 4]

    arrow_fs = _feature_set("value__doubled")
    arrow_fs.implementation = PyArrowDoubled
    out = DoubledFeatureGroup.calculate_feature(pa.table({"value": [1, 2]}), arrow_fs)
    assert out["value__doubled"].to_pylist() == [2, 4]


test_each_implementation_computes_the_operation()
```

## Single-Framework Groups

A group that runs on one framework only still uses `compute_framework_rule()`:

```python
from mloda.provider import ComputeFramework, FeatureGroup
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class PandasOnlyGroup(FeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}
```

`None` (the default) allows any framework. Aggregation, window aggregation, scalar aggregate, frame aggregate, point/scalar arithmetic, lookup, time bucketization, ffill, EMA, resample, sessionization, rank, offset, percentile, string, datetime and binning are converted; a new backend for one of them is an `Implementation[Group]`, not a subclass of the group. For a third-party group served by one FeatureGroup subclass per framework, the base must keep its compute hook `@abstractmethod` (from `abc`), so mloda never selects the base on a framework without a subclass; mloda raises `FeatureResolutionError` instead of failing at compute time.

## Available Frameworks

| Framework | Import Path |
|-----------|-------------|
| PythonDict | `mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework.PythonDictFramework` |
| Pandas | `mloda_plugins.compute_framework.base_implementations.pandas.dataframe.PandasDataFrame` |
| Polars | `mloda_plugins.compute_framework.base_implementations.polars.dataframe.PolarsDataFrame` |
| Polars Lazy | `mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe.PolarsLazyDataFrame` |
| PyArrow | `mloda_plugins.compute_framework.base_implementations.pyarrow.table.PyArrowTable` |
| DuckDB | `mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework.DuckDBFramework` |
| SQLite | `mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework.SqliteFramework` |
| Spark | `mloda_plugins.compute_framework.base_implementations.spark.spark_framework.SparkFramework` |

## Real Implementations

| File | Description |
|------|-------------|
| [aggregation/base.py](../../../mloda/community/feature_groups/data_operations/aggregation/base.py) | `AggregationFeatureGroup`, the group and its shared `aggregate` loop |
| [aggregation/pandas_aggregation.py](../../../mloda/community/feature_groups/data_operations/aggregation/pandas_aggregation.py) | `PandasAggregation`, with `divergent` |
| [aggregation/pyarrow_aggregation.py](../../../mloda/community/feature_groups/data_operations/aggregation/pyarrow_aggregation.py) | `PyArrowAggregation`, with `supported` |
| [aggregation/sqlite_aggregation.py](../../../mloda/community/feature_groups/data_operations/aggregation/sqlite_aggregation.py) | `SqliteAggregation`, with `supported` and `divergent` |
| [aggregation/polars_lazy_aggregation.py](../../../mloda/community/feature_groups/data_operations/aggregation/polars_lazy_aggregation.py) | `PolarsLazyAggregation` |
| [aggregation/duckdb_aggregation.py](../../../mloda/community/feature_groups/data_operations/aggregation/duckdb_aggregation.py) | `DuckdbAggregation` |
| [aggregation/python_dict_aggregation.py](../../../mloda/community/feature_groups/data_operations/aggregation/python_dict_aggregation.py) | `PythonDictAggregation` |
| [window_aggregation/base.py](../../../mloda/community/feature_groups/data_operations/row_preserving/window_aggregation/base.py) | `WindowAggregationFeatureGroup`, the group and its shared `aggregate` loop |
| [scalar_aggregate/base.py](../../../mloda/community/feature_groups/data_operations/row_preserving/scalar_aggregate/base.py) | `ScalarAggregateFeatureGroup`, the group and its shared `aggregate` loop |
| [frame_aggregate/base.py](../../../mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/base.py) | `FrameAggregateFeatureGroup`, the group and its shared `aggregate` loop |
| [point_arithmetic/base.py](../../../mloda/community/feature_groups/data_operations/row_preserving/point_arithmetic/base.py) | `PointArithmeticFeatureGroup`, shared loop with two keyword-only callables |
| [point_arithmetic/sqlite_point_arithmetic.py](../../../mloda/community/feature_groups/data_operations/row_preserving/point_arithmetic/sqlite_point_arithmetic.py) | `SqlitePointArithmetic`, with `divergent` |
| [row_preserving/rank/base.py](../../../mloda/community/feature_groups/data_operations/row_preserving/rank/base.py) | `RankFeatureGroup`, shared `rank` loop, `SUBTYPES` with parametric families |
| [row_preserving/rank/pandas_rank.py](../../../mloda/community/feature_groups/data_operations/row_preserving/rank/pandas_rank.py) | `PandasRank`, with `divergent` |
| [row_preserving/offset/base.py](../../../mloda/community/feature_groups/data_operations/row_preserving/offset/base.py) | `OffsetFeatureGroup`, shared `offset` loop, `SUBTYPES` with parametric families |
| [row_preserving/percentile/base.py](../../../mloda/community/feature_groups/data_operations/row_preserving/percentile/base.py) | `PercentileFeatureGroup`, shared `percentile` loop |
| [string/sqlite_string.py](../../../mloda/community/feature_groups/data_operations/string/sqlite_string.py) | `SqliteStringOps`, with `supported` |
| [core aggregated_feature_group/base.py](https://github.com/mloda-ai/mloda/blob/0.18.0/mloda_plugins/feature_group/experimental/aggregated_feature_group/base.py) | Core's group, shared loop |
| [core aggregated_feature_group/pandas.py](https://github.com/mloda-ai/mloda/blob/0.18.0/mloda_plugins/feature_group/experimental/aggregated_feature_group/pandas.py) | Core's Pandas Implementation |
| [core aggregated_feature_group/pyarrow.py](https://github.com/mloda-ai/mloda/blob/0.18.0/mloda_plugins/feature_group/experimental/aggregated_feature_group/pyarrow.py) | Core's PyArrow Implementation |

## Combines With

- **Chained** (Pattern 3): The group matches names; Implementations only compute
- **Index** (Pattern 7): Framework-specific window functions
- **Artifact** (Pattern 6): Framework-specific serialization
