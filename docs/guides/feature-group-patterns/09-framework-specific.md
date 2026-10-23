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
| `OperationFeatureGroup` | Frameworks are the ones its loaded Implementations serve; no `compute_framework_rule()` |
| `Implementation.feature_group` / `compute_framework` | The pair it serves; required, checked at class definition |
| `Implementation.calculate(group, data, features, framework)` | Runs the feature; call the group's shared loop |
| `Implementation.supported` | `frozenset` of subtypes it can run; `None` (default) = all |
| `Implementation.divergent` | `{subtype: reason}` where it differs from the reference result |
| `Feature(..., implementation=...)` | Pin by class name, class or list of either |

## Complete Example

The shared loop lives on the group as a classmethod that takes the framework step as a keyword-only callable. Each `calculate` passes its own step.

```python
from collections.abc import Callable
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import ComputeFramework, FeatureSet, Implementation, OperationFeatureGroup
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class DoubledFeatureGroup(OperationFeatureGroup):
    """One operation: `<source>__doubled`. Frameworks are the Implementations below."""

    PREFIX_PATTERN = r"^.+__doubled$"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.not_typed(feature_name.name.removesuffix("__doubled"))}

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
```

Capability and differences are declared on the Implementation, next to its code. Both are keyed by the group's canonical subtype (see Pattern 3 and Pattern 14), so they need a group with `SUBTYPES`:

```python
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

```python
from mloda.user import Feature

by_name = Feature("value__doubled", implementation="PandasDoubled")
by_class = Feature("value__doubled", implementation=PandasDoubled)
by_list = Feature("value__doubled", implementation=[PandasDoubled, PyArrowDoubled])
```

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

`None` (the default) allows any framework. Other data-operation families are not yet converted and still use one FeatureGroup subclass per framework.

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
| [core aggregated_feature_group/base.py](https://github.com/mloda-ai/mloda/blob/0.17.0/mloda_plugins/feature_group/experimental/aggregated_feature_group/base.py) | Core's group, shared loop |
| [core aggregated_feature_group/pandas.py](https://github.com/mloda-ai/mloda/blob/0.17.0/mloda_plugins/feature_group/experimental/aggregated_feature_group/pandas.py) | Core's Pandas Implementation |
| [core aggregated_feature_group/pyarrow.py](https://github.com/mloda-ai/mloda/blob/0.17.0/mloda_plugins/feature_group/experimental/aggregated_feature_group/pyarrow.py) | Core's PyArrow Implementation |

## Combines With

- **Chained** (Pattern 3): The group matches names; Implementations only compute
- **Index** (Pattern 7): Framework-specific window functions
- **Artifact** (Pattern 6): Framework-specific serialization
