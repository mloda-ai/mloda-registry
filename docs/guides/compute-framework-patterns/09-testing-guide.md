# Concept 9: Testing Guide

Test your compute framework with mloda's conformance suites instead of hand-written contract tests.

**What**: Shape check, plus one pytest suite per aspect (filter, mask, merge, relation, connection, run-all, ...).
**When**: After implementing any compute framework component.
**Why**: Verify correctness against the shared contract; catch regressions.
**Where**: `mloda.testing` (`compute_framework_shape`, `framework_adapter`, `typed_data`, `compute_framework.*_suite`).
**How**: Write a `FrameworkAdapter` for your framework, then subclass the suites that apply and set `adapter_class`.

The Arrow-backed adapters need `pip install "mloda-testing[pyarrow]"`; `PythonDictAdapter` does not.

## Start With the Shape Check

```py
from mloda.testing.compute_framework_shape import assert_compute_framework_shape, applicable_aspects


def test_shape() -> None:
    assert_compute_framework_shape(MyFramework)
    print(applicable_aspects(MyFramework))  # e.g. ("filter", "mask", "merge")
```

It checks construction (including `uuid=`), `is_available` and the engine accessors. `applicable_aspects` lists which of `filter`, `mask`, `merge` the framework serves; a suite for any other aspect fails as "mis-wired".

## The Adapter

A `FrameworkAdapter` builds frames from `TypedTestData` (columns with a declared `DataType`) and reads them back. Stock adapters: `PythonDictAdapter`, `PyArrowAdapter`, `PandasAdapter`, `PolarsAdapter`, `PolarsLazyAdapter`, `DuckDBAdapter`, `SqliteAdapter`, `IcebergAdapter`, `SparkAdapter`. For a new framework implement:

| Member | Purpose |
|--------|---------|
| `capabilities` | `AdapterCapabilities(typed_columns=True, sql_relation=False)`; `sql_relation=True` serves the relation suites |
| `compute_framework()` | the framework class |
| `from_arrow(table)` | an Arrow table to a native frame (`from_typed` calls it) |
| `column_names(result)`, `row_count(result)`, `column_values(result, column)` | read a frame; nulls as `None` |
| `column_type(result, column)`, `lost_type(result, column)` | native type object; true when the column lost its type (null, object) |

Optional: `open()`, `close()` and `connection` for connection-bound frameworks (the adapter is a context manager), `duplicate_column_frame`, `zero_column_frame`, `rows`, `sync_only_reason` (framework has no THREADING), `no_native_type`, `untyped_may_hold`. A backend that cannot hold a shape raises `self._cannot_hold("...")` from the matching method.

```py
from typing import Any

import my_lib
import pytest

from mloda.testing.framework_adapter import AdapterCapabilities, FrameworkAdapter
from mloda.testing.typed_data import TypedTestData
from mloda.user import DataType


class MyAdapter(FrameworkAdapter):
    capabilities = AdapterCapabilities(typed_columns=True)

    def compute_framework(self) -> type[MyFramework]:
        return MyFramework

    def from_arrow(self, table: Any) -> Any:
        return my_lib.from_arrow(table)

    def column_names(self, result: Any) -> list[str]:
        return list(result.columns)

    def row_count(self, result: Any) -> int:
        return int(result.height)

    def column_values(self, result: Any, column: str) -> list[Any]:
        return list(result[column].to_list())

    def column_type(self, result: Any, column: str) -> Any:
        return result.schema[column]

    def lost_type(self, result: Any, column: str) -> bool:
        return result.schema[column] in (my_lib.Null, my_lib.Object)


def test_adapter_round_trip() -> None:
    data = TypedTestData({"id": (DataType.INT64, [1, 2, None])})
    with MyAdapter() as adapter:
        frame = adapter.from_typed(data)
        assert adapter.column_values(frame, "id") == [1, 2, None]
```

## Hosts

A host subclasses a suite and sets `adapter_class`; the suite builds its own `TypedTestData` and runs the contract through the adapter. Hosts must be named `Test*`. A host that omits `adapter_class` or a hook in the suite's `required_hooks` fails at class creation with `HookError`, as does a misspelled hook name.

| Suite | Aspect, served when | The host sets |
|-------|---------------------|---------------|
| `FilterEngineSuite` | `filter` | `adapter_class` |
| `MaskEngineSuite` | `mask` | `evaluate_mask`, `is_boolean_mask`, `select_rows` |
| `SqlMaskEngineSuite` | `mask`, engine returns SQL conditions | `evaluate_mask`, `select_rows` |
| `MergeEngineSuite` | `merge` | `adapter_class` |
| `AsofMergeEngineSuite` | `asof merge`, the merge engine overrides `merge_asof` | `adapter_class` |
| `EmptyResultSuite`, `AppendColumnsSuite` | `empty result`, `append columns`, always | `adapter_class` (`AppendColumnsSuite`: `single_column`, `joined_frame` where lazy) |
| `DtypeExtractionSuite`, `DataTypeValidatorSuite` | `dtype extraction`, `datatype validator`, always | `adapter_class` |
| `OutputSchemaSuite` | `output schema`, always | `adapter_class` |
| `RelationSuite`, `SqlRelationWindowSuite` | `sql_relation`, the adapter has `sql_relation=True` | `adapter_class` (`raw_projection` if the relation has no `select(_raw_sql=...)`) |
| `ConnectionPickSuite` | `connection`, the framework overrides `_connection_matches` | `adapter_class` |
| `ConnectionEndToEndSuite` | `connection`, as above | `destination_fg_class`, `extract_doubled`, `double_relation` |
| `RunAllSuite` family: `EmptyResultRunAllSuite`, `SiblingDictRunAllSuite`, `MissingColumnRunAllSuite`, `AsofRunAllSuite` | run-all, always (`AsofRunAllSuite` only with `merge_asof`) | `adapter_class` (`MissingColumnRunAllSuite`: `undeclared_reads`) |

Import each suite from its module: `filter_suite`, `mask_suite`, `merge_suite`, `frame_suite`, `dtype_suite`, `output_schema_suite`, `relation_suite`, `connection_suite`, `run_all_suite`, all under `mloda.testing.compute_framework`. Hook signatures take the adapter first, for example `evaluate_mask(self, adapter, mask, data) -> list[bool]`. Other optional hooks: `arrow_types(name)`, `per_filter_frame`, `apply_filters_names`, `decimal_dtype` (filter), `empty_schema_frame` (output schema), `zero_column_expectation` (empty result run-all).

```py
from typing import Any

from mloda.testing.compute_framework.filter_suite import FilterEngineSuite
from mloda.testing.compute_framework.mask_suite import MaskEngineSuite
from mloda.testing.compute_framework.merge_suite import MergeEngineSuite
from mloda.testing.framework_adapter import FrameworkAdapter


class TestMyFilterEngine(FilterEngineSuite):
    adapter_class = MyAdapter


class TestMyMergeEngine(MergeEngineSuite):
    adapter_class = MyAdapter


class TestMyMaskEngine(MaskEngineSuite):
    adapter_class = MyAdapter

    def evaluate_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> list[bool]:
        return list(mask.to_list())

    def is_boolean_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> bool:
        return mask.dtype == my_lib.Boolean

    def select_rows(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> Any:
        return data.filter(mask)
```

## Declaring What the Backend Cannot Do

Never emulate a missing capability in Python to pass a test. Declare it with a `Cause` (`NOT_IMPLEMENTED`, `BACKEND`, `ENVIRONMENT`, `OTHER`) and a one-line reason:

- `unsupported_cases`: per-case skip, for suites that list case ids (`DataTypeValidatorSuite`). Unknown ids fail at class creation.
- `switched_off`: when the host redefines, blanks or skips a contract test, map the test name to a `Cause`. An undeclared override fails at class creation.

```py
import pytest

from mloda.provider import Cause
from mloda.testing.compute_framework.dtype_suite import DataTypeValidatorSuite
from mloda.testing.compute_framework.output_schema_suite import OutputSchemaSuite
from mloda.testing.framework_adapter import FrameworkAdapter


class TestMyDataTypeValidator(DataTypeValidatorSuite):
    adapter_class = MyAdapter
    unsupported_cases = {"timestamp_s_col": (Cause.BACKEND, "no second-precision timestamp")}


class TestMyOutputSchema(OutputSchemaSuite):
    adapter_class = MyAdapter
    switched_off = {"test_duplicate_column_names_collapse_to_one_entry": Cause.BACKEND}

    def empty_schema_frame(self, adapter: FrameworkAdapter) -> Any:
        return my_lib.DataFrame()

    def test_duplicate_column_names_collapse_to_one_entry(self) -> None:
        pytest.skip("frames cannot hold duplicate column names")
```

## Connection-Bound Frameworks

The adapter owns the connection (see `DuckDBAdapter` and `SqliteAdapter`). `ConnectionEndToEndSuite` needs a destination feature group that reads `raw_val` and writes `tfs_doubled` on your framework:

```py
from typing import Any

from mloda.provider import FeatureGroup
from mloda.testing.compute_framework.connection_suite import ConnectionEndToEndSuite, ConnectionPickSuite
from mloda.testing.compute_framework.relation_suite import SqlRelationWindowSuite
from mloda.testing.compute_framework.run_all_suite import EmptyResultRunAllSuite, PolicyRaises
from mloda.testing.framework_adapter import DuckDBAdapter
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework


class DoubledFeatureGroup(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("raw_val")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tfs_doubled"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[DuckDBFramework]]:
        return {DuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return data.project('"raw_val", "raw_val" * 2 AS "tfs_doubled"')


class TestDuckDBConnectionPick(ConnectionPickSuite):
    adapter_class = DuckDBAdapter


class TestDuckDBConnectionEndToEnd(ConnectionEndToEndSuite):
    adapter_class = DuckDBAdapter
    destination_fg_class = DoubledFeatureGroup

    def extract_doubled(self, final: Any) -> list[int]:
        return list(final.column("tfs_doubled").to_pylist())

    @staticmethod
    def double_relation(data: Any, src: str, dst: str) -> Any:
        return data.project(f'"{src}", "{src}" * 2 AS "{dst}"')


class TestDuckDBRelationWindow(SqlRelationWindowSuite):
    adapter_class = DuckDBAdapter

    def raw_projection(self, relation: Any, sql: str) -> Any:
        return relation.project(sql)


class TestDuckDBEmptyResultRunAll(EmptyResultRunAllSuite):
    adapter_class = DuckDBAdapter

    def zero_column_expectation(self) -> PolicyRaises:
        return PolicyRaises(match_substring="must have at least one column")
```

Run-all suites run each case in SYNC and THREADING; a framework without THREADING sets `sync_only_reason`.

## Manual Test Examples

For behavior the suites do not cover, start from the `Test` blocks in [Stateless eager](01-stateless-eager.md#test) and [Framework transformer](08-framework-transformer.md#test).
