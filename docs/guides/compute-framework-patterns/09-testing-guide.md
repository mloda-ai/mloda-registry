# Concept 9: Testing Guide

Test your compute framework implementation with this structured approach.

**What**: Testing framework, merge engine, filter engine, and transformer.
**When**: After implementing any compute framework component.
**Why**: Verify correctness; catch regressions.
**Where**: `tests/test_plugins/compute_framework/`.
**How**: Use pytest with reusable base classes and mixins.

## Test Structure

```text
tests/
├── test_my_framework.py
├── test_my_merge_engine.py
├── test_my_filter_engine.py
└── test_my_transformer.py
```

## Reusable Test Infrastructure

mloda provides base classes that give you comprehensive tests with minimal code.

### FilterEngineTestMixin (11+ tests)

For filter engine tests - implement 3 methods, get 11 tests:

```python
from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import FilterEngineTestMixin


class TestMyFilterEngine(FilterEngineTestMixin):
    @pytest.fixture
    def filter_engine(self) -> Any:
        return MyFilterEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return my_lib.DataFrame({"str_col": ["a", "b"], "int_col": [1, 5]})

    def get_column_values(self, result, column) -> list[Any]:
        return result[column].tolist()
```

### MaskEngineTestMixin

For mask engine tests - set the mask engine class and provide framework-specific data and mask hooks:

```python
from decimal import Decimal
from typing import Any

import pytest

from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import MaskEngineTestMixin


class TestMyMaskEngine(MaskEngineTestMixin):
    mask_engine_class = MyMaskEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return my_lib.DataFrame({"status": ["active", "inactive"], "value": [10, 20]})

    @pytest.fixture
    def empty_data(self) -> Any:
        return my_lib.DataFrame({"status": [], "value": []})

    @pytest.fixture
    def null_data(self) -> Any:
        return my_lib.DataFrame(
            {
                "status": ["active", None],
                "value": [10, 20],
                "score": [Decimal("1.00"), None],
                "ratio": [1.0, None],
            }
        )

    @pytest.fixture
    def decimal_sample_data(self) -> Any:
        return my_lib.DataFrame({"d": [Decimal("12.34"), Decimal("5.50"), None]})

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        return mask.tolist()

    def is_boolean_mask(self, mask: Any, data: Any) -> bool:
        return mask.dtype == bool

    def apply_mask(self, mask: Any, data: Any) -> dict[str, list[Any]]:
        return data[mask].to_dict("list")
```

The mixin expects `mask_engine_class`, `sample_data`, `empty_data`, `null_data`,
`decimal_sample_data`, `evaluate_mask`, `is_boolean_mask`, and `apply_mask`.
For SQL engines that return condition strings, inherit from `SqlMaskEngineTestMixin`
to add SQL condition-shape checks on top of the shared mask engine contract.

### MultiIndexMergeEngineTestBase (6 tests)

For merge engine tests with multi-column indexes:

```python
from tests.test_plugins.compute_framework.test_tooling.multi_index.multi_index_test_base import (
    MultiIndexMergeEngineTestBase,
)


class TestMyMergeEngine(MultiIndexMergeEngineTestBase):
    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return MyMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        return my_lib.DataFrame

    def get_connection(self) -> Any | None:
        return None  # Or connection for stateful frameworks
```

### DataFrameTestBase (6 tests)

For framework-level merge tests:

```python
from tests.test_plugins.compute_framework.test_tooling.dataframe_test_base import DataFrameTestBase


class TestMyFrameworkMerge(DataFrameTestBase):
    @classmethod
    def framework_class(cls) -> type[Any]:
        return MyFramework

    def create_dataframe(self, data: dict) -> Any:
        return my_lib.DataFrame(data)

    def get_connection(self) -> Any | None:
        return None
```

## Test Data Utilities

### DataConverter

Converts test data to any framework format via PyArrow:

```python
from tests.test_plugins.compute_framework.test_tooling.multi_index.test_data_converter import DataConverter

converter = DataConverter()
df = converter.to_framework([{"col": 1}, {"col": 2}], my_lib.DataFrame, connection=None)
```

### SCENARIOS

Framework-agnostic merge scenarios, a `dict[str, MergeScenario]` keyed by scenario name. `MergeScenario` is a `TypedDict` with keys `left`, `right`, `index`, `expected_rows`, `expected_columns`, `description`:

```python
from tests.test_plugins.compute_framework.test_tooling.multi_index.test_scenarios import SCENARIOS


# Use in parametrized tests
@pytest.mark.parametrize("scenario_key", list(SCENARIOS))
def test_merge_scenario(scenario_key, converter, engine):
    scenario = SCENARIOS[scenario_key]
    left = converter.to_framework(scenario["left"], my_lib.DataFrame)
    right = converter.to_framework(scenario["right"], my_lib.DataFrame)
    result = engine.merge_inner(left, right, Index(scenario["index"]), Index(scenario["index"]))
    assert len(result) == scenario["expected_rows"]
```

## Shared Helpers

### Availability Testing

```python
from tests.test_plugins.compute_framework.test_tooling.availability_test_helper import (
    assert_unavailable_when_import_blocked,
)


def test_unavailable_when_not_installed():
    assert_unavailable_when_import_blocked(MyFramework, "my_lib")
```

### Shared Compute Frameworks

For testing transformer chains:

```python
from tests.test_plugins.compute_framework.test_tooling.shared_compute_frameworks import SecondCfw, ThirdCfw, FourthCfw
```

## Shared Fixtures

The compute framework conftest provides:

```python
# tests/test_plugins/compute_framework/base_implementations/conftest.py
@pytest.fixture
def index_obj() -> Any:
    return Index(("idx",))


@pytest.fixture
def dict_data() -> dict[str, list[int]]:
    return {"column1": [1, 2, 3], "column2": [4, 5, 6]}
```

Framework-specific fixtures:

```python
# DuckDB conftest.py
@pytest.fixture
def connection():
    conn = duckdb.connect()
    yield conn
    conn.close()


# Spark conftest.py
@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder.getOrCreate()
    yield spark
    spark.stop()
```

## Manual Test Examples

When not using base classes:

### Framework Tests

```python
@pytest.mark.skipif(my_lib is None, reason="my_lib not installed")
class TestMyFramework:
    def test_is_available(self):
        assert MyFramework.is_available()

    def test_expected_data_framework(self):
        assert MyFramework.expected_data_framework() == my_lib.DataFrame

    def test_transform_dict(self):
        framework = MyFramework(...)
        result = framework.transform({"col": [1, 2]}, {"col"})
        assert isinstance(result, my_lib.DataFrame)
```

### Transformer Tests

```python
@pytest.mark.skipif(my_lib is None or pa is None, reason="deps not installed")
class TestMyTransformer:
    def test_roundtrip(self):
        original = my_lib.DataFrame({"col": [1, 2]})
        arrow = MyPyArrowTransformer.transform_fw_to_other_fw(original)
        restored = MyPyArrowTransformer.transform_other_fw_to_fw(arrow)
        assert restored.equals(original)
```

## Test Checklist

### Framework
- [ ] `is_available()` returns correct boolean
- [ ] `expected_data_framework()` returns correct type
- [ ] `merge_engine()` returns BaseMergeEngine subclass
- [ ] `filter_engine()` returns BaseFilterEngine subclass
- [ ] `mask_engine()` returns BaseMaskEngine subclass
- [ ] `transform()` handles dict input

### Merge Engine (use MultiIndexMergeEngineTestBase)
- [ ] All 6 merge types work with multi-column indexes
- [ ] Connection passed correctly for stateful frameworks

### Filter Engine (use FilterEngineTestMixin)
- [ ] All filter types work (range, min, max, equal, regex, categorical)

### Mask Engine (use MaskEngineTestMixin)
- [ ] Mask comparisons, set membership, boolean combination, null handling, and decimal values work

### Transformer
- [ ] Conversion to PyArrow works
- [ ] Conversion from PyArrow works
- [ ] Roundtrip preserves data

## Real Test Examples

| File | Description |
|------|-------------|
| [filter_engine_test_mixin.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/compute_framework/base_implementations/filter_engine_test_mixin.py) | Filter engine mixin |
| [mask_engine_test_mixin.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/compute_framework/base_implementations/mask_engine_test_mixin.py) | Mask engine mixin |
| [pyarrow/test_pyarrow_mask_engine.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/compute_framework/base_implementations/pyarrow/test_pyarrow_mask_engine.py) | PyArrow mask engine consumer |
| [sql_mask_engine_test_mixin.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/compute_framework/base_implementations/sql_mask_engine_test_mixin.py) | SQL mask engine mixin |
| [multi_index_test_base.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/compute_framework/test_tooling/multi_index/multi_index_test_base.py) | Merge engine base |
| [dataframe_test_base.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/compute_framework/test_tooling/dataframe_test_base.py) | Framework merge base |
| [pandas/test_pandas_dataframe.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/compute_framework/base_implementations/pandas/test_pandas_dataframe.py) | Pandas tests |
