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

## Level 1: Unit Test Example

```python
def test_matching():
    # Class name matching
    assert MyFeature.match_feature_group_criteria("MyFeature", None)
    assert not MyFeature.match_feature_group_criteria("Wrong", None)

    # Chained pattern matching
    assert ScaledFeature.match_feature_group_criteria("price__scaled", None)
    assert not ScaledFeature.match_feature_group_criteria("price", None)

    # Framework restriction
    assert PandasDataFrame in PandasOnlyFeature.compute_framework_rule()
```

## Level 2: Framework Test Example

```python
import pandas as pd

def test_calculate_feature():
    df = pd.DataFrame({"input": [1, 2, 3]})

    class MockFeatures:
        class name_of_one_feature:
            name = "input__scaled"

    result = ScaledFeature.calculate_feature(df, MockFeatures())
    assert len(result) == 3
    assert result.min() >= 0 and result.max() <= 1
```

## Level 3: Integration Test Example

```python
from mloda.user import mloda, Feature

def test_full_pipeline():
    result = mloda.run_all([Feature.not_typed("my_feature")])
    assert "my_feature" in result[0]
```

`run_all()` returns a list of result frames, one per compute framework and feature set, so index into it before asserting on columns.

### Level 3 Only: The Empty-Result Contract

The rule that a FeatureGroup must return columns even at zero rows (see [calculate_feature](12-calculate-feature.md#empty-results)) is enforced during the run, not inside `calculate_feature()`. A Level 2 test that calls `calculate_feature()` directly passes on a group that returns `[]` or `{}` for "nothing found"; the same group raises `EmptyResultError` the first time that path is hit through a run. Cover every reachable empty path (no source files, a query with no hits, a filter that removes every row) at Level 3:

```python
def test_empty_source_keeps_schema():
    result = mloda.run_all(  # no matching data
        [Feature.not_typed("my_feature")],
        compute_frameworks={PythonDictFramework},
    )
    assert result[0] == {"my_feature": []}  # zero rows, schema intact
```

**Tip**: Use `column_ordering="request_order"` for predictable test assertions:

```python
def test_multiple_features():
    result = mloda.run_all(
        [Feature("feature_a"), Feature("feature_b")],
        column_ordering="request_order"
    )
    # Columns guaranteed to be in request order
    assert list(result[0]) == ["feature_a", "feature_b"]
```

Without `column_ordering`, column order is non-deterministic and tests comparing column lists may be flaky.

**Note**: You can also use `stream_all` for integration tests when testing streaming behavior. See [Streaming Results](23-streaming.md).

## Mocking Feature Dependencies

Isolate derived features from expensive upstream computations by disabling dependencies and injecting mock data.

```python
from mloda.user import mloda, Feature, PluginCollector

def test_derived_feature_isolated():
    result = mloda.run_all(
        [Feature.not_typed("my_derived_feature")],
        api_data={"MockData": {"upstream_feature": [100, 200, 300]}},
        plugin_collector=PluginCollector.disabled_feature_groups({ExpensiveUpstreamFeature}),
    )

    assert len(result) > 0
```

Use when upstream features are slow (API calls, ML inference) or you need controlled test data.

## Testing by Pattern

| Pattern | Level 1 Focus | Level 2 Focus |
|---------|---------------|---------------|
| Root | `input_data()` returns instance | Data loading works |
| Derived | `input_features()` correct | Transformation logic |
| Chained | Pattern regex matching | Suffix extraction |
| Multi-input | `&` parsing | Multiple inputs combined |
| Multi-output | Column count | `~` naming correct |
| Artifact | `artifact()` returns class | Save/load cycle |
| Index | `index_columns()` correct | Ordering respected |
| Framework | `compute_framework_rule()` | Framework-specific ops |

## Real Test Examples

| Directory | Pattern |
|-----------|---------|
| [test_base_aggregated_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group) | Chained + Index |
| [test_clustering_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_clustering_feature_group) | Artifact |
| [test_geo_distance_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_geo_distance_feature_group) | Multi-input |
| [test_dimensionality_reduction_feature_group/](https://github.com/mloda-ai/mloda/tree/main/tests/test_plugins/feature_group/experimental/test_dimensionality_reduction_feature_group) | Multi-output + Artifact |
