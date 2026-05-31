# Feature Configuration from JSON

Define feature pipelines using JSON configuration instead of Python code.

**What**: Load feature definitions from JSON strings via `load_features_from_config()`.
**When**: AI agents generating feature requests, config-driven pipelines, external configuration.
**Why**: Decouple feature definition from Python code; enable LLMs to request data without writing code.
**Where**: LLM tool functions, configuration files, dynamic pipeline builders.

## Basic Usage

```python
from mloda.user import load_features_from_config, mloda

config = '''
[
    "customer_id",
    {
        "name": "sales_aggregated",
        "in_features": ["daily_sales"],
        "context_options": {"aggregation_type": "sum", "window_days": 7}
    },
    {"name": "pca_result", "column_index": 0}
]
'''

features = load_features_from_config(config)
result = mloda.run_all(features, compute_frameworks=["PandasDataFrame"])
```

## FeatureConfig Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Feature name |
| `options` | object | No | Simple options dict (cannot combine with group/context options) |
| `in_features` | array | No | Source feature names for chained features |
| `group_options` | object | No | Group parameters (affect FeatureGroup resolution) |
| `context_options` | object | No | Context parameters (metadata, doesn't affect resolution) |
| `column_index` | integer | No | Index for multi-output features (adds `~N` suffix) |
| `propagate_context_keys` | array | No | Context keys to propagate to dependent features |

Items can be plain strings (`"feature_name"`) or feature objects. `options` and `group_options`/`context_options` are mutually exclusive.

## Window / Rank / Percentile Requests

Row-preserving `data_operations` (window aggregation, rank, percentile) cannot be requested by a bare name: the matcher only resolves the FeatureGroup when the request also carries the partition/order options it needs. Put those in `context_options`. The feature name encodes the operation (`{source}__{operation}`); the matcher reads the rest from `context_options`.

```python
config = '''
[
    {"name": "steps__sum_window", "context_options": {"partition_by": ["subject_id"]}},
    {"name": "price__last_window", "context_options": {"partition_by": ["region"], "order_by": "timestamp"}},
    {"name": "sales__row_number_ranked", "context_options": {"partition_by": ["region"], "order_by": "sales"}},
    {"name": "sales__p95_percentile", "context_options": {"partition_by": ["region"]}}
]
'''
```

| Operation | Name pattern | Required `context_options` |
|-----------|--------------|----------------------------|
| Window aggregation | `{source}__{agg}_window` (`sum`, `avg`, `first`, `last`, ...) | `partition_by` (list); `order_by` (string) for order-dependent aggs like `first`/`last` |
| Rank | `{source}__{rank_type}_ranked` (`row_number`, `dense_rank`, `ntile_N`, ...) | `partition_by` (list), `order_by` (string) |
| Percentile | `{source}__p{N}_percentile` (e.g. `p50`, `p95`) | `partition_by` (list) |

Use `context_options` (not `group_options`): the partition/order are operation parameters, not identity that should split the FeatureGroup.

## How Plugins Resolve Config Values

When a JSON config specifies `context_options` like `{"aggregation_type": "sum"}`, the receiving FeatureGroup needs to extract that value at compute time. Plugins using `FeatureChainParserMixin` call `_resolve_operation(feature, config_key)` to handle both string-based names (where the operation is embedded in the feature name) and config-based creation (where the operation comes from options) in one call. See [Chained Features](03-chained-features.md) for the full pattern.

---

## Full Documentation

See [Feature Configuration](https://mloda-ai.github.io/mloda/in_depth/feature-config/) for additional details.
