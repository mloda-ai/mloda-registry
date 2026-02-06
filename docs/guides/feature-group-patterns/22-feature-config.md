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

Items can be plain strings (`"feature_name"`) or feature objects. `options` and `group_options`/`context_options` are mutually exclusive.

## Full Documentation

See [Feature Configuration](https://mloda-ai.github.io/mloda/in_depth/feature-config/) for additional details.
