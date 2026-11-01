# Streaming Results

Consume results incrementally as each feature group completes, rather than waiting for all groups to finish.

**What**: `stream_all` yields each feature group's result as soon as it finishes.
**When**: You request multiple feature groups and want to process results as they arrive.
**Why**: Start processing early, reduce peak memory, display partial results sooner.
**Where**: Dashboards, pipelines, reports (anywhere you consume multiple independent features).
**How**: Replace `mloda.run_all(...)` with `for result in mloda.stream_all(...)`.

## Basic Usage

```python
from mloda.user import PluginLoader, mloda, Feature

PluginLoader.all()

api_data = {"MyKey": {"feature_a": [1, 2], "feature_b": [3, 4]}}

for result in mloda.stream_all([Feature("feature_a"), Feature("feature_b")], api_data=api_data):
    print(result)
```

`stream_all` takes the same parameters as `run_all`, and `list(mloda.stream_all(...))` equals `mloda.run_all(...)`. With a single feature group it behaves identically to `run_all`.

## Per-Group Streaming with `session.stream_run()`

Combines streaming with [realtime execution](24-realtime.md). Same parameters as `session.run()`.

```python
session = mloda.prepare([Feature("feature_a"), Feature("feature_b")], api_data=api_data)

for result in session.stream_run(api_data={"MyKey": {"feature_a": [1, 2], "feature_b": [3, 4]}}):
    print(result)
```

`stream_run()` needs the same set keys and column names as `prepare()`.

## What Streaming Does NOT Do

Each yielded value is a **complete** result for one feature group (e.g. a `pa.Table`): rows are not yielded as they are computed, and one group's computation is never split into chunks.

## Full Documentation

See [Streaming with `stream_all`](https://mloda-ai.github.io/mloda/in_depth/streaming/) for additional details.
