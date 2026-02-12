# Streaming Results

Consume results incrementally as each feature group completes, rather than waiting for all groups to finish.

**What**: `stream_all` yields each feature group's result as soon as it finishes.
**When**: You request multiple feature groups and want to process results as they arrive.
**Why**: Start processing early, reduce peak memory, display partial results sooner.
**Where**: Dashboards, pipelines, reports — anywhere you consume multiple independent features.
**How**: Replace `mloda.run_all(...)` with `for result in stream_all(...)`.

## Basic Usage

```python
from mloda.user import PluginLoader, stream_all, Feature

PluginLoader.all()

for result in stream_all([Feature("feature_a"), Feature("feature_b")]):
    print(result)
```

## Comparison with `run_all`

| | `run_all` | `stream_all` |
|---|---|---|
| **Return type** | `list` (all results at once) | Generator (one result at a time) |
| **Blocking** | Waits for all groups to finish | Yields each group as it completes |
| **Equivalence** | `run_all(...)` | `list(stream_all(...))` |
| **Parameters** | All standard parameters | Same parameters as `run_all` |

## When to Use Which

- **`run_all`** — You need all results before proceeding, or you request a single feature group.
- **`stream_all`** — You request many independent feature groups and want to process results as they arrive.
- **`session.stream_run()`** — Combines streaming with [realtime execution](24-realtime.md): reuse a prepared plan and yield results incrementally. Use when you need both plan reuse and per-group delivery.

If you only request one feature group, `stream_all` behaves identically to `run_all` since there is only one result to yield.

## What Streaming Does NOT Do

- **Row-by-row streaming** — individual rows are not yielded as they are computed.
- **Partial results** — you cannot observe a feature group's output before it has fully completed.
- **Chunked input** — a single feature group's computation is not split into smaller streaming chunks.

Each yielded value is a **complete** result for one feature group (e.g. a `pa.Table`).

## Works With All `run_all` Parameters

`stream_all` accepts the same parameters as `run_all`, including filters, compute frameworks, and links:

```python
from mloda.user import stream_all, Feature, GlobalFilter, SingleFilter, FilterType

for result in stream_all(
    [Feature("feature_a"), Feature("feature_b")],
    compute_frameworks=["PandasDataFrame"],
    global_filter=GlobalFilter({SingleFilter("status", FilterType.EQUAL, "active")}),
):
    print(result)
```

## Full Documentation

See [Streaming with `stream_all`](https://mloda-ai.github.io/mloda/in_depth/streaming/) for additional details.
