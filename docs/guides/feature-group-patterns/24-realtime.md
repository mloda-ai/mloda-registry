# Realtime Execution

Build the execution plan once at startup and reuse it for repeated calls with fresh data.

**What**: `mloda.prepare()` builds an execution plan; `session.run()` executes it with new data.
**When**: You serve the same features repeatedly with different input data.
**Why**: Avoid rebuilding the plan on every call — `prepare` pays the cost once.
**Where**: ML inference endpoints, event-driven pipelines, interactive dashboards.
**How**: Replace `mloda.run_all(...)` with `mloda.prepare(...)` + `session.run(...)`.

## Basic Usage

```python
from mloda.user import PluginLoader, mloda, Feature

PluginLoader.all()

# 1. Prepare once (expensive: builds execution plan)
session = mloda.prepare(
    [Feature("my_feature")],
    compute_frameworks=["PandasDataFrame"],
)

# 2. Run many times (cheap: reuses plan)
result_1 = session.run(api_data={"MyKey": {"col": [1, 2]}})
result_2 = session.run(api_data={"MyKey": {"col": [3, 4]}})
```

## How It Works

1. **`mloda.prepare(features, ...)`** resolves features, builds the dependency graph, and returns a `session` object. This is the expensive step.
2. **`session.run(api_data=...)`** executes the pre-built plan with fresh input data. This is cheap and can be called repeatedly.

## Comparison with `run_all`

| | `run_all` | `prepare` + `run` |
|---|---|---|
| **Plan cost** | Rebuilt every call | Built once in `prepare` |
| **Data flexibility** | Data passed per call | Data passed per `run` |
| **Equivalence** | `run_all(features, api_data=d)` | `prepare(features).run(api_data=d)` |

## When to Use Which

- **`run_all`** — One-off computation or infrequent calls where plan-building cost is negligible.
- **`prepare` + `run`** — Repeated execution of the same features with different data (serving, streaming, interactive).

## Composability

`session.run()` accepts additional parameters beyond `api_data`:

- `parallelization_modes`: override parallelization per run
- `flight_server`: Arrow Flight server for distributed execution
- `artifacts`: a previous run's `get_artifacts()`; matching feature groups switch to load mode
- `carrier`: W3C trace-context carrier forwarded to every hook context; may differ per call
- `child_bootstrap`: picklable no-argument callable run once in each `MULTIPROCESSING` worker before its first command
- `graceful_shutdown_timeout`: seconds (default 2.0), shared by a `MULTIPROCESSING` worker's extenders, for their `close()` before the worker is terminated, see [Pickle Compatibility](../11-create-extender.md#pickle-compatibility)

## Full Documentation

See [Realtime API](https://mloda-ai.github.io/mloda/in_depth/realtime/) for additional details.
