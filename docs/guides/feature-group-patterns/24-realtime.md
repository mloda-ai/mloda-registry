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

# 1. Prepare once (expensive — builds execution plan)
session = mloda.prepare(
    [Feature("my_feature")],
    compute_frameworks=["PandasDataFrame"],
)

# 2. Run many times (cheap — reuses plan)
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

- `parallelization_modes` — Override parallelization per run
- `flight_server` — Arrow Flight server for distributed execution
- `function_extender` — Per-run extender overrides

## Full Documentation

See [Realtime API](https://mloda-ai.github.io/mloda/in_depth/realtime/) for additional details.
