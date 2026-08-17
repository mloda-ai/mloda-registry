[![CI](https://github.com/mloda-ai/mloda-registry/actions/workflows/ci.yaml/badge.svg)](https://github.com/mloda-ai/mloda-registry/actions/workflows/ci.yaml)
[![PyPI](https://img.shields.io/pypi/v/mloda-community.svg)](https://pypi.org/project/mloda-community/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)

# mloda-registry

Community plugins, registry tooling, and development guides for [mloda](https://github.com/mloda-ai/mloda).

> **New to mloda?** Visit [mloda.ai](https://mloda.ai) for business context or the [core repository](https://github.com/mloda-ai/mloda) for technical details.

## Quick start

```bash
pip install mloda-community pandas
```

```python
from mloda.user import Feature, PluginLoader, mloda

PluginLoader.all()  # discover installed plugins

result = mloda.run_all(
    [Feature("name__upper"), Feature("name__length")],
    compute_frameworks=["PandasDataFrame"],
    api_data={"orders": {"name": ["alice", "bob"]}},
)
print(result[0])
#   name__upper  name__length
# 0       ALICE             5
# 1         BOB             3
```

Every plugin is requested by feature name (usually `{column}__{operation}`) and runs on PyArrow, Pandas, Polars lazy, DuckDB, and SQLite, each backend checked against the PyArrow reference; per-plugin coverage is in the [framework support matrix](docs/guides/data-operation-patterns/framework-support-matrix.md). Swap `compute_frameworks` between `PyArrowTable`, `PandasDataFrame`, and `PolarsLazyDataFrame` as-is; `DuckDBFramework` and `SqliteFramework` also need a connection passed via `data_access_collection` ([stateful connections](docs/guides/compute-framework-patterns/03-stateful-connection.md)). See [Use an existing plugin](docs/guides/01-use-existing-plugin.md) for streaming, realtime, and column ordering.

## Plugins

`mloda-community` bundles all plugins below. Each is also published on its own for minimal installs, with the backend as an extra (`pip install "mloda-community-ema[pandas]"`).

| Plugin | Feature name | Guide |
|--------|--------------|-------|
| `mloda-community-aggregation` | `{col}__{agg}_agg` (sum, avg, count, min, max, std, median, ...; one row per group) | [aggregation](docs/guides/data-operation-patterns/framework-support-matrix.md#aggregation) |
| `mloda-community-window-aggregation` | `{col}__{agg}_window` (partitioned aggregate broadcast per row) | [window aggregation](docs/guides/data-operation-patterns/06-window-aggregation.md) |
| `mloda-community-scalar-aggregate` | `{col}__{agg}_scalar` (global aggregate broadcast per row) | [scalar and frame aggregate](docs/guides/data-operation-patterns/08-scalar-and-frame-aggregate.md) |
| `mloda-community-frame-aggregate` | `{col}__sum_rolling_3`, `{col}__cumsum`, `{col}__expanding_avg`, `{col}__avg_7_day_window` | [scalar and frame aggregate](docs/guides/data-operation-patterns/08-scalar-and-frame-aggregate.md) |
| `mloda-community-scalar-arithmetic` | `{col}__{op}_constant` (add, subtract, multiply, divide by a constant) | [scalar and frame aggregate](docs/guides/data-operation-patterns/08-scalar-and-frame-aggregate.md) |
| `mloda-community-point-arithmetic` | `{a}&{b}__{op}_point` (element-wise two-column arithmetic) | [scalar and frame aggregate](docs/guides/data-operation-patterns/08-scalar-and-frame-aggregate.md) |
| `mloda-community-rank` | `{col}__{rank_type}_ranked` (row_number, rank, dense_rank, percent_rank, ntile_N, top_N, bottom_N) | [percentile, rank, offset](docs/guides/data-operation-patterns/07-percentile-rank-offset.md) |
| `mloda-community-offset` | `{col}__lag_1_offset` (lag_N, lead_N, diff_N, pct_change_N, first_value, last_value) | [percentile, rank, offset](docs/guides/data-operation-patterns/07-percentile-rank-offset.md) |
| `mloda-community-percentile` | `{col}__p95_percentile` | [percentile, rank, offset](docs/guides/data-operation-patterns/07-percentile-rank-offset.md) |
| `mloda-community-binning` | `{col}__bin_5`, `{col}__qbin_10` (equal-width, quantile) | [binning](docs/guides/data-operation-patterns/05-binning.md) |
| `mloda-community-datetime` | `{col}__year`, `{col}__dayofweek`, ... | [datetime](docs/guides/data-operation-patterns/framework-support-matrix.md#datetime) |
| `mloda-community-string` | `{col}__upper`, `{col}__lower`, `{col}__trim`, `{col}__length`, `{col}__reverse` | [string operations](docs/guides/data-operation-patterns/09-string-operations.md) |
| `mloda-community-time-bucketization` | `{col}__floor_1_day`, `{col}__ceil_15_minute`, `{col}__round_5_minute` | [time bucketization](docs/guides/data-operation-patterns/11-time-bucketization.md) |
| `mloda-community-ffill` | `{col}__ffill` (forward fill across time gaps, per partition) | [forward fill](docs/guides/data-operation-patterns/12-ffill-by-time.md) |
| `mloda-community-ema` | `{col}__ema_10` (exponential moving average, per partition) | [EMA](docs/guides/data-operation-patterns/13-ema.md) |
| `mloda-community-sessionization` | `{ts}__sessionize_30_minute` (gap-threshold session id) | [sessionization](docs/guides/data-operation-patterns/15-sessionization.md) |
| `mloda-community-resample` | `{col}__resample_1_hour_mean` (events onto a regular time grid) | [resample](docs/guides/data-operation-patterns/14-resample.md) |

Options such as `partition_by` and `order_by`, plus the shared contracts, are in the [data operation patterns](docs/guides/data-operation-patterns/index.md). Also published: `mloda-community-data-operations` (the shared base classes) and the example packages `mloda-community-example` and `mloda-community-example-a`. `config/packages.toml` is the source of truth for the package list.

## PyPI packages

| Package | Description | License | Install |
|---------|-------------|---------|---------|
| `mloda-community` | All community plugins (bundle) | Apache 2.0 | `pip install mloda-community` |
| `mloda-community-<plugin>` | One plugin from the table above | Apache 2.0 | `pip install "mloda-community-rank[pandas]"` |
| `mloda-registry` | Plugin discovery and search | Apache 2.0 | `pip install mloda-registry` |
| `mloda-testing` | Test utilities for plugin development | Apache 2.0 | `pip install mloda-testing` |
| `mloda-enterprise` | All enterprise plugins (bundle) | [Source-available](mloda/enterprise/LICENSE) ([Get license](https://mloda.ai/enterprise)) | `pip install mloda-enterprise` |

> **Note:** Only `mloda/enterprise/` and its PyPI package require a license. Everything else in this repository is Apache 2.0 (see [LICENSE](LICENSE)).

The remaining example packages are not on PyPI; install them from git, replacing the subdirectory with the package `path` from `config/packages.toml`:

```bash
pip install "git+https://github.com/mloda-ai/mloda-registry.git#subdirectory=mloda/community/feature_groups/example/example_b"
```

## Guides

[`docs/guides/`](docs/guides/index.md) covers the whole plugin journey: using and discovering plugins, creating FeatureGroups in-project or as a package, sharing with a team or the community, and the advanced ComputeFramework and Extender plugins. Pattern references: [feature groups](docs/guides/feature-group-patterns/index.md), [compute frameworks](docs/guides/compute-framework-patterns/index.md), [data operations](docs/guides/data-operation-patterns/index.md).

## Development

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once
uv venv
source .venv/bin/activate
uv sync --all-extras
tox                                               # the gate: pytest, ruff, mypy --strict, bandit
```

Use `uv run tox` instead of `tox` after changing dependencies. Per-package envs (`tox -e registry`, `tox -e testing`, ...) and `tox -e lint-docs` are listed in `tox.ini`.

Every `pyproject.toml` is generated: edit `config/shared.toml` or `config/packages.toml`, then run `python scripts/generate_pyproject.py`. Maintainer docs: [packaging](docs/packaging.md), [releasing](docs/releasing.md).

## Contributing

New plugins, plugin improvements, guide fixes, and registry tooling fixes are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) for the PR workflow and code style; scaffold a standalone plugin with the [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template).

- [Open an issue](https://github.com/mloda-ai/mloda-registry/issues/new/choose) for bugs and feature requests
- Looking for a first task? Browse [`good first issue`](https://github.com/mloda-ai/mloda-registry/labels/good%20first%20issue) and [`help wanted`](https://github.com/mloda-ai/mloda-registry/labels/help%20wanted)
- [Code of Conduct](CODE_OF_CONDUCT.md) and [Security policy](SECURITY.md)

## Related repositories

- **[mloda](https://github.com/mloda-ai/mloda)**: the core library. Declare what data you need, not how to get it; mloda resolves features, dependencies, and compute frameworks.
- **[mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template)**: GitHub template for standalone FeatureGroup, ComputeFramework, and Extender packages.
