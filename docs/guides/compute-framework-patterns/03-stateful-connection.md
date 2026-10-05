# Category 3: Stateful Connection Frameworks

Stateful frameworks require a connection or session object to operate.

**What**: Frameworks requiring a connection/session to process data.
**When**: Database connections, distributed compute engines, SQL-based queries.
**Why**: External engine handles computation; enables distributed processing.
**Where**: DuckDB, SQLite, Spark, Trino, Dask distributed.
**How**: Same as Category 1, plus implement `set_framework_connection_object()`, `connection_requirement()` and `_connection_matches()`.

## Key Difference from Stateless

| Aspect | Stateless (Category 1) | Stateful (Category 3) |
|--------|------------------------|------------------------|
| Connection | Not needed | **Required** or self-managed |
| Critical methods | - | `set_framework_connection_object()`, `connection_requirement()`, `_connection_matches()` |
| Data location | In-memory | External engine/database |
| Transform | Direct conversion | Requires connection |

## What's Different

Only this method is added:

```python
def set_framework_connection_object(self, framework_connection_object: Any | None = None) -> None:
    """CRITICAL: Set or validate connection object."""
    if self.framework_connection_object is None:
        if framework_connection_object is not None:
            # Validate type
            if not isinstance(framework_connection_object, ExpectedConnectionType):
                raise ValueError(f"Expected connection type, got {type(framework_connection_object)}")
            self.framework_connection_object = framework_connection_object
        # Optional: auto-create connection
        # else:
        #     self.framework_connection_object = library.connect()
```

And `transform()` must check the connection:

```python
def transform(self, data: Any, feature_names: set[str]) -> Any:
    if self.framework_connection_object is None:
        raise ValueError("Connection not set.")
    # Use connection for conversion...
```

## Declare the Connection Requirement

Override `connection_requirement()` and `_connection_matches()`; both default to "no connection" (`NONE`, `False`):

```python
from mloda.provider import ConnectionRequirement


@classmethod
def connection_requirement(cls) -> ConnectionRequirement:
    return ConnectionRequirement.REQUIRED


@classmethod
def _connection_matches(cls, conn: Any) -> bool:
    return isinstance(conn, ExpectedConnectionType)
```

| Value | Pick when | Core examples |
|-------|-----------|---------------|
| `NONE` (default) | The framework runs without any connection | Pandas, PyArrow, Polars |
| `SELF_MANAGED` | It uses a supplied connection but still runs without one (opens its own session, or works on the tables it is given) | Spark, Iceberg |
| `REQUIRED` | It cannot run without a supplied connection | DuckDB, SQLite |

Planning ranks frameworks by this value (`NONE` first) after any explicit `compute_frameworks` order, and skips a `REQUIRED` framework for an unpinned feature without a matching connection when another framework fits. That skip checks only the feature's options (keyed by the FeatureGroup's class name), not `data_access_collection`, so pin such features with `compute_frameworks`. A connection-needing framework left at `NONE` ranks next to Pandas, so unpinned runs can land on it and fail for lack of a connection.

## Usage

Pass connections via `data_access_collection`. The engine picks the connection your `_connection_matches()` accepts at setup and calls `set_framework_connection_object()` on the framework for you:

```python
conn = duckdb.connect()
result = mloda.run_all(
    features=[...],
    compute_frameworks=["MyStatefulFramework"],
    data_access_collection=DataAccessCollection(connections={conn}),
)
```

## Real Implementations

| File | Description |
|------|-------------|
| [duckdb/duckdb_framework.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/duckdb/duckdb_framework.py) | DuckDB |
| [sqlite/sqlite_framework.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/sqlite/sqlite_framework.py) | SQLite |
| [spark/spark_framework.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/spark/spark_framework.py) | Spark |

## Combines With

- **Category 1**: Inherits base structure
- **Merge Engine** (Concept 6): Connection passed to merge engine
- **MatchData**: Match feature groups to connections
