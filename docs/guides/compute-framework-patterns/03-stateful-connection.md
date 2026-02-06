# Category 3: Stateful Connection Frameworks

Stateful frameworks require a connection or session object to operate.

**What**: Frameworks requiring a connection/session to process data.
**When**: Database connections, distributed compute engines, SQL-based queries.
**Why**: External engine handles computation; enables distributed processing.
**Where**: DuckDB, Spark, Trino, Dask distributed.
**How**: Same as Category 1, plus implement `set_framework_connection_object()`.

## Key Difference from Stateless

| Aspect | Stateless (Category 1) | Stateful (Category 3) |
|--------|------------------------|------------------------|
| Connection | Not needed | **Required** |
| Critical method | - | `set_framework_connection_object()` |
| Data location | In-memory | External engine/database |
| Transform | Direct conversion | Requires connection |

## What's Different

Only this method is added:

```python
def set_framework_connection_object(self, framework_connection_object: Optional[Any] = None) -> None:
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

And `transform()` must check connection:

```python
def transform(self, data: Any, feature_names: Set[str]) -> Any:
    if self.framework_connection_object is None:
        raise ValueError("Connection not set.")
    # Use connection for conversion...
```

## Usage

Pass connections via `data_connections`:

```python
conn = duckdb.connect()
result = mloda.run_all(
    features=[...],
    compute_frameworks=["MyStatefulFramework"],
    data_connections=[conn],
)
```

## Real Implementations

| File | Description |
|------|-------------|
| [duckdb/duckdb_framework.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/duckdb/duckdb_framework.py) | DuckDB |
| [spark/spark_framework.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/spark/spark_framework.py) | Spark |

## Combines With

- **Category 1**: Inherits base structure
- **Merge Engine** (Concept 6): Connection passed to merge engine
- **ConnectionMatcherMixin**: Match feature groups to connections
