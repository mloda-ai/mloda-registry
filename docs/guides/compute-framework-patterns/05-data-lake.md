# Category 5: Data Lake Frameworks

Data lake frameworks work with catalog-based table formats for large-scale data.

**What**: Frameworks for data lake table formats with metadata management.
**When**: Schema evolution, time travel, partitioned data, data catalogs.
**Why**: ACID transactions on data lakes; efficient metadata-driven queries.
**Where**: Apache Iceberg, Delta Lake, Apache Hudi.
**How**: Like Category 3, but accepts catalog OR table; uses PyArrow interchange.

## Key Difference from Category 3

| Aspect | Category 3 (Stateful) | Category 5 (Data Lake) |
|--------|----------------------|------------------------|
| Connection type | Single type | Catalog **or** Table |
| Merge engine | Usually supported | Often **not supported** |
| Data interchange | Library-specific | PyArrow |
| Type validation | Strict | Relaxed (accepts PyArrow too) |

## What's Different

Flexible connection - accepts catalog or table:

```python
def set_framework_connection_object(self, framework_connection_object: Any | None = None) -> None:
    if self.framework_connection_object is None and framework_connection_object is not None:
        if hasattr(framework_connection_object, "load_table"):
            self.framework_connection_object = framework_connection_object  # Catalog
        elif isinstance(framework_connection_object, IcebergTable):
            self.framework_connection_object = framework_connection_object  # Table
        else:
            raise ValueError(f"Expected catalog or table")
```

Declare the requirement as in [Category 3](03-stateful-connection.md#declare-the-connection-requirement): `SELF_MANAGED` when the framework still runs without a supplied catalog (Iceberg works on the table objects it is given), `REQUIRED` when it cannot, with `_connection_matches()` accepting a catalog or a table:

```python
from mloda.provider import ConnectionRequirement


@classmethod
def connection_requirement(cls) -> ConnectionRequirement:
    return ConnectionRequirement.SELF_MANAGED


@classmethod
def _connection_matches(cls, conn: Any) -> bool:
    return hasattr(conn, "load_table") or isinstance(conn, IcebergTable)
```

Merge typically not supported:

```python
@classmethod
def merge_engine(cls) -> type[BaseMergeEngine]:
    raise NotImplementedError("Use catalog-level operations for merging.")
```

PyArrow as interchange format:

```python
def transform(self, data: Any, feature_names: set[str]) -> Any:
    if isinstance(data, dict):
        return pa.Table.from_pydict(data)  # PyArrow interchange
    if isinstance(data, (IcebergTable, pa.Table)):
        return data
    raise ValueError(f"Data type {type(data)} not supported")
```

## Real Implementations

| File | Description |
|------|-------------|
| [iceberg/iceberg_framework.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/iceberg/iceberg_framework.py) | Iceberg |
| [iceberg/iceberg_filter_engine.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/iceberg/iceberg_filter_engine.py) | Iceberg filter |

## Combines With

- **Category 3**: Inherits connection pattern
- **Filter Engine** (Concept 7): Partition-aware filtering
- **Transformer** (Concept 8): PyArrow conversion
