"""Re-export of mloda._testing.framework_adapter."""

from mloda._testing.framework_adapter import (
    AdapterCapabilities,
    DuckDBAdapter,
    FrameworkAdapter,
    IcebergAdapter,
    PandasAdapter,
    PolarsAdapter,
    PolarsLazyAdapter,
    PyArrowAdapter,
    PythonDictAdapter,
    SparkAdapter,
    SqliteAdapter,
    to_arrow_table,
)

__all__ = [
    "AdapterCapabilities",
    "DuckDBAdapter",
    "FrameworkAdapter",
    "IcebergAdapter",
    "PandasAdapter",
    "PolarsAdapter",
    "PolarsLazyAdapter",
    "PyArrowAdapter",
    "PythonDictAdapter",
    "SparkAdapter",
    "SqliteAdapter",
    "to_arrow_table",
]
