"""Base class for scalar arithmetic feature groups.

Computes an element-wise arithmetic operation (add, subtract, multiply,
divide) between a single source column and a numeric constant supplied
via ``Options(context={"constant": <value>})``. Supports DuckDB, SQLite,
Pandas, Polars, and PyArrow backends.

Pattern: ``{col}__{op}_constant``

Example: ``value_int__divide_constant`` with ``constant=2`` divides every
non-null value in ``value_int`` by 2.

The ``constant`` option carries ``strict_validation=False`` so that
pattern-only matches (``{col}__{op}_constant``) succeed without it; the
missing-constant check then fires at compute time with a clear error.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys, FeatureGroup

from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns

ARITHMETIC_OPERATIONS: dict[str, str] = {
    "add": "Element-wise addition of a constant",
    "subtract": "Element-wise subtraction of a constant",
    "multiply": "Element-wise multiplication by a constant",
    "divide": "Element-wise division by a constant",
}


class ScalarArithmeticFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__([\w]+)_constant$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    ARITHMETIC_OP = "arithmetic_op"
    CONSTANT = "constant"

    PROPERTY_MAPPING = {
        ARITHMETIC_OP: {
            **ARITHMETIC_OPERATIONS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Single source feature column for the arithmetic operation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        CONSTANT: {
            "explanation": "Numeric constant applied element-wise to the source column",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        return operation_config in ARITHMETIC_OPERATIONS

    @classmethod
    def get_arithmetic_op(cls, feature_name: str) -> str:
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract arithmetic operation from feature name: {feature_name}")

    @classmethod
    def _extract_arithmetic_op(cls, feature: Feature) -> str:
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        op = feature.options.get(cls.ARITHMETIC_OP)
        if op is None:
            raise ValueError(f"Could not extract arithmetic operation for {feature_name}")
        return str(op)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        _feature_name = str(feature_name)

        prefix_patterns = self._get_prefix_patterns()
        operation_config, source_feature = FeatureChainParser.parse_feature_name(_feature_name, prefix_patterns)

        if operation_config and source_feature:
            return {Feature(source_feature)}

        in_features_set = options.get_in_features()
        self._validate_in_feature_count(list(in_features_set), _feature_name)
        return set(in_features_set)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        """Extract and validate the single source feature for the arithmetic op.

        Returns a one-element list containing the source column name.
        Raises ValueError if more than one source feature is found, since
        this package only supports single-column arithmetic.
        """
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()

        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)

        if operation_config and source_feature:
            return [source_feature]

        in_features_set = feature.options.get_in_features()
        source_names: list[str] = [str(f.name) for f in in_features_set]

        if len(source_names) > cls.MAX_IN_FEATURES:
            raise ValueError(
                f"Scalar arithmetic supports at most {cls.MAX_IN_FEATURES} source feature, "
                f"but got {len(source_names)}: {source_names}"
            )

        return source_names

    @staticmethod
    def _input_columns_and_framework(data: Any) -> tuple[list[str], str]:
        """Return ``(column_names, framework_label)`` polymorphically across backends.

        Probes in this order so each branch lands on its native accessor:

        - ``data.column_names`` (a ``list``) for PyArrow tables.
        - ``data.collect_schema().names()`` for Polars lazy frames.
        - ``data.columns`` for Pandas DataFrames (an ``Index``) and the
          DuckDB / SQLite relation wrappers (each exposes ``list[str]``).

        Keeping the probe local avoids importing the optional backends
        (``pyarrow``, ``polars``, ``pandas``, ``duckdb``) into this base
        module, which would tie the package to dependencies it does not need.
        The framework label is best-effort: for SQL relation wrappers it
        falls back to ``"SQL"`` since DuckDB and SQLite share the same
        ``columns: list[str]`` shape.
        """
        column_names = getattr(data, "column_names", None)
        if isinstance(column_names, list):
            return column_names, "PyArrow"

        collect_schema = getattr(data, "collect_schema", None)
        if callable(collect_schema):
            return list(collect_schema().names()), "Polars"

        columns = getattr(data, "columns", None)
        if columns is not None:
            try:
                import pandas as pd

                if isinstance(data, pd.DataFrame):
                    return list(columns), "Pandas"
            except ImportError:  # pragma: no cover - defensive
                pass
            type_name = type(data).__name__
            if type_name == "DuckdbRelation":
                return list(columns), "DuckDB"
            if type_name == "SqliteRelation":
                return list(columns), "SQLite"
            return list(columns), "SQL"

        raise TypeError(
            f"Cannot determine column names for object of type {type(data).__name__}; "
            "scalar arithmetic supports PyArrow, Polars (lazy), Pandas, DuckDB, and SQLite inputs."
        )

    @staticmethod
    def _assert_source_column_is_numeric(data: Any, source_col: str) -> None:
        """Reject non-numeric source columns with a clear ``ValueError``.

        Covers PyArrow, Polars (lazy), Pandas, DuckDB, and SQLite. Each backend is
        inspected at its native API: ``collect_schema`` for Polars LazyFrame,
        ``.column().type`` for PyArrow, ``.dtypes`` for Pandas, ``_relation.types``
        for DuckDB, and ``PRAGMA table_info`` for SQLite.

        SQLite caveat: ``SqliteRelation.from_arrow`` stores boolean columns with
        SQLite ``INTEGER`` affinity, so a boolean source column is indistinguishable
        from ``int64`` after materialization. The columnar backends and DuckDB
        preserve the boolean type and reject it; SQLite accepts it and performs
        arithmetic on the 0/1 storage.

        Branch order matters: Polars is checked before Pandas because Polars
        LazyFrame also exposes ``.dtypes`` / ``.columns`` (each triggering a
        ``PerformanceWarning`` about schema resolution).
        """
        # Polars LazyFrame: schema via ``collect_schema``.
        if callable(getattr(data, "collect_schema", None)):
            try:
                import polars as pl
            except ImportError:  # pragma: no cover - defensive
                return
            dtype = data.collect_schema()[source_col]
            if dtype == pl.Boolean or not dtype.is_numeric():
                raise ValueError(f"Source column {source_col!r} must be numeric for scalar arithmetic; got {dtype}.")
            return

        # PyArrow Table: detect via ``column_names`` (a ``list``) plus ``column`` accessor.
        if isinstance(getattr(data, "column_names", None), list) and callable(getattr(data, "column", None)):
            try:
                import pyarrow as pa
            except ImportError:  # pragma: no cover - defensive
                return
            arrow_type = data.column(source_col).type
            if pa.types.is_boolean(arrow_type) or not (
                pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type)
            ):
                raise ValueError(
                    f"Source column {source_col!r} must be numeric for scalar arithmetic; got {arrow_type}."
                )
            return

        # Pandas DataFrame: detect via ``dtypes`` plus ``columns``.
        if hasattr(data, "dtypes") and hasattr(data, "columns"):
            try:
                import pandas as pd
            except ImportError:  # pragma: no cover - defensive
                return
            if isinstance(data, pd.DataFrame):
                series = data[source_col]
                if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
                    raise ValueError(
                        f"Source column {source_col!r} must be numeric for scalar arithmetic; got {series.dtype}."
                    )
                return

        # DuckDB: inspect the wrapped relation's declared types directly. Cheap
        # (~4 microseconds) vs. ``data.to_arrow_table().schema`` which materializes.
        if type(data).__name__ == "DuckdbRelation":
            underlying = getattr(data, "_relation", None)
            if underlying is not None:
                try:
                    duckdb_types = [str(t) for t in underlying.types]
                    duckdb_columns = list(underlying.columns)
                except Exception:  # pragma: no cover - defensive
                    return
                type_by_column = dict(zip(duckdb_columns, duckdb_types))
                dtype_str = type_by_column.get(source_col)
                if dtype_str is not None:
                    numeric_prefixes = (
                        "TINYINT",
                        "SMALLINT",
                        "INTEGER",
                        "BIGINT",
                        "HUGEINT",
                        "UTINYINT",
                        "USMALLINT",
                        "UINTEGER",
                        "UBIGINT",
                        "UHUGEINT",
                        "FLOAT",
                        "DOUBLE",
                        "REAL",
                        "DECIMAL",
                        "NUMERIC",
                        "BIGNUM",
                    )
                    if not any(dtype_str == p or dtype_str.startswith(p + "(") for p in numeric_prefixes):
                        raise ValueError(
                            f"Source column {source_col!r} must be numeric for scalar arithmetic; got {dtype_str}."
                        )
            return

        # SQLite: inspect ``PRAGMA table_info`` for declared affinity. Cheap
        # (~15 microseconds) vs. ``to_arrow_table().schema`` which fully
        # materializes the relation. Caveat: ``SqliteRelation.from_arrow`` maps
        # arrow booleans to SQLite INTEGER affinity, so a boolean source column
        # is indistinguishable from int64 at the relation level. The shared test
        # ``test_boolean_source_column_rejected`` is correspondingly skipped for
        # SQLite via the ``detects_non_numeric_source`` test-class override.
        if type(data).__name__ == "SqliteRelation":
            conn = getattr(data, "connection", None)
            table_name = getattr(data, "table_name", None)
            if conn is not None and table_name is not None:
                # Identifier escape: double internal double-quotes; never user-controlled
                # since SqliteRelation generates the name itself.
                safe_table = '"' + str(table_name).replace('"', '""') + '"'
                try:
                    rows = conn.execute(f"PRAGMA table_info({safe_table})").fetchall()
                except Exception:  # pragma: no cover - defensive
                    return
                affinity_by_column = {row[1]: (row[2] or "").upper() for row in rows}
                affinity = affinity_by_column.get(source_col)
                if affinity is not None:
                    if (
                        "INT" in affinity
                        or "REAL" in affinity
                        or "FLOA" in affinity
                        or "DOUB" in affinity
                        or "NUMERIC" in affinity
                    ):
                        return
                    raise ValueError(
                        f"Source column {source_col!r} must be numeric for scalar arithmetic; "
                        f"got SQLite affinity {affinity!r}."
                    )
            return

        return

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute an element-wise arithmetic operation per source column.

        Each feature produces one new column containing ``source {op} constant``.
        Null values in the source propagate to the result. Divide-by-zero and
        missing constant are rejected before dispatching to the backend.

        Reserved-column guard runs first so callers that omit ``constant`` (such
        as the shared ``ReservedColumnsTestMixin``) see the reserved-column
        error rather than the missing-constant one.
        """
        column_names, framework_label = cls._input_columns_and_framework(data)
        assert_no_reserved_columns(column_names, framework=framework_label, operation="scalar arithmetic")

        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            op = cls._extract_arithmetic_op(feature)

            cls._assert_source_column_is_numeric(data, source_col)

            constant = feature.options.get(cls.CONSTANT)
            if constant is None:
                raise ValueError(f"Missing required option 'constant' for feature {feature_name!r}")
            if isinstance(constant, bool) or not isinstance(constant, (int, float)):
                raise ValueError(
                    f"Option 'constant' for feature {feature_name!r} must be int or float, "
                    f"got {type(constant).__name__}"
                )
            if op == "divide" and constant == 0:
                raise ValueError(f"Cannot divide by zero for feature {feature_name!r}")

            table = cls._compute_arithmetic(table, feature_name, source_col, op, constant)

        return table

    @classmethod
    def _compute_arithmetic(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
        constant: int | float,
    ) -> Any:
        raise NotImplementedError
