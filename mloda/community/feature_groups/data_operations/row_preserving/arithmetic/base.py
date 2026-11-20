"""Shared arithmetic-op parsing and non-numeric rejection message.

Both arithmetic operation groups (point and scalar) mix ``ArithmeticOpMixin`` in
for op parsing and the shared non-numeric rejection message. Each Implementation
passes its framework's ``<fw>_non_numeric_descriptor`` to the group's ``apply_arithmetic``.
"""

from __future__ import annotations

from mloda.provider import Cause, Divergence, DivergenceKind, FeatureChainParser, FeatureChainParserMixin
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.base import op_token_value

ARITHMETIC_OP_NAMES: frozenset[str] = frozenset({"add", "subtract", "multiply", "divide"})

#: Shared SQL operator map, imported by the DuckDB and SQLite Implementations.
SQL_ARITHMETIC_OPS: dict[str, str] = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}

#: Shared NaN-as-null variant record for the Pandas and SQLite Implementations.
NAN_AS_NULL_DIVERGENCE = Divergence(
    kind=DivergenceKind.VARIANT,
    variant="nan_as_null",
    cause=Cause.BACKEND,
    reason="turns a NaN operand or result into null",
)


def _overflow_gap(reason: str) -> Divergence:
    return Divergence(
        kind=DivergenceKind.GAP,
        cause=Cause.BACKEND,
        subtypes=frozenset({"add", "subtract", "multiply"}),
        reason=reason,
    )


OVERFLOW_WRAPS_OPERAND_WIDTH = _overflow_gap("wraps an integer overflow at the operand width instead of raising")
OVERFLOW_WRAPS_INT64_WIDENS = _overflow_gap("wraps an int64 overflow and widens a narrower result instead of raising")
OVERFLOW_UNBOUNDED = _overflow_gap(
    "returns the unbounded integer result because rows carry no integer width instead of raising"
)
OVERFLOW_SQLITE_REAL = _overflow_gap("keeps 64-bit integers and turns an int64 overflow into REAL instead of raising")


class ArithmeticOpMixin(FeatureChainParserMixin):
    ARITHMETIC_OP = "arithmetic_op"

    #: Operation label used in the numeric-source rejection message.
    #: Subclasses override with ``"point arithmetic"`` / ``"scalar arithmetic"``.
    OPERATION_LABEL: str = ""

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        return operation_config in ARITHMETIC_OP_NAMES

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
        return op_token_value(op)

    @classmethod
    def _raise_non_numeric_source(cls, source_col: str, got: object) -> None:
        """Shared error format for the numeric-source contract.

        ``got`` is the framework descriptor (dtype, affinity string, ...), inlined verbatim.
        """
        raise ValueError(f"Source column {source_col!r} must be numeric for {cls.OPERATION_LABEL}; got {got}.")
