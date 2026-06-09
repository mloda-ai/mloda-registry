"""Shared skeleton for the arithmetic feature-group families.

The point-arithmetic and scalar-arithmetic families share the same
arithmetic-op extraction and numeric-source contract: they parse the same four
operation names (``add``/``subtract``/``multiply``/``divide``) out of the
feature name or options, and they reject non-numeric source columns with the
same error format. Issue #214 had this logic duplicated byte-for-byte across
the two families' ``base.py`` files.

``ArithmeticFeatureGroupBase`` holds the identical parts. The families subclass
it and override ``OPERATION_LABEL`` (so the rejection message names the right
operation) plus the family-specific bits (operand count, constant handling,
PROPERTY_MAPPING). Per-backend numeric introspection lives in
``numeric_source``; the per-backend ``_assert_source_column_is_numeric``
implementations delegate to it and call ``_raise_non_numeric_source`` here.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.provider import FeatureGroup

ARITHMETIC_OP_NAMES: frozenset[str] = frozenset({"add", "subtract", "multiply", "divide"})


class ArithmeticFeatureGroupBase(FeatureChainParserMixin, FeatureGroup):
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
        return str(op)

    @classmethod
    def _raise_non_numeric_source(cls, source_col: str, got: object) -> None:
        """Shared error format for the numeric-source contract.

        Backend overrides of ``_assert_source_column_is_numeric`` call this
        helper so the message stays uniform across all backends. ``got`` is
        inlined verbatim, allowing backends to pass a native dtype, an
        affinity string, or any other descriptor.
        """
        raise ValueError(f"Source column {source_col!r} must be numeric for {cls.OPERATION_LABEL}; got {got}.")

    @classmethod
    def _input_columns_and_framework(cls, data: Any) -> tuple[list[str], str]:
        """Return ``(column_names, framework_label)`` for ``data``.

        Backend-specific; implemented per backend so the base class has no
        compile-time or import-time dependency on any compute framework.
        """
        raise NotImplementedError

    @classmethod
    def _assert_source_column_is_numeric(cls, data: Any, source_col: str) -> None:
        """Reject non-numeric source columns with a clear ``ValueError``.

        Backend-specific; implemented per backend. Implementations should
        call ``cls._raise_non_numeric_source(source_col, <native dtype>)`` to
        keep the message format uniform.
        """
        raise NotImplementedError
