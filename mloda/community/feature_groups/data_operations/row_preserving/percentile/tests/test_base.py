"""Tests for PercentileFeatureGroup base class."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import pytest
from mloda.provider import FeatureChainParserMixin, FeatureSet, collect_match_rejections
from mloda.user import Feature, Options

from mloda.community.feature_groups.data_operations.base import KeyColumnInputsMixin
from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase, TokenCase
from mloda.testing.feature_groups.data_operations.operation_group import ImplSpec, OperationGroupTestBase


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(PercentileFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(PercentileFeatureGroup.PREFIX_PATTERN, str)

    def test_min_in_features_is_one(self) -> None:
        assert PercentileFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert PercentileFeatureGroup.MAX_IN_FEATURES == 1

    def test_percentile_constant_defined(self) -> None:
        assert PercentileFeatureGroup.PERCENTILE == "percentile"

    def test_partition_by_constant_defined(self) -> None:
        assert PercentileFeatureGroup.PARTITION_BY == "partition_by"


_ENGINE = "mloda_plugins.compute_framework.base_implementations"


class TestPercentileOperationGroup(OperationGroupTestBase):
    @classmethod
    def group_class(cls) -> Any:
        return PercentileFeatureGroup

    @classmethod
    def backend_package(cls) -> str:
        return "mloda.community.feature_groups.data_operations.row_preserving.percentile"

    @classmethod
    def implementations(cls) -> list[ImplSpec]:
        return [
            ImplSpec(
                "pandas",
                "pandas_percentile",
                "PandasPercentile",
                f"{_ENGINE}.pandas.dataframe",
                "PandasDataFrame",
                dependency="pandas",
            ),
            ImplSpec(
                "polars_lazy",
                "polars_lazy_percentile",
                "PolarsLazyPercentile",
                f"{_ENGINE}.polars.lazy_dataframe",
                "PolarsLazyDataFrame",
                dependency="polars",
            ),
            ImplSpec(
                "duckdb",
                "duckdb_percentile",
                "DuckdbPercentile",
                f"{_ENGINE}.duckdb.duckdb_framework",
                "DuckDBFramework",
                dependency="duckdb",
            ),
            ImplSpec(
                "python_dict",
                "python_dict_percentile",
                "PythonDictPercentile",
                f"{_ENGINE}.python_dict.python_dict_framework",
                "PythonDictFramework",
            ),
        ]

    @classmethod
    def subtype_key(cls) -> None:
        """The percentile key is a float, not a subtype dimension."""
        return None

    @classmethod
    def loop_method(cls) -> str:
        return "percentile"

    @classmethod
    def compute_method(cls) -> str:
        return "_compute_percentile"

    @classmethod
    def probe_feature_set(cls) -> FeatureSet:
        return make_feature_set("value_int__p50_percentile", ["region"])

    @classmethod
    def mixins(cls) -> tuple[type, ...]:
        return (KeyColumnInputsMixin, FeatureChainParserMixin)

    @classmethod
    def universe(cls) -> frozenset[str]:
        return frozenset()

    @classmethod
    def name_probes(cls) -> list[tuple[str, str]]:
        return []

    @classmethod
    def option_probes(cls) -> list[tuple[str, Options, str]]:
        return []

    @classmethod
    def unresolved_probes(cls) -> list[tuple[str, Options]]:
        return [("value_int__p50_percentile", Options())]


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__p0_percentile",
            "value_int__p25_percentile",
            "value_int__p50_percentile",
            "value_int__p75_percentile",
            "value_int__p90_percentile",
            "value_int__p95_percentile",
            "value_int__p99_percentile",
            "value_int__p100_percentile",
        ],
    )
    def test_matches_valid_percentiles(self, feature_name: str) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param("value_int__p50_grouped", id="wrong_suffix"),
            pytest.param("value_int__p50", id="no_suffix"),
            pytest.param("p50_percentile", id="no_source_column"),
            pytest.param("value_int__p101_percentile", id="invalid_percentile_too_high"),
            pytest.param("value_int__p-1_percentile", id="invalid_percentile_negative"),
            pytest.param("value_int__pfoo_percentile", id="non_numeric"),
        ],
    )
    def test_no_match(self, feature_name: str) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False


class TestPatternParsing:
    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            pytest.param("value_int__p50_percentile", 0.5, id="p50"),
            pytest.param("value_int__p25_percentile", 0.25, id="p25"),
            pytest.param("value_int__p0_percentile", 0.0, id="p0"),
            pytest.param("value_int__p100_percentile", 1.0, id="p100"),
        ],
    )
    def test_parse_percentile_value(self, feature_name: str, expected: float) -> None:
        result = PercentileFeatureGroup.get_percentile_value(feature_name)
        assert result == expected

    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            pytest.param("value_int__p50_percentile", 0.5, id="p50"),
            pytest.param("value_int__p0_percentile", 0.0, id="p0"),
            pytest.param("value_int__p100_percentile", 1.0, id="p100"),
        ],
    )
    def test_resolve_feature_name_binds_converted_percentile(self, feature_name: str, expected: float) -> None:
        resolution = PercentileFeatureGroup.resolve_feature_name(feature_name)
        assert resolution.bindings == {"percentile": expected}

    @pytest.mark.parametrize(
        "token",
        [pytest.param("p150", id="p150"), pytest.param("p" + "9" * 5000, id="huge_digits")],
    )
    def test_resolve_feature_name_out_of_range_percentile_is_rejected_binding(self, token: str) -> None:
        resolution = PercentileFeatureGroup.resolve_feature_name(f"value_int__{token}_percentile")
        assert "percentile" not in resolution.bindings
        assert resolution.rejected_bindings == {"percentile": token}

    def test_parse_source_feature(self) -> None:
        feature = Feature(
            "value_int__p50_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = PercentileFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_with_underscores(self) -> None:
        feature = Feature(
            "my_value__p75_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = PercentileFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_value"]


class TestConfigBasedFeatures:
    @pytest.mark.parametrize(
        ("percentile", "expected"),
        [
            pytest.param(0.75, True, id="valid"),
            pytest.param(1.5, False, id="rejects_invalid_percentile_too_high"),
            pytest.param(-0.1, False, id="rejects_invalid_percentile_negative"),
            pytest.param(True, False, id="rejects_bool_true"),
            pytest.param(False, False, id="rejects_bool_false"),
            pytest.param(1, True, id="boundary_int_one"),
            pytest.param(0, True, id="boundary_int_zero"),
            pytest.param(1.0, True, id="boundary_float_one"),
            pytest.param(0.0, True, id="boundary_float_zero"),
        ],
    )
    def test_config_based_match(self, percentile: float | int | bool, expected: bool) -> None:
        options = Options(
            context={
                "percentile": percentile,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is expected

    def test_config_based_match_rejects_missing_partition_by(self) -> None:
        options = Options(
            context={
                "percentile": 0.5,
                "in_features": "value_int",
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestConfigBasedExtraction:
    def test_extract_rejects_bool_true(self) -> None:
        feature = Feature(
            "my_result",
            options=Options(context={"percentile": True, "partition_by": ["region"]}),
        )
        with pytest.raises(ValueError):
            PercentileFeatureGroup._extract_percentile(feature)

    def test_extract_rejects_bool_false(self) -> None:
        feature = Feature(
            "my_result",
            options=Options(context={"percentile": False, "partition_by": ["region"]}),
        )
        with pytest.raises(ValueError):
            PercentileFeatureGroup._extract_percentile(feature)

    def test_extract_float_value(self) -> None:
        feature = Feature(
            "my_result",
            options=Options(context={"percentile": 0.75, "partition_by": ["region"]}),
        )
        result = PercentileFeatureGroup._extract_percentile(feature)
        assert result == 0.75

    def test_extract_int_value(self) -> None:
        feature = Feature(
            "my_result",
            options=Options(context={"percentile": 1, "partition_by": ["region"]}),
        )
        result = PercentileFeatureGroup._extract_percentile(feature)
        assert result == 1.0


class TestRejectionReporting:
    """A wrong-typed percentile is a reported strict-validation rejection, not a silent non-match."""

    def test_wrong_typed_percentile_reports_a_reason(self) -> None:
        options = Options(
            context={
                "percentile": "fifty",
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        reason = PercentileFeatureGroup._strict_validation_rejection_reason("my_result", options)
        assert reason is not None
        assert "percentile" in reason

    def test_out_of_range_percentile_reports_a_reason(self) -> None:
        """An out-of-range percentile is a reported strict-validation rejection, not a silent non-match."""
        for value in (50, 1.5):
            options = Options(
                context={
                    "percentile": value,
                    "in_features": "value_int",
                    "partition_by": ["region"],
                }
            )
            assert PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None) is False
            reason = PercentileFeatureGroup._strict_validation_rejection_reason("my_result", options)
            assert reason is not None
            assert "percentile" in reason

    def test_out_of_range_percentile_name_records_the_rejected_binding(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        with collect_match_rejections() as reasons:
            assert (
                PercentileFeatureGroup.match_feature_group_criteria("value_int__p150_percentile", options, None)
                is False
            )
        reason = reasons[PercentileFeatureGroup.get_class_name()].reason
        assert "percentile" in reason
        assert "'p150'" in reason

    def test_valid_percentile_reports_nothing(self) -> None:
        options = Options(
            context={
                "percentile": 0.5,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        reason = PercentileFeatureGroup._strict_validation_rejection_reason("my_result", options)
        assert reason is None


class TestPercentileNameAgreement:
    """A ``percentile`` (forwarded or own) that contradicts the name-parsed value is rejected, not silently ignored."""

    @pytest.mark.parametrize("path", ["group", "context"])
    def test_mismatched_forwarded_percentile_raises(self, path: str) -> None:
        child_options = Options(context={"partition_by": ["region"]})
        if path == "group":
            child_options.inherit_from(Options(group={"percentile": 0.75}))
        else:
            child_options.inherit_from(
                Options(context={"percentile": 0.75}), inherit_context_keys=frozenset({"percentile"})
            )
        assert ("percentile" in child_options.inherited_context_keys) == (path == "context")

        with pytest.raises(ValueError, match="percentile") as exc_info:
            PercentileFeatureGroup.match_feature_group_criteria("value_int__p50_percentile", child_options, None)
        assert "was forwarded from a consumer" in str(exc_info.value)

    def test_matching_forwarded_integer_percentile_is_accepted(self) -> None:
        consumer_options = Options(group={"percentile": 1})
        child_options = Options(context={"partition_by": ["region"]})
        child_options.inherit_from(consumer_options)

        result = PercentileFeatureGroup.match_feature_group_criteria("value_int__p100_percentile", child_options, None)
        assert result is True

    @pytest.mark.parametrize("wrap", [False, True], ids=["scalar", "singleton_tuple"])
    @pytest.mark.parametrize("path", ["group", "context"])
    def test_matching_forwarded_percentile_is_accepted(self, path: str, wrap: bool) -> None:
        value: Any = (0.5,) if wrap else 0.5
        child_options = Options(context={"partition_by": ["region"]})
        if path == "group":
            child_options.inherit_from(Options(group={"percentile": value}))
        else:
            child_options.inherit_from(
                Options(context={"percentile": value}), inherit_context_keys=frozenset({"percentile"})
            )

        result = PercentileFeatureGroup.match_feature_group_criteria("value_int__p50_percentile", child_options, None)
        assert result is True

    @pytest.mark.parametrize("placement", ["group", "context"])
    def test_own_contradictory_percentile_raises(self, placement: str) -> None:
        if placement == "group":
            options = Options(group={"percentile": 0.75}, context={"partition_by": ["region"]})
        else:
            options = Options(context={"percentile": 0.75, "partition_by": ["region"]})

        with pytest.raises(
            ValueError,
            match=r"option 'percentile' is float 0\.75, but the feature name encodes 'p50'\. The name is authoritative",
        ):
            PercentileFeatureGroup.match_feature_group_criteria("value_int__p50_percentile", options, None)

    @pytest.mark.parametrize(
        ("feature_name", "percentile"),
        [
            ("value_int__p50_percentile", 0.5),
            ("value_int__p50_percentile", (0.5,)),
            ("value_int__p100_percentile", 1),
            ("value_int__p100_percentile", (1,)),
        ],
    )
    @pytest.mark.parametrize("placement", ["group", "context"])
    def test_own_matching_percentile_is_accepted(self, placement: str, feature_name: str, percentile: Any) -> None:
        if placement == "group":
            options = Options(group={"percentile": percentile}, context={"partition_by": ["region"]})
        else:
            options = Options(context={"percentile": percentile, "partition_by": ["region"]})

        assert PercentileFeatureGroup.match_feature_group_criteria(feature_name, options, None) is True

    @pytest.mark.parametrize("origin", ["forwarded", "own"])
    def test_mismatched_percentile_env_downgrade_is_accepted(
        self, origin: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        options = Options(context={"partition_by": ["region"]})
        if origin == "forwarded":
            options.inherit_from(Options(group={"percentile": 0.75}))
        else:
            options = Options(context={"percentile": 0.75, "partition_by": ["region"]})

        with caplog.at_level(logging.WARNING):
            result = PercentileFeatureGroup.match_feature_group_criteria("value_int__p50_percentile", options, None)
        assert result is True
        assert any(r.levelno == logging.WARNING and "percentile" in r.getMessage() for r in caplog.records)

    def test_config_based_feature_ignores_unrelated_forwarded_percentile(self) -> None:
        consumer_options = Options(group={"percentile": 0.75})
        child_options = Options(context={"in_features": "value_int", "partition_by": ["region"]})
        child_options.inherit_from(consumer_options)

        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", child_options, None)
        assert result is True


class TestPercentileMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return PercentileFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return {"p0", "p25", "p50", "p75", "p90", "p95", "p99", "p100"}

    @classmethod
    def config_key(cls) -> str:
        return "percentile"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_percentile"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "p50_percentile"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "partition_by": ["region"]}

    @classmethod
    def pattern_match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"]})

    @classmethod
    def config_value(cls, operation: str) -> Any:
        # The name path uses pN tokens, the config path a float in [0.0, 1.0].
        return int(operation[1:]) / 100.0

    @classmethod
    def options_reject_invalid_types(cls) -> bool:
        # percentile is strict with an element validator, so wrong-typed config values are rejected.
        return True

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        # A percentile is one float in [0.0, 1.0]: out of range, a bool, and an int too large
        # for float all stay out. The last one must be a plain non-match rather than an
        # OverflowError, which the mixin does not catch and which would abort discovery for
        # every feature group.
        primary = super().token_cases()[0]
        return [replace(primary, invalid=(1.5, True, 10**400))]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        return [PercentileFeatureGroup._extract_percentile(Feature("my_result", options=options))]
