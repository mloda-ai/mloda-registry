"""Tests for BinningFeatureGroup base class."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from mloda.provider import FeatureSet
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BINNING_OPS,
    BinningFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase, TokenCase
from mloda.testing.feature_groups.data_operations.operation_group import ImplSpec, OperationGroupTestBase


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(BinningFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(BinningFeatureGroup.PREFIX_PATTERN, str)

    def test_binning_ops_contains_all_operations(self) -> None:
        expected_ops = {"bin", "qbin"}
        for op in expected_ops:
            assert op in BINNING_OPS, f"Missing operation: {op}"

    def test_min_in_features_is_one(self) -> None:
        assert BinningFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert BinningFeatureGroup.MAX_IN_FEATURES == 1


_ENGINE = "mloda_plugins.compute_framework.base_implementations"


class TestBinningOperationGroup(OperationGroupTestBase):
    @classmethod
    def group_class(cls) -> Any:
        return BinningFeatureGroup

    @classmethod
    def backend_package(cls) -> str:
        return "mloda.community.feature_groups.data_operations.row_preserving.binning"

    @classmethod
    def implementations(cls) -> list[ImplSpec]:
        return [
            ImplSpec(
                "pyarrow",
                "pyarrow_binning",
                "PyArrowBinning",
                f"{_ENGINE}.pyarrow.table",
                "PyArrowTable",
            ),
            ImplSpec(
                "pandas",
                "pandas_binning",
                "PandasBinning",
                f"{_ENGINE}.pandas.dataframe",
                "PandasDataFrame",
                dependency="pandas",
            ),
            ImplSpec(
                "polars_lazy",
                "polars_lazy_binning",
                "PolarsLazyBinning",
                f"{_ENGINE}.polars.lazy_dataframe",
                "PolarsLazyDataFrame",
                dependency="polars",
            ),
            ImplSpec(
                "duckdb",
                "duckdb_binning",
                "DuckdbBinning",
                f"{_ENGINE}.duckdb.duckdb_framework",
                "DuckDBFramework",
                dependency="duckdb",
            ),
            ImplSpec(
                "sqlite",
                "sqlite_binning",
                "SqliteBinning",
                f"{_ENGINE}.sqlite.sqlite_framework",
                "SqliteFramework",
            ),
            ImplSpec(
                "python_dict",
                "python_dict_binning",
                "PythonDictBinning",
                f"{_ENGINE}.python_dict.python_dict_framework",
                "PythonDictFramework",
            ),
        ]

    @classmethod
    def subtype_key(cls) -> str:
        return "binning_op"

    @classmethod
    def loop_method(cls) -> str:
        return "assign_bins"

    @classmethod
    def compute_method(cls) -> str:
        return "_compute_binning"

    @classmethod
    def probe_feature_set(cls) -> FeatureSet:
        return make_feature_set("value_int__bin_3")

    @classmethod
    def name_probes(cls) -> list[tuple[str, str]]:
        return [("value__bin_5", "bin"), ("value__qbin_3", "qbin")]

    @classmethod
    def option_probes(cls) -> list[tuple[str, Options, str]]:
        return [("my_result", Options(context={"binning_op": "qbin", "n_bins": 3, "in_features": "value"}), "qbin")]


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__bin_3",
            "value_int__bin_5",
            "value_int__bin_10",
            "value_float__bin_4",
            "value_int__qbin_3",
            "value_int__qbin_5",
            "value_float__qbin_4",
        ],
    )
    def test_matches_valid_binning_features(self, feature_name: str) -> None:
        options = Options()
        result = BinningFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param("value_int__bucket_3", id="wrong_suffix"),
            pytest.param("value_int__bin", id="no_number"),
            pytest.param("bin_3", id="no_source_column"),
            pytest.param("value_int__bin_" + "9" * 5000, id="huge_digits"),
        ],
    )
    def test_no_match(self, feature_name: str) -> None:
        options = Options()
        result = BinningFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__bin_0",
            "value_int__qbin_0",
        ],
    )
    def test_no_match_zero_bins(self, feature_name: str) -> None:
        # n_bins must be >= 1, so the prefix pattern rejects a trailing 0.
        options = Options()
        result = BinningFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False, f"Should not match (n_bins must be >= 1): {feature_name}"


class TestPatternParsing:
    def test_parse_bin_operation(self) -> None:
        op, n_bins = BinningFeatureGroup.get_binning_params("value_int__bin_5")
        assert op == "bin"
        assert n_bins == 5

    def test_parse_qbin_operation(self) -> None:
        op, n_bins = BinningFeatureGroup.get_binning_params("value_int__qbin_10")
        assert op == "qbin"
        assert n_bins == 10

    @pytest.mark.parametrize(
        ("feature_name", "expected_op", "expected_n_bins"),
        [("value_int__bin_5", "bin", 5), ("value_int__qbin_10", "qbin", 10)],
    )
    def test_resolve_feature_name_bindings(self, feature_name: str, expected_op: str, expected_n_bins: int) -> None:
        resolution = BinningFeatureGroup.resolve_feature_name(feature_name)
        assert resolution.bindings == {"binning_op": expected_op, "n_bins": expected_n_bins}
        assert isinstance(resolution.bindings["n_bins"], int)

    @pytest.mark.parametrize(
        "feature_name",
        [
            # bin_0 no longer matches the prefix pattern (n_bins must be >= 1)
            pytest.param("value_int__bin_0", id="bin_0"),
            pytest.param("value_int__qbin_0", id="qbin_0"),
            pytest.param("value_int__bin_" + "9" * 5000, id="huge_digits"),
        ],
    )
    def test_unextractable_bin_suffix_raises(self, feature_name: str) -> None:
        with pytest.raises(ValueError, match="Could not extract binning parameters"):
            BinningFeatureGroup.get_binning_params(feature_name)

    def test_huge_digit_name_does_not_fall_back_to_options(self) -> None:
        feature = Feature("value_int__bin_" + "9" * 5000, Options(context={"binning_op": "bin", "n_bins": 3}))
        with pytest.raises(ValueError, match="Could not extract binning parameters"):
            BinningFeatureGroup._extract_binning_params(feature)

    def test_parse_source_feature(self) -> None:
        from mloda.user import Feature

        feature = Feature("value_int__bin_3", options=Options())
        source_features = BinningFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_with_underscores(self) -> None:
        from mloda.user import Feature

        feature = Feature("my_value_int__bin_5", options=Options())
        source_features = BinningFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_value_int"]

    def test_multi_source_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="allows at most 1 in_feature"):
            BinningFeatureGroup._extract_single_source_feature(Feature("a&b__bin_3"))


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "binning_op": "bin",
                "n_bins": 5,
                "in_features": "value_int",
            }
        )
        result = BinningFeatureGroup.match_feature_group_criteria("my_binned_result", options, None)
        assert result is True

    def test_config_based_match_rejects_invalid_op(self) -> None:
        options = Options(
            context={
                "binning_op": "invalid_op",
                "n_bins": 5,
                "in_features": "value_int",
            }
        )
        result = BinningFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    @pytest.mark.parametrize("invalid_n_bins", [0, -1, "abc"])
    def test_config_based_match_rejects_invalid_n_bins(self, invalid_n_bins: Any) -> None:
        # n_bins must be a positive integer; 0, negatives and non-ints must not match.
        options = Options(
            context={
                "binning_op": "bin",
                "n_bins": invalid_n_bins,
                "in_features": "value_int",
            }
        )
        result = BinningFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestNBinsCoercion:
    """The ``n_bins`` check the arity harness cannot express.

    Every arity verdict for ``n_bins`` (bare, wrapped, multi-element, wrong type and the
    values outside its value space) is declared on ``TestBinningMatchValidation``; what
    stays here is the return type a direct extractor call owes its caller.
    """

    @pytest.mark.parametrize("n_bins", [5, (5,), 5.9, (5.9,)])
    def test_extract_binning_params_returns_an_int(self, n_bins: Any) -> None:
        """The signature says ``tuple[str, int]``: a direct call must coerce, not just unwrap.

        ``is_positive_int`` keeps a float out at match time, so only a direct call can
        reach here with one, and it must still hand the backend an int bin count.
        """
        options = Options(context={"binning_op": "bin", "n_bins": n_bins, "in_features": "value_int"})
        _, extracted = BinningFeatureGroup._extract_binning_params(Feature("my_result", options=options))
        assert extracted == 5
        assert isinstance(extracted, int)


class TestReturnDataTypeRule:
    """return_data_type_rule should fix the output type for deterministic ops.

    Both bin and qbin emit integer bin indices, so the rule returns INT64.
    """

    def test_bin_returns_int64(self) -> None:
        feature = Feature("value_int__bin_5", options=Options())
        assert BinningFeatureGroup.return_data_type_rule(feature) == DataType.INT64

    def test_qbin_returns_int64(self) -> None:
        feature = Feature("value_int__qbin_4", options=Options())
        assert BinningFeatureGroup.return_data_type_rule(feature) == DataType.INT64


class TestNBinsNameAgreement:
    """An ``n_bins`` (forwarded or own) that contradicts the name-parsed value is rejected, not silently ignored."""

    @pytest.mark.parametrize("path", ["group", "context"])
    def test_mismatched_forwarded_n_bins_raises(self, path: str) -> None:
        child_options = Options()
        if path == "group":
            child_options.inherit_from(Options(group={"n_bins": 10}))
        else:
            child_options.inherit_from(Options(context={"n_bins": 10}), inherit_context_keys=frozenset({"n_bins"}))
        assert ("n_bins" in child_options.inherited_context_keys) == (path == "context")

        with pytest.raises(ValueError, match="n_bins") as exc_info:
            BinningFeatureGroup.match_feature_group_criteria("value_int__bin_5", child_options, None)
        assert "was forwarded from a consumer" in str(exc_info.value)

    def test_matching_forwarded_n_bins_is_accepted(self) -> None:
        consumer_options = Options(group={"n_bins": 5})
        child_options = Options()
        child_options.inherit_from(consumer_options)

        result = BinningFeatureGroup.match_feature_group_criteria("value_int__bin_5", child_options, None)
        assert result is True

    @pytest.mark.parametrize("feature_name", ["value_int__bin_5", "value_int__qbin_5"])
    @pytest.mark.parametrize("placement", ["group", "context"])
    @pytest.mark.parametrize("value", [10, 0], ids=["contradicts", "guard_invalid_checked_first"])
    def test_own_contradictory_n_bins_raises(self, placement: str, feature_name: str, value: int) -> None:
        options = Options(group={"n_bins": value}) if placement == "group" else Options(context={"n_bins": value})

        with pytest.raises(
            ValueError,
            match=rf"option 'n_bins' is int {value}, but the feature name encodes '5'\. The name is authoritative",
        ):
            BinningFeatureGroup.match_feature_group_criteria(feature_name, options, None)

    @pytest.mark.parametrize("n_bins", [5, (5,)])
    def test_own_matching_n_bins_is_accepted(self, n_bins: Any) -> None:
        options = Options(group={"n_bins": n_bins})

        assert BinningFeatureGroup.match_feature_group_criteria("value_int__bin_5", options, None) is True

    @pytest.mark.parametrize("origin", ["forwarded", "own"])
    def test_mismatched_n_bins_env_downgrade_is_accepted(
        self, origin: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        if origin == "forwarded":
            consumer_options = Options(group={"n_bins": 10})
            options = Options()
            options.inherit_from(consumer_options)
        else:
            options = Options(group={"n_bins": 10})

        with caplog.at_level(logging.WARNING):
            result = BinningFeatureGroup.match_feature_group_criteria("value_int__bin_5", options, None)
        assert result is True
        assert any(r.levelno == logging.WARNING and "n_bins" in r.getMessage() for r in caplog.records)


class TestBinningMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return BinningFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(BINNING_OPS)

    @classmethod
    def config_key(cls) -> str:
        return "binning_op"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_5"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "equal_width_5"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "n_bins": 5}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        # n_bins is scalar too: one positive int, so zero, a bool and a digit string stay out.
        return [*super().token_cases(), TokenCase("n_bins", 5, 10, invalid=(0, True, "5"))]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        return list(BinningFeatureGroup._extract_binning_params(Feature("my_result", options=options)))
