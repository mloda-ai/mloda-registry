"""Tests for OffsetFeatureGroup base class."""

from __future__ import annotations

import importlib.util
import inspect
from typing import Any

import pytest
from mloda.provider import (
    FeatureChainParserMixin,
    FeatureGroup,
    Implementation,
    OperationFeatureGroup,
)
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.base import KeyColumnInputsMixin
from mloda.community.feature_groups.data_operations.row_preserving.offset.base import OffsetFeatureGroup
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import extract_column, feature_set_for, make_feature_set
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase, TokenCase
from mloda.testing.feature_groups.data_operations.row_preserving.offset.reference import reference_offset


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(OffsetFeatureGroup, "PREFIX_PATTERN")

    def test_offset_types_exists(self) -> None:
        assert hasattr(OffsetFeatureGroup, "OFFSET_TYPES")
        assert "first_value" in OffsetFeatureGroup.OFFSET_TYPES
        assert "last_value" in OffsetFeatureGroup.OFFSET_TYPES

    def test_supports_lag(self) -> None:
        assert OffsetFeatureGroup._supports_offset_type("lag_1")
        assert OffsetFeatureGroup._supports_offset_type("lag_5")

    @pytest.mark.parametrize(
        "offset_type",
        [
            pytest.param("lead_1", id="lead"),
            pytest.param("diff_1", id="diff"),
            pytest.param("pct_change_1", id="pct_change"),
            pytest.param("first_value", id="first_value"),
            pytest.param("last_value", id="last_value"),
        ],
    )
    def test_supports_offset_type(self, offset_type: str) -> None:
        assert OffsetFeatureGroup._supports_offset_type(offset_type)

    def test_rejects_invalid(self) -> None:
        assert not OffsetFeatureGroup._supports_offset_type("lag_0")
        assert not OffsetFeatureGroup._supports_offset_type("unknown")

    def test_rejects_non_numeric_suffix(self) -> None:
        assert not OffsetFeatureGroup._supports_offset_type("lag_abc")
        assert not OffsetFeatureGroup._supports_offset_type("lead_")

    def test_rejects_empty_string(self) -> None:
        assert not OffsetFeatureGroup._supports_offset_type("")

    def test_min_max_in_features(self) -> None:
        assert OffsetFeatureGroup.MIN_IN_FEATURES == 1
        assert OffsetFeatureGroup.MAX_IN_FEATURES == 1


_IMPLEMENTATIONS = [
    pytest.param(
        "pandas_offset",
        "PandasOffset",
        "mloda_plugins.compute_framework.base_implementations.pandas.dataframe",
        "PandasDataFrame",
        id="pandas",
    ),
    pytest.param(
        "polars_lazy_offset",
        "PolarsLazyOffset",
        "mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe",
        "PolarsLazyDataFrame",
        id="polars_lazy",
    ),
    pytest.param(
        "duckdb_offset",
        "DuckdbOffset",
        "mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework",
        "DuckDBFramework",
        id="duckdb",
    ),
    pytest.param(
        "sqlite_offset",
        "SqliteOffset",
        "mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework",
        "SqliteFramework",
        id="sqlite",
    ),
    pytest.param(
        "python_dict_offset",
        "PythonDictOffset",
        "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework",
        "PythonDictFramework",
        id="python_dict",
    ),
]

_OFFSET_PACKAGE = "mloda.community.feature_groups.data_operations.row_preserving.offset"


def _implementation(module: str, name: str) -> Any:
    """The named offset Implementation; skips when its optional backend is not installed."""
    return getattr(pytest.importorskip(f"{_OFFSET_PACKAGE}.{module}"), name)


class TestOperationGroupShape:
    """OffsetFeatureGroup is one OperationFeatureGroup served by one Implementation per framework."""

    def test_group_is_a_concrete_operation_group(self) -> None:
        assert issubclass(OffsetFeatureGroup, OperationFeatureGroup)
        assert not inspect.isabstract(OffsetFeatureGroup)

    def test_group_mro_puts_the_mixins_before_the_operation_group(self) -> None:
        assert OffsetFeatureGroup.__mro__[1:5] == (
            KeyColumnInputsMixin,
            FeatureChainParserMixin,
            OperationFeatureGroup,
            FeatureGroup,
        )

    def test_group_declares_the_offset_type_subtype_key(self) -> None:
        declaration = OffsetFeatureGroup.SUBTYPES
        assert declaration is not None
        assert declaration.key == "offset_type"

    def test_group_owns_the_loop_and_no_per_framework_compute(self) -> None:
        assert callable(OffsetFeatureGroup.offset)
        assert not hasattr(OffsetFeatureGroup, "_compute_offset")

    def test_group_needs_an_implementation_on_the_feature_set(self) -> None:
        """calculate_feature dispatches to the step's implementation; none on the feature set is an error."""
        fs = make_feature_set("value_int__lag_1_offset", ["region"], "value_int")
        with pytest.raises(ValueError, match="no implementation"):
            OffsetFeatureGroup.calculate_feature(PyArrowDataOpsTestDataCreator.create(), fs)

    @pytest.mark.parametrize(("module", "name", "framework_module", "framework_name"), _IMPLEMENTATIONS)
    def test_implementation_names_its_pair(
        self, module: str, name: str, framework_module: str, framework_name: str
    ) -> None:
        impl = _implementation(module, name)
        framework = getattr(pytest.importorskip(framework_module), framework_name)
        assert issubclass(impl, Implementation)
        assert not issubclass(impl, FeatureGroup)
        assert impl.feature_group is OffsetFeatureGroup
        assert impl.compute_framework is framework
        assert impl.__module__ == f"{_OFFSET_PACKAGE}.{module}"

    @pytest.mark.parametrize(("module", "name", "framework_module", "framework_name"), _IMPLEMENTATIONS)
    def test_implementation_keeps_its_compute_offset(
        self, module: str, name: str, framework_module: str, framework_name: str
    ) -> None:
        assert callable(_implementation(module, name)._compute_offset)


class TestSubtypes:
    """The offset_type subtype dimension, including the parametric families."""

    def test_universe_is_the_named_types_and_family_names(self) -> None:
        assert OffsetFeatureGroup.subtype_universe() == {
            "first_value",
            "last_value",
            "lag",
            "lead",
            "diff",
            "pct_change",
        }

    @pytest.mark.parametrize(
        ("instance", "family"),
        [
            ("lag_2", "lag"),
            ("lead_1", "lead"),
            ("diff_4", "diff"),
            ("pct_change_3", "pct_change"),
            ("first_value", "first_value"),
        ],
    )
    def test_canonical_subtype_collapses_a_parametric_instance(self, instance: str, family: str) -> None:
        assert OffsetFeatureGroup.canonical_subtype(instance) == family

    def test_resolves_from_the_feature_name(self) -> None:
        assert OffsetFeatureGroup.resolve_subtype("x__lag_2_offset", Options()) == "lag_2"

    def test_resolves_from_the_config_option(self) -> None:
        options = Options(
            context={"offset_type": "pct_change_3", "in_features": "x", "partition_by": ["g"], "order_by": "x"}
        )
        assert OffsetFeatureGroup.resolve_subtype("my_result", options) == "pct_change_3"

    @pytest.mark.parametrize("feature_name", ["value_int__lag_x_offset", "value_int__lag_0_offset"])
    def test_malformed_parametric_instance_is_rejected_at_match(self, feature_name: str) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert OffsetFeatureGroup.match_feature_group_criteria(feature_name, options, None) is False


class TestSupportedAndDivergent:
    """No offset Implementation restricts its subtypes or declares a divergence."""

    @pytest.mark.parametrize(("module", "name", "framework_module", "framework_name"), _IMPLEMENTATIONS)
    def test_no_implementation_restricts_supported(
        self, module: str, name: str, framework_module: str, framework_name: str
    ) -> None:
        assert _implementation(module, name).supported is None

    @pytest.mark.parametrize(("module", "name", "framework_module", "framework_name"), _IMPLEMENTATIONS)
    def test_no_implementation_declares_a_divergence(
        self, module: str, name: str, framework_module: str, framework_name: str
    ) -> None:
        assert not _implementation(module, name).divergent


class TestManifest:
    """The entry-point manifest lists the group and its Implementations."""

    def test_manifest_lists_the_group_and_every_installed_implementation(self) -> None:
        from mloda.community.feature_groups.data_operations.row_preserving.offset.manifest import FEATURE_GROUPS

        assert OffsetFeatureGroup in FEATURE_GROUPS
        listed = {cls.__name__ for cls in FEATURE_GROUPS if issubclass(cls, Implementation)}
        expected = {"SqliteOffset", "PythonDictOffset"}
        for name, dependency in (
            ("PandasOffset", "pandas"),
            ("PolarsLazyOffset", "polars"),
            ("DuckdbOffset", "duckdb"),
        ):
            if importlib.util.find_spec(dependency) is not None:
                expected.add(name)
        assert listed == expected
        implementations: list[Any] = [cls for cls in FEATURE_GROUPS if cls is not OffsetFeatureGroup]
        assert all(cls.feature_group is OffsetFeatureGroup for cls in implementations)


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__lag_1_offset",
            "value_int__lead_1_offset",
            "value_int__diff_1_offset",
            "value_int__pct_change_1_offset",
            "value_int__first_value_offset",
            "value_int__last_value_offset",
        ],
    )
    def test_matches_offset_types(self, feature_name: str) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        result = OffsetFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True

    def test_no_match_wrong_suffix(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_ranked", options, None)

    def test_no_match_invalid_type(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__unknown_offset", options, None)


class TestPatternParsing:
    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            pytest.param("value_int__lag_1_offset", "lag_1", id="lag"),
            pytest.param("value_int__lead_3_offset", "lead_3", id="lead"),
            pytest.param("value_int__first_value_offset", "first_value", id="first_value"),
        ],
    )
    def test_parse_offset_type(self, feature_name: str, expected: str) -> None:
        assert OffsetFeatureGroup.get_offset_type(feature_name) == expected

    def test_parse_source_feature(self) -> None:
        from mloda.user import Feature

        feature = Feature(
            "value_int__lag_1_offset",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )
        assert OffsetFeatureGroup._extract_source_features(feature) == ["value_int"]


class TestConfigValidation:
    def test_partition_by_required(self) -> None:
        options = Options(context={"order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_order_by_required(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_valid_config(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_partition_by_as_tuple(self) -> None:
        """partition_by should accept tuples (mloda converts lists to tuples internally)."""
        options = Options(context={"partition_by": ("region",), "order_by": "value_int"})
        assert OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_partition_by_rejects_string(self) -> None:
        options = Options(context={"partition_by": "region", "order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_partition_by_rejects_non_string_items(self) -> None:
        options = Options(context={"partition_by": [123], "order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_partition_by_rejects_empty_list(self) -> None:
        options = Options(context={"partition_by": [], "order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_partition_by_rejects_empty_tuple(self) -> None:
        options = Options(context={"partition_by": (), "order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_order_by_rejects_non_string(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": 123})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "offset_type": "lag_1",
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )
        assert OffsetFeatureGroup.match_feature_group_criteria("my_result", options, None)

    def test_config_based_calculate_feature(self) -> None:
        import pyarrow as pa
        from mloda.provider import FeatureSet
        from mloda.user import Feature

        from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
        from mloda.testing.feature_groups.data_operations.row_preserving.offset.reference import reference_offset

        table = PyArrowDataOpsTestDataCreator.create()
        feature = Feature(
            "my_lag",
            options=Options(
                context={
                    "offset_type": "lag_1",
                    "in_features": "value_int",
                    "partition_by": ["region"],
                    "order_by": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = reference_offset(table, fs)
        assert isinstance(result, pa.Table)
        assert "my_lag" in result.column_names
        assert result.num_rows == 12


class TestConfigBasedOffsetTypeValidation:
    """The config path must reject exactly the offset types the feature-name path rejects.

    Matching may not defer offset-type validation to compute time: an unsupported
    offset_type has to be a non-match at discovery.
    """

    def _options(self, offset_type: Any) -> Options:
        return Options(
            context={
                "offset_type": offset_type,
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )

    @pytest.mark.parametrize(
        "offset_type",
        ["lag_1", "lead_2", "diff_1", "pct_change_3", "first_value", "last_value"],
    )
    def test_config_based_match_accepts_supported_offset_types(self, offset_type: str) -> None:
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", self._options(offset_type), None)
        assert result is True, f"Config path should accept: {offset_type}"

    @pytest.mark.parametrize("offset_type", ["banana", "lag_0", "lag_-1", "lag_abc", "lead_", ""])
    def test_config_based_match_rejects_unsupported_offset_types(self, offset_type: str) -> None:
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", self._options(offset_type), None)
        assert result is False, f"Config path should reject: {offset_type!r}"


class TestDigitLikeOffsetSuffixes:
    """A suffix that str.isdigit accepts is not automatically an int; both paths must reject it without raising."""

    def _pattern_options(self) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "value_int"})

    def _config_options(self, offset_type: str) -> Options:
        return Options(
            context={
                "offset_type": offset_type,
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )

    @pytest.mark.parametrize("feature_name", ["value_int__lag_²_offset", "value_int__pct_change_²_offset"])
    def test_name_path_rejects_superscript_digit(self, feature_name: str) -> None:
        """Superscript two is isdigit-true but int()-invalid."""
        result = OffsetFeatureGroup.match_feature_group_criteria(feature_name, self._pattern_options(), None)
        assert result is False, f"Name path should reject: {feature_name}"

    @pytest.mark.parametrize("offset_type", ["lag_²", "pct_change_²"])
    def test_config_path_rejects_superscript_digit(self, offset_type: str) -> None:
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", self._config_options(offset_type), None)
        assert result is False, f"Config path should reject: {offset_type}"

    def test_name_path_rejects_non_ascii_digit(self) -> None:
        """Arabic-Indic three is isdigit-true and int()-valid, but still not an ASCII offset suffix."""
        options = self._pattern_options()
        result = OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_٣_offset", options, None)
        assert result is False

    def test_config_path_rejects_non_ascii_digit(self) -> None:
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", self._config_options("lag_٣"), None)
        assert result is False

    @pytest.mark.parametrize("offset_type", ["lag_²", "pct_change_²", "lag_٣"])
    def test_supports_offset_type_rejects_digit_like_suffixes(self, offset_type: str) -> None:
        assert not OffsetFeatureGroup._supports_offset_type(offset_type)


class TestHostileInFeatures:
    """A hostile in_features value is a plain non-match; no exception may escape the matcher."""

    @pytest.mark.parametrize("in_features", ["", 0, 3.5, True, {"a": 1}, []])
    def test_rejects_hostile_in_features(self, in_features: Any) -> None:
        options = Options(
            context={
                "offset_type": "lag_1",
                "in_features": in_features,
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False, f"Config path should reject in_features: {in_features!r}"


class TestReturnDataTypeRule:
    """return_data_type_rule should fix the output type only for deterministic ops.

    pct_change_N always returns a fractional ratio (DOUBLE). lag / lead / diff /
    first_value / last_value preserve the input column type, so the rule must
    return None for them.
    """

    def test_pct_change_returns_double(self) -> None:
        feature = Feature(
            "value_int__pct_change_1_offset",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )
        assert OffsetFeatureGroup.return_data_type_rule(feature) == DataType.DOUBLE

    @pytest.mark.parametrize(
        "offset_type",
        ["lag_1", "lead_1", "diff_1", "first_value", "last_value"],
    )
    def test_input_dependent_ops_return_none(self, offset_type: str) -> None:
        feature = Feature(
            f"value_int__{offset_type}_offset",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )
        assert OffsetFeatureGroup.return_data_type_rule(feature) is None


class TestForwardedOffsetTypeMismatch:
    """A group-forwarded ``offset_type`` that contradicts the name-parsed type is rejected, not silently ignored."""

    def test_mismatched_forwarded_offset_type_raises(self) -> None:
        consumer_options = Options(group={"offset_type": "lag_5"})
        child_options = Options(context={"partition_by": ["region"], "order_by": "ts"})
        child_options.inherit_from(consumer_options)

        with pytest.raises(ValueError, match="offset_type"):
            OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", child_options, None)

    def test_matching_forwarded_offset_type_is_accepted(self) -> None:
        consumer_options = Options(group={"offset_type": "lag_1"})
        child_options = Options(context={"partition_by": ["region"], "order_by": "ts"})
        child_options.inherit_from(consumer_options)

        result = OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", child_options, None)
        assert result is True

    def test_mismatched_forwarded_fixed_offset_type_raises(self) -> None:
        consumer_options = Options(group={"offset_type": "first_value"})
        child_options = Options(context={"partition_by": ["region"], "order_by": "ts"})
        child_options.inherit_from(consumer_options)

        with pytest.raises(ValueError, match="offset_type"):
            OffsetFeatureGroup.match_feature_group_criteria("value_int__last_value_offset", child_options, None)


class TestOffsetMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return OffsetFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return {"first_value", "last_value", "lag_1", "lead_1"}

    @classmethod
    def config_key(cls) -> str:
        return "offset_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_offset"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "lag_1_offset"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "partition_by": ["region"], "order_by": "timestamp"}

    @classmethod
    def pattern_match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "value_int"})

    @classmethod
    def options_reject_invalid_types(cls) -> bool:
        return True

    @classmethod
    def parity_operations(cls) -> set[str]:
        # Widen, never narrow: valid_operations() covers the fixed offset types plus one lag/lead
        # instance, so add the remaining parametric families and a second lead arity on top of it.
        return cls.valid_operations() | {"lag_1", "lead_2", "diff_1", "pct_change_3"}

    @classmethod
    def malformed_operations(cls) -> set[str]:
        return {"banana", "lag_0", "lag_abc", "lead_"}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        # order_by names one column, and offset requires it on both paths.
        return [*super().token_cases(), TokenCase("order_by", "timestamp", "region")]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        return [OffsetFeatureGroup._extract_offset_type(Feature("my_result", options=options))]

    @classmethod
    def compute_values(cls, options: Options) -> list[Any] | None:
        # Offset reads order_by inline in calculate_feature rather than through an extractor,
        # so only a run through the backend shows a container reaching it unwrapped.
        result = reference_offset(PyArrowDataOpsTestDataCreator.create(), feature_set_for("my_lag", options))
        return extract_column(result, "my_lag")
