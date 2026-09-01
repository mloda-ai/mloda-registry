"""Tests for ``BinaryExampleFeatureGroup``: the enterprise example FeatureGroup that mixes in
``BinaryModelMixin`` to run the "hash" operation via an external binary (pattern 28, Binary-Backed
Features; see ``docs/guides/feature-group-patterns/28-binary-backed-features.md``).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import pytest

from mloda.provider import ApiInputDataFeature, FeatureSet, PropertySpec
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.binary_model.binary import clear_capability_cache
from mloda.community.feature_groups.binary_model.errors import BinaryUnavailableError, LicenseMissingError
from mloda.enterprise.feature_groups.binary_example import BinaryExampleFeatureGroup
from mloda.enterprise.feature_groups.binary_example import manifest as binary_example_manifest
from mloda.testing.base import FeatureGroupTestBase
from mloda.testing.binary_model.hash_reference import compute_expected_hash_column
from mloda.testing.binary_model.license_vectors import license_token_text

STUB_CMD = [sys.executable, "-m", "mloda.testing.binary_model.simulated_binary"]
VALID_LICENSE_KEY = license_token_text("valid", ["example_binary"])


class StubExample(BinaryExampleFeatureGroup):
    """Points ``BinaryExampleFeatureGroup`` at the simulated binary with a valid placeholder license."""

    BINARY_COMMAND_OVERRIDE = STUB_CMD
    LICENSE_KEY_OVERRIDE = VALID_LICENSE_KEY


@pytest.fixture(autouse=True)
def _clear_capability_cache_before_each_test() -> None:
    clear_capability_cache()


def _feature_set(*features: Feature) -> FeatureSet:
    feature_set = FeatureSet()
    for feature in features:
        feature_set.add(feature)
    return feature_set


def _hash_feature(name: str, columns: Sequence[str], parameters: dict[str, Any] | None = None) -> Feature:
    context: dict[str, Any] = {"binary_operation": "hash", "binary_input_columns": list(columns)}
    if parameters is not None:
        context["binary_parameters"] = parameters
    return Feature(name, Options(context=context))


# -------------------------------------------------------------------------------------------
# 0. FeatureGroupTestBase smoke check (mirrors the sibling enterprise example)
# -------------------------------------------------------------------------------------------


class TestBinaryExampleFeatureGroupClassSet(FeatureGroupTestBase):
    feature_group_class = BinaryExampleFeatureGroup

    def test_feature_group_class_set(self) -> None:
        assert self.feature_group_class is BinaryExampleFeatureGroup


# -------------------------------------------------------------------------------------------
# 1. Level 1: compute framework rule, PROPERTY_MAPPING, matching, input_features, manifest
# -------------------------------------------------------------------------------------------


class TestComputeFrameworkRule:
    def test_restricted_to_pyarrow_table(self) -> None:
        assert BinaryExampleFeatureGroup.compute_framework_rule() == {PyArrowTable}


class TestPropertyMapping:
    def test_property_mapping_has_exactly_the_three_option_keys(self) -> None:
        expected = {
            BinaryExampleFeatureGroup.OPERATION,
            BinaryExampleFeatureGroup.INPUT_COLUMNS,
            BinaryExampleFeatureGroup.PARAMETERS,
        }
        assert set(BinaryExampleFeatureGroup.PROPERTY_MAPPING) == expected

    def test_every_property_mapping_value_is_a_property_spec(self) -> None:
        for value in BinaryExampleFeatureGroup.PROPERTY_MAPPING.values():
            assert isinstance(value, PropertySpec)

    def test_binary_plugin_id_is_example_binary(self) -> None:
        assert BinaryExampleFeatureGroup.BINARY_PLUGIN_ID == "example_binary"


class TestMatchFeatureGroupCriteria:
    def test_matches_hash_with_a_single_column(self) -> None:
        options = Options(context={"binary_operation": "hash", "binary_input_columns": ["col_a"]})
        assert BinaryExampleFeatureGroup.match_feature_group_criteria("hashed", options)

    def test_matches_hash_with_multiple_columns(self) -> None:
        options = Options(context={"binary_operation": "hash", "binary_input_columns": ["col_a", "col_b"]})
        assert BinaryExampleFeatureGroup.match_feature_group_criteria("hashed", options)

    def test_matches_with_input_columns_as_a_tuple(self) -> None:
        options = Options(context={"binary_operation": "hash", "binary_input_columns": ("col_a", "col_b")})
        assert BinaryExampleFeatureGroup.match_feature_group_criteria("hashed", options)

    def test_matches_with_a_valid_parameters_dict(self) -> None:
        options = Options(
            context={
                "binary_operation": "hash",
                "binary_input_columns": ["col_a"],
                "binary_parameters": {"key": "k"},
            }
        )
        assert BinaryExampleFeatureGroup.match_feature_group_criteria("hashed", options)

    def test_rejects_missing_operation_key(self) -> None:
        options = Options(context={"binary_input_columns": ["col_a"]})
        assert not BinaryExampleFeatureGroup.match_feature_group_criteria("hashed", options)

    def test_rejects_unknown_operation(self) -> None:
        options = Options(context={"binary_operation": "unhash", "binary_input_columns": ["col_a"]})
        assert not BinaryExampleFeatureGroup.match_feature_group_criteria("hashed", options)

    def test_rejects_missing_input_columns_key(self) -> None:
        options = Options(context={"binary_operation": "hash"})
        assert not BinaryExampleFeatureGroup.match_feature_group_criteria("hashed", options)

    def test_rejects_empty_input_columns(self) -> None:
        options = Options(context={"binary_operation": "hash", "binary_input_columns": []})
        assert not BinaryExampleFeatureGroup.match_feature_group_criteria("hashed", options)

    def test_rejects_non_string_entries(self) -> None:
        options = Options(context={"binary_operation": "hash", "binary_input_columns": [1, 2]})
        assert not BinaryExampleFeatureGroup.match_feature_group_criteria("hashed", options)

    def test_rejects_bare_class_name_request_with_no_options(self) -> None:
        feature = Feature("BinaryExampleFeatureGroup")
        assert not BinaryExampleFeatureGroup.match_feature_group_criteria(feature.name, feature.options)

    def test_rejects_non_dict_parameters(self) -> None:
        options = Options(
            context={
                "binary_operation": "hash",
                "binary_input_columns": ["col_a"],
                "binary_parameters": "not_a_dict",
            }
        )
        assert not BinaryExampleFeatureGroup.match_feature_group_criteria("hashed", options)

    def test_rejects_parameters_with_non_string_keys(self) -> None:
        options = Options(
            context={
                "binary_operation": "hash",
                "binary_input_columns": ["col_a"],
                "binary_parameters": {1: "x"},
            }
        )
        assert not BinaryExampleFeatureGroup.match_feature_group_criteria("hashed", options)


class TestInputFeatures:
    def test_returns_features_named_after_the_configured_columns(self) -> None:
        options = Options(context={"binary_input_columns": ["col_a", "col_b"]})
        instance = BinaryExampleFeatureGroup()
        result = instance.input_features(options, FeatureName("hashed"))
        assert result is not None
        assert {feature.name for feature in result} == {"col_a", "col_b"}

    def test_returns_one_feature_for_a_single_column(self) -> None:
        options = Options(context={"binary_input_columns": ["col_a"]})
        instance = BinaryExampleFeatureGroup()
        result = instance.input_features(options, FeatureName("hashed"))
        assert result is not None
        assert {feature.name for feature in result} == {"col_a"}


class TestManifest:
    def test_manifest_lists_exactly_the_feature_group(self) -> None:
        assert binary_example_manifest.FEATURE_GROUPS == [BinaryExampleFeatureGroup]

    def test_wheel_is_not_installed_precondition(self) -> None:
        assert importlib.util.find_spec("example_binary") is None

    def test_importing_the_manifest_never_imports_the_binary_wheel(self) -> None:
        importlib.import_module("mloda.enterprise.feature_groups.binary_example.manifest")
        assert "example_binary" not in sys.modules


# -------------------------------------------------------------------------------------------
# 2. Level 2: calculate_feature against the simulated binary, and its up-front rejections
# -------------------------------------------------------------------------------------------


class TestCalculateFeature:
    def test_hash_matches_the_reference_algorithm(self) -> None:
        rows: dict[str, list[Any]] = {"col_a": ["alpha", "beta", "gamma"], "col_b": ["x", "y", "z"]}
        table = pa.table({**rows, "other": [1, 2, 3]})
        result = StubExample.calculate_feature(table, _feature_set(_hash_feature("hashed", ["col_a", "col_b"])))

        expected = compute_expected_hash_column(rows, ["col_a", "col_b"], None)
        assert result.column("hashed").to_pylist() == expected
        assert result.num_rows == table.num_rows
        assert set(result.schema.names) == {"col_a", "col_b", "other", "hashed"}

    def test_column_order_is_honoured(self) -> None:
        rows: dict[str, list[Any]] = {"col_a": ["alpha", "beta"], "col_b": ["x", "y"]}
        table = pa.table(rows)

        forward = StubExample.calculate_feature(table, _feature_set(_hash_feature("hashed", ["col_a", "col_b"])))
        reversed_ = StubExample.calculate_feature(table, _feature_set(_hash_feature("hashed", ["col_b", "col_a"])))

        assert forward.column("hashed").to_pylist() == compute_expected_hash_column(rows, ["col_a", "col_b"], None)
        assert reversed_.column("hashed").to_pylist() == compute_expected_hash_column(rows, ["col_b", "col_a"], None)
        assert forward.column("hashed").to_pylist() != reversed_.column("hashed").to_pylist()

    def test_binary_parameters_change_the_result(self) -> None:
        rows: dict[str, list[Any]] = {"col_a": ["alpha", "beta"]}
        table = pa.table(rows)
        feature = _hash_feature("hashed", ["col_a"], parameters={"key": "k"})
        result = StubExample.calculate_feature(table, _feature_set(feature))
        expected = compute_expected_hash_column(rows, ["col_a"], "k")
        assert result.column("hashed").to_pylist() == expected

    def test_two_requested_features_both_get_appended(self) -> None:
        rows: dict[str, list[Any]] = {"col_a": ["alpha", "beta"], "col_b": ["x", "y"]}
        table = pa.table(rows)
        feature_a = _hash_feature("hash_a", ["col_a"])
        feature_b = _hash_feature("hash_b", ["col_b"])
        result = StubExample.calculate_feature(table, _feature_set(feature_a, feature_b))

        assert result.column("hash_a").to_pylist() == compute_expected_hash_column(rows, ["col_a"], None)
        assert result.column("hash_b").to_pylist() == compute_expected_hash_column(rows, ["col_b"], None)


class TestCalculateFeatureRejections:
    def test_production_class_without_override_raises_binary_unavailable(self) -> None:
        table = pa.table({"col_a": ["alpha"]})
        feature = _hash_feature("hashed", ["col_a"])
        with pytest.raises(BinaryUnavailableError, match="example_binary"):
            BinaryExampleFeatureGroup.calculate_feature(table, _feature_set(feature))

    def test_no_license_raises_license_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLODA_LICENSE_FILE", raising=False)
        monkeypatch.delenv("MLODA_LICENSE_KEY", raising=False)

        class _NoLicenseExample(BinaryExampleFeatureGroup):
            BINARY_COMMAND_OVERRIDE = STUB_CMD

        table = pa.table({"col_a": ["alpha"]})
        feature = _hash_feature("hashed", ["col_a"])
        with pytest.raises(LicenseMissingError):
            _NoLicenseExample.calculate_feature(table, _feature_set(feature))


# -------------------------------------------------------------------------------------------
# 3. Level 3: mloda.run_all end-to-end
# -------------------------------------------------------------------------------------------


class TestIntegration:
    def test_single_hashed_feature_end_to_end(self) -> None:
        rows: dict[str, list[Any]] = {"col_a": ["alpha", "beta", "gamma"], "col_b": ["x", "y", "z"]}
        feature = _hash_feature("hashed", ["col_a", "col_b"])
        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            api_data={"BinaryExampleData": rows},
            plugin_collector=PluginCollector.enabled_feature_groups({ApiInputDataFeature, StubExample}),
        )
        expected = compute_expected_hash_column(rows, ["col_a", "col_b"], None)
        found = False
        for table in results:
            if isinstance(table, pa.Table) and "hashed" in table.column_names:
                assert table.column("hashed").to_pylist() == expected
                found = True
        assert found, "hashed column not found in any result table"

    def test_two_hashed_features_end_to_end(self) -> None:
        rows: dict[str, list[Any]] = {"col_a": ["alpha", "beta"], "col_b": ["x", "y"]}
        feature_a = _hash_feature("hash_a", ["col_a"])
        feature_b = _hash_feature("hash_b", ["col_b"])
        results = mloda.run_all(
            [feature_a, feature_b],
            compute_frameworks={PyArrowTable},
            api_data={"BinaryExampleData": rows},
            plugin_collector=PluginCollector.enabled_feature_groups({ApiInputDataFeature, StubExample}),
            column_ordering="request_order",
        )
        found_a = False
        found_b = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "hash_a" in table.column_names:
                assert table.column("hash_a").to_pylist() == compute_expected_hash_column(rows, ["col_a"], None)
                found_a = True
            if "hash_b" in table.column_names:
                assert table.column("hash_b").to_pylist() == compute_expected_hash_column(rows, ["col_b"], None)
                found_b = True
        assert found_a, "hash_a column not found in any result table"
        assert found_b, "hash_b column not found in any result table"

    def test_production_class_in_a_run_raises(self) -> None:
        feature = _hash_feature("hashed", ["col_a"])
        with pytest.raises(Exception, match="example_binary"):
            mloda.run_all(
                [feature],
                compute_frameworks={PyArrowTable},
                api_data={"BinaryExampleData": {"col_a": ["alpha"]}},
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {ApiInputDataFeature, BinaryExampleFeatureGroup}
                ),
            )
