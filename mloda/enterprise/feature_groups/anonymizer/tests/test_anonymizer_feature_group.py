"""Tests for ``AnonymizerFeatureGroup``: the enterprise FeatureGroup that mixes in ``BinaryModelMixin`` to
pseudonymize a utf8 column with keyed HMAC-SHA256 via an external binary (pattern 29, Binary-Backed
Features; see ``docs/guides/feature-group-patterns/29-binary-backed-features.md``).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import pyarrow as pa
import pytest
from mloda.provider import ApiInputDataFeature, FeatureSet, PropertySpec
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.binary_model import mixin
from mloda.community.feature_groups.binary_model.binary import clear_capability_cache
from mloda.community.feature_groups.binary_model.errors import (
    BinaryUnavailableError,
    BinaryUsageError,
    LicenseInvalidError,
    LicenseMissingError,
)
from mloda.enterprise.feature_groups.anonymizer import manifest as anonymizer_manifest
from mloda.enterprise.feature_groups.anonymizer.anonymizer_feature_group import AnonymizerFeatureGroup
from mloda.testing.base import FeatureGroupTestBase
from mloda.testing.binary_model.hmac_sha256_reference import (
    KNOWN_ANSWER_DIGEST,
    KNOWN_ANSWER_KEY,
    KNOWN_ANSWER_VALUE,
    compute_expected_hmac_sha256_column,
)
from mloda.testing.binary_model.license_vectors import valid_license_token
from mloda.testing.tests._hmac_sha256_fake_binary import PLUGIN_ID as HMAC_FAKE_PLUGIN_ID

STUB_CMD = [sys.executable, "-m", "mloda.testing.tests._hmac_sha256_fake_binary"]
VALID_LICENSE_KEY = valid_license_token([HMAC_FAKE_PLUGIN_ID])
KEY_ENV = "ANONYMIZER_TEST_PII_KEY"
UNSET_KEY_ENV = "ANONYMIZER_TEST_UNSET_PII_KEY"


class StubAnonymizer(AnonymizerFeatureGroup):
    """Points ``AnonymizerFeatureGroup`` at the hmac fake binary with a valid placeholder license."""

    BINARY_PLUGIN_ID = HMAC_FAKE_PLUGIN_ID
    BINARY_COMMAND_OVERRIDE = STUB_CMD
    LICENSE_KEY_OVERRIDE = VALID_LICENSE_KEY


@pytest.fixture(autouse=True)
def _clear_capability_cache_before_each_test() -> None:
    clear_capability_cache()


@pytest.fixture
def key_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(KEY_ENV, KNOWN_ANSWER_KEY)
    return KEY_ENV


def _feature_set(*features: Feature) -> FeatureSet:
    feature_set = FeatureSet()
    for feature in features:
        feature_set.add(feature)
    return feature_set


def _config_feature(name: str, source: str = "email", key_env_value: Any = KEY_ENV) -> Feature:
    context: dict[str, Any] = {
        "pseudonymization_algorithm": "hmac_sha256",
        "pii_key_env": key_env_value,
        "in_features": source,
    }
    return Feature(name, Options(context=context))


def _string_feature(source: str = "email") -> Feature:
    return Feature(f"{source}__hmac_sha256_pseudonymized", Options(context={"pii_key_env": KEY_ENV}))


def _matches(name: str, options: Options) -> bool:
    return bool(AnonymizerFeatureGroup.match_feature_group_criteria(name, options))


def _is_rejected(name: str, options: Options) -> bool:
    """A strict spec may reject by returning False or by raising ``ValueError`` (``PropertyValueRejection``)."""
    try:
        return not _matches(name, options)
    except ValueError:
        return True


def _result_column(results: list[Any], name: str) -> list[Any]:
    for table in results:
        if isinstance(table, pa.Table) and name in table.column_names:
            return list(table.column(name).to_pylist())
    raise AssertionError(f"{name} column not found in any result table")


# -------------------------------------------------------------------------------------------
# 0. FeatureGroupTestBase smoke check (mirrors the sibling enterprise example)
# -------------------------------------------------------------------------------------------


class TestAnonymizerFeatureGroupClassSet(FeatureGroupTestBase):
    feature_group_class = AnonymizerFeatureGroup

    def test_feature_group_class_set(self) -> None:
        assert self.feature_group_class is AnonymizerFeatureGroup


# -------------------------------------------------------------------------------------------
# 1. Level 1: compute framework rule, PROPERTY_MAPPING, matching, manifest
# -------------------------------------------------------------------------------------------


class TestComputeFrameworkRule:
    def test_restricted_to_pyarrow_table(self) -> None:
        assert AnonymizerFeatureGroup.compute_framework_rule() == {PyArrowTable}


class TestPropertyMapping:
    def test_property_mapping_has_exactly_the_three_option_keys(self) -> None:
        expected = {
            AnonymizerFeatureGroup.ALGORITHM,
            AnonymizerFeatureGroup.KEY_ENV,
            "in_features",
        }
        assert set(AnonymizerFeatureGroup.PROPERTY_MAPPING) == expected

    def test_every_property_mapping_value_is_a_property_spec(self) -> None:
        for value in AnonymizerFeatureGroup.PROPERTY_MAPPING.values():
            assert isinstance(value, PropertySpec)

    def test_key_env_spec_is_not_strict_and_has_no_match_guard(self) -> None:
        """A strict spec or a match_guard would let mloda's validators echo a mistaken value, which
        may be the key itself."""
        spec = AnonymizerFeatureGroup.PROPERTY_MAPPING[AnonymizerFeatureGroup.KEY_ENV]
        assert isinstance(spec, PropertySpec)
        assert not spec.strict_validation
        assert spec.match_guard is None

    def test_binary_plugin_id_is_anonymizer_binary(self) -> None:
        assert AnonymizerFeatureGroup.BINARY_PLUGIN_ID == "anonymizer_binary"

    def test_binary_wheel_distribution_is_mloda_anonymizer_binary(self) -> None:
        assert AnonymizerFeatureGroup.BINARY_WHEEL_DISTRIBUTION == "mloda-anonymizer-binary"

    def test_binary_install_extra_is_the_enterprise_anonymizer_extra(self) -> None:
        assert AnonymizerFeatureGroup.BINARY_INSTALL_EXTRA == "mloda-enterprise[anonymizer]"


class TestMatchFeatureGroupCriteria:
    def test_config_path_matches(self) -> None:
        options = Options(
            context={"pseudonymization_algorithm": "hmac_sha256", "pii_key_env": KEY_ENV, "in_features": "email"}
        )
        assert _matches("pseudonymized_email", options)

    def test_string_path_matches_with_key_env(self) -> None:
        assert _matches("email__hmac_sha256_pseudonymized", Options(context={"pii_key_env": KEY_ENV}))

    def test_rejects_another_algorithm(self) -> None:
        options = Options(context={"pseudonymization_algorithm": "md5", "pii_key_env": KEY_ENV, "in_features": "email"})
        assert _is_rejected("pseudonymized_email", options)

    def test_rejects_missing_key_env_on_the_config_path(self) -> None:
        options = Options(context={"pseudonymization_algorithm": "hmac_sha256", "in_features": "email"})
        assert _is_rejected("pseudonymized_email", options)

    def test_rejects_missing_key_env_on_the_string_path(self) -> None:
        assert _is_rejected("email__hmac_sha256_pseudonymized", Options())

    def test_rejects_two_in_features(self) -> None:
        options = Options(
            context={
                "pseudonymization_algorithm": "hmac_sha256",
                "pii_key_env": KEY_ENV,
                "in_features": ["email", "phone"],
            }
        )
        assert _is_rejected("pseudonymized_email", options)

    def test_rejects_bare_class_name_request_with_no_options(self) -> None:
        feature = Feature("AnonymizerFeatureGroup")
        assert _is_rejected(feature.name, feature.options)


class TestManifest:
    def test_manifest_lists_exactly_the_feature_group(self) -> None:
        assert anonymizer_manifest.FEATURE_GROUPS == [AnonymizerFeatureGroup]


# -------------------------------------------------------------------------------------------
# 2. Level 2: calculate_feature against the hmac fake binary, and its up-front rejections
# -------------------------------------------------------------------------------------------


class TestCalculateFeature:
    def test_known_answer(self, key_env: str) -> None:
        table = pa.table({"email": [KNOWN_ANSWER_VALUE]})
        result = StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))
        assert result.column("pseudonymized_email").to_pylist() == [KNOWN_ANSWER_DIGEST]

    def test_matches_the_reference_algorithm_and_keeps_nulls(self, key_env: str) -> None:
        rows: list[str | None] = ["alpha", None, "gamma"]
        table = pa.table({"email": rows, "other": [1, 2, 3]})
        result = StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))

        expected = compute_expected_hmac_sha256_column(rows, KNOWN_ANSWER_KEY)
        assert result.column("pseudonymized_email").to_pylist() == expected
        assert expected[1] is None
        assert result.num_rows == table.num_rows
        assert set(result.schema.names) == {"email", "other", "pseudonymized_email"}

    def test_string_path_reads_the_source_from_the_name(self, key_env: str) -> None:
        rows: list[str | None] = ["alpha", "beta"]
        table = pa.table({"email": rows})
        result = StubAnonymizer.calculate_feature(table, _feature_set(_string_feature()))
        expected = compute_expected_hmac_sha256_column(rows, KNOWN_ANSWER_KEY)
        assert result.column("email__hmac_sha256_pseudonymized").to_pylist() == expected

    def test_key_with_surrounding_whitespace_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(KEY_ENV, KNOWN_ANSWER_KEY + "\n")
        table = pa.table({"email": [KNOWN_ANSWER_VALUE]})
        result = StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))
        assert result.column("pseudonymized_email").to_pylist() == [KNOWN_ANSWER_DIGEST]

    def test_two_features_in_one_feature_set_each_use_their_own_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        key_a, key_b = KNOWN_ANSWER_KEY, "22" * 32
        monkeypatch.setenv("ANONYMIZER_TEST_KEY_A", key_a)
        monkeypatch.setenv("ANONYMIZER_TEST_KEY_B", key_b)
        rows_a: list[str | None] = ["alpha", None]
        rows_b: list[str | None] = ["beta", "gamma"]
        table = pa.table({"col_a": rows_a, "col_b": rows_b})
        feature_a = _config_feature("pseudo_a", source="col_a", key_env_value="ANONYMIZER_TEST_KEY_A")
        feature_b = _config_feature("pseudo_b", source="col_b", key_env_value="ANONYMIZER_TEST_KEY_B")
        result = StubAnonymizer.calculate_feature(table, _feature_set(feature_a, feature_b))
        assert result.column("pseudo_a").to_pylist() == compute_expected_hmac_sha256_column(rows_a, key_a)
        assert result.column("pseudo_b").to_pylist() == compute_expected_hmac_sha256_column(rows_b, key_b)

    def test_key_never_appears_in_a_log_record(self, key_env: str, caplog: pytest.LogCaptureFixture) -> None:
        table = pa.table({"email": [KNOWN_ANSWER_VALUE]})
        with caplog.at_level(logging.DEBUG):
            StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))
        assert caplog.records, "fixture assumption: DEBUG logging captured at least one record"
        assert all(KNOWN_ANSWER_KEY not in record.getMessage() for record in caplog.records)


class TestCalculateFeatureKeyRejections:
    """The error never echoes the variable name, nor a key a user mistakenly put in ``pii_key_env``."""

    def test_unset_env_var_raises_without_naming_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(UNSET_KEY_ENV, raising=False)
        table = pa.table({"email": ["alpha"]})
        feature = _config_feature("pseudonymized_email", key_env_value=UNSET_KEY_ENV)
        with pytest.raises(BinaryUsageError) as excinfo:
            StubAnonymizer.calculate_feature(table, _feature_set(feature))
        assert UNSET_KEY_ENV not in str(excinfo.value)

    def test_empty_env_var_raises_without_naming_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(KEY_ENV, "")
        table = pa.table({"email": ["alpha"]})
        with pytest.raises(BinaryUsageError) as excinfo:
            StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))
        assert KEY_ENV not in str(excinfo.value)

    def test_non_str_key_env_raises(self, key_env: str) -> None:
        table = pa.table({"email": ["alpha"]})
        feature = _config_feature("pseudonymized_email", key_env_value=12345)
        with pytest.raises(BinaryUsageError) as excinfo:
            StubAnonymizer.calculate_feature(table, _feature_set(feature))
        assert "12345" not in str(excinfo.value)

    def test_a_key_planted_in_key_env_is_never_echoed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        planted_key = "ab" * 32
        monkeypatch.delenv(planted_key, raising=False)
        table = pa.table({"email": ["alpha"]})
        feature = _config_feature("pseudonymized_email", key_env_value=planted_key)
        with pytest.raises(BinaryUsageError) as excinfo:
            StubAnonymizer.calculate_feature(table, _feature_set(feature))
        assert planted_key not in str(excinfo.value)

    @pytest.mark.parametrize("malformed_key", ["11" * 31 + "1", "11" * 31 + "zz", "11" * 33])
    def test_malformed_key_raises_before_the_binary_runs_without_echoing_it(
        self, monkeypatch: pytest.MonkeyPatch, malformed_key: str
    ) -> None:
        runs: list[int] = []

        def recording_run_binary(*args: Any, **kwargs: Any) -> bytes:
            runs.append(1)
            raise AssertionError("the binary must not run for a malformed key")

        monkeypatch.setattr(mixin, "run_binary", recording_run_binary)
        monkeypatch.setenv(KEY_ENV, malformed_key + "\n")
        table = pa.table({"email": ["alpha"]})
        with pytest.raises(BinaryUsageError) as excinfo:
            StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))
        assert not runs
        assert malformed_key not in str(excinfo.value)
        assert KEY_ENV not in str(excinfo.value)


class TestCalculateFeatureRejections:
    def test_production_class_without_override_raises_binary_unavailable(self, key_env: str) -> None:
        table = pa.table({"email": ["alpha"]})
        feature = _config_feature("pseudonymized_email")
        with pytest.raises(BinaryUnavailableError, match="anonymizer_binary"):
            AnonymizerFeatureGroup.calculate_feature(table, _feature_set(feature))

    def test_no_license_raises_license_missing(self, key_env: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLODA_LICENSE_FILE", raising=False)
        monkeypatch.delenv("MLODA_LICENSE_KEY", raising=False)

        class _NoLicenseAnonymizer(AnonymizerFeatureGroup):
            BINARY_PLUGIN_ID = HMAC_FAKE_PLUGIN_ID
            BINARY_COMMAND_OVERRIDE = STUB_CMD

        table = pa.table({"email": ["alpha"]})
        with pytest.raises(LicenseMissingError):
            _NoLicenseAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))

    def test_license_for_another_plugin_id_raises_license_invalid(self, key_env: str) -> None:
        class _WrongLicenseAnonymizer(StubAnonymizer):
            LICENSE_KEY_OVERRIDE = valid_license_token(["example_binary"])

        table = pa.table({"email": ["alpha"]})
        with pytest.raises(LicenseInvalidError):
            _WrongLicenseAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))


# -------------------------------------------------------------------------------------------
# 3. Level 3: mloda.run_all end-to-end
# -------------------------------------------------------------------------------------------


class TestIntegration:
    def test_config_form_end_to_end(self, key_env: str) -> None:
        rows: dict[str, list[Any]] = {"email": ["alpha", None, "gamma"]}
        results = mloda.run_all(
            [_config_feature("pseudonymized_email")],
            compute_frameworks=[PyArrowTable],
            api_data={"AnonymizerData": rows},
            plugin_collector=PluginCollector.enabled_feature_groups({ApiInputDataFeature, StubAnonymizer}),
        )
        expected = compute_expected_hmac_sha256_column(rows["email"], KNOWN_ANSWER_KEY)
        assert _result_column(results, "pseudonymized_email") == expected

    def test_string_form_end_to_end(self, key_env: str) -> None:
        rows: dict[str, list[Any]] = {"email": ["alpha", "beta"]}
        results = mloda.run_all(
            [_string_feature()],
            compute_frameworks=[PyArrowTable],
            api_data={"AnonymizerData": rows},
            plugin_collector=PluginCollector.enabled_feature_groups({ApiInputDataFeature, StubAnonymizer}),
        )
        expected = compute_expected_hmac_sha256_column(rows["email"], KNOWN_ANSWER_KEY)
        assert _result_column(results, "email__hmac_sha256_pseudonymized") == expected
