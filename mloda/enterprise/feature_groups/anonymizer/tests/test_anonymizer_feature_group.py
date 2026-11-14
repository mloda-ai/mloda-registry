"""Tests for ``AnonymizerFeatureGroup``: the enterprise FeatureGroup that mixes in ``BinaryModelMixin`` to
pseudonymize a utf8 column with keyed HMAC-SHA256 via an external binary (pattern 29, Binary-Backed
Features; see ``docs/guides/feature-group-patterns/29-binary-backed-features.md``).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pytest
from mloda.provider import (
    ApiDataFG,
    FeatureSet,
    PropertySpec,
    credential_handle_spec,
    credentials_scope,
)
from mloda.user import Credential, DataAccessCollection, Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.binary_model import mixin
from mloda.community.feature_groups.binary_model.binary import clear_capability_cache
from mloda.community.feature_groups.binary_model.errors import (
    BinaryNotInstalledError,
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

STUB_CMD = [
    sys.executable,
    "-m",
    "mloda.testing.binary_model.hmac_fake_binary",
    "--plugin-id",
    AnonymizerFeatureGroup.BINARY_PLUGIN_ID,
]
VALID_LICENSE_KEY = valid_license_token([AnonymizerFeatureGroup.BINARY_PLUGIN_ID])
LIST_UTF8_ROWS: list[list[str | None] | None] = [["alpha", "beta"], None, [], ["gamma", None]]
KEY_HANDLE = "anonymizer-test-pii-key"
LEFTOVER_ENV_VALUE = "LEFTOVER_PII_KEY_ENV_VALUE"


class StubAnonymizer(AnonymizerFeatureGroup):
    """Points ``AnonymizerFeatureGroup`` at the shipped hmac fake binary with a valid license."""

    BINARY_COMMAND_OVERRIDE = STUB_CMD
    LICENSE_KEY_OVERRIDE = VALID_LICENSE_KEY


@pytest.fixture(autouse=True)
def _clear_capability_cache_before_each_test() -> None:
    clear_capability_cache()


@pytest.fixture
def key_handle() -> Iterator[str]:
    with credentials_scope({KEY_HANDLE: {"key": KNOWN_ANSWER_KEY}}):
        yield KEY_HANDLE


def _key_collection() -> DataAccessCollection:
    return DataAccessCollection(credentials={KEY_HANDLE: Credential(key=KNOWN_ANSWER_KEY)})


def _feature_set(*features: Feature) -> FeatureSet:
    feature_set = FeatureSet()
    for feature in features:
        feature_set.add(feature)
    return feature_set


def _config_feature(name: str, source: str = "email", key_handle_value: Any = KEY_HANDLE) -> Feature:
    context: dict[str, Any] = {"pseudonymization_algorithm": "hmac_sha256", "in_features": source}
    return Feature(name, Options(group={"pii_key_handle": key_handle_value}, context=context))


def _string_feature(source: str = "email") -> Feature:
    return Feature(f"{source}__hmac_sha256_pseudonymized", Options(group={"pii_key_handle": KEY_HANDLE}))


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


def _assert_list_utf8_pseudonymized(out: list[Any]) -> None:
    assert [None if r is None else len(r) for r in out] == [None if r is None else len(r) for r in LIST_UTF8_ROWS]
    flat_in = [v for row in LIST_UTF8_ROWS if row for v in row]
    flat_out = [v for row in out if row for v in row]
    assert flat_out == compute_expected_hmac_sha256_column(flat_in, KNOWN_ANSWER_KEY)


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
            AnonymizerFeatureGroup.KEY_HANDLE,
            "in_features",
        }
        assert set(AnonymizerFeatureGroup.PROPERTY_MAPPING) == expected

    def test_every_property_mapping_value_is_a_property_spec(self) -> None:
        for value in AnonymizerFeatureGroup.PROPERTY_MAPPING.values():
            assert isinstance(value, PropertySpec)

    def test_key_handle_spec_is_a_credential_handle_spec_requiring_the_key_field(self) -> None:
        spec = AnonymizerFeatureGroup.PROPERTY_MAPPING[AnonymizerFeatureGroup.KEY_HANDLE]
        assert isinstance(spec, type(credential_handle_spec("x")))
        assert spec.required_fields == ("key",)

    def test_binary_plugin_id_is_anonymizer_binary(self) -> None:
        assert AnonymizerFeatureGroup.BINARY_PLUGIN_ID == "anonymizer_binary"

    def test_binary_wheel_distribution_is_mloda_anonymizer_binary(self) -> None:
        assert AnonymizerFeatureGroup.BINARY_WHEEL_DISTRIBUTION == "mloda-anonymizer-binary"

    def test_binary_install_extra_is_the_enterprise_anonymizer_extra(self) -> None:
        assert AnonymizerFeatureGroup.BINARY_INSTALL_EXTRA == "mloda-enterprise[anonymizer]"


class TestMatchFeatureGroupCriteria:
    def test_config_path_matches(self) -> None:
        options = Options(
            group={"pii_key_handle": KEY_HANDLE},
            context={"pseudonymization_algorithm": "hmac_sha256", "in_features": "email"},
        )
        assert _matches("pseudonymized_email", options)

    def test_string_path_matches_with_key_handle(self) -> None:
        assert _matches("email__hmac_sha256_pseudonymized", Options(group={"pii_key_handle": KEY_HANDLE}))

    def test_rejects_another_algorithm(self) -> None:
        options = Options(
            group={"pii_key_handle": KEY_HANDLE},
            context={"pseudonymization_algorithm": "md5", "in_features": "email"},
        )
        assert _is_rejected("pseudonymized_email", options)

    def test_rejects_missing_key_handle_on_the_config_path(self) -> None:
        options = Options(context={"pseudonymization_algorithm": "hmac_sha256", "in_features": "email"})
        assert _is_rejected("pseudonymized_email", options)

    def test_rejects_missing_key_handle_on_the_string_path(self) -> None:
        assert _is_rejected("email__hmac_sha256_pseudonymized", Options())

    def test_rejects_two_in_features(self) -> None:
        options = Options(
            group={"pii_key_handle": KEY_HANDLE},
            context={"pseudonymization_algorithm": "hmac_sha256", "in_features": ["email", "phone"]},
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
    def test_known_answer(self, key_handle: str) -> None:
        table = pa.table({"email": [KNOWN_ANSWER_VALUE]})
        result = StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))
        assert result.column("pseudonymized_email").to_pylist() == [KNOWN_ANSWER_DIGEST]

    def test_matches_the_reference_algorithm_and_keeps_nulls(self, key_handle: str) -> None:
        rows: list[str | None] = ["alpha", None, "gamma"]
        table = pa.table({"email": rows, "other": [1, 2, 3]})
        result = StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))

        expected = compute_expected_hmac_sha256_column(rows, KNOWN_ANSWER_KEY)
        assert result.column("pseudonymized_email").to_pylist() == expected
        assert expected[1] is None
        assert result.num_rows == table.num_rows
        assert set(result.schema.names) == {"email", "other", "pseudonymized_email"}

    def test_string_path_reads_the_source_from_the_name(self, key_handle: str) -> None:
        rows: list[str | None] = ["alpha", "beta"]
        table = pa.table({"email": rows})
        result = StubAnonymizer.calculate_feature(table, _feature_set(_string_feature()))
        expected = compute_expected_hmac_sha256_column(rows, KNOWN_ANSWER_KEY)
        assert result.column("email__hmac_sha256_pseudonymized").to_pylist() == expected

    def test_list_utf8_source_is_pseudonymized_element_wise(self, key_handle: str) -> None:
        table = pa.table({"email": pa.array(LIST_UTF8_ROWS, type=pa.list_(pa.string()))})
        result = StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))
        out = result.column("pseudonymized_email")
        assert pa.types.is_list(out.type)
        _assert_list_utf8_pseudonymized(out.to_pylist())

    def test_key_with_surrounding_whitespace_is_stripped(self) -> None:
        table = pa.table({"email": [KNOWN_ANSWER_VALUE]})
        with credentials_scope({KEY_HANDLE: {"key": KNOWN_ANSWER_KEY + "\n"}}):
            result = StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))
        assert result.column("pseudonymized_email").to_pylist() == [KNOWN_ANSWER_DIGEST]

    def test_two_features_in_one_feature_set_each_use_their_own_key(self) -> None:
        key_a, key_b = KNOWN_ANSWER_KEY, "22" * 32
        rows_a: list[str | None] = ["alpha", None]
        rows_b: list[str | None] = ["beta", "gamma"]
        table = pa.table({"col_a": rows_a, "col_b": rows_b})
        feature_a = _config_feature("pseudo_a", source="col_a", key_handle_value="anonymizer-test-key-a")
        feature_b = _config_feature("pseudo_b", source="col_b", key_handle_value="anonymizer-test-key-b")
        with credentials_scope({"anonymizer-test-key-a": {"key": key_a}, "anonymizer-test-key-b": {"key": key_b}}):
            result = StubAnonymizer.calculate_feature(table, _feature_set(feature_a, feature_b))
        assert result.column("pseudo_a").to_pylist() == compute_expected_hmac_sha256_column(rows_a, key_a)
        assert result.column("pseudo_b").to_pylist() == compute_expected_hmac_sha256_column(rows_b, key_b)

    def test_key_never_appears_in_a_log_record(self, key_handle: str, caplog: pytest.LogCaptureFixture) -> None:
        table = pa.table({"email": [KNOWN_ANSWER_VALUE]})
        with caplog.at_level(logging.DEBUG):
            StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))
        assert caplog.records, "fixture assumption: DEBUG logging captured at least one record"
        assert all(KNOWN_ANSWER_KEY not in record.getMessage() for record in caplog.records)


class TestCalculateFeatureKeyRejections:
    """The error never echoes a rejected key value."""

    def _calculate(self, credential: dict[str, Any]) -> pa.Table:
        table = pa.table({"email": ["alpha"]})
        with credentials_scope({KEY_HANDLE: credential}):
            return StubAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))

    def test_non_str_key_raises_without_echoing_it(self) -> None:
        with pytest.raises(BinaryUsageError) as excinfo:
            self._calculate({"key": 12345})
        assert "12345" not in str(excinfo.value)

    @pytest.mark.parametrize(
        ("key", "secrets"),
        [
            ("", ()),
            ("  \n", ()),
            ("11" * 31 + "1", ("11" * 31 + "1",)),
            ("11" * 31 + "zz", ("11" * 31 + "zz",)),
            ("11" * 33, ("11" * 33,)),
        ],
    )
    def test_malformed_key_raises_before_the_binary_runs_without_echoing_it(
        self, monkeypatch: pytest.MonkeyPatch, key: str, secrets: tuple[str, ...]
    ) -> None:
        runs: list[int] = []

        def recording_run_binary(*args: Any, **kwargs: Any) -> bytes:
            runs.append(1)
            raise AssertionError("the binary must not run for a malformed key")

        monkeypatch.setattr(mixin, "run_binary", recording_run_binary)
        with pytest.raises(BinaryUsageError) as excinfo:
            self._calculate({"key": key + "\n"})
        assert not runs
        assert all(secret not in str(excinfo.value) for secret in secrets)


class TestCalculateFeatureRejections:
    def test_production_class_without_override_raises_binary_unavailable(self, key_handle: str) -> None:
        table = pa.table({"email": ["alpha"]})
        feature = _config_feature("pseudonymized_email")
        with pytest.raises(BinaryNotInstalledError, match="anonymizer_binary"):
            AnonymizerFeatureGroup.calculate_feature(table, _feature_set(feature))

    def test_no_license_raises_license_missing(self, key_handle: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLODA_LICENSE_FILE", raising=False)
        monkeypatch.delenv("MLODA_LICENSE_KEY", raising=False)

        class _NoLicenseAnonymizer(AnonymizerFeatureGroup):
            BINARY_COMMAND_OVERRIDE = STUB_CMD

        table = pa.table({"email": ["alpha"]})
        with pytest.raises(LicenseMissingError):
            _NoLicenseAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))

    def test_license_for_another_plugin_id_raises_license_invalid(self, key_handle: str) -> None:
        class _WrongLicenseAnonymizer(StubAnonymizer):
            LICENSE_KEY_OVERRIDE = valid_license_token(["example_binary"])

        table = pa.table({"email": ["alpha"]})
        with pytest.raises(LicenseInvalidError):
            _WrongLicenseAnonymizer.calculate_feature(table, _feature_set(_config_feature("pseudonymized_email")))


# -------------------------------------------------------------------------------------------
# 3. Level 3: mloda.run_all end-to-end
# -------------------------------------------------------------------------------------------


class TestIntegration:
    def test_config_form_end_to_end(self) -> None:
        rows: dict[str, list[Any]] = {"email": ["alpha", None, "gamma"]}
        results = mloda.run_all(
            [_config_feature("pseudonymized_email")],
            compute_frameworks=[PyArrowTable],
            api_data={"AnonymizerData": rows},
            data_access_collection=_key_collection(),
            plugin_collector=PluginCollector.enabled_feature_groups({ApiDataFG, StubAnonymizer}),
        )
        expected = compute_expected_hmac_sha256_column(rows["email"], KNOWN_ANSWER_KEY)
        assert _result_column(results, "pseudonymized_email") == expected

    def test_list_utf8_source_end_to_end(self) -> None:
        rows: dict[str, list[Any]] = {"email": LIST_UTF8_ROWS}
        results = mloda.run_all(
            [_config_feature("pseudonymized_email")],
            compute_frameworks=[PyArrowTable],
            api_data={"AnonymizerData": rows},
            data_access_collection=_key_collection(),
            plugin_collector=PluginCollector.enabled_feature_groups({ApiDataFG, StubAnonymizer}),
        )
        out = _result_column(results, "pseudonymized_email")
        _assert_list_utf8_pseudonymized(out)

    def test_string_form_end_to_end(self) -> None:
        rows: dict[str, list[Any]] = {"email": ["alpha", "beta"]}
        results = mloda.run_all(
            [_string_feature()],
            compute_frameworks=[PyArrowTable],
            api_data={"AnonymizerData": rows},
            data_access_collection=_key_collection(),
            plugin_collector=PluginCollector.enabled_feature_groups({ApiDataFG, StubAnonymizer}),
        )
        expected = compute_expected_hmac_sha256_column(rows["email"], KNOWN_ANSWER_KEY)
        assert _result_column(results, "email__hmac_sha256_pseudonymized") == expected

    @pytest.mark.parametrize(
        "feature",
        [
            Feature(
                "pseudonymized_email",
                Options(
                    context={
                        "pseudonymization_algorithm": "hmac_sha256",
                        "in_features": "email",
                        "pii_key_env": LEFTOVER_ENV_VALUE,
                    }
                ),
            ),
            Feature("email__hmac_sha256_pseudonymized", Options(context={"pii_key_env": LEFTOVER_ENV_VALUE})),
        ],
        ids=["config", "string"],
    )
    def test_leftover_pii_key_env_is_refused_naming_the_replacement(self, feature: Feature) -> None:
        with pytest.raises(Exception) as excinfo:
            mloda.run_all(
                [feature],
                compute_frameworks=[PyArrowTable],
                api_data={"AnonymizerData": {"email": ["alpha"]}},
                data_access_collection=_key_collection(),
                plugin_collector=PluginCollector.enabled_feature_groups({ApiDataFG, StubAnonymizer}),
            )
        message = str(excinfo.value)
        assert "pii_key_env" in message
        assert "pii_key_handle" in message
        assert LEFTOVER_ENV_VALUE not in message
