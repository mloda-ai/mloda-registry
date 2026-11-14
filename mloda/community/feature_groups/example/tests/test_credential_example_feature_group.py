"""Tests for CredentialExampleFeatureGroup (guide 11, Secrets)."""

from typing import Any

from mloda.provider import DataCreatorFeatureGroup, FeatureSet, credentials_scope
from mloda.user import Credential, DataAccessCollection, Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

from mloda.community.feature_groups.example.credential_example_feature_group import CredentialExampleFeatureGroup
from mloda.testing.base import FeatureGroupTestBase
from mloda.testing.feature_group_shape import assert_feature_group_shape

KEY = "example-signing-key"
HANDLE = "example-signing-handle"
EMAILS: list[str | None] = ["a@example.com", None]
DIGEST = "86c777b5aa5df370810ded5d15fc5552ede68b28d9906d9e27b5aa7b31291284"  # HMAC-SHA256 of "a@example.com" under KEY
EXPECTED: list[str | None] = [DIGEST, None]


class EmailSource(DataCreatorFeatureGroup):
    """Provides an ``email`` column."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"email"}

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return {"email": list(EMAILS)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[Any]]:
        return {PythonDictFramework}


class TestCredentialExampleFeatureGroup(FeatureGroupTestBase):
    feature_group_class = CredentialExampleFeatureGroup

    def test_feature_group_shape(self) -> None:
        assert_feature_group_shape(
            CredentialExampleFeatureGroup,
            base_options=Options(group={"signing_key_handle": "h"}),
            requests=["email__hmac_signed"],
        )

    def test_matches_name_with_handle_option(self) -> None:
        options = Options(group={"signing_key_handle": "h"})
        assert CredentialExampleFeatureGroup.match_feature_group_criteria("email__hmac_signed", options)

    def test_refuses_name_without_handle_option(self) -> None:
        assert not CredentialExampleFeatureGroup.match_feature_group_criteria("email__hmac_signed", Options())

    def test_compute_framework_rule_is_python_dict(self) -> None:
        assert CredentialExampleFeatureGroup.compute_framework_rule() == {PythonDictFramework}

    def test_calculate_feature_signs_with_the_scoped_credential(self) -> None:
        feature = Feature("email__hmac_signed", Options(group={"signing_key_handle": HANDLE}))
        feature_set = _feature_set(feature)
        with credentials_scope({HANDLE: {"key": KEY}}):
            result = CredentialExampleFeatureGroup.calculate_feature({"email": list(EMAILS)}, feature_set)
        assert result["email__hmac_signed"] == EXPECTED

    def test_run_all_signs_the_column(self) -> None:
        feature = Feature("email__hmac_signed", Options(group={"signing_key_handle": HANDLE}))
        results = mloda.run_all(
            [feature],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups({CredentialExampleFeatureGroup, EmailSource}),
            data_access_collection=DataAccessCollection(credentials={HANDLE: Credential(key=KEY)}),
        )
        signed = [r["email__hmac_signed"] for r in results if "email__hmac_signed" in r]
        assert signed == [EXPECTED]


def _feature_set(*features: Feature) -> FeatureSet:
    feature_set = FeatureSet()
    for feature in features:
        feature_set.add(feature)
    return feature_set
