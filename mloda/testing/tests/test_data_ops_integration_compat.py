"""Published-base subclasses must still run: legacy _plugin_collector, family bases, credential handles."""

from abc import abstractmethod
from typing import Any

import pyarrow.compute as pc
import pytest
from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    credential_handle_spec,
    property_spec,
    read_credential,
)
from mloda.user import Credential, DataAccessCollection, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.percentile.tests import (
    test_integration as percentile,
)
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase


class _LegacyCollectorIntegration(percentile.TestPercentileIntegration):
    __test__ = False

    def _plugin_collector(self) -> PluginCollector:  # type: ignore[override]
        return PluginCollector.enabled_feature_groups({self.data_creator_class(), self.feature_group_class()})


def test_legacy_collector_runs_primary_path() -> None:
    _LegacyCollectorIntegration().test_primary_feature_through_pipeline()


def test_legacy_collector_runs_mask_path() -> None:
    _LegacyCollectorIntegration().test_mask_feature_through_pipeline()


# -- the family-base test runs for a third-party family base --------------------


class _ToyFamilyBase(FeatureChainParserMixin, FeatureGroup):
    """An abstract family base outside ``mloda.community``."""

    PREFIX_PATTERN = r".*__toy_double$"

    @classmethod
    @abstractmethod
    def double(cls, data: Any, source: str, name: str) -> Any:
        """Append the doubled source column."""


class _ToyFamily(_ToyFamilyBase):
    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PyArrowTable}

    @classmethod
    def double(cls, data: Any, source: str, name: str) -> Any:
        return data.append_column(name, pc.multiply(data[source], 2))

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data = cls.double(data, cls._extract_single_source_feature(feature), feature.name)
        return data


class _ToyConcreteParent(FeatureChainParserMixin, FeatureGroup):
    """A concrete parent group declaring a pattern: not a family base."""

    PREFIX_PATTERN = r".*__toy_triple$"


class _ToyTakeover(_ToyConcreteParent):
    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PyArrowTable}


class _ToyIntegration(DataOpsIntegrationTestBase):
    __test__ = False
    _group: Any = _ToyFamily
    _primary = "value_int__toy_double"

    @classmethod
    def feature_group_class(cls) -> Any:
        return cls._group

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return cls._primary

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return []

    @classmethod
    def secondary_feature_name(cls) -> str:
        return cls._primary

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return []

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return [cls._primary]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return []

    @classmethod
    def match_options(cls) -> Options:
        return Options()


class _ToyTakeoverIntegration(_ToyIntegration):
    _group: Any = _ToyTakeover
    _primary = "value_int__toy_triple"


def test_family_base_test_runs_for_a_third_party_abstract_base() -> None:
    try:
        _ToyIntegration().test_family_base_is_not_selected_without_an_implementation()
    except pytest.skip.Exception:
        pytest.fail("the family-base test skipped for an abstract base outside mloda.community")


def test_family_base_test_skips_for_a_concrete_parent_group() -> None:
    with pytest.raises(pytest.skip.Exception):
        _ToyTakeoverIntegration().test_family_base_is_not_selected_without_an_implementation()


def test_missing_config_feature_name_defaults_to_the_primary_name() -> None:
    assert _ToyIntegration.missing_config_feature_name() == _ToyIntegration.primary_feature_name()


def test_match_rejects_missing_config_probes_the_hook() -> None:
    class _NonMatching(_ToyIntegration):
        @classmethod
        def missing_config_feature_name(cls) -> str:
            return "plain_feature"

    class _Matching(_ToyIntegration):
        @classmethod
        def missing_config_feature_name(cls) -> str:
            return "other__toy_double"

    _NonMatching().test_match_rejects_missing_config()
    with pytest.raises(AssertionError):
        _Matching().test_match_rejects_missing_config()


# -- a feature group declaring a credential handle runs the inherited Level 3 tests --


class _ToyKeyedBase(_ToyFamilyBase):
    HANDLE = "toy_key_handle"
    PREFIX_PATTERN = r".*__toy_keyed$"
    RECOGNITION_ONLY_PATTERN = True
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1
    PROPERTY_MAPPING = {
        HANDLE: credential_handle_spec("Credential handle with a 'key' field", required_fields=("key",)),
        DefaultOptionKeys.in_features: property_spec("Column to double"),
    }


class _ToyKeyed(_ToyKeyedBase, _ToyFamily):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            assert read_credential(feature.options.get(cls.HANDLE))["key"] == "not-a-secret"
        return super().calculate_feature(data, features)


class _ToyKeyedIntegration(_ToyIntegration):
    _group: Any = _ToyKeyed
    _primary = "value_int__toy_keyed"
    _doubled = [20, -10, 0, 40, None, 100, 60, 120, 30, 30, 80, -20]

    @classmethod
    def group_options(cls) -> dict[str, Any]:
        return {_ToyKeyedBase.HANDLE: "toy-key"}

    @classmethod
    def data_access_collection(cls) -> DataAccessCollection | None:
        return DataAccessCollection(credentials={"toy-key": Credential(key="not-a-secret")})

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return cls._doubled

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return cls._doubled


@pytest.mark.parametrize(
    "name",
    [
        "test_primary_feature_through_pipeline",
        "test_secondary_feature_through_pipeline",
        "test_primary_feature_with_column_selective_source",
        "test_family_base_is_not_selected_without_an_implementation",
    ],
)
def test_credential_group_runs_inherited_tests(name: str) -> None:
    try:
        getattr(_ToyKeyedIntegration(), name)()
    except pytest.skip.Exception:
        pytest.fail(f"{name} skipped for a feature group declaring a credential handle")


def test_unregistered_credential_handle_is_rejected() -> None:
    class _NoCollection(_ToyKeyedIntegration):
        @classmethod
        def data_access_collection(cls) -> DataAccessCollection | None:
            return None

    with pytest.raises(ValueError, match="not a registered credential"):
        _NoCollection().test_primary_feature_through_pipeline()
