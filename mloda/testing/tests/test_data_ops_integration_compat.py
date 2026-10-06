"""A published-base subclass whose _plugin_collector has no creator parameter must still run."""

from mloda.user import PluginCollector

from mloda.community.feature_groups.data_operations.row_preserving.percentile.tests import (
    test_integration as percentile,
)


class _LegacyCollectorIntegration(percentile.TestPercentileIntegration):
    __test__ = False

    def _plugin_collector(self) -> PluginCollector:  # type: ignore[override]
        return PluginCollector.enabled_feature_groups({self.data_creator_class(), self.feature_group_class()})


def test_legacy_collector_runs_primary_path() -> None:
    _LegacyCollectorIntegration().test_primary_feature_through_pipeline()


def test_legacy_collector_runs_mask_path() -> None:
    _LegacyCollectorIntegration().test_mask_feature_through_pipeline()
