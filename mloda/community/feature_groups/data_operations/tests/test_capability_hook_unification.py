"""Behavioral guard for the unified match-time capability surface (issue #299).

The aggregation, rank, and frame-aggregate families previously hand-rolled their own
``supports_compute_framework`` plus a differently-shaped supported-types method. Issue
#299 folds these into one shared ``SubtypeCapabilityHook`` mixin: a family declares its
supported subtypes once via ``supported_op_subtypes(secondary)`` and the mixin turns that
single declaration into match-time rejection through ``supports_compute_framework``.

These tests pin that shared *behaviour* through the public hook (True/False), not through
class structure: restricting the one declaration method rejects the unlisted subtype and
accepts the listed one for every family, the unrestricted default accepts everything,
rank's parametric families stay open, and frame-aggregate's optional secondary axis
(frame_type) keys the restriction. Asserting on observable capability keeps them
protecting the contract across future refactors of the class hierarchy.

``SqliteFramework`` is a neutral driver: ``supports_compute_framework`` derives capability
from the class's own ``supported_op_subtypes``, not from the framework argument, so the same
framework exercises all three families.
"""

from __future__ import annotations

import pytest

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

from mloda.core.abstract_plugins.components.options import Options

from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
from mloda.community.feature_groups.data_operations.capability_hook import SubtypeCapabilityHook
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
    SqliteFrameAggregate,
)
from mloda.community.feature_groups.data_operations.row_preserving.rank.base import RankFeatureGroup


class TestUnifiedRestrictionSurface:
    """Restricting the single ``supported_op_subtypes`` method drives match-time rejection for every family.

    The three families resolve their discriminator (agg type / rank type / frame agg type)
    from three different feature-name shapes, but all inherit the same
    ``supported_op_subtypes`` -> ``supports_compute_framework`` wiring.
    """

    def test_aggregation_restriction(self) -> None:
        """An aggregation subclass advertising only sum rejects median and accepts sum."""

        class _Restricted(AggregationFeatureGroup):
            @classmethod
            def supported_op_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
                return frozenset({"sum"})

        assert _Restricted.supports_compute_framework("value__median_agg", Options(), SqliteFramework) is False
        assert _Restricted.supports_compute_framework("value__sum_agg", Options(), SqliteFramework) is True

    def test_rank_restriction(self) -> None:
        """A rank subclass advertising only row_number rejects percent_rank and accepts row_number."""

        class _Restricted(RankFeatureGroup):
            @classmethod
            def supported_op_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
                return frozenset({"row_number"})

        assert _Restricted.supports_compute_framework("value__percent_rank_ranked", Options(), SqliteFramework) is False
        assert _Restricted.supports_compute_framework("value__row_number_ranked", Options(), SqliteFramework) is True

    def test_frame_aggregate_restriction(self) -> None:
        """A frame-aggregate subclass advertising only sum rejects median rolling and accepts sum rolling."""

        class _Restricted(FrameAggregateFeatureGroup):
            @classmethod
            def supported_op_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
                return frozenset({"sum"})

        assert _Restricted.supports_compute_framework("value__median_rolling_3", Options(), SqliteFramework) is False
        assert _Restricted.supports_compute_framework("value__sum_rolling_3", Options(), SqliteFramework) is True


class TestUnrestrictedDefault:
    """With no ``supported_op_subtypes`` override, the shared default leaves every subtype unrestricted."""

    @pytest.mark.parametrize(
        "base, feature_name",
        [
            (AggregationFeatureGroup, "value__median_agg"),
            (RankFeatureGroup, "value__percent_rank_ranked"),
            (FrameAggregateFeatureGroup, "value__median_rolling_3"),
        ],
    )
    def test_default_accepts_everything(self, base: type[SubtypeCapabilityHook], feature_name: str) -> None:
        """The base families ship no restriction, so the shared hook accepts every subtype (conservative)."""
        assert base.supports_compute_framework(feature_name, Options(), SqliteFramework) is True


class TestRankParametricStaysOpen:
    """Rank parametric families are never subtype-checked, even under a restriction."""

    def test_parametric_family_open_under_restriction(self) -> None:
        """ntile_N stays open through the shared hook even though it is not in the restricted set."""

        class _Restricted(RankFeatureGroup):
            @classmethod
            def supported_op_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
                return frozenset({"row_number"})

        assert _Restricted.supports_compute_framework("value__ntile_4_ranked", Options(), SqliteFramework) is True


class TestFrameAggregateSecondaryAxis:
    """Frame aggregate keys its restriction on the optional secondary axis (frame_type)."""

    def test_restriction_keyed_by_frame_type(self) -> None:
        """median is allowed on the unrestricted rolling axis but rejected on the restricted cumulative axis."""

        class _Restricted(FrameAggregateFeatureGroup):
            @classmethod
            def supported_op_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
                if secondary in ("cumulative", "expanding"):
                    return frozenset({"sum"})
                return None

        assert _Restricted.supports_compute_framework("value__median_rolling_3", Options(), SqliteFramework) is True
        assert _Restricted.supports_compute_framework("value__cummedian", Options(), SqliteFramework) is False


class TestUnrestrictedSkipsSubtypeResolution:
    """An unrestricted family must consult ``supported_op_subtypes`` first and never resolve its subtype."""

    def test_aggregation_unrestricted_skips_resolution(self) -> None:
        """An unrestricted aggregation subclass accepts median without resolving the agg type."""

        class _Unrestricted(AggregationFeatureGroup):
            @classmethod
            def _capability_subtype(cls, feature_name: str, options: Options) -> str | None:
                raise AssertionError("subtype must not be resolved when the backend is unrestricted")

        assert _Unrestricted.supports_compute_framework("value__median_agg", Options(), SqliteFramework) is True

    def test_rank_unrestricted_skips_resolution(self) -> None:
        """An unrestricted rank subclass accepts percent_rank without resolving the rank type."""

        class _Unrestricted(RankFeatureGroup):
            @classmethod
            def _capability_subtype(cls, feature_name: str, options: Options) -> str | None:
                raise AssertionError("subtype must not be resolved when the backend is unrestricted")

        assert (
            _Unrestricted.supports_compute_framework("value__percent_rank_ranked", Options(), SqliteFramework) is True
        )


class TestLegacyMethodNamesRejected:
    """Pre-#299 supported-types method names must fail loudly at class definition, not be silently ignored."""

    def test_legacy_supported_agg_types_rejected(self) -> None:
        """Overriding the pre-#299 ``supported_agg_types`` raises TypeError naming ``supported_op_subtypes``."""
        with pytest.raises(TypeError, match="supported_op_subtypes"):

            class _Legacy(AggregationFeatureGroup):
                @classmethod
                def supported_agg_types(cls) -> frozenset[str]:
                    return frozenset({"sum"})

    def test_legacy_supported_rank_types_rejected(self) -> None:
        """Overriding the pre-#299 ``supported_rank_types`` raises TypeError naming ``supported_op_subtypes``."""
        with pytest.raises(TypeError, match="supported_op_subtypes"):

            class _Legacy(RankFeatureGroup):
                @classmethod
                def supported_rank_types(cls) -> frozenset[str]:
                    return frozenset({"row_number"})


class TestRestrictionRequiresResolver:
    """Declaring a restriction without a subtype resolver must fail at class definition."""

    def test_restriction_without_resolver_rejected(self) -> None:
        """Overriding ``supported_op_subtypes`` while inheriting the unresolved default raises TypeError."""
        with pytest.raises(TypeError, match="_capability_subtype"):

            class _Restricted(SubtypeCapabilityHook):
                @classmethod
                def supported_op_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
                    return frozenset({"sum"})

    def test_guard_only_family_defines_cleanly(self) -> None:
        """A guard-only subclass that never restricts subtypes needs no resolver and defines cleanly."""

        class _GuardOnly(SubtypeCapabilityHook):
            @classmethod
            def _capability_guard(cls, feature_name: str, options: Options) -> bool:
                return True

        assert _GuardOnly._capability_guard("anything", Options()) is True

    def test_non_callable_supported_subtypes_rejected(self) -> None:
        """Binding supported_op_subtypes to a plain (non-callable) value raises TypeError at class definition."""
        with pytest.raises(TypeError, match="supported_op_subtypes"):

            class _Broken(AggregationFeatureGroup):
                supported_op_subtypes = frozenset({"sum"})  # type: ignore[assignment]

        class _Valid(AggregationFeatureGroup):
            @classmethod
            def supported_op_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
                return frozenset({"sum"})

        assert _Valid.supported_op_subtypes() == frozenset({"sum"})


class TestUnresolvedAxisStaysConservative:
    """An unresolved secondary axis must skip the subtype check so restricted backends stay conservative."""

    def test_sqlite_frame_aggregate_unresolved_axis_accepts_median(self) -> None:
        """frame_type is unresolved, so the hook must not consult supported_op_subtypes(None), which would
        return SQLite's restricted set and wrongly reject median."""
        assert (
            SqliteFrameAggregate.supports_compute_framework(
                "totally_unrelated", Options(context={"aggregation_type": "median"}), SqliteFramework
            )
            is True
        )
