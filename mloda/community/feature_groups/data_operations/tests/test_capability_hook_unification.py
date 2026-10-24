"""Behavioral guard for the unified match-time capability surface (issue #299).

The aggregation, rank, and frame-aggregate families previously hand-rolled their own
``supports_compute_framework`` plus a differently-shaped supported-types method. They were folded into one shared ``SubtypeCapabilityHook`` mixin (all three have since moved
to ``Implementation.supported`` and ``accepts``, so no family is on the hook): a hook subclass declares its
supported subtypes once via ``supported_op_subtypes(secondary)`` and the mixin turns that
single declaration into match-time rejection through ``supports_compute_framework``.

These tests pin that shared *behaviour* through the public hook (True/False), not through
class structure, using a small local frame-like subclass: restricting the one declaration method
rejects the unlisted subtype and accepts the listed one, the unrestricted default accepts everything,
and the hook's optional secondary axis keys the restriction. Asserting on observable capability keeps them
protecting the contract across future refactors of the class hierarchy.

Aggregation's own restriction (``Implementation.supported``) is covered in
``aggregation/tests/test_base.py``.

``SqliteFramework`` is a neutral driver: ``supports_compute_framework`` derives capability
from the class's own ``supported_op_subtypes``, not from the framework argument.
"""

from __future__ import annotations

import pytest
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

from mloda.community.feature_groups.data_operations.capability_hook import SubtypeCapabilityHook
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.integration import load_group_implementations


class _FrameAxisHook(SubtypeCapabilityHook):
    """Minimal hook subclass with a secondary axis (frame_type), so the hook's axis code stays covered."""

    _CAPABILITY_HAS_AXIS = True

    @classmethod
    def _capability_subtype(cls, feature_name: str, options: Options) -> str | None:
        parsed = FrameAggregateFeatureGroup._parse_frame_feature(feature_name)
        if parsed is not None:
            return str(parsed["agg_type"])
        agg_type = options.get("aggregation_type")
        return None if agg_type is None else str(agg_type)

    @classmethod
    def _capability_secondary(cls, feature_name: str, options: Options) -> str | None:
        parsed = FrameAggregateFeatureGroup._parse_frame_feature(feature_name)
        if parsed is not None:
            return str(parsed["frame_type"])
        frame_type = options.get("frame_type")
        return None if frame_type is None else str(frame_type)


class TestUnifiedRestrictionSurface:
    """Restricting the single ``supported_op_subtypes`` method drives match-time rejection.

    Frame aggregate resolves its discriminator (frame agg type) from the feature name and inherits the
    ``supported_op_subtypes`` -> ``supports_compute_framework`` wiring.
    """

    def test_frame_like_restriction(self) -> None:
        """A frame-like hook advertising only sum rejects median rolling and accepts sum rolling."""

        class _Restricted(_FrameAxisHook):
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
            (_FrameAxisHook, "value__median_rolling_3"),
        ],
    )
    def test_default_accepts_everything(self, base: type[SubtypeCapabilityHook], feature_name: str) -> None:
        """The base families ship no restriction, so the shared hook accepts every subtype (conservative)."""
        assert base.supports_compute_framework(feature_name, Options(), SqliteFramework) is True


class TestFrameLikeSecondaryAxis:
    """A hook with a secondary axis (frame_type) keys its restriction on it."""

    def test_restriction_keyed_by_frame_type(self) -> None:
        """median is allowed on the unrestricted rolling axis but rejected on the restricted cumulative axis."""

        class _Restricted(_FrameAxisHook):
            @classmethod
            def supported_op_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
                if secondary in ("cumulative", "expanding"):
                    return frozenset({"sum"})
                return None

        assert _Restricted.supports_compute_framework("value__median_rolling_3", Options(), SqliteFramework) is True
        assert _Restricted.supports_compute_framework("value__cummedian", Options(), SqliteFramework) is False


class TestUnrestrictedSkipsSubtypeResolution:
    """An unrestricted family must consult ``supported_op_subtypes`` first and never resolve its subtype."""

    def test_frame_aggregate_unrestricted_skips_resolution(self) -> None:
        """An unrestricted frame-like hook accepts median without resolving the agg type."""

        class _Unrestricted(_FrameAxisHook):
            @classmethod
            def _capability_subtype(cls, feature_name: str, options: Options) -> str | None:
                raise AssertionError("subtype must not be resolved when the backend is unrestricted")

        assert _Unrestricted.supports_compute_framework("value__median_rolling_3", Options(), SqliteFramework) is True


class TestLegacyMethodNamesRejected:
    """Pre-#299 supported-types method names must fail loudly at class definition, not be silently ignored."""

    def test_legacy_supported_agg_types_rejected(self) -> None:
        """Overriding the pre-#299 ``supported_agg_types`` raises TypeError naming ``supported_op_subtypes``."""
        with pytest.raises(TypeError, match="supported_op_subtypes"):

            class _Legacy(_FrameAxisHook):
                @classmethod
                def supported_agg_types(cls) -> frozenset[str]:
                    return frozenset({"sum"})

    def test_legacy_supported_rank_types_rejected(self) -> None:
        """Overriding the pre-#299 ``supported_rank_types`` raises TypeError naming ``supported_op_subtypes``."""
        with pytest.raises(TypeError, match="supported_op_subtypes"):

            class _Legacy(_FrameAxisHook):
                @classmethod
                def supported_rank_types(cls) -> frozenset[str]:
                    return frozenset({"row_number"})

    def test_legacy_supported_subtypes_rejected(self) -> None:
        """Overriding the stale ``supported_subtypes`` raises TypeError naming ``supported_op_subtypes``."""
        with pytest.raises(TypeError, match="supported_op_subtypes"):

            class _Legacy(_FrameAxisHook):
                @classmethod
                def supported_subtypes(cls) -> frozenset[str]:
                    return frozenset({"sum"})


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

            class _Broken(_FrameAxisHook):
                supported_op_subtypes = frozenset({"sum"})  # type: ignore[assignment]

        class _Valid(_FrameAxisHook):
            @classmethod
            def supported_op_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
                return frozenset({"sum"})

        assert _Valid.supported_op_subtypes() == frozenset({"sum"})


class TestUnresolvedAxisStaysConservative:
    """An unresolved secondary axis must skip the subtype check so restricted hooks stay conservative."""

    def test_unresolved_axis_accepts_median(self) -> None:
        """frame_type is unresolved, so the hook must not consult supported_op_subtypes(None), which would
        return the restricted set and wrongly reject median."""

        class _Restricted(_FrameAxisHook):
            @classmethod
            def supported_op_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
                return frozenset({"sum"})

        assert (
            _Restricted.supports_compute_framework(
                "totally_unrelated", Options(context={"aggregation_type": "median"}), SqliteFramework
            )
            is True
        )

    def test_frame_group_has_no_axis_so_the_resolved_agg_type_decides(self) -> None:
        """The frame group declares capability per agg type only (SQLite lacks median), with no axis to leave open."""
        load_group_implementations(FrameAggregateFeatureGroup)
        assert (
            FrameAggregateFeatureGroup.supports_compute_framework(
                "totally_unrelated", Options(context={"aggregation_type": "median"}), SqliteFramework
            )
            is False
        )
