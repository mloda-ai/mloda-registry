"""Shared test helpers for the effective-classification tests (not collected)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mloda.provider import FeatureGroup
from mloda.steward import PlanStep
from mloda.user import Options

from mloda.community.extenders.shared.classification import CLASSIFICATION_KEY


def declaring(level: str | None) -> Mapping[str, Any]:
    return {} if level is None else {CLASSIFICATION_KEY: level}


class InternalGroup(FeatureGroup):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return declaring("internal")


class PiiGroup(FeatureGroup):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return declaring("pii")


class UnknownGroup(FeatureGroup):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return declaring("secret")


def classified_step(
    group: type[FeatureGroup] | None,
    names: tuple[str, ...],
    edges: Mapping[str, tuple[str, ...]] | None = None,
    *,
    kind: str = "compute",
    options: Options | None = None,
) -> PlanStep:
    return PlanStep(
        step_kind=kind,  # type: ignore[arg-type]
        feature_names=names,
        feature_group=group,
        compute_framework=None,
        source_feature_group=None,
        source_compute_framework=None,
        requested_feature_names=names,
        input_feature_edges=edges or {},
        feature_set_options=options,
    )
