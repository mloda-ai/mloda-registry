"""Build a real FeatureSet for calling calculate_feature directly in tests."""

from __future__ import annotations

from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import Feature


def build_feature_set(
    feature_group: type[FeatureGroup] | None,
    *features: Feature | str,
    compute_framework: type[ComputeFramework] | None = None,
) -> FeatureSet:
    """Build a FeatureSet the way a run does before calling ``calculate_feature``."""
    fs = FeatureSet()
    for feature in features:
        fs.add(Feature(feature) if isinstance(feature, str) else feature)
    if feature_group is not None:
        fs.materialize_option_defaults(feature_group)
    if compute_framework is not None:
        framework = compute_framework()
        framework.set_filter_engine(fs)
        framework.set_mask_engine(fs)
    return fs
