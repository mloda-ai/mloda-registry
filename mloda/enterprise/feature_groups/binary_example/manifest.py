"""Entry-point manifest for mloda-enterprise-binary-example.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .binary_example_feature_group import BinaryExampleFeatureGroup

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    BinaryExampleFeatureGroup,
]
