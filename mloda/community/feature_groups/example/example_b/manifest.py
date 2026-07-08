"""Entry-point manifest for mloda-community-example-b.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .example_b_feature_group import ExampleBFeatureGroup

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    ExampleBFeatureGroup,
]
