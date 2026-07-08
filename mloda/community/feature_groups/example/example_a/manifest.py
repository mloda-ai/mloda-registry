"""Entry-point manifest for mloda-community-example-a.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .example_a_feature_group import ExampleAFeatureGroup

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    ExampleAFeatureGroup,
]
