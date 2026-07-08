"""Entry-point manifest for mloda-enterprise-example.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .enterprise_example_feature_group import EnterpriseExampleFeatureGroup

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    EnterpriseExampleFeatureGroup,
]
