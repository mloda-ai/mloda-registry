"""Entry-point manifest for mloda-enterprise-anonymizer.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

FEATURE_GROUPS: list[type[FeatureGroup]]

try:
    from .anonymizer_feature_group import AnonymizerFeatureGroup
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] == "pyarrow":
        FEATURE_GROUPS = []
    else:
        raise
else:
    FEATURE_GROUPS = [AnonymizerFeatureGroup]
