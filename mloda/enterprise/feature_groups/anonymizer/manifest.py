"""Entry-point manifest for mloda-enterprise-anonymizer.

Lists the concrete FeatureGroup classes that mloda discovers via the
``mloda.feature_groups`` entry point.
"""

from __future__ import annotations

import importlib.util
import logging

from mloda.provider import FeatureGroup

FEATURE_GROUPS: list[type[FeatureGroup]]

try:
    from .anonymizer_feature_group import AnonymizerFeatureGroup
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] == "pyarrow":
        FEATURE_GROUPS = []
        try:
            wheel_installed = importlib.util.find_spec("anonymizer_binary") is not None
        except Exception:
            wheel_installed = False
        logging.getLogger(__name__).log(
            logging.WARNING if wheel_installed else logging.DEBUG,
            'pyarrow is missing, so AnonymizerFeatureGroup is not registered; pip install "mloda-enterprise[anonymizer]"',
        )
    else:
        raise
else:
    FEATURE_GROUPS = [AnonymizerFeatureGroup]
