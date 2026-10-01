"""Entry-point manifest for mloda-community-german-ledger.

Lists the concrete FeatureGroup classes for the ``mloda.feature_groups`` entry point. The entry
point is NOT declared yet (see pyproject.toml): entry points are only loaded by
``PluginLoader.all()``, which also loads the stock ``ReadFileFeature`` and collides with these
groups until mloda#1745 is fixed. Enabling it is one line in config/packages.toml.

AdmissibilityPolicyGroup is absent on purpose: the base is inert, and the live policy is a
host subclass that the host registers by importing its own module.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .policy import DatevJournalFeatureGroup, JournalFeatureGroup
from .skr import SkrAccountFeatureGroup
from .sources import SourcesFeatureGroup

FEATURE_GROUPS: list[type[FeatureGroup]] = [
    JournalFeatureGroup,
    DatevJournalFeatureGroup,
    SkrAccountFeatureGroup,
    SourcesFeatureGroup,
]
