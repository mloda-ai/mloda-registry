"""Entry-point manifest for mloda-community-german-ledger.

The policy base is absent because the host's subclass is the policy; JournalFeatureGroup and
DatevJournalFeatureGroup are aliases of the two readers.
"""

from __future__ import annotations

from mloda.provider import FeatureGroup

from .datev import DatevExtfReader
from .reader import GdpduReader
from .skr import SkrAccountFeatureGroup
from .sources import SourcesFeatureGroup

FEATURE_GROUPS: list[type[FeatureGroup]] = [GdpduReader, DatevExtfReader, SkrAccountFeatureGroup, SourcesFeatureGroup]
