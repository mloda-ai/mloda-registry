"""Isolated: what happens WITHOUT the match_feature_group_criteria gate.

Run as a subprocess -- defining the ungated subclass registers it globally
via subclass discovery, which would poison every later resolution.
"""

import sys
from pathlib import Path
from typing import Any

from mloda.provider import FeatureGroup, FeatureResolutionError
from mloda.user import DataAccessCollection, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.german_ledger.policy import JournalFeatureGroup
from mloda.community.feature_groups.german_ledger.skr import SkrAccountFeatureGroup  # noqa: F401
from mloda.community.feature_groups.german_ledger.sources import SourcesFeatureGroup  # noqa: F401


class UngatedJournal(JournalFeatureGroup):
    """Identical, except it does not gate on its own name first.

    The reader-backed root is the journal now; SkrAccountFeatureGroup used to be
    the root, and this hazard moved with the reader.
    """

    @classmethod
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, data_access_collection: Any = None) -> bool:
        # The inherited rule, called on this class: skips JournalFeatureGroup's name gate.
        unbound: Any = FeatureGroup.match_feature_group_criteria
        return bool(unbound.__func__(cls, feature_name, options, data_access_collection))


try:
    mloda.run_all(
        features=["revenue__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(
            folders={str(Path(__file__).parent.parent / "fixtures" / "dossier_a")}
        ),
    )
    print("UNEXPECTED: no collision")
    sys.exit(1)
except FeatureResolutionError as exc:
    first = str(exc).splitlines()[0]
    print(f"COLLISION REPRODUCED: {first}")
    sys.exit(0)
