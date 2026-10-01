"""Isolated: a subclass that quietly takes over a concept is named in the receipt (F1).

A subprocess because the subclass registers process-wide and would shadow the concept for every
later test.

mloda keeps a matching subclass over its parent, so a host module that defines a subclass of
SkrAccountFeatureGroup -- here one that merely re-states the same catalogue -- answers `revenue`
instead of the class the package ships. The total does not change, so nothing else shows it.
The receipt does: `chosen` names the subclass and `also_matched` names the parent it shadowed.
That is the question only mloda can be asked -- why this producer, and who else could have
answered -- and the receipt now carries the answer.
"""

import sys
from pathlib import Path

from mloda.user import DataAccessCollection
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.german_ledger.policy import AdmissibilityPolicyGroup
from mloda.community.feature_groups.german_ledger.receipt import run_with_receipts
from mloda.community.feature_groups.german_ledger.skr import SkrAccountFeatureGroup
from mloda.community.feature_groups.german_ledger.sources import SourcesFeatureGroup  # noqa: F401

DOSSIER_A = Path(__file__).parent.parent / "fixtures" / "dossier_a"


class Closing2025(AdmissibilityPolicyGroup):
    LOCK_DATE = __import__("datetime").date(2026, 1, 15)
    PERIOD_END = __import__("datetime").date(2025, 12, 31)


class QuietRevision(SkrAccountFeatureGroup):
    """Same catalogue, different class: nothing in the total gives it away."""


_, [receipt] = run_with_receipts(
    ["revenue__sources"],
    compute_frameworks={PyArrowTable},
    data_access_collection=DataAccessCollection(folders={str(DOSSIER_A)}),
)
revenue = next(r for r in receipt["resolution"] if r["feature"] == "revenue")
print(f"total={receipt['total']}")
print(f"revenue: chosen={revenue['chosen']} also_matched={revenue['also_matched']}")
if receipt["total"] != "45385.06" or revenue["chosen"] != ["QuietRevision"]:
    sys.exit(1)
if revenue["also_matched"] != ["SkrAccountFeatureGroup"]:
    sys.exit(1)
print("RESOLUTION RECEIPT: the shadowing subclass and the class it shadowed are both named")
sys.exit(0)
