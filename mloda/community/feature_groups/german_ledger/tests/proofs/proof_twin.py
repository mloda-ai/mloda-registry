"""Isolated: the same books as a GDPdU dossier and a DATEV batch give the same totals.

A subprocess because the host's policy registers process-wide, and exactly one may be live.

The twin (fixtures/twin_2025, see make_twin.py) is four SKR04 bookings for FY2025. GDPdU lists
each account in its own direction; DATEV books Konto against Gegenkonto, +Soll/-Haben. Two
things must hold for the totals to meet:
  - one policy both formats can be judged by. The late-entry cutoff needs an Erfassungsdatum
    DATEV lacks, and Festschreibung needs a flag GDPdU lacks, so this host bounds the period;
  - one sign per concept. Revenue is reported credit-positive whatever the journal's sign.
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

from mloda.user import DataAccessCollection, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.german_ledger.policy import AdmissibilityPolicyGroup, PeriodBound
from mloda.community.feature_groups.german_ledger.skr import SkrAccountFeatureGroup
from mloda.community.feature_groups.german_ledger.sources import SourcesFeatureGroup  # noqa: F401

TWIN = Path(__file__).parent.parent / "fixtures" / "twin_2025"
EXPECTED = {"revenue": Decimal("22485.06"), "receivables": Decimal("7500.00")}


class TwinClosing2025(AdmissibilityPolicyGroup):
    """FY2025, judged by the one column both formats carry: the booking date."""

    RULES = (PeriodBound(date(2025, 1, 1), date(2025, 12, 31)),)


SkrAccountFeatureGroup.CHART = "04"


def _total(folder: Path, concept: str) -> tuple[object, list[str]]:
    results = mloda.run_all(
        features=[f"{concept}__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(folder)}),
    )
    got = {k: v for t in results for k, v in t.to_pydict().items()}
    [value] = got[f"{concept}__sources~value"]
    [origins] = got[f"{concept}__sources~origins"]
    return value, list(origins or [])


failed = False
for concept, expected in EXPECTED.items():
    gdpdu, gdpdu_cited = _total(TWIN / "gdpdu", concept)
    datev, datev_cited = _total(TWIN / "datev", concept)
    ok = gdpdu == datev == expected and gdpdu_cited and len(gdpdu_cited) == len(datev_cited)
    failed |= not ok
    print(f"{'SAME' if ok else 'DIFFERENT'} {concept}: GDPdU {gdpdu} / DATEV {datev} (expected {expected})")
    print(f"     GDPdU cites {', '.join(c.rsplit(':', 1)[1] for c in gdpdu_cited)}")
    print(f"     DATEV cites {', '.join(c.rsplit(':', 1)[1] for c in datev_cited)}")

if failed:
    sys.exit(1)
print("TWIN: one set of books, two formats, the same totals, both cited")
sys.exit(0)
