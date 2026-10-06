"""Isolated: dossier_c end to end, through its own profile, to a real total.

A subprocess because configuring a second profile registers it globally via subclass
discovery and would shadow the base group for every later resolution -- which is the
hazard proof_profile_shadow.py exists to demonstrate.

The point: the same money, re-declared under a second (synthetic) exporter profile in a
different file shape, reaches the same cited total through the whole pipeline -- not
merely the same decoded column out of the reader.
"""

import sys
from datetime import date
from pathlib import Path

from mloda.user import DataAccessCollection, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.experimental.german_ledger.policy import AdmissibilityPolicyGroup
from mloda.community.feature_groups.experimental.german_ledger.skr import (
    SkrAccountFeatureGroup,  # noqa: F401  (registers by import)
)
from mloda.community.feature_groups.experimental.german_ledger.sources import SourcesFeatureGroup  # noqa: F401

FIX = Path(__file__).parent.parent / "fixtures"


class Closing2025C(AdmissibilityPolicyGroup):
    """This host's policy: the clocks are declared under different names in dossier_c."""

    LOCK_DATE = date(2026, 1, 15)
    PERIOD_END = date(2025, 12, 31)
    BOOKING_COLUMN = "Buchung"
    KEYING_COLUMN = "Erfassung"


# No subclass, no profile override. The profile is chosen from the columns dossier_c
# declares, so a second exporter shape costs the caller nothing at all.

results = mloda.run_all(
    features=["revenue__sources", "receivables__sources"],
    compute_frameworks=[PyArrowTable],
    data_access_collection=DataAccessCollection(folders={str(FIX / "dossier_c")}),
)
got = {c: t.column(c).to_pylist()[0] for t in results for c in t.column_names}

revenue = str(got["revenue__sources~value"])
receivables = str(got["receivables__sources~value"])
n_rev = len(got["revenue__sources~origins"])

if revenue != "45385.06" or receivables != "5000.00" or n_rev != 5:
    print(f"UNEXPECTED: revenue={revenue} receivables={receivables} origins={n_rev}")
    sys.exit(1)
n_rec = len(got["receivables__sources~origins"])
if n_rec != 1:
    print(f"UNEXPECTED: receivables origins={n_rec}")
    sys.exit(1)
every_citation = got["revenue__sources~origins"] + got["receivables__sources~origins"]
if not all(o.startswith("JOURNAL.csv@") for o in every_citation):
    print(f"UNEXPECTED: citations do not name this dossier's data file: {every_citation}")
    sys.exit(1)

print(
    f"SECOND PROFILE END TO END: revenue={revenue} ({n_rev} origins), "
    f"receivables={receivables} ({n_rec} origin); all citations name JOURNAL.csv"
)
print("selected from the dossier's declared columns -- no subclass, no override")
sys.exit(0)
