"""Isolated: one host reads GDPdU and DATEV, each judged by the rules its format can carry.

A subprocess because the host's policy registers process-wide, and exactly one may be live.

The late-entry cutoff needs an Erfassungsdatum (GDPdU has one, DATEV does not); Festschreibung
needs a lock flag (DATEV has one, GDPdU does not). Scoped with ForFormat, both live in the one
policy beside a period bound that applies to every row. The twin books (fixtures/twin_2025)
then give the same total from both formats -- once the DATEV batch is locked; open, it is
outside the Festschreibung, and no number comes back.
"""

import shutil
import sys
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

from mloda.user import DataAccessCollection, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.german_ledger.policy import (
    AdmissibilityPolicyGroup,
    Festschreibung,
    ForFormat,
    LateEntryCutoff,
    PeriodBound,
)
from mloda.community.feature_groups.german_ledger.skr import SkrAccountFeatureGroup
from mloda.community.feature_groups.german_ledger.sources import InadmissibleTotal, SourcesFeatureGroup  # noqa: F401

TWIN = Path(__file__).parent.parent / "fixtures" / "twin_2025"


class MixedClosing2025(AdmissibilityPolicyGroup):
    """FY2025 for a host that receives both formats."""

    RULES = (
        ForFormat("gdpdu", LateEntryCutoff(lock_date=date(2026, 1, 15), period_end=date(2025, 12, 31))),
        ForFormat("datev", Festschreibung()),
        PeriodBound(date(2025, 1, 1), date(2025, 12, 31)),
    )


SkrAccountFeatureGroup.CHART = "04"


def _revenue(folder: Path) -> object:
    results = mloda.run_all(
        features=["revenue__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(folder)}),
    )
    [value] = [v for t in results for v in t.column("revenue__sources~value").to_pylist()]
    return value


def _root(e: BaseException) -> BaseException:
    while (deeper := e.__cause__ or e.__context__) is not None:
        e = deeper
    return e


with tempfile.TemporaryDirectory() as tmp:
    locked = Path(tmp) / "datev"
    shutil.copytree(TWIN / "datev", locked)
    batch = locked / "EXTF_Buchungsstapel.csv"
    head, rest = batch.read_bytes().split(b"\r\n", 1)
    fields = head.split(b";")
    fields[20] = b"1"
    batch.write_bytes(b";".join(fields) + b"\r\n" + rest)

    gdpdu = _revenue(TWIN / "gdpdu")
    datev = _revenue(locked)
    print(f"GDPdU:  revenue {gdpdu}  (late-entry cutoff + period bound)")
    print(f"DATEV:  revenue {datev}  (Festschreibung + period bound)")
    if not gdpdu == datev == Decimal("22485.06"):
        sys.exit(1)

    try:
        _revenue(TWIN / "datev")
    except Exception as e:
        root = _root(e)
        if not isinstance(root, InadmissibleTotal) or "outside=datev.festschreibung" not in str(root):
            print(f"UNEXPECTED on the open batch: {type(root).__name__}: {str(root)[:200]}")
            sys.exit(1)
        print("OPEN DATEV: InadmissibleTotal (outside=datev.festschreibung)")
    else:
        print("UNEXPECTED: an open DATEV batch produced a total")
        sys.exit(1)

print("FORMAT RULES: one host, both formats, each judged by what it carries")
sys.exit(0)
