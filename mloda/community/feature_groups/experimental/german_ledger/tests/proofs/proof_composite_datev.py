"""Isolated: a DATEV host's composite policy, end to end, through the one policy slot.

A subprocess because the host's policy registers process-wide, and exactly one may be live.

The point: DATEV carries no Erfassungsdatum, so the late-entry cutoff refuses every DATEV
journal. A DATEV host lists the rules it can evaluate instead -- a period bound and the
Festschreibung -- in ONE policy group, and they write one combined stamp:
  - ledermann as shipped (Festschreibung 0): outside the lock, so no number;
  - the same batch locked (header field 21 = 1): admitted by both rules, one stamp per leg,
    and then refused by the SKR03 concept because 8400 is booked gross;
  - locked with BU key 40 on that row (automatic lifted): a cited total.
"""

import shutil
import sys
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

from mloda.provider import FeatureSet
from mloda.user import DataAccessCollection, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.experimental.german_ledger.datev import DatevExtfReader
from mloda.community.feature_groups.experimental.german_ledger.policy import (
    AdmissibilityPolicyGroup,
    Festschreibung,
    PeriodBound,
)
from mloda.community.feature_groups.experimental.german_ledger.reader import ADMISSIBILITY_COLUMN, is_admitted
from mloda.community.feature_groups.experimental.german_ledger.skr import GrossRevenueRefused, SkrAccountFeatureGroup
from mloda.community.feature_groups.experimental.german_ledger.sources import (  # noqa: F401
    InadmissibleTotal,
    SourcesFeatureGroup,
)

LEDERMANN = Path(__file__).parent.parent / "fixtures" / "datev_ledermann"


class DatevClosing2018(AdmissibilityPolicyGroup):
    """This host's policy: FY2018, and only what the Festschreibung vouches for."""

    RULES = (PeriodBound(date(2018, 1, 1), date(2018, 12, 31)), Festschreibung())


# ledermann's header leaves SKR empty; the books are SKR03, and the host says so.
SkrAccountFeatureGroup.CHART = "03"


def _run(folder: Path, feature: str) -> list[dict[str, list[object]]]:
    results = mloda.run_all(
        features=[feature],
        compute_frameworks=[PyArrowTable],
        data_access_collection=DataAccessCollection(folders={str(folder)}),
    )
    return [t.to_pydict() for t in results]


def _root(e: BaseException) -> BaseException:
    while (deeper := e.__cause__ or e.__context__) is not None:
        e = deeper
    return e


def _copy(src: Path, dst: Path, *, locked: bool, bu_8400: bytes | None = None) -> Path:
    shutil.copytree(src, dst)
    batch = dst / "EXTF_Buchungsstapel.csv"
    lines = batch.read_bytes().split(b"\n")
    if locked:
        head = lines[0].split(b";")
        head[20] = b"1"
        lines[0] = b";".join(head)
    if bu_8400 is not None:
        row = lines[3].split(b";")
        row[8] = bu_8400
        lines[3] = b";".join(row)
    batch.write_bytes(b"\n".join(lines))
    return dst


with tempfile.TemporaryDirectory() as tmp:
    shipped = _copy(LEDERMANN, Path(tmp) / "shipped", locked=False)
    locked = _copy(LEDERMANN, Path(tmp) / "locked", locked=True)
    net = _copy(LEDERMANN, Path(tmp) / "net", locked=True, bu_8400=b'"40"')

    try:
        _run(shipped, "revenue__sources")
    except Exception as e:
        root = _root(e)
        if not isinstance(root, InadmissibleTotal) or "outside-scope:all-of" not in str(root):
            print(f"UNEXPECTED on the open batch: {type(root).__name__}: {str(root)[:200]}")
            sys.exit(1)
        print("OPEN BATCH:   InadmissibleTotal (outside-scope:all-of ... outside=festschreibung)")
    else:
        print("UNEXPECTED: an open batch produced a total")
        sys.exit(1)

    # Through the chain: the policy admits every leg, and the SKR03 concept then refuses the
    # 8400 leg by name, because DATEV books an automatic account gross.
    try:
        _run(locked, "revenue__sources")
    except Exception as e:
        root = _root(e)
        if not isinstance(root, GrossRevenueRefused) or "automatic account 8400" not in str(root):
            print(f"UNEXPECTED on the locked batch: {type(root).__name__}: {str(root)[:200]}")
            sys.exit(1)
        print("LOCKED BATCH: admitted, then GrossRevenueRefused (automatic account 8400)")
    else:
        print("UNEXPECTED: a gross automatic-account booking produced a total")
        sys.exit(1)

    # BU 40 lifts the automatic: the booked amount is net, and the chain yields a cited total.
    # Haben on the Gegenkonto, so the leg is -5950; revenue is reported credit-positive.
    got = {k: v for t in _run(net, "revenue__sources") for k, v in t.items()}
    value, origins = got["revenue__sources~value"], got["revenue__sources~origins"]
    cited = origins[0] if len(origins) == 1 and isinstance(origins[0], list) else []
    if value != [Decimal("5950.00")] or len(cited) != 1 or not str(cited[0]).endswith(":4/G"):
        print(f"UNEXPECTED on the net batch: {got}")
        sys.exit(1)
    print(f"NET BATCH:    revenue {value[0]} cited to {cited[0]}")

    # The stamp itself: core returns only the requested feature's columns, so read it from the
    # policy's own step on the same rows.
    stamps = set(
        DatevClosing2018.clear(DatevExtfReader.load_data(str(locked), FeatureSet()))
        .column(ADMISSIBILITY_COLUMN)
        .to_pylist()
    )
    stamp = stamps.pop() if len(stamps) == 1 else None
    if (
        stamp is None
        or not is_admitted(stamp)
        or not stamp.startswith("admitted:all-of;rules=period-bound,festschreibung;")
    ):
        print(f"UNEXPECTED stamps on the locked batch: {stamps or stamp}")
        sys.exit(1)
    print(f"              every leg carries one combined admission: {stamp}")

print("COMPOSITE POLICY: several rules, one policy slot, one stamp")
sys.exit(0)
