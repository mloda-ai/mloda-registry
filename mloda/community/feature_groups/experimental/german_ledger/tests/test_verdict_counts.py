"""Verdict counts on refusal: a refused run still says how many rows got which verdict.

A refusal is the moment the numbers matter most, so it must not be the moment they are
dropped. Each admissibility refusal carries `verdicts` -- per row, how many were admitted,
outside the scope, refused or could not be evaluated -- and says it in its message. A failed
run reaches it by walking the exception chain.
"""

import shutil
from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest
from mloda.user import DataAccessCollection, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.experimental.german_ledger.policy import (
    AdmissibilityRefused,
    Festschreibung,
    LateEntryCutoff,
    LateEntryRefused,
    PeriodBound,
    apply_rules,
)
from mloda.community.feature_groups.experimental.german_ledger.reader import GdpduReader
from mloda.community.feature_groups.experimental.german_ledger.sources import InadmissibleTotal

from ._host import PLUGINS, _cause

DOSSIER_A = Path(__file__).parent / "fixtures" / "dossier_a"
CUTOFF = LateEntryCutoff(lock_date=date(2026, 1, 15), period_end=date(2025, 12, 31))
FY2025 = PeriodBound(start=date(2025, 1, 1), end=date(2025, 12, 31))


def _journal(**columns: list[object]) -> pa.Table:
    rows = len(next(iter(columns.values())))
    return pa.table({**columns, GdpduReader.ORIGIN_COLUMN: [f"row{i}" for i in range(rows)]})


def _dossier(tmp_path: Path, extra_line: str) -> Path:
    """dossier_a plus one ledger line."""
    folder = tmp_path / "dossier"
    shutil.copytree(DOSSIER_A, folder)
    with (folder / "GL.txt").open("a", encoding="cp1252") as f:
        f.write(extra_line + "\n")
    return folder


def _run(folder: Path) -> None:
    mloda.run_all(
        features=["revenue__sources"],
        compute_frameworks=[PyArrowTable],
        data_access_collection=DataAccessCollection(folders={str(folder)}),
        plugin_collector=PLUGINS,
    )


# --- the policy step -------------------------------------------------------------------------


def test_a_policy_refusal_counts_every_row() -> None:
    table = _journal(
        Buchungsdatum=[date(2025, 3, 1), date(2025, 3, 1), date(2026, 2, 1)],
        Erfassungsdatum=[date(2025, 3, 2), date(2026, 2, 1), date(2026, 2, 2)],
    )
    with pytest.raises(LateEntryRefused) as info:
        apply_rules((CUTOFF, FY2025), table)
    assert info.value.verdicts == {"admitted": 1, "outside-scope": 1, "refused": 1}
    assert "verdicts over 3 row(s): 1 admitted, 1 outside-scope, 1 refused" in str(info.value), str(info.value)


def test_rows_a_rule_could_not_judge_are_counted_as_unevaluated() -> None:
    """Festschreibung needs a column this journal lacks: no row is cleared, none is refused."""
    table = _journal(Buchungsdatum=[date(2025, 3, 1)] * 2, Erfassungsdatum=[date(2025, 3, 2)] * 2)
    with pytest.raises(AdmissibilityRefused) as info:
        apply_rules((CUTOFF, Festschreibung()), table)
    assert info.value.verdicts == {"unevaluated": 2}


def test_a_refused_row_outranks_an_unevaluated_one() -> None:
    table = _journal(Buchungsdatum=[date(2025, 3, 1), date(2025, 3, 1)], Erfassungsdatum=[date(2025, 3, 2), None])
    with pytest.raises(AdmissibilityRefused) as info:
        apply_rules((CUTOFF, Festschreibung()), table)
    assert info.value.verdicts == {"refused": 1, "unevaluated": 1}


# --- the total -------------------------------------------------------------------------------


def test_a_total_refused_for_an_outside_row_counts_its_verdicts(tmp_path: Path) -> None:
    """Booked after the period end: outside the scope, so no total -- and the count says why."""
    folder = _dossier(tmp_path, "8;4000;100,00;05.01.2026;06.01.2026;Umsatz Januar")
    with pytest.raises(Exception) as info:
        _run(folder)
    refusal = _cause(info.value, InadmissibleTotal)
    assert isinstance(refusal, InadmissibleTotal), info.value
    assert refusal.verdicts == {"admitted": 7, "outside-scope": 1}
    assert "7 admitted, 1 outside-scope" in str(refusal), str(refusal)


def test_a_late_entry_refusal_in_a_run_counts_its_verdicts(tmp_path: Path) -> None:
    folder = _dossier(tmp_path, "8;4000;100,00;20.12.2025;20.01.2026;nachgebucht")
    with pytest.raises(Exception) as info:
        _run(folder)
    refusal = _cause(info.value, LateEntryRefused)
    assert isinstance(refusal, LateEntryRefused), info.value
    assert refusal.verdicts == {"admitted": 7, "refused": 1}


def test_a_total_over_unstamped_rows_counts_them() -> None:
    """No policy stamped these rows: the refusal counts them as unstamped."""
    from decimal import Decimal

    from mloda.provider import FeatureSet
    from mloda.user import Feature

    from mloda.community.feature_groups.experimental.german_ledger.sources import SourcesFeatureGroup

    rows = pa.table(
        {
            "revenue~value": pa.array([Decimal("1.00")] * 3, type=pa.decimal128(38, 2)),
            "revenue~origins": ["GL.txt@4564dc0deef2:1", "GL.txt@4564dc0deef2:2", "GL.txt@4564dc0deef2:3"],
        }
    )
    fs = FeatureSet()
    fs.add(Feature("revenue__sources"))
    with pytest.raises(InadmissibleTotal) as info:
        SourcesFeatureGroup.calculate_feature(rows, fs)
    assert info.value.verdicts == {"unstamped": 3}
