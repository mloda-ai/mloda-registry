"""The composite policy: several admissibility rules, one step, one stamp.

Only one policy group may be live per process, so a host that needs more than the late-entry
cutoff -- a DATEV host needs Festschreibung and a period bound -- lists rules in RULES
instead of adding a second group. These tests call `apply_rules` directly: defining a second
configured policy subclass here would put two live policies into the process and break every
end-to-end test after it. The chain itself is proven in proofs/proof_composite_datev.py.
"""

import gc
import shutil
import subprocess  # nosec
import sys
from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest
from mloda.provider import FeatureSet

from mloda.community.feature_groups.german_ledger.datev import DatevExtfReader
from mloda.community.feature_groups.german_ledger.policy import (
    AdmissibilityPolicyGroup,
    AdmissibilityRefused,
    Festschreibung,
    LateEntryCutoff,
    LateEntryRefused,
    PeriodBound,
    apply_rules,
)
from mloda.community.feature_groups.german_ledger.reader import ADMISSIBILITY_COLUMN, GdpduReader, is_admitted

from ._host import TestClosing2025 as Closing2025

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
LEDERMANN = FIX / "datev_ledermann"
CUTOFF = LateEntryCutoff(lock_date=date(2026, 1, 15), period_end=date(2025, 12, 31))
FY2025 = PeriodBound(start=date(2025, 1, 1), end=date(2025, 12, 31))


def _dossier_a() -> pa.Table:
    return GdpduReader.load_data(str(FIX / "dossier_a"), FeatureSet())


def _stamps(table: pa.Table) -> list[str]:
    return list(table.column(ADMISSIBILITY_COLUMN).to_pylist())


def _journal(**columns: list[object]) -> pa.Table:
    rows = len(next(iter(columns.values())))
    return pa.table({**columns, GdpduReader.ORIGIN_COLUMN: [f"row{i}" for i in range(rows)]})


# --- one rule: nothing changes for today's hosts -------------------------------------------


def test_the_lock_date_shorthand_is_one_late_entry_rule() -> None:
    assert Closing2025.rules() == (CUTOFF,)


def test_one_rule_stamps_exactly_as_the_shorthand_policy_does() -> None:
    assert _stamps(apply_rules((CUTOFF,), _dossier_a())) == _stamps(Closing2025.clear(_dossier_a()))


# --- several rules: one combined stamp -----------------------------------------------------


def test_several_rules_write_one_combined_admission() -> None:
    stamps = set(_stamps(apply_rules((CUTOFF, FY2025), _dossier_a())))
    assert len(stamps) == 1, stamps
    stamp = stamps.pop()
    assert is_admitted(stamp), stamp
    assert stamp.startswith("admitted:all-of;rules=late-entry-cutoff,period-bound;"), stamp
    for expected in (
        "late-entry-cutoff.lock=2026-01-15",
        "late-entry-cutoff.period_end=2025-12-31",
        "period-bound.start=2025-01-01",
        "period-bound.end=2025-12-31",
    ):
        assert expected in stamp, (expected, stamp)


def test_a_row_one_rule_does_not_vouch_for_is_not_admitted() -> None:
    """Admitted by the cutoff, outside a 2025-H1 bound: the row is outside, and says whose."""
    h1 = PeriodBound(start=date(2025, 1, 1), end=date(2025, 6, 30))
    table = _journal(
        Buchungsdatum=[date(2025, 3, 1), date(2025, 9, 1)],
        Erfassungsdatum=[date(2025, 3, 2), date(2025, 9, 2)],
    )
    inside, outside = _stamps(apply_rules((CUTOFF, h1), table))
    assert is_admitted(inside), inside
    assert not is_admitted(outside), outside
    assert outside.startswith("outside-scope:all-of;rules=late-entry-cutoff,period-bound;outside=period-bound;"), (
        outside
    )


def test_every_rule_that_refuses_is_named_in_one_refusal() -> None:
    """The caller should see every problem at once, not fix one and meet the next."""
    table = _journal(
        Buchungsdatum=[date(2025, 3, 1), None],
        Erfassungsdatum=[date(2026, 2, 1), date(2025, 3, 2)],
    )
    with pytest.raises(AdmissibilityRefused) as info:
        apply_rules((CUTOFF, Festschreibung()), table)
    message = str(info.value)
    assert "late-entry cutoff" in message and "keyed in after 2026-01-15 (row0)" in message, message
    assert "carrying no booking date (row1)" in message, message
    assert "'Festschreibung'" in message, message


def test_a_lone_late_entry_refusal_keeps_its_own_type() -> None:
    table = _journal(Buchungsdatum=[date(2025, 3, 1)], Erfassungsdatum=[date(2026, 2, 1)])
    with pytest.raises(LateEntryRefused):
        apply_rules((CUTOFF, FY2025), table)


def test_two_rules_of_one_kind_are_refused() -> None:
    """Their parameters would share keys in the stamp; which cutoff was meant is unclear."""
    with pytest.raises(ValueError, match="late-entry-cutoff"):
        apply_rules((CUTOFF, LateEntryCutoff(lock_date=date(2026, 3, 31), period_end=date(2025, 12, 31))), _dossier_a())


def test_no_rules_is_refused() -> None:
    with pytest.raises(AdmissibilityRefused, match="no admissibility rule"):
        apply_rules((), _dossier_a())


def test_a_host_sets_either_rules_or_the_lock_date_not_both() -> None:
    with pytest.raises(TypeError, match="RULES"):

        class Ambiguous(AdmissibilityPolicyGroup):
            RULES = (FY2025,)
            LOCK_DATE = date(2026, 1, 15)
            PERIOD_END = date(2025, 12, 31)

    # The TypeError is raised after Python created the class, and its traceback kept it alive.
    # mloda finds groups through __subclasses__(), so until it is collected it is a second
    # live policy for every end-to-end test that runs later in this process.
    gc.collect()
    assert "Ambiguous" not in {c.__name__ for c in AdmissibilityPolicyGroup.__subclasses__()}


# --- Festschreibung ------------------------------------------------------------------------


def _ledermann(tmp_path: Path, locked: bool) -> pa.Table:
    """ledermann's header says Festschreibung 0; the locked twin flips header field 21."""
    folder = tmp_path / ("locked" if locked else "open")
    shutil.copytree(LEDERMANN, folder)
    batch = folder / "EXTF_Buchungsstapel.csv"
    raw = batch.read_bytes()
    if locked:
        head, rest = raw.split(b"\n", 1)
        fields = head.split(b";")
        assert fields[20] == b"0", fields[20]
        fields[20] = b"1"
        raw = b";".join(fields) + b"\n" + rest
        batch.write_bytes(raw)
    return DatevExtfReader.load_data(str(folder), FeatureSet())


def test_a_locked_datev_batch_is_admitted_by_festschreibung(tmp_path: Path) -> None:
    stamps = set(_stamps(apply_rules((Festschreibung(),), _ledermann(tmp_path, locked=True))))
    assert stamps == {"admitted:festschreibung;column=Festschreibung"}, stamps


def test_an_open_datev_batch_is_outside_the_lock(tmp_path: Path) -> None:
    """Not refused: an open batch is not wrong, but a lock rule cannot vouch for it."""
    stamps = set(_stamps(apply_rules((Festschreibung(),), _ledermann(tmp_path, locked=False))))
    assert stamps == {"outside-scope:festschreibung;column=Festschreibung"}, stamps


def test_festschreibung_fails_closed_on_a_missing_or_unset_flag() -> None:
    with pytest.raises(AdmissibilityRefused, match="carries no 'Festschreibung'"):
        apply_rules((Festschreibung(),), _dossier_a())
    with pytest.raises(AdmissibilityRefused, match=r"carrying no Festschreibung flag \(row1\)"):
        apply_rules((Festschreibung(),), _journal(Festschreibung=[True, None]))


# --- period bound --------------------------------------------------------------------------


def test_the_period_bound_admits_inside_and_scopes_out_the_rest() -> None:
    table = _journal(Buchungsdatum=[date(2024, 12, 31), date(2025, 1, 1), date(2025, 12, 31), date(2026, 1, 1)])
    verdicts = [is_admitted(s) for s in _stamps(apply_rules((FY2025,), table))]
    assert verdicts == [False, True, True, False]


def test_the_period_bound_refuses_an_unbooked_line() -> None:
    with pytest.raises(AdmissibilityRefused, match=r"carrying no booking date \(row0\)"):
        apply_rules((FY2025,), _journal(Buchungsdatum=[None]))


def test_an_inverted_period_is_refused_where_it_is_written() -> None:
    with pytest.raises(ValueError, match="after"):
        PeriodBound(start=date(2025, 12, 31), end=date(2025, 1, 1))


# --- end to end ----------------------------------------------------------------------------


def test_a_datev_host_runs_its_rules_through_the_one_policy_slot() -> None:
    """Its own process: the DATEV host's policy registers process-wide, and one may be live."""
    # Runs our own proof script, not external input.
    script = str(HERE / "proofs" / "proof_composite_datev.py")
    proof = subprocess.run([sys.executable, script], capture_output=True, text=True)  # nosec B603
    assert proof.returncode == 0, proof.stdout + proof.stderr[-2000:]
    assert "OPEN BATCH:   InadmissibleTotal" in proof.stdout, proof.stdout
    assert "LOCKED BATCH: admitted, then GrossRevenueRefused" in proof.stdout, proof.stdout
    assert "NET BATCH:    revenue 5950.00 cited to EXTF_Buchungsstapel.csv@" in proof.stdout, proof.stdout
