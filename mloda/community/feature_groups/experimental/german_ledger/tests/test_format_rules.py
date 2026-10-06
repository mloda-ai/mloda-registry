"""Rules per format: one host that reads GDPdU and DATEV, each judged by what it carries.

The late-entry cutoff needs an Erfassungsdatum that DATEV does not carry, and Festschreibung
needs a flag that GDPdU does not carry. A host that reads both lists each rule for its format,
`ForFormat("gdpdu", LateEntryCutoff(...))` and `ForFormat("datev", Festschreibung())`. Such a
rule judges only its own format's rows and does not apply to the others; a row that no rule
applies to is refused, because nothing vouched for it. The format of a row is read from its
citation: a DATEV citation always names a leg, a GDPdU citation never does.

These tests call `apply_rules` directly; the chain is proven in proofs/proof_format_rules.py.
"""

import shutil
import subprocess  # nosec
import sys
from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest
from mloda.provider import FeatureSet

from mloda.community.feature_groups.experimental.german_ledger.datev import DatevExtfReader
from mloda.community.feature_groups.experimental.german_ledger.policy import (
    AdmissibilityRefused,
    Festschreibung,
    ForFormat,
    LateEntryCutoff,
    PeriodBound,
    apply_rules,
)
from mloda.community.feature_groups.experimental.german_ledger.reader import (
    ADMISSIBILITY_COLUMN,
    GdpduReader,
    is_admitted,
)

from ._host import TestClosing2025 as Closing2025

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
CUTOFF = LateEntryCutoff(lock_date=date(2026, 1, 15), period_end=date(2025, 12, 31))
MIXED = (ForFormat("gdpdu", CUTOFF), ForFormat("datev", Festschreibung()))


def _gdpdu() -> pa.Table:
    return GdpduReader.load_data(str(FIX / "dossier_a"), FeatureSet())


def _datev(tmp_path: Path, *, locked: bool) -> pa.Table:
    folder = tmp_path / "batch"
    shutil.copytree(FIX / "datev_ledermann", folder)
    batch = folder / "EXTF_Buchungsstapel.csv"
    head, rest = batch.read_bytes().split(b"\n", 1)
    fields = head.split(b";")
    fields[20] = b"1" if locked else b"0"
    batch.write_bytes(b";".join(fields) + b"\n" + rest)
    return DatevExtfReader.load_data(str(folder), FeatureSet())


def _stamps(table: pa.Table) -> set[str]:
    return set(table.column(ADMISSIBILITY_COLUMN).to_pylist())


def test_a_gdpdu_journal_is_judged_by_its_own_rule_only() -> None:
    """No Festschreibung column, and no refusal for it: that rule does not apply here."""
    [stamp] = _stamps(apply_rules(MIXED, _gdpdu()))
    assert is_admitted(stamp), stamp
    assert stamp.startswith("admitted:all-of;rules=gdpdu.late-entry-cutoff,datev.festschreibung;"), stamp


def test_a_datev_journal_is_judged_by_its_own_rule_only(tmp_path: Path) -> None:
    """No Erfassungsdatum, and no refusal for it: the cutoff does not apply to DATEV rows."""
    [stamp] = _stamps(apply_rules(MIXED, _datev(tmp_path, locked=True)))
    assert is_admitted(stamp), stamp


def test_an_open_datev_batch_is_outside_its_rule(tmp_path: Path) -> None:
    [stamp] = _stamps(apply_rules(MIXED, _datev(tmp_path, locked=False)))
    assert not is_admitted(stamp)
    assert "outside=datev.festschreibung" in stamp, stamp


def test_an_unscoped_rule_still_judges_every_format(tmp_path: Path) -> None:
    fy2018 = PeriodBound(date(2018, 1, 1), date(2018, 12, 31))
    [stamp] = _stamps(apply_rules((*MIXED, fy2018), _datev(tmp_path, locked=True)))
    assert is_admitted(stamp) and "period-bound.start=2018-01-01" in stamp, stamp


def test_a_row_no_rule_applies_to_is_refused() -> None:
    """Only a DATEV rule, a GDPdU journal: nothing vouched for these rows, so none is admitted."""
    with pytest.raises(AdmissibilityRefused, match="no rule applies") as info:
        apply_rules((ForFormat("datev", Festschreibung()),), _gdpdu())
    assert info.value.verdicts == {"unevaluated": 7}


def test_the_inner_rule_still_refuses_its_own_rows() -> None:
    late = _gdpdu()
    keyed = late.column("Erfassungsdatum").to_pylist()
    keyed[0] = date(2026, 2, 1)
    late = late.set_column(late.column_names.index("Erfassungsdatum"), "Erfassungsdatum", pa.array(keyed))
    with pytest.raises(AdmissibilityRefused, match="keyed in after 2026-01-15"):
        apply_rules(MIXED, late)


def test_an_unknown_format_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="'csv'"):
        ForFormat("csv", CUTOFF)


def test_a_rule_scoped_twice_to_one_format_is_one_kind() -> None:
    with pytest.raises(ValueError, match="gdpdu.late-entry-cutoff"):
        apply_rules((ForFormat("gdpdu", CUTOFF), ForFormat("gdpdu", CUTOFF)), _gdpdu())


def test_one_host_reads_both_formats_end_to_end() -> None:
    """Its own process: the mixed host's policy registers process-wide."""
    script = str(HERE / "proofs" / "proof_format_rules.py")
    proof = subprocess.run([sys.executable, script], capture_output=True, text=True)  # nosec B603
    assert proof.returncode == 0, proof.stdout + proof.stderr[-2000:]
    assert "GDPdU:  revenue 22485.06" in proof.stdout, proof.stdout
    assert "DATEV:  revenue 22485.06" in proof.stdout, proof.stdout
    assert "OPEN DATEV: InadmissibleTotal" in proof.stdout, proof.stdout


def test_a_policy_with_scoped_rules_declares_the_stamp_every_admitted_row_carries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RULES swapped on the one live policy for this test; a second subclass would leak."""
    monkeypatch.setattr(Closing2025, "RULES", MIXED)
    [stamp] = _stamps(Closing2025.clear(_gdpdu()))
    assert stamp.startswith("admitted:all-of;rules=gdpdu.late-entry-cutoff,datev.festschreibung;"), stamp
    assert Closing2025.declared_attributes(None) == {"policy.verdict": stamp}
    assert _stamps(Closing2025.clear(_datev(tmp_path, locked=True))) == {stamp}
