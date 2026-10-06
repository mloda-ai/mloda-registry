"""Two charts of accounts, one concept: SKR03 beside SKR04, and the two refusals they need.

The same account number means different things in the two charts. 1200 is Bank in SKR03 and
Forderungen in SKR04, so which chart applies is the first fact a concept needs. The host
states it (`SkrAccountFeatureGroup.CHART`). A DATEV header may also state it (field 27), and
when both speak and disagree, the concept refuses instead of picking one.

On a DATEV source, a revenue leg on an automatic account (SKR03 8400 = Erlöse 19 % USt) is
booked GROSS: DATEV splits the tax out on its own. Summing it would report VAT as revenue,
so the concept refuses it by name until a tax-key table exists.

These tests call `_map_accounts` on the reader's rows directly; the chain is proven in
proofs/proof_composite_datev.py.
"""

import json
import shutil
import subprocess  # nosec
import sys
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from mloda.provider import FeatureSet
from mloda.user import Feature

from mloda.community.feature_groups.experimental.german_ledger import skr
from mloda.community.feature_groups.experimental.german_ledger.datev import SOLL_POSITIVE, VORZEICHEN, DatevExtfReader
from mloda.community.feature_groups.experimental.german_ledger.reader import ADMISSIBILITY_COLUMN, GdpduReader
from mloda.community.feature_groups.experimental.german_ledger.skr import (
    SKR03_2025,
    SKR04_2025,
    AccountLengthUnsupported,
    ChartConflict,
    GrossRevenueRefused,
    LedgerProfile,
    SkrAccountFeatureGroup,
)

LEDERMANN = Path(__file__).parent / "fixtures" / "datev_ledermann"
TWIN = Path(__file__).parent / "fixtures" / "twin_2025"
DOSSIER_A = Path(__file__).parent / "fixtures" / "dossier_a"


@pytest.fixture
def chart() -> Iterator[None]:
    """Restore the host's chart after each test; it is process-wide host configuration."""
    original = SkrAccountFeatureGroup.CHART
    yield
    SkrAccountFeatureGroup.CHART = original


@pytest.fixture
def catalogue() -> Iterator[None]:
    """Restore the SKR04 catalogue after a test edits it; it is process-wide."""
    original = dict(skr.CHARTS)
    yield
    skr.CHARTS.clear()
    skr.CHARTS.update(original)


def _features(*names: str) -> FeatureSet:
    fs = FeatureSet()
    for n in names:
        fs.add(Feature(n))
    return fs


def _ledermann(
    tmp_path: Path, *, skr: bytes | None = None, bu_8400: bytes | None = None, length: bytes | None = None
) -> pa.Table:
    """The ledermann batch, optionally with header field 27 (SKR), field 14 (Sachkontenlänge)
    or row 2's BU key changed."""
    folder = tmp_path / "batch"
    shutil.copytree(LEDERMANN, folder)
    batch = folder / "EXTF_Buchungsstapel.csv"
    lines = batch.read_bytes().split(b"\n")
    if skr is not None:
        head = lines[0].split(b";")
        head[26] = skr
        lines[0] = b";".join(head)
    if length is not None:
        head = lines[0].split(b";")
        head[13] = length
        lines[0] = b";".join(head)
    if bu_8400 is not None:
        row = lines[3].split(b";")
        assert row[7] == b"8400", row[:9]
        row[8] = bu_8400
        lines[3] = b";".join(row)
    batch.write_bytes(b"\n".join(lines))
    return DatevExtfReader.load_data(str(folder), FeatureSet())


def _values(mapped: pa.Table, name: str) -> list[tuple[object, object]]:
    pairs = zip(mapped.column(f"{name}~value").to_pylist(), mapped.column(f"{name}~origins").to_pylist())
    return [(v, o) for v, o in pairs if o is not None]


def _basis(concept: str) -> dict[str, Any]:
    mapped = SkrAccountFeatureGroup._map_accounts(
        GdpduReader.load_data(str(DOSSIER_A), FeatureSet()), _features(concept)
    )
    [basis] = set(mapped.column(f"{concept}~basis").to_pylist())
    parsed: dict[str, Any] = json.loads(basis)
    return parsed


# --- the concept's basis ---------------------------------------------------------------------


def test_a_concept_states_the_basis_it_was_computed_on() -> None:
    basis = _basis("revenue")
    assert basis["profile"] == {"account_column": "Konto", "amount_column": "Betrag", "sign": None}
    assert basis["chart"] == "SKR04"
    assert basis["catalogue"]["name"] == "SKR04_2025"
    assert basis["catalogue"]["accounts"] == [[4000, 4499]]
    assert basis["sign"] == "as-declared"
    assert len(basis["catalogue"]["fingerprint"]) == 12


def test_a_changed_catalogue_changes_the_basis(catalogue: None) -> None:
    before = _basis("revenue")["catalogue"]
    skr.CHARTS["04"] = {**skr.SKR04_2025, "revenue": range(4000, 4800)}
    after = _basis("revenue")["catalogue"]
    assert after["accounts"] == [[4000, 4799]]
    assert after["fingerprint"] != before["fingerprint"]


# --- the declared attributes: what is fixed before any row is read ---------------------------------


@pytest.mark.parametrize("features", [None, _features("revenue")], ids=["no-features", "features"])
def test_the_declared_attributes_name_the_chart_and_catalogue(features: FeatureSet | None) -> None:
    declared = SkrAccountFeatureGroup.declared_attributes(features)
    assert set(declared) == {"chart", "catalogue", "catalogue.fingerprint"}
    assert declared["chart"] == "SKR04"
    assert declared["catalogue"] == "SKR04_2025"


def test_the_declared_attributes_follow_the_hosts_chart(chart: None) -> None:
    SkrAccountFeatureGroup.CHART = "03"
    declared = SkrAccountFeatureGroup.declared_attributes(None)
    assert (declared["chart"], declared["catalogue"]) == ("SKR03", "SKR03_2025")


def test_the_declared_fingerprint_is_the_one_in_the_basis() -> None:
    declared = SkrAccountFeatureGroup.declared_attributes(None)
    assert declared["catalogue.fingerprint"] == _basis("revenue")["catalogue"]["fingerprint"]
    assert declared["chart"] == _basis("revenue")["chart"]


def test_a_changed_catalogue_changes_the_declared_fingerprint(catalogue: None) -> None:
    before = SkrAccountFeatureGroup.declared_attributes(None)["catalogue.fingerprint"]
    skr.CHARTS["04"] = {**skr.SKR04_2025, "revenue": range(4000, 4800)}
    after = SkrAccountFeatureGroup.declared_attributes(None)["catalogue.fingerprint"]
    assert after != before
    assert after == _basis("revenue")["catalogue"]["fingerprint"]


# --- the catalogue ---------------------------------------------------------------------------


def test_the_two_charts_answer_the_same_concepts() -> None:
    assert set(SKR03_2025) == set(SKR04_2025) == SkrAccountFeatureGroup.feature_names_supported()


def test_1200_is_receivables_only_in_skr04() -> None:
    """The classic mix-up, pinned in both directions."""
    assert 1200 in SKR04_2025["receivables"] and 1200 not in SKR03_2025["receivables"]
    assert 1400 in SKR03_2025["receivables"] and 1400 not in SKR04_2025["receivables"]
    assert 8400 in SKR03_2025["revenue"] and 8400 not in SKR04_2025["revenue"]


def test_an_unknown_host_chart_is_a_configuration_error(chart: None, tmp_path: Path) -> None:
    SkrAccountFeatureGroup.CHART = "05"
    with pytest.raises(ValueError, match="'05'"):
        SkrAccountFeatureGroup._map_accounts(_ledermann(tmp_path), _features("receivables"))


# --- header vs host ----------------------------------------------------------------------------


def test_an_empty_header_leaves_the_chart_to_the_host(chart: None, tmp_path: Path) -> None:
    """ledermann's header names no SKR. The host's word is the only one, so it decides.

    Under SKR03, the 1200 leg is Bank and no receivable. Under SKR04, the same leg would be
    a receivable, which is why the host must state its chart.
    """
    table = _ledermann(tmp_path)
    SkrAccountFeatureGroup.CHART = "03"
    assert _values(SkrAccountFeatureGroup._map_accounts(table, _features("receivables")), "receivables") == []
    SkrAccountFeatureGroup.CHART = "04"
    got = _values(SkrAccountFeatureGroup._map_accounts(table, _features("receivables")), "receivables")
    assert [o for _, o in got] == [o for o in table.column("__origin").to_pylist() if o.endswith(":3/K")]


def test_a_header_chart_that_contradicts_the_host_is_refused(chart: None, tmp_path: Path) -> None:
    table = _ledermann(tmp_path, skr=b'"04"')
    SkrAccountFeatureGroup.CHART = "03"
    with pytest.raises(ChartConflict, match="SKR04.*SKR03"):
        SkrAccountFeatureGroup._map_accounts(table, _features("receivables"))


def test_a_header_chart_that_agrees_with_the_host_is_read(chart: None, tmp_path: Path) -> None:
    table = _ledermann(tmp_path, skr=b'"03"')
    SkrAccountFeatureGroup.CHART = "03"
    mapped = SkrAccountFeatureGroup._map_accounts(table, _features("receivables"))
    assert _values(mapped, "receivables") == []


# --- Sachkontenlänge -----------------------------------------------------------------------------


def test_a_sachkonto_longer_than_the_catalogue_is_refused_not_guessed(chart: None, tmp_path: Path) -> None:
    """Sachkontenlänge 5 makes 10000 a five-digit Sachkonto. The catalogue states 4-digit SKR
    accounts, and how a longer Sachkonto maps onto them is not modelled, so the concept
    refuses by name rather than return an unattested null or a guessed mapping."""
    SkrAccountFeatureGroup.CHART = "03"
    with pytest.raises(AccountLengthUnsupported, match="10000"):
        SkrAccountFeatureGroup._map_accounts(_ledermann(tmp_path, length=b"5"), _features("receivables"))


def test_a_long_personenkonto_is_not_a_sachkonto_and_passes(chart: None, tmp_path: Path) -> None:
    """With Sachkontenlänge 4, 10000 is a Debitor: not in any SKR range, and no refusal."""
    SkrAccountFeatureGroup.CHART = "03"
    SkrAccountFeatureGroup._map_accounts(_ledermann(tmp_path), _features("receivables"))


# --- gross automatic-account revenue -------------------------------------------------------------


def test_revenue_on_an_automatic_account_is_refused_as_gross(chart: None, tmp_path: Path) -> None:
    """ledermann's 5950,00 on 8400 includes 19 % VAT. It is refused, not totalled as revenue."""
    SkrAccountFeatureGroup.CHART = "03"
    with pytest.raises(GrossRevenueRefused, match=r"gross automatic-account booking.*:4/G.*8400"):
        SkrAccountFeatureGroup._map_accounts(_ledermann(tmp_path), _features("revenue"))


def test_bu_key_40_lifts_the_automatic_so_the_leg_is_net(chart: None, tmp_path: Path) -> None:
    """BU 40 (Aufhebung der Automatik): DATEV splits no tax out, so the booked amount stands.

    The revenue leg is the Gegenkonto, on the Haben side, so the reader signs it -5950. The
    concept reports revenue credit-positive, so the same leg reads +5950 as revenue.
    """
    SkrAccountFeatureGroup.CHART = "03"
    mapped = SkrAccountFeatureGroup._map_accounts(_ledermann(tmp_path, bu_8400=b'"40"'), _features("revenue"))
    [(value, origin)] = _values(mapped, "revenue")
    assert value == Decimal("5950.00")
    assert str(origin).endswith(":4/G"), origin


def test_any_other_bu_key_on_a_revenue_leg_is_refused_as_gross(chart: None, tmp_path: Path) -> None:
    """A tax key asks DATEV to split tax out of the booked amount, on any account."""
    SkrAccountFeatureGroup.CHART = "03"
    with pytest.raises(GrossRevenueRefused, match="BU key '3'"):
        SkrAccountFeatureGroup._map_accounts(_ledermann(tmp_path, bu_8400=b'"3"'), _features("revenue"))


def test_receivables_are_gross_by_nature_and_not_refused(chart: None, tmp_path: Path) -> None:
    """A receivable includes the VAT owed, so the gross check belongs to revenue alone."""
    SkrAccountFeatureGroup.CHART = "04"
    got = _values(SkrAccountFeatureGroup._map_accounts(_ledermann(tmp_path), _features("receivables")), "receivables")
    assert len(got) == 1


def test_a_gross_leg_outside_the_policy_is_left_to_the_aggregation(chart: None) -> None:
    """Admissibility comes first: a row no policy admitted is refused by the total, not here."""
    SkrAccountFeatureGroup.CHART = "03"
    table = pa.table(
        {
            "Konto": ["8400"],
            "Betrag": pa.array([Decimal("-5950.00")], type=pa.decimal128(38, 2)),
            "BU-Schlüssel": [""],
            "SKR": [""],
            "__origin": ["EXTF_x.csv@abc:4/G"],
            ADMISSIBILITY_COLUMN: ["outside-scope:festschreibung"],
        }
    )
    mapped = SkrAccountFeatureGroup._map_accounts(table, _features("revenue"))
    assert mapped.column(ADMISSIBILITY_COLUMN).to_pylist() == ["outside-scope:festschreibung"]


def test_a_gdpdu_journal_carries_no_bu_key_and_is_not_checked(chart: None) -> None:
    """GDPdU lines are booked as declared; SKR04 4400 there is a net revenue line."""
    table = pa.table(
        {
            "Konto": ["4400"],
            "Betrag": pa.array([Decimal("100.00")], type=pa.decimal128(38, 2)),
            "__origin": ["GL.txt@abc:1"],
        }
    )
    mapped = SkrAccountFeatureGroup._map_accounts(table, _features("revenue"))
    assert _values(mapped, "revenue") == [(Decimal("100.00"), "GL.txt@abc:1")]


# --- sign convention -----------------------------------------------------------------------------
#
# A concept reports in its natural direction: revenue credit-positive, receivables
# debit-positive. A DATEV journal states that it is signed +Soll/-Haben (`Vorzeichen`); a GDPdU
# journal states nothing, and its amounts are taken as declared, which is how the stock
# Sachkonten export writes them (revenue lines positive).


def test_the_datev_reader_states_its_sign_convention(tmp_path: Path) -> None:
    assert set(_ledermann(tmp_path).column(VORZEICHEN).to_pylist()) == {SOLL_POSITIVE}


def test_a_debit_positive_receivable_keeps_its_sign(chart: None, tmp_path: Path) -> None:
    """1200 under SKR04 is a receivable, booked 24,95 Haben: a receivable reduced, so -24.95."""
    SkrAccountFeatureGroup.CHART = "04"
    [(value, _)] = _values(
        SkrAccountFeatureGroup._map_accounts(_ledermann(tmp_path), _features("receivables")), "receivables"
    )
    assert value == Decimal("-24.95")


def test_an_unknown_sign_convention_is_refused(chart: None) -> None:
    table = pa.table(
        {
            "Konto": ["4400"],
            "Betrag": pa.array([Decimal("100.00")], type=pa.decimal128(38, 2)),
            VORZEICHEN: ["haben-positiv"],
            "__origin": ["x@abc:1"],
        }
    )
    with pytest.raises(ValueError, match="'haben-positiv'"):
        SkrAccountFeatureGroup._map_accounts(table, _features("revenue"))


# --- the twin fixture ---------------------------------------------------------------------------


def test_the_same_books_in_both_formats_give_the_same_cited_totals() -> None:
    """Its own process: the twin host's policy registers process-wide, and one may be live."""
    # Runs our own proof script, not external input.
    script = str(Path(__file__).parent / "proofs" / "proof_twin.py")
    proof = subprocess.run([sys.executable, script], capture_output=True, text=True)  # nosec B603
    assert proof.returncode == 0, proof.stdout + proof.stderr[-2000:]
    assert "SAME revenue: GDPdU 22485.06 / DATEV 22485.06" in proof.stdout, proof.stdout
    assert "SAME receivables: GDPdU 7500.00 / DATEV 7500.00" in proof.stdout, proof.stdout


def test_the_twin_fixture_is_what_its_generator_writes(tmp_path: Path) -> None:
    """The checked-in files are reproducible: a hand edit would break the twin silently."""
    sys.path.insert(0, str(TWIN))
    try:
        import make_twin
    finally:
        sys.path.remove(str(TWIN))
    make_twin.write_gdpdu(tmp_path / "gdpdu")
    make_twin.write_datev(tmp_path / "datev")
    for rel in ("gdpdu/GL.txt", "gdpdu/index.xml", "datev/EXTF_Buchungsstapel.csv"):
        assert (tmp_path / rel).read_bytes() == (TWIN / rel).read_bytes(), rel


# --- a stated sign convention for a GDPdU exporter ------------------------------------------------


def _gdpdu_lines(*amounts: str) -> pa.Table:
    return pa.table(
        {
            "Konto": ["4400"] * len(amounts),
            "Betrag": pa.array([Decimal(a) for a in amounts], type=pa.decimal128(38, 2)),
            "__origin": [f"GL.txt@4564dc0deef2:{i + 1}" for i in range(len(amounts))],
        }
    )


@pytest.fixture
def profiles() -> Iterator[None]:
    original = SkrAccountFeatureGroup.PROFILES
    yield
    SkrAccountFeatureGroup.PROFILES = original


def test_a_debit_positive_gdpdu_exporter_is_stated_on_its_profile(profiles: None) -> None:
    """An exporter that writes revenue as a negative Haben amount says so once, on its profile."""
    SkrAccountFeatureGroup.PROFILES = (LedgerProfile("Konto", "Betrag", sign=SOLL_POSITIVE),)
    mapped = SkrAccountFeatureGroup._map_accounts(_gdpdu_lines("-100.00", "-20.00"), _features("revenue"))
    assert [v for v, _ in _values(mapped, "revenue")] == [Decimal("100.00"), Decimal("20.00")]
    assert json.loads(mapped.column("revenue~basis")[0].as_py())["sign"] == SOLL_POSITIVE


def test_without_a_stated_convention_amounts_are_taken_as_declared() -> None:
    mapped = SkrAccountFeatureGroup._map_accounts(_gdpdu_lines("100.00"), _features("revenue"))
    assert [v for v, _ in _values(mapped, "revenue")] == [Decimal("100.00")]
    assert json.loads(mapped.column("revenue~basis")[0].as_py())["sign"] == "as-declared"


def test_a_profile_that_contradicts_the_journal_is_refused(profiles: None, tmp_path: Path) -> None:
    """The DATEV legs state soll-positiv; a profile claiming as-declared for them is wrong."""
    SkrAccountFeatureGroup.PROFILES = (LedgerProfile("Konto", "Betrag", sign="as-declared"),)
    with pytest.raises(ValueError, match="contradicts"):
        SkrAccountFeatureGroup._map_accounts(_ledermann(tmp_path), _features("receivables"))


def test_an_unknown_profile_convention_is_refused_when_the_profile_is_made() -> None:
    with pytest.raises(ValueError, match="'haben-positiv'"):
        LedgerProfile("Konto", "Betrag", sign="haben-positiv")


# --- which revenue accounts are automatic -----------------------------------------------------------


@pytest.mark.parametrize(
    ("chart_key", "account", "gross"),
    [
        ("03", 8315, True),  # Erlöse aus im Inland steuerpflichtigen EU-Lieferungen 19 % USt
        ("03", 8410, True),  # Erlöse 19 % USt
        ("03", 8125, False),  # steuerfreie innergemeinschaftliche Lieferungen: 0 %, booked net
        ("03", 8130, False),  # Dreiecksgeschäft, erster Abnehmer: 0 % despite its tax key's name
        ("04", 4200, True),  # Erlöse, default 19 % USt
        ("04", 4120, False),  # steuerfreie Umsätze § 4 Nr. 1a UStG
    ],
)
def test_the_automatic_accounts_are_those_with_a_taxable_default(chart_key: str, account: int, gross: bool) -> None:
    from mloda.community.feature_groups.experimental.german_ledger.skr import AUTOMATIC_REVENUE

    assert (account in AUTOMATIC_REVENUE[chart_key]) is gross
