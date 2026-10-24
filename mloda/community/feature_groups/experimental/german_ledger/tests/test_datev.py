"""DatevExtfReader: DATEV-Format EXTF 700 Buchungsstapel -> signed legs, refusals by name.

Decisions of 30 Sep 2026: version 700 only; Soll/Haben AND
Generalumkehr are modelled in the reader; old versions (510/300) stay refused. The reader
does not decide admissibility or concepts -- it produces one row per LEG, signed from the
point of view of that leg's account, each carrying its own citation.
"""

import re
import shutil
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from mloda.provider import FeatureSet
from mloda.user import Feature, Options

from mloda.community.feature_groups.experimental.german_ledger.datev import DatevExtfReader, DatevRefusal
from mloda.community.feature_groups.experimental.german_ledger.reader import JOURNAL, GdpduReader

from .isolated import claimant, source_match

HERE = Path(__file__).parent
LEDERMANN = HERE / "fixtures" / "datev_ledermann"

# The column-name line of a real 700/21/13 batch (125 fields, en dashes and all).
NAMES = (LEDERMANN / "EXTF_Buchungsstapel.csv").read_bytes().decode("cp1252").splitlines()[1]
NAME_LIST = NAMES.split(";")
# DATEV's spelling of the names the reader checks, as both exports write them; positions unchanged.
SPEC_NAMES = (
    NAMES.replace("\u2013", "-").replace("Basisumsatz", "Basis-Umsatz").replace("Generalumkehr", "Generalumkehr (GU)")
)


def _load(folder: Path | str) -> pa.Table:
    """The reader ignores the FeatureSet: a batch always yields every leg column."""
    table: pa.Table = DatevExtfReader.load_data(str(folder), FeatureSet())
    return table


def _header(**over: str) -> str:
    h = {
        "marker": '"EXTF"',
        "version": "700",
        "category": "21",
        "format_name": '"Buchungsstapel"',
        "format_version": "13",
        "created": "20250110120000000",
        "berater": "1001",
        "mandant": "456",
        "wj": "20250101",
        "skl": "4",
        "von": "20250101",
        "bis": "20251231",
        "buchungstyp": "1",
        "zweck": "0",
        "festschreibung": "0",
        "wkz": '"EUR"',
        "skr": "",
    }
    h.update(over)
    fields = [
        h["marker"],
        h["version"],
        h["category"],
        h["format_name"],
        h["format_version"],
        h["created"],
        "",
        '"XY"',
        '"Chief Accounting Officer"',
        "",
        h["berater"],
        h["mandant"],
        h["wj"],
        h["skl"],
        h["von"],
        h["bis"],
        '"Test"',
        "",
        h["buchungstyp"],
        h["zweck"],
        h["festschreibung"],
        h["wkz"],
        "",
        "",
        "",
        "",
        h["skr"],
        "",
        "",
        "",
        "",
    ]
    return ";".join(fields)


def _row(
    umsatz: str,
    sh: str,
    konto: str,
    gegen: str,
    belegdatum: str,
    *,
    bu: str = "",
    text: str = "t",
    wkz: str = "",
    kurs: str = "",
    basis: str = "",
    wkz_basis: str = "",
    beleg: str = "RE1",
    **by_name: str,
) -> str:
    fields = [""] * len(NAME_LIST)
    for pos, v in {
        1: umsatz,
        2: f'"{sh}"',
        3: wkz,
        4: kurs,
        5: basis,
        6: wkz_basis,
        7: konto,
        8: gegen,
        9: f'"{bu}"' if bu else "",
        10: belegdatum,
        11: f'"{beleg}"',
        14: f'"{text}"',
    }.items():
        fields[pos - 1] = v
    for name, v in by_name.items():
        fields[NAME_LIST.index(name)] = v
    return ";".join(fields)


def _batch(
    folder: Path,
    rows: list[str],
    name: str = "EXTF_Buchungsstapel.csv",
    *,
    encoding: str = "cp1252",
    newline: str = "\r\n",
    names: str = NAMES,
    fields: int = 31,
    **header: str,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    head = ";".join(_header(**header).split(";")[:fields])
    path.write_bytes((newline.join([head, names, *rows]) + newline).encode(encoding))
    return path


def _legs(table: pa.Table) -> list[tuple[Any, ...]]:
    return list(zip(*(table.column(c).to_pylist() for c in ("Konto", "Gegenkonto", "Betrag", "Leg"))))


# --- the ledermann batch: acceptance criterion 1 (§3.4) ---------------------------------


def test_ledermann_reads_two_rows_as_four_signed_legs() -> None:
    t = _load(str(LEDERMANN))
    assert _legs(t) == [
        ("1200", "4940", Decimal("-24.95"), "K"),  # 24,95 H on 1200
        ("4940", "1200", Decimal("24.95"), "G"),
        ("10000", "8400", Decimal("5950.00"), "K"),  # 5950,00 S on 10000
        ("8400", "10000", Decimal("-5950.00"), "G"),
    ]
    assert sum(t.column("Betrag").to_pylist()) == 0, "every row balances across its two legs"
    assert t.column("Buchungsdatum").to_pylist() == [date(2018, 2, 21)] * 2 + [date(2018, 2, 22)] * 2
    assert pa.types.is_decimal(t.schema.field("Betrag").type)
    assert pa.types.is_date(t.schema.field("Buchungsdatum").type)


def test_each_leg_carries_its_own_citation() -> None:
    t = _load(str(LEDERMANN))
    origins = t.column(DatevExtfReader.ORIGIN_COLUMN).to_pylist()
    pattern = re.compile(r"EXTF_Buchungsstapel\.csv@[0-9a-f]{12}:(\d+)/([KG])\Z")
    matches = [pattern.match(o) for o in origins]
    assert all(matches), origins
    # Record 1 is the header, record 2 the column names: bookings start at record 3.
    assert [m.groups() for m in matches if m] == [("3", "K"), ("3", "G"), ("4", "K"), ("4", "G")]


def test_personenkonten_are_told_apart_by_length() -> None:
    """Sachkontenlänge 4: 10000 is a Personenkonto (a Debitor), 1200 a Sachkonto."""
    t = _load(str(LEDERMANN))
    kinds = dict(zip(t.column("Konto").to_pylist(), t.column("Kontoart").to_pylist()))
    assert kinds == {"1200": "Sachkonto", "4940": "Sachkonto", "10000": "Personenkonto", "8400": "Sachkonto"}


def test_festschreibung_and_header_skr_travel_with_each_leg() -> None:
    t = _load(str(LEDERMANN))
    assert t.column("Festschreibung").to_pylist() == [False] * 4  # header field 21 = 0
    assert t.column("SKR").to_pylist() == [""] * 4, "empty in the header; the concept decides"


@pytest.mark.parametrize(
    ("header", "carried"), [("1", True), ("0", False), ("", None)], ids=["header-1", "header-0", "header-empty"]
)
def test_a_rows_festschreibung_wins_and_a_row_without_one_takes_the_headers(
    tmp_path: Path, header: str, carried: bool | None
) -> None:
    _batch(
        tmp_path,
        [
            _row("1,00", "S", "4400", "10000", "1503", Festschreibung="0"),
            _row("1,00", "S", "4400", "10000", "1503", Festschreibung="1"),
            _row("1,00", "S", "4400", "10000", "1503"),
        ],
        festschreibung=header,
    )
    t = _load(str(tmp_path))
    assert t.column("Festschreibung").to_pylist() == [False] * 2 + [True] * 2 + [carried] * 2


def test_stammdaten_are_never_parsed(tmp_path: Path) -> None:
    """Category 16 holds names, addresses and IBANs. It is skipped by its header, unread."""
    d = tmp_path / "books"
    shutil.copytree(LEDERMANN, d)
    stamm = d / "EXTF_Stammdaten.csv"
    head = stamm.read_bytes().split(b"\n", 1)[0]
    stamm.write_bytes(head + b"\n\xff\xfe garbage that would crash any parser")
    t = _load(str(d))
    assert t.num_rows == 4
    assert not any(
        "Stammdaten" in o or "Kontenbeschriftungen" in o for o in t.column(DatevExtfReader.ORIGIN_COLUMN).to_pylist()
    )


# --- Generalumkehr and Soll/Haben ---------------------------------------------------------


@pytest.mark.parametrize(
    "gu_marker",
    [
        {"Generalumkehr": "1"},  # the field (strobelm rows 48-56)
        {"bu": "20"},  # legacy two-digit key, Berichtigungsschlüssel 2
        {"Generalumkehr": "1", "bu": "20"},  # both say the same thing: one reversal, not two
    ],
)
def test_generalumkehr_reverses_the_sign_on_the_same_side(tmp_path: Path, gu_marker: dict[str, str]) -> None:
    bu = gu_marker.get("bu", "")
    extra = {k: v for k, v in gu_marker.items() if k != "bu"}
    _batch(tmp_path, [_row("100,00", "S", "4400", "10000", "1503", bu=bu, **extra)])
    t = _load(str(tmp_path))
    assert _legs(t) == [("4400", "10000", Decimal("-100.00"), "K"), ("10000", "4400", Decimal("100.00"), "G")]
    assert t.column("Generalumkehr").to_pylist() == [True, True]


@pytest.mark.parametrize("bu", ["3", "9", "40", "501", "6501", "2105"])
def test_longer_tax_keys_are_not_read_as_generalumkehr(tmp_path: Path, bu: str) -> None:
    """DATEV's own sample carries 3- and 4-digit keys; a leading 2 there is a tax key."""
    _batch(tmp_path, [_row("100,00", "S", "4400", "10000", "1503", bu=bu)])
    t = _load(str(tmp_path))
    assert t.column("Betrag").to_pylist()[0] == Decimal("100.00")
    assert t.column("Generalumkehr").to_pylist() == [False, False]
    assert t.column("BU-Schlüssel").to_pylist() == [bu, bu]


def test_an_unknown_generalumkehr_value_is_refused(tmp_path: Path) -> None:
    _batch(tmp_path, [_row("100,00", "S", "4400", "10000", "1503", Generalumkehr="G")])
    with pytest.raises(DatevRefusal, match="Generalumkehr"):
        _load(str(tmp_path))


# --- amounts, currency, dates -----------------------------------------------------------


def test_an_empty_umsatz_is_unpriced_not_refused(tmp_path: Path) -> None:
    """OCA row 93: carry it as null on both legs so the total becomes unstateable."""
    _batch(tmp_path, [_row("", "S", "4400", "10000", "1503"), _row("10,00", "H", "4400", "10000", "1503")])
    t = _load(str(tmp_path))
    assert t.column("Betrag").to_pylist() == [None, None, Decimal("-10.00"), Decimal("10.00")]


def test_foreign_currency_takes_the_basisumsatz(tmp_path: Path) -> None:
    _batch(
        tmp_path, [_row("110,00", "S", "4400", "10000", "1503", wkz="USD", kurs="1,1", basis="100,00", wkz_basis="EUR")]
    )
    t = _load(str(tmp_path))
    assert t.column("Betrag").to_pylist() == [Decimal("100.00"), Decimal("-100.00")]


def test_leistungsdatum_is_carried_when_present(tmp_path: Path) -> None:
    _batch(
        tmp_path,
        [
            _row("1,00", "S", "4400", "10000", "1503", Leistungsdatum="28022025"),
            _row("1,00", "S", "4400", "10000", "1503"),
        ],
    )
    t = _load(str(tmp_path))
    assert t.column("Leistungsdatum").to_pylist() == [date(2025, 2, 28)] * 2 + [None] * 2


def test_several_batches_of_one_client_are_read_together(tmp_path: Path) -> None:
    _batch(tmp_path, [_row("1,00", "S", "4400", "10000", "1501")], name="EXTF_jan.csv", von="20250101", bis="20250131")
    _batch(tmp_path, [_row("2,00", "S", "4400", "10000", "1502")], name="EXTF_feb.csv", von="20250201", bis="20250228")
    t = _load(str(tmp_path))
    # In period order, not file-name order ("feb" sorts before "jan").
    assert t.column("Betrag").to_pylist() == [Decimal(1), Decimal(-1), Decimal(2), Decimal(-2)]
    assert [o.split("@")[0] for o in t.column(DatevExtfReader.ORIGIN_COLUMN).to_pylist()] == ["EXTF_jan.csv"] * 2 + [
        "EXTF_feb.csv"
    ] * 2


# --- refusals by name: acceptance criterion 3 (§3.4) ------------------------------------

REFUSALS: dict[str, tuple[dict[str, Any], list[dict[str, str]], str]] = {
    "version 510": ({"version": "510", "format_version": "7"}, [], "Versionsnummer 510"),
    "version 300": ({"version": "300", "format_version": "2"}, [], "Versionsnummer 300"),
    "category 22 (OCA writer bug)": ({"category": "22"}, [], "Kategorie 22"),
    "Buchungstyp 2": ({"buchungstyp": "2"}, [], "Buchungstyp"),
    "Rechnungslegungszweck": ({"zweck": "30"}, [], "Rechnungslegungszweck"),
    "SKR 3": ({"skr": "3"}, [], "SKR"),
    "header currency": ({"wkz": '"USD"'}, [], "WKZ"),
    "vom after bis": ({"von": "20251231", "bis": "20250101"}, [], "Datum vom"),
    "Sachkontenlänge 9": ({"skl": "9"}, [], "Sachkontenlänge"),
    "non-EUR without Basisumsatz": (
        {},
        [dict(umsatz="10,00", sh="S", konto="4400", gegen="10000", belegdatum="1503", wkz="USD")],
        "Basisumsatz",
    ),
    "flag X": ({}, [dict(umsatz="10,00", sh="X", konto="4400", gegen="10000", belegdatum="1503")], "Soll/Haben"),
    "negative Umsatz": (
        {},
        [dict(umsatz="-10,00", sh="S", konto="4400", gegen="10000", belegdatum="1503")],
        "negative",
    ),
    "dot in Umsatz": ({}, [dict(umsatz="1.000,00", sh="S", konto="4400", gegen="10000", belegdatum="1503")], "'\\.'"),
    "three decimals": (
        {},
        [dict(umsatz="1,005", sh="S", konto="4400", gegen="10000", belegdatum="1503")],
        "decimal places",
    ),
    "empty Gegenkonto": ({}, [dict(umsatz="1,00", sh="S", konto="4400", gegen="", belegdatum="1503")], "Gegenkonto"),
    "account too long": (
        {},
        [dict(umsatz="1,00", sh="S", konto="4400", gegen="1000000", belegdatum="1503")],
        "Sachkontenlänge",
    ),
    "Belegdatum not TTMM": (
        {},
        [dict(umsatz="1,00", sh="S", konto="4400", gegen="10000", belegdatum="20250315")],
        "TTMM",
    ),
    "Belegdatum outside period": (
        {},
        [dict(umsatz="1,00", sh="S", konto="4400", gegen="10000", belegdatum="1513")],
        "Belegdatum",
    ),
    "header Festschreibung 2": ({"festschreibung": "2"}, [], "header Festschreibung '2'"),
    "row Festschreibung 2": (
        {},
        [dict(umsatz="1,00", sh="S", konto="4400", gegen="10000", belegdatum="1503", Festschreibung="2")],
        "record 3: Festschreibung '2'",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_each_unhonoured_declaration_is_refused_by_name(tmp_path: Path, case: str) -> None:
    header, rows, message = REFUSALS[case]
    _batch(tmp_path, [_row(**r) for r in rows] or [_row("1,00", "S", "4400", "10000", "1503")], **header)
    with pytest.raises(DatevRefusal, match=message):
        _load(str(tmp_path))


def test_a_date_ambiguous_across_the_year_boundary_is_refused(tmp_path: Path) -> None:
    """A batch spanning two calendar years makes a TTMM date name two days."""
    _batch(tmp_path, [_row("1,00", "S", "4400", "10000", "1503")], von="20250101", bis="20260630")
    with pytest.raises(DatevRefusal, match="2 date"):
        _load(str(tmp_path))


def test_overlapping_batches_of_one_client_are_refused(tmp_path: Path) -> None:
    """The same bookings exported twice would be counted twice."""
    _batch(tmp_path, [_row("1,00", "S", "4400", "10000", "1501")], name="EXTF_a.csv", von="20250101", bis="20250331")
    _batch(tmp_path, [_row("1,00", "S", "4400", "10000", "1502")], name="EXTF_b.csv", von="20250201", bis="20250430")
    with pytest.raises(DatevRefusal, match="overlap"):
        _load(str(tmp_path))


def test_two_clients_in_one_dossier_are_refused(tmp_path: Path) -> None:
    _batch(
        tmp_path,
        [_row("1,00", "S", "4400", "10000", "1501")],
        name="EXTF_a.csv",
        mandant="456",
        von="20250101",
        bis="20250131",
    )
    _batch(
        tmp_path,
        [_row("1,00", "S", "4400", "10000", "1502")],
        name="EXTF_b.csv",
        mandant="789",
        von="20250201",
        bis="20250228",
    )
    with pytest.raises(DatevRefusal, match="Mandant"):
        _load(str(tmp_path))


def test_a_comma_delimited_resave_is_refused(tmp_path: Path) -> None:
    p = _batch(tmp_path, [_row("1,00", "S", "4400", "10000", "1503")])
    p.write_bytes(p.read_bytes().replace(b";", b","))
    with pytest.raises(DatevRefusal, match="delimiter"):
        _load(str(tmp_path))
    # Next to a good batch it must still refuse, not be skipped: skipping undercounts.
    _batch(tmp_path, [_row("1,00", "S", "4400", "10000", "1503")], name="EXTF_good.csv")
    with pytest.raises(DatevRefusal, match="delimiter"):
        _load(str(tmp_path))


def test_a_moved_core_column_is_refused(tmp_path: Path) -> None:
    p = _batch(tmp_path, [_row("1,00", "S", "4400", "10000", "1503")])
    p.write_bytes(p.read_bytes().replace("Konto;".encode(), b"Kontonummer;", 1))
    with pytest.raises(DatevRefusal, match="column 7"):
        _load(str(tmp_path))


def test_a_folder_with_no_booking_batch_is_refused(tmp_path: Path) -> None:
    shutil.copy(LEDERMANN / "EXTF_Stammdaten.csv", tmp_path)
    with pytest.raises(DatevRefusal, match="no Buchungsstapel"):
        _load(str(tmp_path))


# --- claiming a folder, identity ---------------------------------------------------------


def _claimant(folder: Path | str, name: str = JOURNAL) -> Any:
    """The reader that resolution gives `name` over `folder`, or None, with both readers enabled."""
    return claimant([folder], name, {GdpduReader, DatevExtfReader})


def test_the_reader_claims_a_batch_folder_and_nothing_else(tmp_path: Path) -> None:
    assert _claimant(LEDERMANN) is DatevExtfReader
    gdpdu = HERE / "fixtures" / "dossier_a"
    assert _claimant(gdpdu) is GdpduReader
    assert _claimant(LEDERMANN, "x") is None
    only_stamm = tmp_path / "stamm"
    only_stamm.mkdir()
    shutil.copy(LEDERMANN / "EXTF_Stammdaten.csv", only_stamm)
    assert _claimant(only_stamm) is None


def test_identity_names_the_folder_and_fingerprint_and_no_personal_data(tmp_path: Path) -> None:
    ident = DatevExtfReader.data_access_identity(source_match(LEDERMANN))
    assert re.fullmatch(r"datev:datev_ledermann@[0-9a-f]{12}", ident), ident
    for leak in ("1001", "456", "XY", "Chief", str(HERE)):
        assert leak not in ident
    t = _load(str(LEDERMANN))
    batch_hash = t.column(DatevExtfReader.ORIGIN_COLUMN).to_pylist()[0].split("@")[1].split(":")[0]
    assert len(batch_hash) == 12

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "EXTF_x.csv").write_bytes(b"\xff\xfe")
    assert DatevExtfReader.data_access_identity(source_match(broken)).startswith("datev:broken")
    assert DatevExtfReader.data_access_identity(source_match(tmp_path / "absent")) == "datev:absent"


def test_the_identity_moves_when_a_batch_is_edited(tmp_path: Path) -> None:
    d = tmp_path / "books"
    shutil.copytree(LEDERMANN, d)
    before = DatevExtfReader.data_access_identity(source_match(d))
    p = d / "EXTF_Buchungsstapel.csv"
    p.write_bytes(p.read_bytes().replace(b"5950,00", b"5950,01"))
    assert DatevExtfReader.data_access_identity(source_match(d)) != before


# --- third-party export shapes, on invented bookings -------------------------------------


def test_a_batch_shaped_like_winfo_reads_end_to_end(tmp_path: Path) -> None:
    """The shape of https://github.com/Winfo2024Kuhn/ERP-System-fuer-Handwerksbetriebe/blob/ae43e8c/src/test/resources/datev/buchungsstapel-golden.csv"""
    quoted_names = ";".join(f'"{n}"' for n in SPEC_NAMES.split(";"))  # csv handles it (shape only)
    rows = [
        _row("321,00", "S", "1000", "8300", "0408", bu="2", wkz="EUR", Festschreibung="1"),
        _row("18,50", "S", "4910", "1000", "0408", bu="9", wkz="EUR", Festschreibung="1"),
        _row("400,00", "S", "1000", "1360", "1108", wkz="EUR", Festschreibung="1"),
    ]
    _batch(
        tmp_path,
        [r.replace('"', "") for r in rows],  # S/H, WKZ, BU and Belegfeld 1 unquoted (shape only)
        encoding="utf-8",  # no BOM
        newline="\n",  # LF line ends (shape only)
        names=quoted_names,
        fields=22,  # header ends at WKZ: no SKR field
        festschreibung="1",
        zweck="",
        von="20250801",
        bis="20250831",
    )
    t = _load(str(tmp_path))
    assert _legs(t) == [
        ("1000", "8300", Decimal("321.00"), "K"),
        ("8300", "1000", Decimal("-321.00"), "G"),
        ("4910", "1000", Decimal("18.50"), "K"),
        ("1000", "4910", Decimal("-18.50"), "G"),
        ("1000", "1360", Decimal("400.00"), "K"),
        ("1360", "1000", Decimal("-400.00"), "G"),
    ]
    assert sum(t.column("Betrag").to_pylist()) == 0
    assert t.column("Festschreibung").to_pylist() == [True] * 6
    assert t.column("SKR").to_pylist() == [""] * 6
    assert t.column("Buchungsdatum").to_pylist() == [date(2025, 8, 4)] * 4 + [date(2025, 8, 11)] * 2


def test_a_batch_shaped_like_strobelm_reads_end_to_end(tmp_path: Path) -> None:
    """The shape of https://github.com/strobelm/datev/blob/8abf388/data/EXTF_Buchungsstapel.csv (DATEV's sample client)."""
    # `_row(**by_name)` keys resolve against NAME_LIST, so Generalumkehr lands in the column
    # SPEC_NAMES calls "Generalumkehr (GU)".
    rows = [
        _row("750", "S", "4500", "75", "2701", wkz='""'),  # Umsatz without decimal comma; 2-digit Sachkonto
        _row("42,30", "S", "4650", "410", "2701", wkz='""'),  # 3-digit Sachkonto
        # GU written quoted "0"/"1": a reversed S row paired with an H row
        _row("60,00", "H", "70500", "4980", "1502", bu="503", wkz='""', Generalumkehr='"0"'),
        _row("60,00", "S", "70500", "4980", "1502", bu="503", wkz='""', Generalumkehr='"1"'),
        # Leistungsdatum years before the batch period: carried, not refused
        _row("95,00", "H", "70600", "3200", "0904", wkz='""', Leistungsdatum="03062019"),
    ]
    _batch(tmp_path, rows, encoding="utf-8", names=SPEC_NAMES, skr='"03"', von="20250101", bis="20250831")
    t = _load(str(tmp_path))
    assert _legs(t) == [
        ("4500", "75", Decimal("750"), "K"),
        ("75", "4500", Decimal("-750"), "G"),
        ("4650", "410", Decimal("42.30"), "K"),
        ("410", "4650", Decimal("-42.30"), "G"),
        ("70500", "4980", Decimal("-60.00"), "K"),
        ("4980", "70500", Decimal("60.00"), "G"),
        ("70500", "4980", Decimal("-60.00"), "K"),
        ("4980", "70500", Decimal("60.00"), "G"),
        ("70600", "3200", Decimal("-95.00"), "K"),
        ("3200", "70600", Decimal("95.00"), "G"),
    ]
    assert sum(t.column("Betrag").to_pylist()) == 0
    assert t.column("Generalumkehr").to_pylist() == [False] * 6 + [True, True] + [False, False]
    kinds = dict(zip(t.column("Konto").to_pylist(), t.column("Kontoart").to_pylist()))
    assert (kinds["75"], kinds["410"]) == ("Sachkonto", "Sachkonto")
    assert (kinds["70500"], kinds["70600"]) == ("Personenkonto", "Personenkonto")
    assert t.column("SKR").to_pylist() == ["03"] * 10
    assert t.column("Leistungsdatum").to_pylist() == [None] * 8 + [date(2019, 6, 3)] * 2


# --- through the resolution chain ---------------------------------------------------------


def _run(folder: Path, features: list[str | Feature]) -> Any:
    from mloda.user import DataAccessCollection, mloda
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

    # `from ... import`, never `import mloda.community...`: that rebinds the name `mloda` to the
    # namespace package and hides the mloda.user.mloda API imported just above.
    from mloda.community.feature_groups.experimental.german_ledger import skr, sources  # noqa: F401

    from ._host import PLUGINS

    return mloda.run_all(
        features=features,
        compute_frameworks=[PyArrowTable],
        data_access_collection=DataAccessCollection(folders={str(folder)}),
        plugin_collector=PLUGINS,
    )


def test_a_datev_folder_resolves_to_the_datev_journal() -> None:
    from mloda.community.feature_groups.experimental.german_ledger.policy import DatevJournalFeatureGroup

    assert _claimant(LEDERMANN) is DatevJournalFeatureGroup
    assert _claimant(LEDERMANN, "revenue__sources") is None


def test_the_late_entry_cutoff_refuses_a_datev_batch_it_cannot_evaluate() -> None:
    """DATEV has no Erfassungsdatum. The GDPdU cutoff must fail closed, not pass (§3.1 item 8)."""
    from mloda.community.feature_groups.experimental.german_ledger.policy import LateEntryRefused

    from ._host import _cause

    with pytest.raises(Exception) as info:
        _run(LEDERMANN, ["revenue__sources"])
    refusal = _cause(info.value, LateEntryRefused)
    assert refusal is not None, info.value
    assert "Erfassungsdatum" in str(refusal)


def test_a_folder_holding_both_formats_is_refused_not_shadowed(tmp_path: Path) -> None:
    """Neither reader is pinned, so both readers claim the feature and core refuses the ambiguity."""
    both = tmp_path / "both"
    shutil.copytree(HERE / "fixtures" / "dossier_a", both)
    shutil.copy(LEDERMANN / "EXTF_Buchungsstapel.csv", both)
    with pytest.raises(Exception) as info:
        _run(both, ["revenue__sources"])
    message = str(info.value)
    assert "GdpduReader" in message and "DatevExtfReader" in message, message


@pytest.mark.parametrize("reader", [GdpduReader, DatevExtfReader])
def test_the_reader_declares_its_data_classification(reader: Any) -> None:
    from mloda.community.extenders.shared.classification import CLASSIFICATION_KEY

    assert CLASSIFICATION_KEY == "mloda.data.classification"
    assert reader.declared_attributes(None) == {"mloda.data.classification": "confidential"}

    from mloda.community.feature_groups.experimental.german_ledger.reader import ClassifiedReader

    lowered: type[ClassifiedReader] = type("Lowered", (ClassifiedReader,), {"CLASSIFICATION": "internal"})
    assert lowered.declared_attributes(None) == {CLASSIFICATION_KEY: "internal"}


@pytest.mark.parametrize("reader", ["GdpduReader", "DatevExtfReader"])
def test_a_folder_holding_both_formats_resolves_when_one_reader_is_pinned(tmp_path: Path, reader: str) -> None:
    both = tmp_path / "both"
    shutil.copytree(HERE / "fixtures" / "dossier_a", both)
    shutil.copy(LEDERMANN / "EXTF_Buchungsstapel.csv", both)
    feature = Feature("gdpdu_journal", Options({reader: str(both)}))
    results = _run(both, [feature])
    assert results
