"""DatevExtfReader: DATEV-Format EXTF 700 Buchungsstapel -> signed legs, refusals by name.

Decisions of 30 Sep 2026: version 700 only; Soll/Haben AND
Generalumkehr are modelled in the reader; old versions (510/300) stay refused. The reader
does not decide admissibility or concepts -- it produces one row per LEG, signed from the
point of view of that leg's account, each carrying its own citation.
"""

import os
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

from mloda.community.feature_groups.german_ledger.datev import DatevExtfReader, DatevRefusal
from mloda.community.feature_groups.german_ledger.reader import GdpduReader

HERE = Path(__file__).parent
LEDERMANN = HERE / "fixtures" / "datev_ledermann"
# Third-party DATEV files, kept outside this repository: not all are licensed for
# redistribution, so they are never committed here. Point GERMAN_LEDGER_CORPUS at them to run.
CORPUS = Path(os.environ.get("GERMAN_LEDGER_CORPUS", HERE / "corpus-not-configured"))

# The column-name line of a real 700/21/13 batch (125 fields, en dashes and all).
NAMES = (LEDERMANN / "EXTF_Buchungsstapel.csv").read_bytes().decode("cp1252").splitlines()[1]
NAME_LIST = NAMES.split(";")


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


def _batch(folder: Path, rows: list[str], name: str = "EXTF_Buchungsstapel.csv", **header: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(("\r\n".join([_header(**header), NAMES, *rows]) + "\r\n").encode("cp1252"))
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

REFUSALS = {
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


def test_the_reader_claims_a_batch_folder_and_nothing_else(tmp_path: Path) -> None:
    assert DatevExtfReader.match_subclass_data_access(str(LEDERMANN), ["x"], Options()) == str(LEDERMANN)
    gdpdu = HERE / "fixtures" / "dossier_a"
    assert DatevExtfReader.match_subclass_data_access(str(gdpdu), ["x"], Options()) is None
    assert GdpduReader.match_subclass_data_access(str(LEDERMANN), ["x"], Options()) is None
    only_stamm = tmp_path / "stamm"
    only_stamm.mkdir()
    shutil.copy(LEDERMANN / "EXTF_Stammdaten.csv", only_stamm)
    assert DatevExtfReader.match_subclass_data_access(str(only_stamm), ["x"], Options()) is None


def test_identity_names_the_folder_and_fingerprint_and_no_personal_data(tmp_path: Path) -> None:
    ident = DatevExtfReader.data_access_identity(str(LEDERMANN))
    assert re.fullmatch(r"datev:datev_ledermann@[0-9a-f]{12}", ident), ident
    for leak in ("1001", "456", "XY", "Chief", str(HERE)):
        assert leak not in ident
    t = _load(str(LEDERMANN))
    batch_hash = t.column(DatevExtfReader.ORIGIN_COLUMN).to_pylist()[0].split("@")[1].split(":")[0]
    assert len(batch_hash) == 12

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "EXTF_x.csv").write_bytes(b"\xff\xfe")
    assert DatevExtfReader.data_access_identity(str(broken)).startswith("datev:broken")
    assert DatevExtfReader.data_access_identity(str(tmp_path / "absent")) == "datev:absent"


def test_the_identity_moves_when_a_batch_is_edited(tmp_path: Path) -> None:
    d = tmp_path / "books"
    shutil.copytree(LEDERMANN, d)
    before = DatevExtfReader.data_access_identity(str(d))
    p = d / "EXTF_Buchungsstapel.csv"
    p.write_bytes(p.read_bytes().replace(b"5950,00", b"5950,01"))
    assert DatevExtfReader.data_access_identity(str(d)) != before


# --- the corpus: files we did not write --------------------------------------------------


@pytest.mark.skipif(not CORPUS.is_dir(), reason="GERMAN_LEDGER_CORPUS not set")
@pytest.mark.parametrize(
    "path, legs",
    [
        ("datev-extf-ledermann", 4),
        ("datev-winfo", 10),
        ("datev-nolicence/strobelm", 108),  # DATEV's own sample client: carries GU and BU 20
    ],
)
def test_collected_batches_read(path: str, legs: int) -> None:
    folder = CORPUS / path
    if not folder.is_dir():
        pytest.skip(f"{path} not collected here")
    t = _load(str(folder))
    assert t.num_rows == legs
    assert sum(v for v in t.column("Betrag").to_pylist() if v is not None) == 0


@pytest.mark.skipif(not CORPUS.is_dir(), reason="GERMAN_LEDGER_CORPUS not set")
@pytest.mark.parametrize(
    "path, message",
    [
        ("datev-banana", "Versionsnummer 300"),
        ("datev-kontor", "Versionsnummer 510"),
    ],
)
def test_collected_old_versions_stay_refused(path: str, message: str) -> None:
    with pytest.raises(DatevRefusal, match=message):
        _load(str(CORPUS / path))


# --- through the resolution chain ---------------------------------------------------------


def _run(folder: Path, features: list[str | Feature]) -> Any:
    from mloda.user import DataAccessCollection, mloda
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

    # `from ... import`, never `import mloda.community...`: that rebinds the name `mloda` to the
    # namespace package and hides the mloda.user.mloda API imported just above.
    from mloda.community.feature_groups.german_ledger import skr, sources  # noqa: F401

    from ._host import PLUGINS

    return mloda.run_all(
        features=features,
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(folder)}),
        plugin_collector=PLUGINS,
    )


def _cause(exc: BaseException, kind: type[BaseException]) -> BaseException | None:
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, kind):
            return seen
        seen = seen.__cause__ or seen.__context__
    return None


def test_a_datev_folder_resolves_to_the_datev_journal() -> None:
    from mloda.user import DataAccessCollection, Options

    from mloda.community.feature_groups.german_ledger.policy import DatevJournalFeatureGroup, JournalFeatureGroup

    dac = DataAccessCollection(folders={str(LEDERMANN)})
    assert DatevJournalFeatureGroup.match_feature_group_criteria("gdpdu_journal", Options(), dac)
    assert not JournalFeatureGroup.match_feature_group_criteria("gdpdu_journal", Options(), dac)
    assert not DatevJournalFeatureGroup.match_feature_group_criteria("revenue__sources", Options(), dac)


def test_the_late_entry_cutoff_refuses_a_datev_batch_it_cannot_evaluate() -> None:
    """DATEV has no Erfassungsdatum. The GDPdU cutoff must fail closed, not pass (§3.1 item 8)."""
    from mloda.community.feature_groups.german_ledger.policy import LateEntryRefused

    with pytest.raises(Exception) as info:
        _run(LEDERMANN, ["revenue__sources"])
    refusal = _cause(info.value, LateEntryRefused)
    assert refusal is not None, info.value
    assert "Erfassungsdatum" in str(refusal)


def test_a_folder_holding_both_formats_is_refused_not_shadowed(tmp_path: Path) -> None:
    """Core refuses before resolution: both journals claim the feature, each with its own
    reader, and mloda 0.14.0 aborts on the conflicting readers rather than picking one."""
    both = tmp_path / "both"
    shutil.copytree(HERE / "fixtures" / "dossier_a", both)
    shutil.copy(LEDERMANN / "EXTF_Buchungsstapel.csv", both)
    with pytest.raises(Exception) as info:
        _run(both, ["revenue__sources"])
    message = str(info.value)
    assert ("GdpduReader" in message and "DatevExtfReader" in message) or (
        "JournalFeatureGroup" in message and "DatevJournalFeatureGroup" in message
    ), message
