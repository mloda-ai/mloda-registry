"""DatevExtfReader: DATEV-Format EXTF 700 Buchungsstapel -> one signed row per leg.

Grown out of a spike that tested the design against DATEV files we did not write. Decided on
30 Sep 2026:

- Only Versionsnummer 700 has a checked field table. 510 and 300 are refused by name.
- Soll/Haben is modelled: `Umsatz` is unsigned and the flag refers to `Konto`.
- Generalumkehr is modelled, not refused: a GU row is booked on the same side with the
  opposite effect, so the signed amount flips. That is what lets DATEV's own sample books
  (strobelm, Berater 29098 / Mandant 55003) be read at all.

Every booking row becomes TWO legs -- `Konto` (K) and `Gegenkonto` (G) -- each signed from
the point of view of its own account, each with its own citation. Revenue usually sits on
the Gegenkonto, so a concept that summed rows by `Konto` would miss it; one that sums legs
cannot. The reader stops there: it does not decide admissibility (DATEV has no
Erfassungsdatum, so the late-entry cutoff refuses every batch) and it does not know which
accounts are Automatikkonten -- that is a property of the chart, handled by the concept.

Deliberately strict, like the probe: no delimiter sniffing, no encoding guessing beyond the
two the format is written in, no repair of spreadsheet damage.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow as pa
from mloda.provider import FeatureSet
from mloda.user import DataAccessCollection, Options
from mloda_plugins.feature_group.input_data.read_file import ReadFile

from .reader import _ORIGIN_COLUMN


class DatevRefusal(ValueError):
    """The batch makes a declaration this reader does not honour; the message names it."""


# (Versionsnummer, Formatkategorie) -> the Formatversionen with a checked field table.
# 13 = DATEV itself, ledermann, UltraCanvas; 12 = OCA; 9 = erpnext_datev.
BOOKING_FORMATS: dict[tuple[int, int], frozenset[int]] = {(700, 21): frozenset({13, 12, 9})}
BOOKING_CATEGORY = 21

# Core fields by POSITION (1-based) with the names seen for each. Position is what DATEV
# fixes; names drift between writers (hyphen, en dash, "ue" for "ü").
CORE: dict[int, frozenset[str]] = {
    1: frozenset({"Umsatz (ohne Soll/Haben-Kz)"}),
    2: frozenset({"Soll/Haben-Kennzeichen"}),
    3: frozenset({"WKZ Umsatz"}),
    4: frozenset({"Kurs"}),
    5: frozenset({"Basisumsatz", "Basis-Umsatz"}),
    6: frozenset({"WKZ Basisumsatz", "WKZ Basis-Umsatz"}),
    7: frozenset({"Konto"}),
    8: frozenset({"Gegenkonto (ohne BU-Schlüssel)"}),
    9: frozenset({"BU-Schlüssel"}),
    10: frozenset({"Belegdatum"}),
    11: frozenset({"Belegfeld 1"}),
}
_GU_NAMES = ("Generalumkehr (GU)", "Generalumkehr")

# One row per leg. No Buchungstext: it is free text that can carry a customer's name, and
# nothing downstream needs it -- the citation points back at the row for anyone who does.
# The legs' sign convention, stated on every leg so it travels with the rows: +Soll / -Haben
# on the leg's own account. A concept reads it to report in its natural direction; a journal
# that does not state one (GDPdU) is taken as declared.
VORZEICHEN = "Vorzeichen"
SOLL_POSITIVE = "soll-positiv"

COLUMNS: tuple[str, ...] = (
    "Konto",
    "Gegenkonto",
    "Kontoart",
    "Betrag",
    "Buchungsdatum",
    "Leistungsdatum",
    "Leg",
    "BU-Schlüssel",
    "Generalumkehr",
    "Festschreibung",
    "SKR",
    "Belegfeld 1",
    VORZEICHEN,
)
_TYPES: dict[str, pa.DataType] = {
    "Betrag": pa.decimal128(38, 2),
    "Buchungsdatum": pa.date32(),
    "Leistungsdatum": pa.date32(),
    "Generalumkehr": pa.bool_(),
    "Festschreibung": pa.bool_(),
}


@dataclass(frozen=True)
class DatevHeader:
    version: int
    category: int
    format_version: int
    berater: str
    mandant: str
    sachkontenlaenge: int
    datum_von: date
    datum_bis: date
    festschreibung: bool | None
    skr: str


def _decode(raw: bytes, label: str) -> str:
    """UTF-8 (with or without BOM) or Windows-1252: the two the format is written in.

    Strict UTF-8 first. A cp1252 file that decodes as UTF-8 is pure ASCII, where the two
    agree; the reverse does not hold -- cp1252 decodes almost any bytes -- so trying it first
    would silently mojibake every UTF-8 umlaut.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError as e:
        raise DatevRefusal(f"{label}: neither UTF-8 nor Windows-1252 (byte {e.start})") from None


def _first_fields(path: str) -> list[str] | None:
    """The header line's fields, or None when the file is not a DATEV-Format file at all."""
    try:
        with open(path, "rb") as f:
            head = f.readline(8192)
    except OSError:
        return None
    if head.startswith(b"\xef\xbb\xbf"):
        head = head[3:]
    if not (head.startswith((b'"EXTF"', b"EXTF", b'"DTVF"', b"DTVF"))):
        return None
    try:
        line = head.decode("utf-8")
    except UnicodeDecodeError:
        line = head.decode("cp1252", errors="replace")
    return next(csv.reader([line.strip("\r\n")], delimiter=";", quotechar='"'), None)


def _is_booking_batch(fields: list[str]) -> bool:
    """A Buchungsstapel by category OR by name -- OCA writes category 22 named Buchungsstapel,
    and that has to reach the refusal, not be skipped as some other category."""
    return (len(fields) > 3 and fields[2].strip() == str(BOOKING_CATEGORY)) or (
        len(fields) > 3 and fields[3].strip() == "Buchungsstapel"
    )


def _batch_files(folder: str) -> list[str]:
    """Every DATEV booking batch in the folder. Stammdaten (16), Kontenbeschriftungen (20)
    and any other category are skipped by their header line, never parsed: category 16
    holds names, addresses and IBANs, and nothing personal belongs in a journal."""
    out = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not (name.lower().endswith(".csv") and os.path.isfile(path)):
            continue
        fields = _first_fields(path)
        # A DATEV header whose category cannot be read under ';' (a file re-saved with commas)
        # is kept, so read_batch refuses it by name. Skipping it would silently drop its
        # bookings from a folder that holds other batches -- a smaller total, not an error.
        if fields is not None and (len(fields) < 5 or _is_booking_batch(fields)):
            out.append(path)
    return out


def _ymd(raw: str, what: str, label: str) -> date:
    if len(raw) != 8 or not raw.isdigit():
        raise DatevRefusal(f"{label}: header {what} {raw!r} is not JJJJMMTT")
    try:
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
    except ValueError:
        raise DatevRefusal(f"{label}: header {what} {raw!r} is not a calendar date") from None


def parse_header(fields: list[str], label: str) -> DatevHeader:
    if len(fields) < 22:
        raise DatevRefusal(f"{label}: header carries {len(fields)} fields; the format needs 22")
    try:
        version, category, fmt_version = int(fields[1]), int(fields[2]), int(fields[4])
    except ValueError:
        raise DatevRefusal(f"{label}: header version fields {fields[1:5]!r} are not integers") from None
    if fmt_version not in BOOKING_FORMATS.get((version, category), frozenset()):
        raise DatevRefusal(
            f"{label}: format (Versionsnummer {version}, Kategorie {category} {fields[3]!r}, "
            f"Formatversion {fmt_version}) has no checked field table; only Versionsnummer 700 "
            "Kategorie 21 is read"
        )
    try:
        skl = int(fields[13])
    except ValueError:
        raise DatevRefusal(f"{label}: Sachkontenlänge {fields[13]!r} is not an integer") from None
    if not 4 <= skl <= 8:
        raise DatevRefusal(f"{label}: Sachkontenlänge {skl} outside 4..8")
    von = _ymd(fields[14], "Datum vom", label)
    bis = _ymd(fields[15], "Datum bis", label)
    if von > bis:
        raise DatevRefusal(f"{label}: Datum vom {von} is after Datum bis {bis}")
    if fields[18] != "1":
        raise DatevRefusal(
            f"{label}: Buchungstyp {fields[18]!r}: only 1 (Finanzbuchführung) is "
            "read; 2 (Jahresabschluss) would mix closing entries into period totals"
        )
    if fields[19] not in ("", "0"):
        raise DatevRefusal(f"{label}: Rechnungslegungszweck {fields[19]!r}: only purpose-independent batches are read")
    if fields[20] not in ("", "0", "1"):
        raise DatevRefusal(f"{label}: header Festschreibung {fields[20]!r} is neither 0 nor 1")
    if fields[21] not in ("", "EUR"):
        raise DatevRefusal(f"{label}: header WKZ {fields[21]!r}; only EUR books are read")
    skr = fields[26] if len(fields) > 26 else ""
    if skr not in ("", "03", "04"):
        raise DatevRefusal(f"{label}: SKR {skr!r} is neither 03 nor 04")
    return DatevHeader(
        version=version,
        category=category,
        format_version=fmt_version,
        berater=fields[10],
        mandant=fields[11],
        sachkontenlaenge=skl,
        datum_von=von,
        datum_bis=bis,
        festschreibung=None if fields[20] == "" else fields[20] == "1",
        skr=skr,
    )


def _amount(raw: str, where: str, what: str) -> Decimal | None:
    s = raw.strip()
    if not s:
        return None
    # DATEV writes a decimal comma and no grouping. A dot is a foreign writer's decimal point
    # or a German thousands separator -- the two readings differ by 1000x, so refuse.
    if "." in s:
        raise DatevRefusal(f"{where}: {what} {raw!r} contains '.', which DATEV does not write")
    try:
        value = Decimal(s.replace(",", "."))
    except InvalidOperation:
        raise DatevRefusal(f"{where}: {what} {raw!r} is not a decimal") from None
    if not value.is_finite():
        raise DatevRefusal(f"{where}: {what} {raw!r} is not a decimal")
    # Money is not re-scaled in transit: a third decimal place is refused, not rounded away.
    if value.as_tuple().exponent < -2:  # type: ignore[operator]
        raise DatevRefusal(f"{where}: {what} {raw!r} has more than 2 decimal places")
    return value


def _belegdatum(raw: str, h: DatevHeader, where: str) -> date:
    """TTMM, the year taken from the batch period -- refused where that is ambiguous.

    A writer that swaps day and month cannot be caught when both are <= 12; the period bound
    catches such a date only when it falls outside the batch. Disclosed, not fixable here.
    """
    s = raw.strip()
    if len(s) == 3 and s.isdigit():
        s = "0" + s  # a leading zero lost to a numeric column type; TMM is unambiguous
    if len(s) != 4 or not s.isdigit():
        raise DatevRefusal(f"{where}: Belegdatum {raw!r} is not TTMM")
    day, month = int(s[:2]), int(s[2:])
    candidates = []
    for year in range(h.datum_von.year, h.datum_bis.year + 1):
        try:
            d = date(year, month, day)
        except ValueError:
            continue
        if h.datum_von <= d <= h.datum_bis:
            candidates.append(d)
    if len(candidates) != 1:
        raise DatevRefusal(
            f"{where}: Belegdatum {raw!r} gives {len(candidates)} date(s) within the batch "
            f"period {h.datum_von}..{h.datum_bis}; exactly one is needed"
        )
    return candidates[0]


def _leistungsdatum(raw: str, where: str) -> date | None:
    s = raw.strip()
    if not s:
        return None
    if len(s) == 7 and s.isdigit():
        s = "0" + s
    try:
        if len(s) != 8 or not s.isdigit():
            raise ValueError
        return date(int(s[4:]), int(s[2:4]), int(s[:2]))
    except ValueError:
        raise DatevRefusal(f"{where}: Leistungsdatum {raw!r} is not TTMMJJJJ") from None


def _account(raw: str, h: DatevHeader, where: str, what: str) -> tuple[str, str]:
    """The account and its kind. Personenkonten (Debitoren/Kreditoren) are one digit longer
    than the Sachkontenlänge; anything longer than that is not an account of these books."""
    s = raw.strip()
    if not s.isdigit():
        raise DatevRefusal(f"{where}: {what} {raw!r} is not an account number")
    if len(s) <= h.sachkontenlaenge:
        return s, "Sachkonto"
    if len(s) == h.sachkontenlaenge + 1:
        return s, "Personenkonto"
    raise DatevRefusal(
        f"{where}: {what} {s} has {len(s)} digits; Sachkontenlänge "
        f"{h.sachkontenlaenge} allows {h.sachkontenlaenge + 1} at most"
    )


def _generalumkehr(rec: list[str], gu_idx: int | None, bu: str, where: str) -> bool:
    """The GU field, or a legacy two-digit BU key whose first digit (Berichtigungsschlüssel)
    is 2. Only two-digit keys: DATEV's own sample carries 3- and 4-digit tax keys (501,
    6501), where a leading 2 is part of the key. Both markers together are one reversal."""
    flag = rec[gu_idx].strip() if gu_idx is not None and len(rec) > gu_idx else ""
    if flag not in ("", "0", "1"):
        raise DatevRefusal(f"{where}: Generalumkehr {flag!r} is neither 0 nor 1")
    return flag == "1" or (len(bu) == 2 and bu.isdigit() and bu[0] == "2")


def read_batch(path: str) -> tuple[DatevHeader, dict[str, list[Any]]]:
    """One EXTF booking batch -> its header and its legs, column-wise."""
    label = os.path.basename(path)
    raw = Path(path).read_bytes()
    text = _decode(raw, label)
    first = text.split("\n", 1)[0]
    if first.count(";") < 20:
        # Guessing another delimiter is how an amount with a decimal comma splits in two.
        raise DatevRefusal(
            f"{label}: header has {first.count(';')} ';' and {first.count(',')} ','; the DATEV delimiter is ';'"
        )
    reader = csv.reader(io.StringIO(text), delimiter=";", quotechar='"', strict=True)
    try:
        lines = list(reader)
    except csv.Error as e:
        raise DatevRefusal(f"{label}: malformed quoting at line {reader.line_num}: {e}") from None
    if len(lines) < 2:
        raise DatevRefusal(f"{label}: no column-name line")
    h = parse_header(lines[0], label)

    names = [n.strip() for n in lines[1]]
    for pos, allowed in CORE.items():
        got = names[pos - 1] if len(names) >= pos else None
        norm = got.replace("Schluessel", "Schlüssel") if got else got
        if norm not in allowed:
            raise DatevRefusal(f"{label}: column {pos} is {got!r}; expected one of {sorted(allowed)}")
    col = {n: i for i, n in enumerate(names)}
    gu_idx = next((col[n] for n in _GU_NAMES if n in col), None)
    fs_idx = col.get("Festschreibung")
    ld_idx = col.get("Leistungsdatum")

    fingerprint = hashlib.sha256(raw).hexdigest()[:12]
    legs: dict[str, list[Any]] = {c: [] for c in (*COLUMNS, _ORIGIN_COLUMN)}
    for record, rec in enumerate(lines[2:], start=3):
        if not any(f.strip() for f in rec):
            continue
        where = f"{label} record {record}"
        if len(rec) < max(CORE):
            raise DatevRefusal(f"{where}: {len(rec)} fields, fewer than the {max(CORE)} core fields")
        if len(rec) > len(names):
            raise DatevRefusal(
                f"{where}: {len(rec)} fields, more than the {len(names)} named "
                "columns; every field after the break would be misassigned"
            )

        amount = _amount(rec[0], where, "Umsatz")
        if amount is not None and amount < 0:
            raise DatevRefusal(f"{where}: Umsatz {rec[0]!r} is negative; the sign belongs in S/H")
        sh = rec[1].strip()
        if sh not in ("S", "H"):
            raise DatevRefusal(f"{where}: Soll/Haben-Kennzeichen {rec[1]!r} is neither S nor H")
        wkz = rec[2].strip()
        if wkz not in ("", "EUR"):
            if not rec[4].strip():
                raise DatevRefusal(f"{where}: WKZ Umsatz {wkz} without Basisumsatz; no EUR amount")
            if rec[5].strip() not in ("", "EUR"):
                raise DatevRefusal(f"{where}: WKZ Basisumsatz {rec[5].strip()!r} is not EUR")
            amount = _amount(rec[4], where, "Basisumsatz")

        konto, konto_kind = _account(rec[6], h, where, "Konto")
        if not rec[7].strip():
            raise DatevRefusal(f"{where}: empty Gegenkonto; a one-legged row cannot balance")
        gegen, gegen_kind = _account(rec[7], h, where, "Gegenkonto")
        bu = rec[8].strip()
        gu = _generalumkehr(rec, gu_idx, bu, where)
        booked = _belegdatum(rec[9], h, where)
        served = _leistungsdatum(rec[ld_idx], where) if ld_idx is not None and len(rec) > ld_idx else None

        locked = h.festschreibung
        flag = rec[fs_idx].strip() if fs_idx is not None and len(rec) > fs_idx else ""
        if flag not in ("", "0", "1"):
            raise DatevRefusal(f"{where}: Festschreibung {flag!r} is neither 0 nor 1")
        if flag:
            locked = flag == "1"

        # +Soll / -Haben on Konto; a Generalumkehr keeps the side and reverses the effect.
        signed = None
        if amount is not None:
            signed = amount if sh == "S" else -amount
            if gu:
                signed = -signed
        for leg, account, kind, contra, value in (
            ("K", konto, konto_kind, gegen, signed),
            ("G", gegen, gegen_kind, konto, None if signed is None else -signed),
        ):
            legs["Konto"].append(account)
            legs["Gegenkonto"].append(contra)
            legs["Kontoart"].append(kind)
            legs["Betrag"].append(value)
            legs["Buchungsdatum"].append(booked)
            legs["Leistungsdatum"].append(served)
            legs["Leg"].append(leg)
            legs["BU-Schlüssel"].append(bu)
            legs["Generalumkehr"].append(gu)
            legs["Festschreibung"].append(locked)
            legs["SKR"].append(h.skr)
            legs["Belegfeld 1"].append(rec[10].strip())
            legs[VORZEICHEN].append(SOLL_POSITIVE)
            legs[_ORIGIN_COLUMN].append(f"{label}@{fingerprint}:{record}/{leg}")
    return h, legs


class DatevExtfReader(ReadFile):
    """Claims a folder of DATEV-Format booking batches (EXTF/DTVF, category 21)."""

    ORIGIN_COLUMN = _ORIGIN_COLUMN

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        # Vestigial, as in GdpduReader: matching is by folder and header line, not suffix.
        return (".csv",)

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        """Fixed by the field table, not by the batch: every leg has the same columns."""
        return [*COLUMNS, cls.ORIGIN_COLUMN]

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        candidates: list[str] = []
        if isinstance(data_access, DataAccessCollection):
            candidates = list(data_access.folders.values()) + list(data_access.files.values())
        elif isinstance(data_access, (str, Path)):
            candidates = [str(data_access)]
        for candidate in candidates:
            if os.path.isdir(candidate) and _batch_files(candidate):
                return candidate
        return None

    @classmethod
    def data_access_identity(cls, data_access: Any) -> str:
        """`datev:<folder>@<fingerprint>` over exactly the batches load_data reads.

        No Berater, Mandant or header text: those identify a tax advisor and a client, and an
        identity is handed to every extender. Never raises -- a folder that cannot be
        fingerprinted is still named, and the load that follows reports the actual error.
        """
        folder = str(data_access)
        name = os.path.basename(os.path.normpath(folder))
        try:
            files = _batch_files(folder)
            if not files:
                return f"datev:{name}"
            digest = hashlib.sha256()
            for path in files:
                digest.update(os.path.basename(path).encode() + b"\x00")
                digest.update(Path(path).read_bytes() + b"\x00")
        except OSError:
            return f"datev:{name}"
        return f"datev:{name}@{digest.hexdigest()[:12]}"

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        folder = str(data_access)
        files = _batch_files(folder)
        if not files:
            raise DatevRefusal(f"{folder}: no Buchungsstapel (EXTF/DTVF category 21) in this folder")
        batches = sorted((read_batch(p) + (os.path.basename(p),) for p in files), key=lambda b: (b[0].datum_von, b[2]))

        # One dossier is one client's books. Two clients summed into one total is not a
        # total of anything.
        clients = {(h.berater, h.mandant) for h, _, _ in batches}
        if len(clients) > 1:
            raise DatevRefusal(
                f"{folder}: batches of {len(clients)} clients (Berater/Mandant) in one folder; "
                "one dossier is one client's books"
            )
        # The same bookings exported twice would be counted twice.
        for (h1, _, n1), (h2, _, n2) in zip(batches, batches[1:]):
            if h2.datum_von <= h1.datum_bis:
                raise DatevRefusal(
                    f"{folder}: {n1} ({h1.datum_von}..{h1.datum_bis}) and {n2} "
                    f"({h2.datum_von}..{h2.datum_bis}) overlap; the same bookings would count twice"
                )

        arrays: dict[str, pa.Array] = {}
        for c in (*COLUMNS, cls.ORIGIN_COLUMN):
            values = [v for _, legs, _ in batches for v in legs[c]]
            arrays[c] = pa.array(values, type=_TYPES.get(c, pa.string()))
        return pa.table(arrays)
