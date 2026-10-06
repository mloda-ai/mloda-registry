"""Writes the twin fixture: one small set of SKR04 books for FY2025, in both formats.

    python make_twin.py        # rewrites gdpdu/ and datev/ next to this file

The bookings are self-made. The DATEV header and column line are taken from the ledermann
sample (MIT, see ../datev_ledermann/NOTICE) and only the fields named below are changed.

Four bookings, each written once as a DATEV row (Konto against Gegenkonto, +Soll) and as
the GDPdU Sachkonten lines for both of its accounts, each in that account's own direction:

    1  15.10.2025  1200 S / 4000      12.500,00  invoice: receivable and revenue
    2  20.11.2025  1800 S / 4000       8.750,50  cash sale
    3  05.12.2025  1800 S / 4400       1.234,56  BU 40: automatic lifted, so booked net
    4  20.12.2025  1800 S / 1200       5.000,00  the customer pays part of invoice 1

revenue = 12.500,00 + 8.750,50 + 1.234,56 = 22.485,06; receivables = 12.500,00 - 5.000,00 = 7.500,00
"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).parent
LEDERMANN = HERE.parent / "datev_ledermann" / "EXTF_Buchungsstapel.csv"
DOSSIER_A = HERE.parent / "dossier_a" / "index.xml"

# (Belegdatum, Konto, Gegenkonto, BU, amount, Belegfeld 1, text)
BOOKINGS = [
    ("15.10.2025", "1200", "4000", "", "12500,00", "RE2025-101", "Rechnung Kunde A"),
    ("20.11.2025", "1800", "4000", "", "8750,50", "KA2025-17", "Barverkauf"),
    ("05.12.2025", "1800", "4400", "40", "1234,56", "KA2025-22", "Erloese netto"),
    ("20.12.2025", "1800", "1200", "", "5000,00", "BA2025-88", "Zahlung Kunde A"),
]


def _german(amount: str) -> str:
    """1234,56 -> 1.234,56, as the GDPdU descriptor declares the grouping symbol."""
    whole, cents = amount.split(",")
    groups: list[str] = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    return ".".join([whole, *groups]) + "," + cents


def write_datev(folder: Path) -> None:
    lines = LEDERMANN.read_bytes().split(b"\r\n")
    head = lines[0].split(b";")
    head[5] = b"20260110120000000"  # erzeugt am
    head[12] = b"20250101"  # WJ-Beginn
    head[14] = b"20250101"  # Datum vom
    head[15] = b"20251231"  # Datum bis
    head[16] = b'"Zwilling 2025"'  # Bezeichnung
    head[26] = b'"04"'  # SKR
    template = lines[3].split(b";")
    rows = [b";".join(head), lines[1]]
    for booked, konto, gegen, bu, amount, beleg, text in BOOKINGS:
        row = list(template)
        row[0] = amount.encode()
        row[1] = b'"S"'
        row[6] = konto.encode()
        row[7] = gegen.encode()
        row[8] = f'"{bu}"'.encode() if bu else b""
        row[9] = (booked[:2] + booked[3:5]).encode()  # TTMM
        row[10] = f'"{beleg}"'.encode()
        row[13] = f'"{text}"'.encode()
        rows.append(b";".join(row))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "EXTF_Buchungsstapel.csv").write_bytes(b"\r\n".join(rows) + b"\r\n")


def write_gdpdu(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(DOSSIER_A, folder / "index.xml")
    out: list[str] = []
    n = 0
    for booked, konto, gegen, _, amount, _, text in BOOKINGS:
        # Each account in its own direction: the debited Sachkonto gains, the credited one
        # gains too if it is a credit account (revenue), and loses if it is a debit account.
        for account, sign in ((konto, ""), (gegen, "" if gegen.startswith("4") else "-")):
            n += 1
            out.append(f"{n};{account};{sign}{_german(amount)};{booked};{booked};{text}")
    (folder / "GL.txt").write_text("\n".join(out) + "\n", encoding="cp1252")


if __name__ == "__main__":
    write_datev(HERE / "datev")
    write_gdpdu(HERE / "gdpdu")
