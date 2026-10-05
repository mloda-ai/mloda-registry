"""GdpduReader: a descriptor-driven ReadFile subclass for GDPdU/GoBD dossiers."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET
import pyarrow as pa
from defusedxml import DefusedXmlException
from mloda.provider import FeatureSet
from mloda.user import DataAccessCollection, Options
from mloda_plugins.feature_group.input_data.read_file import ReadFile

INDEX_NAME = "index.xml"

# The citation column the reader adds. A descriptor declaring this name would be silently
# overwritten by it, so the name is reserved rather than shared.
_ORIGIN_COLUMN = "__origin"

# The admissibility evidence column. An admissibility policy stamps it; SourcesFeatureGroup
# refuses to total rows without it. Reserved for the same reason as the citation column: a
# descriptor declaring this name would be overwritten by the stamp.
ADMISSIBILITY_COLUMN = "__admissibility"

# A stamp must be an AFFIRMATIVE verdict naming its policy, not merely a non-null value.
# Checking only for non-null let "", "DENIED" and False all read as admitted, which made the
# consumer's demand meaningless. This is an in-process contract, not authentication: a
# determined producer inside the process can still write a conforming string. What it buys is
# that blank, negative and malformed verdicts are rejected rather than silently admitted.
ADMITTED_PREFIX = "admitted:"

# Both readers write `<file>@<sha12>:<record>`, DATEV adding `/<K|G>` for the leg. The
# fingerprint binds the citation to the file's content, not only to a position in it.
_CITATION = re.compile(r"(?P<file>[^@]+)@(?P<fingerprint>[0-9a-f]{12}):(?P<record>[0-9]+)(?:/(?P<leg>[KG]))?\Z")


@dataclass(frozen=True)
class Citation:
    """One cited ledger line, read field by field instead of re-split from a string."""

    file: str
    fingerprint: str
    record: int
    leg: str | None = None


def parse_citation(origin: str) -> Citation:
    """A reader's origin string as a Citation; anything else is refused, not guessed at."""
    m = _CITATION.match(origin)
    if m is None:
        raise ValueError(f"{origin!r} is not a citation: expected <file>@<sha12>:<record>[/K|/G]")
    return Citation(m["file"], m["fingerprint"], int(m["record"]), m["leg"])


def origin_format(origin: object) -> str | None:
    """ "datev" for a citation that names a leg, "gdpdu" for one that does not, else None."""
    m = _CITATION.match(origin) if isinstance(origin, str) else None
    if m is None:
        return None
    return "datev" if m["leg"] else "gdpdu"


# A policy has three things it can say about a row, not two. "I examined this row and it
# passed" is an admission. "I refused it" stops the run. The third -- "this row was never
# mine to judge" -- was being written as an admission carrying a scope note, and the only
# consumer reads the prefix, so a row the cutoff explicitly declined to evaluate was still
# totalled as though it had been cleared. A closed-period guard that lets next year's
# bookings into the answer protects nothing. Out-of-scope is therefore its own verdict, and
# is_admitted is false for it: a total whose rows a policy did not vouch for is not a total
# that policy backs.
OUTSIDE_SCOPE_PREFIX = "outside-scope:"

# The policy identifier must be a real name, not merely non-empty. Checking prefix-plus-length
# let "admitted:;lock=x", "admitted: " and "admitted:\n" all read as admitted -- the same
# too-permissive-predicate mistake one layer down.
_POLICY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


# How many rows got which verdict, in this order. The same shape travels in a receipt's
# `verdicts` and on a refusal, so a refused run is counted the way an answered one is.
VERDICT_KINDS = ("admitted", "outside-scope", "refused", "unevaluated", "malformed", "unstamped")


def verdict_kind(stamp: object) -> str:
    """The kind of one written stamp: admitted, outside-scope, or malformed."""
    if is_admitted(stamp):
        return "admitted"
    if isinstance(stamp, str) and stamp.startswith(OUTSIDE_SCOPE_PREFIX):
        return "outside-scope"
    return "malformed"


def describe_verdicts(verdicts: dict[str, int]) -> str:
    """'verdicts over 8 row(s): 7 admitted, 1 outside-scope' for a refusal's message."""
    parts = [f"{verdicts[k]} {k}" for k in VERDICT_KINDS if verdicts.get(k)]
    return f"verdicts over {sum(verdicts.values())} row(s): " + ", ".join(parts)


def admissibility_verdict(policy: str, **parameters: object) -> str:
    """Build an affirmative verdict naming the policy that issued it.

    The policy name is validated HERE too: a producer that cannot name itself must fail at
    the point of stamping, not hand the consumer a verdict it will reject later.
    """
    if not _POLICY_NAME.match(policy or ""):
        raise ValueError(
            f"admissibility policy name {policy!r} is not a valid identifier "
            f"({_POLICY_NAME.pattern}); a verdict must name the policy that issued it"
        )
    params = ";".join(f"{k}={v}" for k, v in parameters.items())
    return f"{ADMITTED_PREFIX}{policy}" + (f";{params}" if params else "")


def outside_scope_verdict(policy: str, **parameters: object) -> str:
    """Record that a policy saw a row and had no jurisdiction over it.

    Deliberately NOT an admission. It exists so the evidence can say "not mine to judge"
    without that reading as "checked and cleared".
    """
    if not _POLICY_NAME.match(policy or ""):
        raise ValueError(
            f"admissibility policy name {policy!r} is not a valid identifier "
            f"({_POLICY_NAME.pattern}); a verdict must name the policy that issued it"
        )
    params = ";".join(f"{k}={v}" for k, v in parameters.items())
    return f"{OUTSIDE_SCOPE_PREFIX}{policy}" + (f";{params}" if params else "")


def is_admitted(stamp: object) -> bool:
    """Only an affirmative verdict naming a syntactically valid policy admits a row."""
    if not isinstance(stamp, str) or not stamp.startswith(ADMITTED_PREFIX):
        return False
    policy = stamp[len(ADMITTED_PREFIX) :].split(";", 1)[0]
    return bool(_POLICY_NAME.match(policy))


@dataclass(frozen=True)
class TableSpec:
    """Everything about a table that the descriptor -- not the data -- declares."""

    url: str
    name: str
    decimal_symbol: str
    grouping_symbol: str
    delimiter: str
    encapsulator: str | None
    encoding: str
    record_delimiter: str | None
    columns: tuple[str, ...]
    numeric: frozenset[str]
    accuracy: dict[str, int]
    implied: frozenset[str]
    dates: dict[str, str]


# GDPdU declares its code page as an EMPTY element, not as text.
#   <!ELEMENT Table (URL, Name?, ..., (ANSI | Macintosh | OEM | UTF16 | UTF7 | UTF8)?, ...)>
_CODEPAGES: dict[str, str] = {
    "ANSI": "cp1252",
    "Macintosh": "mac_roman",
    "OEM": "cp850",
    "UTF16": "utf-16",
    "UTF7": "utf-7",
    "UTF8": "utf-8",
}

# Declared date formats this reader implements. An unknown one is refused at descriptor-parse
# time, so it cannot escape by happening to meet only empty values in the data.
_DATE_FORMATS: dict[str, str] = {
    "DD.MM.YYYY": "%d.%m.%Y",
    "DD/MM/YYYY": "%d/%m/%Y",
    "DD-MM-YYYY": "%d-%m-%Y",
    "YYYY-MM-DD": "%Y-%m-%d",
    "YYYYMMDD": "%Y%m%d",
}


def _text(node: Any, tag: str, default: str) -> str:
    """Absent element -> the DTD default. Present but empty -> the empty string.

    ``xml.etree`` reports ``.text is None`` for both a missing ``<X>`` and an ``<X/>``, but the
    two are different declarations: ``<DigitGroupingSymbol/>`` says this exporter writes *no*
    thousands separator. Collapsing it into the default ``.`` would make ``_decimal`` strip
    every decimal point, and the inflated total would still look like money -- the same failure
    shape as reading ``<ImpliedAccuracy>`` as ``<Accuracy>``.
    """
    found = node.find(tag)
    if found is None:
        return default
    return "" if found.text is None else found.text


def _int_text(node: Any, tag: str, default: int) -> int:
    """A declared digit count. An empty element carries no count, so the default stands."""
    raw = _text(node, tag, str(default)).strip()
    return int(raw) if raw else default


def parse_descriptor(index_path: str) -> TableSpec:
    """Schema comes from index.xml, per the GDPdU DTD. Nothing here is inferred from the data."""
    return parse_descriptor_bytes(Path(index_path).read_bytes(), index_path)


def parse_descriptor_bytes(descriptor: bytes, index_path: str) -> TableSpec:
    """Parse one *snapshot* of the descriptor.

    Callers that also fingerprint the descriptor must parse and hash the same bytes: reading
    index.xml twice would let a concurrent edit produce a citation naming descriptor B for a
    table interpreted under descriptor A.

    The DTD nests the type inside the column:
        <!ELEMENT VariableColumn (Name, Description?, (Numeric | (AlphaNumeric, MaxLength?) | Date), Map*)>
        <!ELEMENT Numeric ((ImpliedAccuracy | Accuracy)?)>
        <!ELEMENT Date (Format?)>
    """
    # A dossier is third-party input. defusedxml keeps the <!DOCTYPE ... gdpdu-01-09-2004.dtd>
    # every descriptor carries, and refuses entity declarations and external references --
    # the mechanics of an XML bomb. The DTD itself is never fetched either way.
    try:
        root = ET.fromstring(descriptor)
    except DefusedXmlException as exc:
        raise ValueError(
            f"{index_path}: the descriptor declares XML entities or external references "
            f"({type(exc).__name__}); refused, a GDPdU descriptor needs neither"
        ) from exc
    tables = root.findall(".//Table")
    if not tables:
        raise ValueError(f"{index_path}: no <Table> in descriptor")

    # <!ELEMENT Table (URL, Name?, ..., (VariableLength | FixedLength))>. A dossier routinely
    # ships several tables and a fixed-length one may come first, so scan past those instead of
    # refusing the descriptor because table[0] is not the shape this reader implements.
    table = next((t for t in tables if t.find("VariableLength") is not None), None)
    if table is None:
        raise ValueError(
            f"{index_path}: no variable-length <Table> in descriptor; fixed-length tables are not supported"
        )
    var = table.find("VariableLength")
    assert var is not None  # the table was selected above for having one

    encoding = "cp1252"
    for tag, codec in _CODEPAGES.items():
        if table.find(tag) is not None:
            encoding = codec
            break

    columns: list[str] = []
    numeric: set[str] = set()
    accuracy: dict[str, int] = {}
    implied: set[str] = set()
    dates: dict[str, str] = {}

    # VariablePrimaryKey+ then VariableColumn*, or VariableColumn+ -- both are columns here.
    for child in var:
        if child.tag not in ("VariableColumn", "VariablePrimaryKey"):
            continue
        name_node = child.find("Name")
        if name_node is None or name_node.text is None:
            continue
        col = name_node.text.strip()
        # Both of these would otherwise fail far from their cause: a duplicate name collapses in
        # the per-column dict but is appended once per occurrence, so the mismatch only surfaces
        # when Arrow builds the table; a column literally named __origin is parsed correctly and
        # then overwritten by the citation array.
        if col in (_ORIGIN_COLUMN, ADMISSIBILITY_COLUMN):
            raise ValueError(
                f"{index_path}: column name {col!r} is reserved (the citation and its admissibility evidence)"
            )
        if col in columns:
            raise ValueError(f"{index_path}: column {col!r} is declared twice")
        columns.append(col)

        num = child.find("Numeric")
        dat = child.find("Date")
        if num is not None:
            numeric.add(col)
            # <Numeric ((ImpliedAccuracy | Accuracy)?)> -- these are different declarations.
            # Accuracy counts decimal places that are PRESENT in the field. ImpliedAccuracy says
            # the decimal point is absent and the integer must be scaled: 1234 -> 12.34. Treating
            # the second as the first silently multiplies every amount by 10**n.
            # The standard's default accuracy is 0, not 2. Defaulting to 2 misread every
            # conforming dossier that omits the element -- both fixtures declare it, which
            # is exactly why the spike never caught it. 0 is also the safe direction: an
            # unshifted integer is visibly wrong, where a guessed scale silently divides
            # every amount by a hundred and still looks like money.
            imp = num.find("ImpliedAccuracy")
            if imp is not None:
                implied.add(col)
                accuracy[col] = _int_text(num, "ImpliedAccuracy", 0)
            else:
                accuracy[col] = _int_text(num, "Accuracy", 0)
        elif dat is not None:
            fmt = (_text(dat, "Format", "DD.MM.YYYY").strip() or "DD.MM.YYYY").upper()
            if fmt not in _DATE_FORMATS:
                raise ValueError(
                    f"{index_path}: column {col!r} declares date format {fmt!r}, which this reader does not implement"
                )
            dates[col] = fmt
        # else AlphaNumeric (or unspecified) -> string

    # The standard's default TextEncapsulator is the double quote. Defaulting to none split
    # quoted fields containing the delimiter straight down the middle on any dossier that
    # omitted the element.
    encapsulator = _text(var, "TextEncapsulator", '"') or None

    # An explicitly empty <ColumnDelimiter/> is not a declaration this reader can honour:
    # str.split("") raises, and guessing a separator is exactly what the descriptor exists
    # to prevent. Refuse at parse time rather than at the first row.
    delimiter = _text(var, "ColumnDelimiter", ";")
    if not delimiter:
        raise ValueError(f"{index_path}: <ColumnDelimiter/> declares no column delimiter")

    # <Range> selects which lines of the file are data -- it is how an exporter declares a
    # header to skip. Ignoring it turns a header into a booking, or fails a field count far
    # from the cause. This reader does not implement it, so it refuses by name rather than
    # reading the file under a declaration it is disregarding.
    if var.find("Range") is not None or table.find("Range") is not None:
        raise ValueError(
            f"{index_path}: the descriptor declares <Range>, which this reader does not "
            "implement; reading it would silently disregard a declared line range"
        )
    # <SkipNumBytes> is <Range>'s byte-level sibling in the DTD: a header of N bytes in front
    # of the data (OrgaMon declares 154). Ignoring it splits the header into fields and, when
    # the count happens to match, reads it as a booking -- a silent misread. Refused by name.
    if table.find("SkipNumBytes") is not None or var.find("SkipNumBytes") is not None:
        raise ValueError(
            f"{index_path}: the descriptor declares <SkipNumBytes>, which this reader does "
            "not implement; reading it would silently read a declared byte header as rows"
        )

    return TableSpec(
        url=_text(table, "URL", ""),
        name=_text(table, "Name", "table"),
        decimal_symbol=_text(table, "DecimalSymbol", ","),
        grouping_symbol=_text(table, "DigitGroupingSymbol", "."),
        delimiter=delimiter,
        record_delimiter=_text(var, "RecordDelimiter", "") or None,
        encapsulator=encapsulator,
        encoding=encoding,
        columns=tuple(columns),
        numeric=frozenset(numeric),
        accuracy=accuracy,
        implied=frozenset(implied),
        dates=dates,
    )


def _resolve_within(dossier_dir: str, url: str) -> str:
    """A descriptor is untrusted input: keep its URLs inside the dossier directory."""
    base = Path(dossier_dir).resolve()
    target = (base / url).resolve()
    if base != target and base not in target.parents:
        raise ValueError(f"descriptor URL {url!r} escapes the dossier directory")
    return str(target)


# Record delimiters an exporter declares as "a line ending". Declared CRLF is routinely
# written as bare LF (both fixtures here do it), so a literal split on the declared bytes
# would return the whole file as one record and put every field in the wrong column. A
# newline-shaped declaration is honoured as a line ending; anything else is honoured literally.
_NEWLINE_DELIMITERS = frozenset({"\r\n", "\n", "\r", "\n\r"})


def _records(text: str, spec: TableSpec) -> list[str]:
    """Split into records by the declared delimiter, not by assumption."""
    if spec.record_delimiter is None or spec.record_delimiter in _NEWLINE_DELIMITERS:
        return text.splitlines()
    return text.split(spec.record_delimiter)


def _split(line: str, spec: TableSpec) -> list[str]:
    """Honour TextEncapsulator; the DTD allows quoted fields containing the delimiter."""
    if not spec.encapsulator:
        return line.split(spec.delimiter)
    return next(csv.reader([line], delimiter=spec.delimiter, quotechar=spec.encapsulator))


def _decimal(raw: str, spec: TableSpec, column: str) -> Decimal | None:
    """German decimals: '1.234,56' is 1234.56, not 1.234. Decimal, never float.

    An empty field is an absent amount, not a parse error -- the same treatment dates get.
    Under ImpliedAccuracy the field carries no decimal symbol at all and the integer is scaled.
    """
    raw = raw.strip()
    if not raw:
        return None
    # Either symbol may be declared as absent (<DigitGroupingSymbol/>). str.replace("") splices
    # the replacement between every character, so an empty declaration means "do nothing".
    digits = raw.replace(spec.grouping_symbol, "") if spec.grouping_symbol else raw
    if not any(c.isdigit() for c in digits):
        raise ValueError(f"column {column!r}: {raw!r} carries no digits once the declared symbols are removed")
    if column in spec.implied:
        if spec.decimal_symbol and spec.decimal_symbol in digits:
            raise ValueError(f"column {column!r} declares ImpliedAccuracy but {raw!r} carries a decimal symbol")
        cleaned = digits
    else:
        cleaned = digits.replace(spec.decimal_symbol, ".") if spec.decimal_symbol else digits
    # Digits alone do not make a number: "12.500,00 EUR" survives the checks above and then
    # dies inside Decimal. A bare InvalidOperation names neither the column nor the value, so
    # it would surface as a stack trace rather than as a statement about the dossier.
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        raise ValueError(f"column {column!r}: {raw!r} is not a number under the declared symbols") from None
    if column not in spec.implied:
        return value

    # scaleb is ARITHMETIC, so it rounds to the active context -- 28 significant digits by
    # default, narrower than the decimal128(38, s) this reader is allowed to produce. A wide
    # ImpliedAccuracy amount was therefore rounded here, at parse time, and no later context
    # could recover what was already gone. Decimal(str) itself is exact; only the shift needed
    # room. The digits of the value plus the declared shift is the exact requirement.
    shift = spec.accuracy[column]
    with localcontext() as ctx:
        ctx.prec = len(value.as_tuple().digits) + shift + 2
        return value.scaleb(-shift)


def _civil_date(raw: str, fmt: str) -> date | None:
    """A date format carries no timezone, so this stays a civil date.

    An empty field is a genuinely absent date, not a parse error: the lock-date gate has to be
    able to see a missing keying timestamp on a single line. Unknown formats were already
    refused when the descriptor was parsed.
    """
    raw = raw.strip()
    if not raw:
        return None
    return datetime.strptime(raw, _DATE_FORMATS[fmt.strip().upper()]).date()


def _fingerprint(descriptor_bytes: bytes, raw: bytes) -> str:
    """The dossier fingerprint shared by row citations and the data access identity.

    The descriptor is bound into the fingerprint, not just the data: index.xml decides how
    the same bytes are read -- change <DecimalSymbol> and every amount moves while the data
    file is untouched. A citation that only covered the data would survive that edit.
    Truncated to 12 hex characters for readability, which makes it a change detector for
    ordinary edits, not a cryptographic seal against someone hunting a 48-bit collision.
    """
    return hashlib.sha256(descriptor_bytes + b"\x00" + raw).hexdigest()[:12]


class GdpduReader(ReadFile):
    """Claims a dossier *directory* by its index.xml, not a file by suffix."""

    #: public alias; parse_descriptor reserves this name against a descriptor collision
    ORIGIN_COLUMN = _ORIGIN_COLUMN

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        # Vestigial for THIS reader: matching is directory-based (see
        # match_subclass_data_access below), so nothing here is selected by suffix. The value
        # is kept because the stock ReadFile base still calls it on paths this class does not
        # take, and ".xml" is the descriptor a dossier is claimed by.
        return (".xml",)

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        """The columns the DESCRIPTOR declares, plus the origin column this reader adds.

        mloda 0.13.0 made this method load-bearing: ReadFile treats a class that does not
        override it as unable to confirm a chain- or column-separated feature name, and
        declines such a name while matching. This reader CAN confirm one -- index.xml names
        every column before a byte of data is read -- so it answers rather than bypassing the
        guard by omission. `__admissibility` is deliberately absent: a policy adds it, not the
        reader, and claiming a column this class does not produce would be the same silent
        overstatement the rest of this package exists to prevent.
        """
        index_path = os.path.join(file_name, INDEX_NAME) if os.path.isdir(file_name) else file_name
        spec = parse_descriptor_bytes(Path(index_path).read_bytes(), index_path)
        return [*spec.columns, cls.ORIGIN_COLUMN]

    @classmethod
    def data_access_identity(cls, data_access: Any) -> str:
        """`gdpdu:<dossier>@<fingerprint>` -- what mloda 0.14.0 (#1577) hands every extender.

        The default for an existing local path is the path as given, which publishes the
        host's directory layout to every lineage and audit extender. This names the dossier
        folder and the same descriptor-plus-data fingerprint each row citation carries, so an
        extender's record joins with the citations it vouches for. It reads both files once
        more to do so. On mloda 0.13.0 core never calls it.

        A dossier that cannot be fingerprinted still gets a name, without the fingerprint:
        claiming less is honest, and the load that follows reports the actual error.
        """
        dossier_dir = str(data_access)
        name = os.path.basename(os.path.normpath(dossier_dir))
        try:
            descriptor_bytes = Path(os.path.join(dossier_dir, INDEX_NAME)).read_bytes()
            spec = parse_descriptor_bytes(descriptor_bytes, os.path.join(dossier_dir, INDEX_NAME))
            raw = Path(_resolve_within(dossier_dir, spec.url)).read_bytes()
        except (OSError, ValueError, ET.ParseError):
            return f"gdpdu:{name}"
        return f"gdpdu:{name}@{_fingerprint(descriptor_bytes, raw)}"

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        """Stock ReadFile takes the first suffix match in listdir and then demands the requested
        feature name be a source column. A dossier has neither property, so claim the directory."""
        candidates: list[str] = []
        if isinstance(data_access, DataAccessCollection):
            candidates = list(data_access.folders.values()) + list(data_access.files.values())
        elif isinstance(data_access, (str, Path)):
            candidates = [str(data_access)]

        for candidate in candidates:
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, INDEX_NAME)):
                return candidate
            if os.path.basename(candidate) == INDEX_NAME and os.path.isfile(candidate):
                return os.path.dirname(candidate)
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        """Returns a materialised pa.Table -- not a FileSource -- so an INPUT_DATA_LOAD
        extender can inspect actual rows after delegating."""
        dossier_dir = str(data_access)
        index_path = os.path.join(dossier_dir, INDEX_NAME)

        # Read the descriptor ONCE and both parse and hash those exact bytes. Reading it twice
        # would let an edit between the two reads produce a citation naming a descriptor that
        # is not the one the table was interpreted under.
        descriptor_bytes = Path(index_path).read_bytes()
        spec = parse_descriptor_bytes(descriptor_bytes, index_path)
        data_path = _resolve_within(dossier_dir, spec.url)
        # A bare FileNotFoundError names a path; the defect is a declaration with no file.
        if not os.path.isfile(data_path):
            raise ValueError(f"{index_path}: <URL>{spec.url}</URL> names a data file that is absent from the dossier")

        raw = Path(data_path).read_bytes()
        content_hash = _fingerprint(descriptor_bytes, raw)
        file_label = os.path.basename(data_path)

        cols: dict[str, list[Any]] = {c: [] for c in spec.columns}
        origins: list[str] = []

        try:
            text = raw.decode(spec.encoding)
        except UnicodeDecodeError as exc:
            # A crash names a codec and a position; the refusal names the declaration it
            # contradicts. The offset is a byte offset into the data file, not a row.
            raise ValueError(
                f"{file_label}: byte {exc.start} is not valid in the declared encoding "
                f"{spec.encoding} ({exc.reason}); the descriptor's code page does not match "
                "the data"
            ) from exc
        records = _records(text, spec)

        for row_no, line in enumerate(records, start=1):
            if not line.strip():
                continue
            values = _split(line, spec)
            # zip() would truncate to the shorter side and hide both failure directions: a
            # short row leaves the trailing columns one append behind (pa.table then dies on
            # mismatched lengths, far from the cause), and a long row -- an unescaped
            # delimiter inside a field -- is worse, because columns are assigned by position:
            # every field after the break lands under the NEXT declared column and whatever
            # runs past the last one is dropped in silence. The descriptor declares the column
            # count; a row that disagrees with it is a mismatch to report, not a shape to guess.
            if len(values) != len(spec.columns):
                raise ValueError(
                    f"{file_label} record {row_no}: descriptor declares {len(spec.columns)} "
                    f"columns, row carries {len(values)} fields"
                )
            for col, value in zip(spec.columns, values):
                if col in spec.numeric:
                    cols[col].append(_decimal(value, spec, col))
                elif col in spec.dates:
                    cols[col].append(_civil_date(value, spec.dates[col]))
                else:
                    cols[col].append(value.strip())
            # A citation must survive the file being touched, so it carries a content hash.
            origins.append(f"{file_label}@{content_hash}:{row_no}")

        arrays: dict[str, pa.Array] = {}
        for col in spec.columns:
            if col in spec.numeric:
                arrays[col] = pa.array(cols[col], type=pa.decimal128(38, spec.accuracy[col]))
            elif col in spec.dates:
                arrays[col] = pa.array(cols[col], type=pa.date32())
            else:
                arrays[col] = pa.array(cols[col], type=pa.string())
        arrays[cls.ORIGIN_COLUMN] = pa.array(origins, type=pa.string())
        return pa.table(arrays)
