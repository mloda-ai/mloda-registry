"""GDPdU claims: each test pins one claim the package makes.

The subprocess proofs live in proofs/: each one registers classes process-wide, so it runs
in its own process.
"""

import shutil
import subprocess  # nosec
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
from mloda.provider import FeatureSet
from mloda.user import DataAccessCollection, Options, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.german_ledger.policy import (
    AdmissibilityRefused,
    JournalFeatureGroup,
)
from mloda.community.feature_groups.german_ledger.reader import GdpduReader, _resolve_within, parse_descriptor
from mloda.community.feature_groups.german_ledger.skr import SKR04_2025, SkrAccountFeatureGroup
from mloda.community.feature_groups.german_ledger.sources import SourcesFeatureGroup  # noqa: F401

from ._host import PLUGINS, plugins_with
from ._host import TestClosing2025 as Closing2025  # the one live admissibility policy in this process

HERE = Path(__file__).parent
PROOFS = HERE / "proofs"


def _without_column(xml: str, name: str) -> str:
    """Drop one <VariableColumn>...<Name>{name}</Name>...</VariableColumn> block."""
    import re

    return re.sub(
        r"\s*<VariableColumn>\s*<Name>" + re.escape(name) + r"</Name>.*?</VariableColumn>", "", xml, flags=re.S
    )


def _drop_column(dossier: Path, name: str) -> None:
    """Remove a column from the descriptor AND the field it declares from every record.

    The descriptor is authoritative about the column count, so dropping the declaration
    alone would leave a dossier the reader is right to refuse. What these tests are about
    is a dump that genuinely never carried the column.
    """
    idx = dossier / "index.xml"
    position = parse_descriptor(str(idx)).columns.index(name)
    idx.write_text(_without_column(idx.read_text(), name))

    gl = dossier / "GL.txt"
    kept = []
    for line in gl.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split(";")
        del fields[position]
        kept.append(";".join(fields))
    gl.write_text("\n".join(kept) + "\n")


FIX = HERE / "fixtures"


def _run_proof(name: str) -> "subprocess.CompletedProcess[str]":
    """Run one proofs/<name>.py in its own process: each registers classes process-wide."""
    # Runs our own proof scripts, not external input.
    script = str(PROOFS / f"{name}.py")
    return subprocess.run([sys.executable, script], capture_output=True, text=True)  # nosec B603


def test_collision_gate() -> None:
    """The gate is load-bearing, and the failure it prevents is reproducible."""
    assert SkrAccountFeatureGroup.match_feature_group_criteria("revenue", Options()) is True
    assert SkrAccountFeatureGroup.match_feature_group_criteria("revenue__sources", Options()) is False
    # The reader-backed root is now the journal, so that is where the gate has to hold.
    dossier = DataAccessCollection(folders={str(FIX / "dossier_a")})
    assert JournalFeatureGroup.match_feature_group_criteria("gdpdu_journal", Options(), dossier) is True
    assert JournalFeatureGroup.match_feature_group_criteria("revenue__sources", Options(), dossier) is False

    proof = _run_proof("proof_collision")
    assert proof.returncode == 0 and "COLLISION REPRODUCED" in proof.stdout, proof.stdout + proof.stderr


def test_plugin_loader_all() -> None:
    """mloda#1745 pinned: after PluginLoader.all(), stock ReadFileFeature claims our names.

    GdpduReader stays a ReadFile subclass by decision (30 Sep), so the stock family asks
    it too. Entry-point plugins are only discovered through PluginLoader.all(), so this is
    the packaged path, not an edge case. When #1745 lands upstream this fails on purpose:
    flip it to assert the revenue total instead.
    """
    proof = _run_proof("proof_plugin_loader_all")
    assert proof.returncode == 0, proof.stdout + proof.stderr
    assert "COLLISION REPRODUCED" in proof.stdout
    assert "ReadFileFeature" in proof.stdout, "the second claimant must be the stock family"


def test_german_decimal() -> None:
    t = GdpduReader.load_data(str(FIX / "dossier_a"), FeatureSet())
    amounts = [str(v) for v in t.column("Betrag").to_pylist()]
    # Both directions of the German symbol pair: '.' groups and ',' decimates. Reading '.' as
    # the decimal point would turn 12.500,00 into 12.50 and 1.234,56 into 1.234 -- the second
    # still looks like money, which is why it is asserted rather than assumed.
    assert amounts[0] == "12500.00", "12.500,00 must not parse as 12.50"
    assert amounts[3] == "1234.56", "1.234,56 must not parse as 1.234"
    assert pa.types.is_decimal(t.schema.field("Betrag").type)
    assert pa.types.is_date(t.schema.field("Buchungsdatum").type), "civil date, no invented timezone"


def test_path_confinement() -> None:
    try:
        _resolve_within(str(FIX / "dossier_a"), "../../../etc/passwd")
    except ValueError:
        return
    raise AssertionError("path escape was allowed")


def test_origins_survive_aggregation() -> None:
    res = mloda.run_all(
        features=["revenue__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(FIX / "dossier_a")}),
        plugin_collector=PLUGINS,
    )
    cols = {c for t in res for c in t.column_names}
    assert cols == {"revenue__sources~value", "revenue__sources~origins", "revenue__sources~receipt"}, cols
    total = [t.column("revenue__sources~value").to_pylist()[0] for t in res][0]
    origins = [t.column("revenue__sources~origins").to_pylist()[0] for t in res][0]
    assert str(total) == "45385.06", total
    assert len(origins) == 5 and all(":" in o and "@" in o for o in origins)


def test_guard_fails_closed_without_second_clock() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "no_clock"
        shutil.copytree(FIX / "dossier_a", d)
        _drop_column(d, "Erfassungsdatum")
        try:
            mloda.run_all(
                features=["revenue__sources"],
                compute_frameworks={PyArrowTable},
                data_access_collection=DataAccessCollection(folders={str(d)}),
                plugin_collector=PLUGINS,
            )
        except Exception as e:
            assert "cannot be evaluated" in str(e), e
            return
        raise AssertionError("guard failed open")


def test_reader_loads_dossier_without_second_clock() -> None:
    """Fail-closed is the guard's job, not the reader's: the dump must still load."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "no_clock"
        shutil.copytree(FIX / "dossier_a", d)
        _drop_column(d, "Erfassungsdatum")
        t = GdpduReader.load_data(str(d), FeatureSet())
        assert "Erfassungsdatum" not in t.column_names and t.num_rows == 7


def test_both_concepts_in_one_request() -> None:
    """mloda is about declaring SETS of features: two concepts must share one request.

    Filtering per concept gave each a different length and Arrow rejected the table with
    "expected length 1 but got length 5". Common ledger grain + null pairs fixes it.
    """
    res = mloda.run_all(
        features=["revenue__sources", "receivables__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(FIX / "dossier_a")}),
        plugin_collector=PLUGINS,
    )
    got = {c: t.column(c).to_pylist()[0] for t in res for c in t.column_names}
    assert set(got) == {
        "revenue__sources~value",
        "revenue__sources~origins",
        "receivables__sources~value",
        "receivables__sources~origins",
        "revenue__sources~receipt",
        "receivables__sources~receipt",
    }, set(got)
    assert str(got["revenue__sources~value"]) == "45385.06", got["revenue__sources~value"]
    assert str(got["receivables__sources~value"]) == "5000.00", got["receivables__sources~value"]
    assert len(got["revenue__sources~origins"]) == 5
    assert len(got["receivables__sources~origins"]) == 1


def test_skr04_not_skr03() -> None:
    """1400 is abziehbare Vorsteuer in SKR04; SKR03 receivables must not leak in."""
    assert 1200 in SKR04_2025["receivables"] and 1400 not in SKR04_2025["receivables"]
    # 4500-4505 is Sonderbetriebseinnahmen -- still class 4, still not Umsatzerloese.
    # (sonstige betriebliche Ertraege start at 4830; see the SKR04_2025 note in skr.py.)
    assert 4000 in SKR04_2025["revenue"] and 4500 not in SKR04_2025["revenue"]


def test_per_row_missing_keying_date_fails_closed() -> None:
    """A single line in the closed period with an EMPTY keying date must not slip through."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "gap"
        shutil.copytree(FIX / "dossier_a", d)
        gl = d / "GL.txt"
        gl.write_text(gl.read_text() + "9;4000;7.000,00;30.12.2025;;Ohne Erfassungsdatum\n")

        table = GdpduReader.load_data(str(d), FeatureSet())
        assert table.column("Erfassungsdatum").to_pylist()[-1] is None, "empty field must read as null"

        try:
            mloda.run_all(
                features=["revenue__sources"],
                compute_frameworks={PyArrowTable},
                data_access_collection=DataAccessCollection(folders={str(d)}),
                plugin_collector=PLUGINS,
            )
        except Exception as e:
            assert "cannot be cleared" in str(e) and "never keyed" in str(e), e
            return
        raise AssertionError("a line with no keying date failed open")


def test_unimplemented_date_format_is_refused_at_descriptor_parse() -> None:
    """Refusal belongs at parse time; at value time an unknown format escapes on empty fields."""
    from mloda.community.feature_groups.german_ledger.reader import _civil_date, parse_descriptor

    assert _civil_date("20251231", "YYYYMMDD") == date(2025, 12, 31)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "badfmt"
        shutil.copytree(FIX / "dossier_a", d)
        idx = d / "index.xml"
        idx.write_text(
            idx.read_text().replace(
                "<Date><Format>DD.MM.YYYY</Format></Date>", "<Date><Format>MM-DD-YY</Format></Date>", 1
            )
        )
        try:
            parse_descriptor(str(idx))
        except ValueError as e:
            assert "does not implement" in str(e), e
            return
        raise AssertionError("an unknown declared format was accepted")


def test_descriptor_is_dtd_shaped() -> None:
    """The type is nested inside VariableColumn, per the GDPdU DTD -- not a sibling of Name.

    A parser matching <Numeric><Name>..</Name></Numeric> would read a real descriptor as
    all-strings and only ever pass against its own invented fixtures.
    """
    xml = (FIX / "dossier_a" / "index.xml").read_text()
    assert "<VariableColumn>" in xml and "<VariablePrimaryKey>" in xml
    assert "<Numeric><Accuracy>2</Accuracy></Numeric>" in xml
    assert "<ANSI/>" in xml, "code page is an EMPTY element, not text"

    spec = parse_descriptor(str(FIX / "dossier_a" / "index.xml"))
    assert spec.encoding == "cp1252" and spec.encapsulator == '"'
    assert spec.numeric == {"Betrag"} and spec.accuracy["Betrag"] == 2
    assert set(spec.dates) == {"Buchungsdatum", "Erfassungsdatum"}
    # dossier_b declares <UTF8/>, so both code-page branches are exercised
    assert parse_descriptor(str(FIX / "dossier_b" / "index.xml")).encoding == "utf-8"


def test_an_unpriced_contributing_line_makes_the_total_unstateable() -> None:
    """A matching line with an empty amount is an attested row whose amount nobody knows.

    This test previously asserted the opposite -- that the total stayed 45385.06 while the
    unpriced row kept its citation. That is the known subtotal wearing the authority of a
    complete answer: SQL's SUM skips the null, so the caller sees a confident number cited
    to a row that was never priced. It is the sourceless-zero failure with better manners,
    treating unknown as zero. The total is now null; the citations still say what was found.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "gap_amount"
        shutil.copytree(FIX / "dossier_a", d)
        gl = d / "GL.txt"
        gl.write_text(gl.read_text() + "9;4000;;20.10.2025;21.10.2025;Betrag fehlt\n")

        table = GdpduReader.load_data(str(d), FeatureSet())
        assert table.column("Betrag").to_pylist()[-1] is None

        res = mloda.run_all(
            features=["revenue__sources"],
            compute_frameworks={PyArrowTable},
            data_access_collection=DataAccessCollection(folders={str(d)}),
            plugin_collector=PLUGINS,
        )
        got = {c: t.column(c).to_pylist()[0] for t in res for c in t.column_names}
        # six revenue lines now, one of them unpriced: no stateable total, six citations
        assert got["revenue__sources~value"] is None, got["revenue__sources~value"]
        assert len(got["revenue__sources~origins"]) == 6, got["revenue__sources~origins"]


def test_implied_accuracy_shifts_the_decimal_point() -> None:
    """<ImpliedAccuracy> is not <Accuracy>: the point is absent from the field, not present.

    Treating the two as the same declaration multiplies every amount by 10**n and the total
    still looks like money, which is the worst kind of wrong.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "implied"
        shutil.copytree(FIX / "dossier_a", d)
        idx = d / "index.xml"
        idx.write_text(
            idx.read_text().replace(
                "<Numeric><Accuracy>2</Accuracy></Numeric>",
                "<Numeric><ImpliedAccuracy>2</ImpliedAccuracy></Numeric>",
                1,
            )
        )
        gl = d / "GL.txt"
        gl.write_text("1;4000;1250000;15.10.2025;16.10.2025;Umsatz ohne Komma\n")

        spec = parse_descriptor(str(idx))
        assert "Betrag" in spec.implied and spec.accuracy["Betrag"] == 2

        t = GdpduReader.load_data(str(d), FeatureSet())
        assert str(t.column("Betrag").to_pylist()[0]) == "12500.00", "1250000 must scale to 12500.00"


def test_declared_crlf_record_delimiter_survives_lf_data() -> None:
    """Exporters declare CRLF and write LF. A literal split makes the file one record."""
    from mloda.community.feature_groups.german_ledger.reader import _records

    spec = parse_descriptor(str(FIX / "dossier_a" / "index.xml"))
    assert spec.record_delimiter == "\r\n", "the fixture really does declare CRLF"

    lf_only = "a;b\nc;d\n"
    assert _records(lf_only, spec) == ["a;b", "c;d"]

    t = GdpduReader.load_data(str(FIX / "dossier_a"), FeatureSet())
    assert t.num_rows > 1, "declared CRLF over LF data must not collapse to a single record"


def test_missing_booking_date_fails_closed() -> None:
    """A null booking date cannot be shown to fall outside the closed period, so it cannot pass."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "unbooked"
        shutil.copytree(FIX / "dossier_a", d)
        gl = d / "GL.txt"
        gl.write_text(gl.read_text() + "9;4000;7.000,00;;16.01.2026;Ohne Buchungsdatum\n")

        try:
            mloda.run_all(
                features=["revenue__sources"],
                compute_frameworks={PyArrowTable},
                data_access_collection=DataAccessCollection(folders={str(d)}),
                plugin_collector=PLUGINS,
            )
        except Exception as e:
            assert "no booking date" in str(e), e
            return
        raise AssertionError("a line with no booking date failed open")


def test_first_variable_length_table_wins_over_a_leading_fixed_length_one() -> None:
    """A descriptor may ship several tables, and the first need not be the readable one.

    Taking ``.//Table``[0] and then raising on a missing <VariableLength> turns a dossier
    whose first table is fixed-length into a hard failure instead of a skip.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "mixed"
        shutil.copytree(FIX / "dossier_a", d)
        idx = d / "index.xml"
        fixed = (
            "    <Table>\n"
            "      <URL>STAMM.txt</URL>\n"
            "      <Name>Stammdaten</Name>\n"
            "      <ANSI/>\n"
            "      <FixedLength>\n"
            "        <Length>80</Length>\n"
            "        <FixedColumn><Name>Konto</Name><AlphaNumeric/><Length>8</Length></FixedColumn>\n"
            "      </FixedLength>\n"
            "    </Table>\n"
        )
        # inject the fixed-length table BEFORE the variable-length one
        idx.write_text(idx.read_text().replace("    <Table>\n", fixed + "    <Table>\n", 1))

        spec = parse_descriptor(str(idx))
        assert spec.url == "GL.txt", spec.url
        assert spec.columns[0] == "Satznr" and "Betrag" in spec.numeric

        t = GdpduReader.load_data(str(d), FeatureSet())
        assert t.num_rows == 7


def test_no_variable_length_table_is_refused_not_silently_read() -> None:
    """A fixed-length-only descriptor is out of scope, and must say so."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "fixed_only"
        shutil.copytree(FIX / "dossier_a", d)
        idx = d / "index.xml"
        xml = idx.read_text()
        xml = (
            xml[: xml.index("<VariableLength>")]
            + "<FixedLength><Length>80</Length></FixedLength>\n    </Table>\n  </Media>\n</DataSet>\n"
        )
        idx.write_text(xml)
        try:
            parse_descriptor(str(idx))
        except ValueError as e:
            assert "no variable-length" in str(e), e
            return
        raise AssertionError("a fixed-length-only descriptor was accepted")


def test_empty_element_is_not_an_absent_element() -> None:
    """<DigitGroupingSymbol/> declares NO thousands separator; absent declares the default.

    xml.etree gives .text is None for both. Reading the empty element as the default '.'
    strips every decimal point from the file: 12.500,00 -> 1250000, still shaped like money.
    """
    import defusedxml.ElementTree as ET

    from mloda.community.feature_groups.german_ledger.reader import _text

    node = ET.fromstring("<Table><DigitGroupingSymbol/></Table>")
    assert _text(node, "DigitGroupingSymbol", ".") == "", "empty element must not take the default"
    assert _text(node, "DecimalSymbol", ",") == ",", "absent element must take the default"

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "no_grouping"
        shutil.copytree(FIX / "dossier_a", d)
        idx = d / "index.xml"
        # An exporter that writes no thousands separator and uses '.' as the decimal symbol.
        idx.write_text(
            idx.read_text()
            .replace("<DigitGroupingSymbol>.</DigitGroupingSymbol>", "<DigitGroupingSymbol/>")
            .replace("<DecimalSymbol>,</DecimalSymbol>", "<DecimalSymbol>.</DecimalSymbol>")
        )
        gl = d / "GL.txt"
        gl.write_text("1;4000;12500.00;15.10.2025;16.10.2025;Umsatz\n")

        spec = parse_descriptor(str(idx))
        assert spec.grouping_symbol == "" and spec.decimal_symbol == "."

        t = GdpduReader.load_data(str(d), FeatureSet())
        assert str(t.column("Betrag").to_pylist()[0]) == "12500.00", t.column("Betrag").to_pylist()


def test_row_field_count_must_match_the_declared_column_count() -> None:
    """zip() truncates to the shorter side and hides both directions of a mismatch.

    A short row leaves the trailing columns one append behind, so pa.table dies on unequal
    array lengths far from the cause. A long row -- an unescaped delimiter inside a field --
    is worse: columns are assigned by position, so every field after the break lands under the
    next declared column and zip drops whatever runs past the last one, leaving a row that
    still parses and an amount that still looks like money.
    """
    for extra, shape in (
        ("9;4000;7.000,00;30.12.2025\n", "6 columns, row carries 4"),
        ("9;4000;7.000,00;30.12.2025;31.12.2025;a;b\n", "6 columns, row carries 7"),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "ragged"
            shutil.copytree(FIX / "dossier_a", d)
            gl = d / "GL.txt"
            gl.write_text(gl.read_text() + extra)
            try:
                GdpduReader.load_data(str(d), FeatureSet())
            except ValueError as e:
                assert shape in str(e), (shape, str(e))
                continue
            raise AssertionError(f"a ragged row was accepted: {extra!r}")


def test_all_null_amounts_return_null_not_a_sourced_zero() -> None:
    """Lines found but no amount known is the sourceless-zero hole wearing citations.

    Summing an all-null set to Decimal(0) answers "zero" to a question nobody can answer.
    The origins must still come back: we did find the lines.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "amountless"
        shutil.copytree(FIX / "dossier_a", d)
        gl = d / "GL.txt"
        # every revenue line keeps its account and dates, and loses its amount
        kept = []
        for line in gl.read_text().splitlines():
            f = line.split(";")
            if f[1].startswith("4"):
                f[2] = ""
            kept.append(";".join(f))
        gl.write_text("\n".join(kept) + "\n")

        res = mloda.run_all(
            features=["revenue__sources"],
            compute_frameworks={PyArrowTable},
            data_access_collection=DataAccessCollection(folders={str(d)}),
            plugin_collector=PLUGINS,
        )
        got = {c: t.column(c).to_pylist()[0] for t in res for c in t.column_names}
        assert got["revenue__sources~value"] is None, got["revenue__sources~value"]
        assert len(got["revenue__sources~origins"]) == 5, got["revenue__sources~origins"]


def test_citation_binds_the_descriptor_not_only_the_data() -> None:
    """index.xml decides how the same bytes are read, so it belongs in the fingerprint.

    Change <DecimalSymbol> and every amount moves while GL.txt is untouched. A citation
    covering only the data file would survive that edit and still look authoritative.
    """

    def origins_of(dossier: Path) -> list[str]:
        origins: list[str] = GdpduReader.load_data(str(dossier), FeatureSet()).column("__origin").to_pylist()
        return origins

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base"
        shutil.copytree(FIX / "dossier_a", base)
        before = origins_of(base)

        touched_descriptor = Path(tmp) / "descriptor_edited"
        shutil.copytree(FIX / "dossier_a", touched_descriptor)
        idx = touched_descriptor / "index.xml"
        idx.write_text(idx.read_text().replace("<Name>Sachkonten</Name>", "<Name>Sachkonten </Name>"))
        assert origins_of(touched_descriptor) != before, "descriptor edit left the citation unchanged"

        touched_data = Path(tmp) / "data_edited"
        shutil.copytree(FIX / "dossier_a", touched_data)
        gl = touched_data / "GL.txt"
        gl.write_text(gl.read_text().replace("Umsatz Oktober", "Umsatz Oktobers"))
        assert origins_of(touched_data) != before, "data edit left the citation unchanged"


def test_a_dossier_running_past_the_closed_period_yields_no_number() -> None:
    """This test previously asserted the contradiction at the heart of the chain.

    It added a February 2026 revenue line and asserted the total ROSE to 52385.06. So the
    guard protected 2025 rows from late keying while 2026 revenue walked into the answer
    under that same cutoff's authority. Either `revenue` is the period's total, and that
    was wrong, or it is an all-time total, and the cutoff story described nothing.

    An empty Erfassungsdatum is still only the guard's business on a line booked into the
    closed period -- a 2026 booking is not a late entry into 2025, and is not refused.
    But it is not admitted either: the policy had no jurisdiction, says so with an
    out-of-scope verdict, and the aggregation totals only rows a policy affirmatively
    vouched for. So the request returns no number rather than a total mixing vetted and
    unvetted rows. Scoping a total to a period the feature name carries is the work this
    points at, and it is not in this spike.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "after_period"
        shutil.copytree(FIX / "dossier_a", d)
        gl = d / "GL.txt"
        gl.write_text(gl.read_text() + "9;4000;7.000,00;03.02.2026;;Buchung nach Periodenende\n")

        from mloda.community.feature_groups.german_ledger.sources import InadmissibleTotal

        try:
            mloda.run_all(
                features=["revenue__sources"],
                compute_frameworks={PyArrowTable},
                data_access_collection=DataAccessCollection(folders={str(d)}),
                plugin_collector=PLUGINS,
            )
        except Exception as exc:
            root: BaseException = exc
            while (deeper := root.__cause__ or root.__context__) is not None:
                root = deeper
            assert isinstance(root, InadmissibleTotal), f"{type(root).__name__}: {root}"
            assert "outside-scope" in str(root), root
            return
        raise AssertionError("a dossier running past the closed period produced a total")


def test_empty_column_delimiter_is_refused_at_descriptor_parse() -> None:
    """str.split("") raises, and guessing a separator is what the descriptor exists to prevent."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "no_delim"
        shutil.copytree(FIX / "dossier_a", d)
        idx = d / "index.xml"
        idx.write_text(idx.read_text().replace("<ColumnDelimiter>;</ColumnDelimiter>", "<ColumnDelimiter/>"))
        try:
            parse_descriptor(str(idx))
        except ValueError as e:
            assert "no column delimiter" in str(e), e
            return
        raise AssertionError("an empty <ColumnDelimiter/> was accepted")


def test_reserved_and_duplicate_column_names_are_refused() -> None:
    """Both would otherwise fail far from their cause, inside Arrow table construction."""
    reserved = (
        "<VariableColumn>\n          <Name>__origin</Name>\n          <AlphaNumeric/>\n        </VariableColumn>\n"
    )
    duplicate = "<VariableColumn>\n          <Name>Konto</Name>\n          <AlphaNumeric/>\n        </VariableColumn>\n"
    for injected, expected in ((reserved, "is reserved"), (duplicate, "declared twice")):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "badnames"
            shutil.copytree(FIX / "dossier_a", d)
            idx = d / "index.xml"
            idx.write_text(
                idx.read_text().replace("      </VariableLength>", "        " + injected + "      </VariableLength>", 1)
            )
            try:
                parse_descriptor(str(idx))
            except ValueError as e:
                assert expected in str(e), (expected, str(e))
                continue
            raise AssertionError(f"a descriptor declaring {expected!r} was accepted")


def test_a_field_of_only_separators_is_not_a_number() -> None:
    """'.' with '.' declared as the grouping symbol strips to '', and Decimal('') raises opaquely."""
    from mloda.community.feature_groups.german_ledger.reader import _decimal

    spec = parse_descriptor(str(FIX / "dossier_a" / "index.xml"))
    assert _decimal("", spec, "Betrag") is None, "an empty field is an absent amount"
    try:
        _decimal(".", spec, "Betrag")
    except ValueError as e:
        assert "carries no digits" in str(e), e
        return
    raise AssertionError("a separator-only field was accepted as a number")


def test_descriptor_is_parsed_from_the_bytes_that_are_fingerprinted() -> None:
    """Reading index.xml twice would let a citation name a descriptor the table never used."""
    from mloda.community.feature_groups.german_ledger.reader import parse_descriptor_bytes

    raw = (FIX / "dossier_a" / "index.xml").read_bytes()
    assert (
        parse_descriptor_bytes(raw, "<snapshot>").columns
        == parse_descriptor(str(FIX / "dossier_a" / "index.xml")).columns
    )
    import inspect

    body = inspect.getsource(GdpduReader.load_data)
    assert body.count("read_bytes()") == 2, "descriptor and data are each read exactly once"
    assert "parse_descriptor_bytes(descriptor_bytes" in body, "the parsed bytes must be the hashed bytes"


def test_a_field_with_digits_and_junk_names_its_column() -> None:
    """Digits alone do not make a number, and Decimal's own error names neither the column
    nor the value -- it would reach the caller as a stack trace, not a statement about the
    dossier. Same shape of message as the digit-free case."""
    from mloda.community.feature_groups.german_ledger.reader import _decimal

    spec = parse_descriptor(str(FIX / "dossier_a" / "index.xml"))
    assert str(_decimal("12.500,00", spec, "Betrag")) == "12500.00", "the good case still parses"
    try:
        _decimal("12.500,00 EUR", spec, "Betrag")
    except ValueError as e:
        assert "is not a number" in str(e) and "Betrag" in str(e), e
        return
    raise AssertionError("a numeric field carrying trailing text was accepted")


def test_second_exporter_profile_reads_without_reader_changes() -> None:
    """The descriptor drives the reader, so a second exporter SHAPE costs it nothing.

    dossier_c is synthetic, like the others, and carries dossier_a's seven bookings
    re-declared under a second profile: '|' delimiter, an omitted TextEncapsulator (which
    takes the standard's default of '"'), UTF-8, Anglo decimal and grouping symbols,
    <ImpliedAccuracy> instead of <Accuracy>, ISO dates, different column names. Same
    amounts. This is format independence, not evidence of real-world exporter coverage.
    """
    spec = parse_descriptor(str(FIX / "dossier_c" / "index.xml"))
    assert spec.delimiter == "|" and spec.encapsulator == '"' and spec.encoding == "utf-8"
    assert spec.decimal_symbol == "." and spec.grouping_symbol == ","
    assert spec.implied == {"Umsatz"} and spec.dates["Buchung"] == "YYYY-MM-DD"

    t = GdpduReader.load_data(str(FIX / "dossier_c"), FeatureSet())
    assert [str(v) for v in t.column("Umsatz").to_pylist()[:4]] == ["12500.00", "8750.50", "3100.00", "1234.56"]
    a = GdpduReader.load_data(str(FIX / "dossier_a"), FeatureSet())
    assert [str(v) for v in t.column("Umsatz").to_pylist()] == [str(v) for v in a.column("Betrag").to_pylist()], (
        "same money, different file shape"
    )


def test_second_profile_reaches_the_same_total_end_to_end() -> None:
    """Same decoded column is not the same claim as same cited total.

    Run in a subprocess: configuring a second profile registers it globally and would
    shadow the base group for every later resolution here.
    """
    proof = _run_proof("proof_dossier_c")
    assert proof.returncode == 0, proof.stdout + proof.stderr
    assert "SECOND PROFILE END TO END" in proof.stdout, proof.stdout
    assert "revenue=45385.06 (5 origins)" in proof.stdout, proof.stdout
    assert "receivables=5000.00 (1 origin)" in proof.stdout, proof.stdout
    assert "all citations name JOURNAL.csv" in proof.stdout, proof.stdout


def test_a_profile_naming_undeclared_columns_refuses_by_name() -> None:
    """The mapping from declared column to meaning is per exporter, so it can be wrong.

    Reading straight through gave KeyError('Field "Konto" does not exist in schema')
    from inside Arrow, which names neither the profile nor the dossier.
    """
    from mloda.provider import FeatureSet
    from mloda.user import Feature

    from mloda.community.feature_groups.german_ledger.skr import LedgerProfile

    # Overriding PROFILE on the base class and restoring it, rather than defining a
    # subclass: a subclass registers globally via subclass discovery and would shadow the
    # base group for every later resolution in this process -- which is the very hazard
    # proof_profile_shadow.py demonstrates. Warning about that and then doing it would
    # leave this suite order-dependent.
    table = GdpduReader.load_data(str(FIX / "dossier_a"), FeatureSet())
    fs = FeatureSet()
    fs.add(Feature("revenue"))

    from mloda.community.feature_groups.german_ledger.skr import NoProfileForDossier

    original = SkrAccountFeatureGroup.PROFILES
    try:
        SkrAccountFeatureGroup.PROFILES = (LedgerProfile(account_column="Sachkonto", amount_column="Umsatz"),)
        SkrAccountFeatureGroup._map_accounts(table, fs)
    except NoProfileForDossier as e:
        assert "Sachkonto" in str(e), e
        assert "Konto" in str(e), "the error must show what the dossier DOES declare"
        return
    finally:
        SkrAccountFeatureGroup.PROFILES = original
    raise AssertionError("a dossier no profile fits was accepted")


def test_profile_shadowing_refuses_instead_of_answering_wrongly() -> None:
    """The framework rule is unchanged; the design no longer turns it into a number.

    mloda keeps a matching subclass over its parent, so modelling one exporter per
    subclass let a second profile shadow the first and answer under the wrong mapping.
    Profiles are no longer subclasses: `select_profile` picks the one the DOSSIER fits.
    A shadowing subclass therefore cannot map the wrong dossier -- it refuses, naming the
    columns the dossier actually declares. Three siblings still collide outright.

    Still not solved, and disclosed: two profiles whose columns are named identically but
    mean different things are indistinguishable on names alone.
    """
    proof = _run_proof("proof_profile_shadow")
    assert proof.returncode == 0, proof.stdout + proof.stderr
    assert "SHADOWING REFUSES INSTEAD OF GUESSING" in proof.stdout, proof.stdout
    # two is the dangerous count precisely because three is the loud one
    assert "THIRD PROFILE COLLIDES" in proof.stdout, proof.stdout
    assert "Multiple feature groups found" in proof.stdout, proof.stdout


def test_removing_the_admissibility_policy_yields_no_number() -> None:
    """The claim that makes four plugins one chain rather than four shipped together.

    Subprocess: the unguarded run must not be confused with the guarded runs above.
    """
    proof = _run_proof("proof_inseparable")
    assert proof.returncode == 0, proof.stdout + proof.stderr
    assert "WITH POLICY:    45385.06 and 5 citations" in proof.stdout, proof.stdout
    # Option A: with no policy class the verdict feature does not resolve at all; the
    # InadmissibleTotal backstop now catches the concept wired past the policy instead.
    assert "WITHOUT POLICY: FeatureResolutionError" in proof.stdout, proof.stdout
    assert "BYPASS:         InadmissibleTotal" in proof.stdout, proof.stdout
    assert "TWO POLICIES:   FeatureResolutionError" in proof.stdout, proof.stdout
    assert "CHAIN IS INSEPARABLE" in proof.stdout, proof.stdout


def test_admissibility_evidence_names_the_policy_that_attested() -> None:
    """Evidence, not a boolean: a citation attests admitted origin, so the verdict says
    which policy admitted it and under which cutoff."""
    from mloda.community.feature_groups.german_ledger.reader import ADMISSIBILITY_COLUMN

    table = Closing2025.clear(GdpduReader.load_data(str(FIX / "dossier_a"), FeatureSet()))

    assert ADMISSIBILITY_COLUMN in table.column_names
    stamps = set(table.column(ADMISSIBILITY_COLUMN).to_pylist())
    assert len(stamps) == 1, stamps
    stamp = stamps.pop()
    for expected in (
        "late-entry-cutoff",
        "lock=2026-01-15",
        "period_end=2025-12-31",
        "booking=Buchungsdatum",
        "keying=Erfassungsdatum",
    ):
        assert expected in stamp, (expected, stamp)


def test_the_evidence_column_name_is_reserved_against_a_descriptor() -> None:
    """A descriptor declaring __admissibility would have its column overwritten by the stamp."""
    from mloda.community.feature_groups.german_ledger.reader import ADMISSIBILITY_COLUMN

    injected = (
        f"<VariableColumn>\n          <Name>{ADMISSIBILITY_COLUMN}</Name>\n"
        "          <AlphaNumeric/>\n        </VariableColumn>\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "reserved_adm"
        shutil.copytree(FIX / "dossier_a", d)
        idx = d / "index.xml"
        idx.write_text(
            idx.read_text().replace("      </VariableLength>", "        " + injected + "      </VariableLength>", 1)
        )
        try:
            parse_descriptor(str(idx))
        except ValueError as e:
            assert "is reserved" in str(e), e
            return
        raise AssertionError("a descriptor declaring the evidence column was accepted")


def test_a_blank_negative_or_malformed_stamp_is_not_an_admission() -> None:
    """Checking only for non-null admitted "", "DENIED" and False alike.

    An affirmative verdict naming its policy is the contract. This is an in-process
    convention, not authentication -- a producer inside the process can still write a
    conforming string -- but a blank, negative or malformed stamp must never total.
    """
    import pyarrow as pa

    from mloda.community.feature_groups.german_ledger.policy import AdmissibilityPolicyGroup
    from mloda.community.feature_groups.german_ledger.reader import ADMISSIBILITY_COLUMN
    from mloda.community.feature_groups.german_ledger.sources import InadmissibleTotal

    # A stand-in policy in the host's place. It was an INPUT_DATA_LOAD extender, and on
    # mloda 0.14.0 (#1549) core discards what an extender returns: the fake stamp never
    # reached the aggregation, and this test passed without testing anything.
    class FakeStamp(AdmissibilityPolicyGroup):
        STAMP: tuple[object, pa.DataType] | None = None  # set while this test runs

        @classmethod
        def configured(cls) -> bool:
            return cls.STAMP is not None

        @classmethod
        def clear(cls, table: pa.Table) -> pa.Table:
            assert cls.STAMP is not None
            value, arrow_type = cls.STAMP
            return table.append_column(ADMISSIBILITY_COLUMN, pa.array([value] * table.num_rows, type=arrow_type))

    # the last three are the grammar cases: prefix-plus-length admitted all of them
    try:
        for value, arrow_type in (
            ("", pa.string()),
            ("DENIED", pa.string()),
            ("made-up", pa.string()),
            (False, pa.bool_()),
            ("admitted:;lock=x", pa.string()),
            ("admitted: ", pa.string()),
            ("admitted:\n", pa.string()),
        ):
            FakeStamp.STAMP = (value, arrow_type)
            try:
                mloda.run_all(
                    features=["revenue__sources"],
                    compute_frameworks={PyArrowTable},
                    data_access_collection=DataAccessCollection(folders={str(FIX / "dossier_a")}),
                    plugin_collector=plugins_with(FakeStamp),
                )
            except Exception as e:
                root: BaseException = e
                while (deeper := root.__cause__ or root.__context__) is not None:
                    root = deeper
                assert isinstance(root, InadmissibleTotal), (value, type(root).__name__, str(root)[:120])
                continue
            raise AssertionError(f"a {value!r} stamp produced a total")
    finally:
        FakeStamp.STAMP = None


def test_two_policies_collide_where_the_cause_is_visible() -> None:
    """append_column permits duplicates, so a second stamp would surface as a KeyError
    deep in the transform. Refuse at the second stamp instead."""
    from mloda.community.feature_groups.german_ledger.reader import ADMISSIBILITY_COLUMN

    once = Closing2025.clear(GdpduReader.load_data(str(FIX / "dossier_a"), FeatureSet()))
    assert ADMISSIBILITY_COLUMN in once.column_names
    try:
        Closing2025.clear(once)
    except AdmissibilityRefused as e:  # the base of every rule's refusal, LateEntryRefused included
        assert "already present" in str(e), e
        return
    raise AssertionError("a second admissibility stamp was accepted")


def test_an_attested_dossier_with_no_matching_rows_still_returns_null() -> None:
    """Presence of the evidence column is the contract, not a non-empty one.

    Requiring non-empty evidence made a zero-contribution concept raise instead of
    returning the promised null -- re-breaking the sourceless-zero guarantee from the
    other direction.
    """
    import pyarrow as pa
    from mloda.provider import FeatureSet
    from mloda.user import Feature

    from mloda.community.feature_groups.german_ledger.reader import ADMISSIBILITY_COLUMN, admissibility_verdict
    from mloda.community.feature_groups.german_ledger.sources import SourcesFeatureGroup

    empty = pa.table(
        {
            "revenue~value": pa.array([], type=pa.decimal128(38, 2)),
            "revenue~origins": pa.array([], type=pa.string()),
            ADMISSIBILITY_COLUMN: pa.array([], type=pa.string()),
        }
    )
    fs = FeatureSet()
    fs.add(Feature("revenue__sources"))
    out = SourcesFeatureGroup.calculate_feature(empty, fs)
    assert out.column("revenue__sources~value").to_pylist() == [None]
    assert out.column("revenue__sources~origins").to_pylist() == [None]
    assert admissibility_verdict("p", a=1).startswith("admitted:")


def test_removing_the_policy_leaves_the_independent_source_with_no_number() -> None:
    """Inseparability is a property of the primitive, so it must hold off the ledger too.

    Same statement as proof_inseparable.py, one source over: strip the admissibility
    evidence and the aggregation returns nothing rather than an unvetted total.
    """
    from decimal import Decimal

    from mloda.provider import FeatureSet
    from mloda.user import Feature

    from mloda.community.feature_groups.german_ledger.sources import InadmissibleTotal

    unstamped = pa.table(
        {
            "pump_a~value": pa.array([Decimal("12.50")], type=pa.decimal128(38, 2)),
            "pump_a~origins": pa.array(["readings.csv#r1"], type=pa.string()),
        }
    )
    fs = FeatureSet()
    fs.add(Feature("pump_a__sources"))
    try:
        SourcesFeatureGroup.calculate_feature(unstamped, fs)
    except InadmissibleTotal as e:
        assert "no admissibility evidence" in str(e), e
        return
    raise AssertionError("an unstamped independent source produced a total")


def test_a_verdict_must_name_a_syntactically_valid_policy() -> None:
    """The producer fails at the point of stamping if it cannot name itself."""
    from mloda.community.feature_groups.german_ledger.reader import admissibility_verdict, is_admitted

    assert is_admitted(admissibility_verdict("late-entry-cutoff", lock="2026-01-15"))
    assert is_admitted("admitted:x")
    for bad in ("admitted:;lock=x", "admitted: ", "admitted:\n", "admitted:", "denied:p", ""):
        assert not is_admitted(bad), bad
    for bad_name in ("", " ", ";", "\n", ".leading-dot"):
        try:
            admissibility_verdict(bad_name, lock="x")
        except ValueError as e:
            assert "is not a valid identifier" in str(e), e
            continue
        raise AssertionError(f"policy name {bad_name!r} was accepted")


def test_a_fully_priced_concept_still_totals() -> None:
    """The guard against over-correction: unknown amounts must not make every total null."""
    res = mloda.run_all(
        features=["revenue__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(FIX / "dossier_a")}),
        plugin_collector=PLUGINS,
    )
    got = {c: t.column(c).to_pylist()[0] for t in res for c in t.column_names}
    assert str(got["revenue__sources~value"]) == "45385.06", got["revenue__sources~value"]
    assert len(got["revenue__sources~origins"]) == 5


def _one_concept_table(values: list[Any], origins: list[Any], scale: int = 2, stamp: str | None = None) -> pa.Table:
    """A ~value/~origins pair at row grain, stamped admissible, fed straight to the primitive.

    SourcesFeatureGroup is offered to the registry as a primitive any producer may feed, so
    its contract has to hold for tables the GDPdU chain would never build.
    """
    from mloda.community.feature_groups.german_ledger.reader import ADMISSIBILITY_COLUMN, admissibility_verdict

    stamp = stamp or admissibility_verdict("test-policy")
    return pa.table(
        {
            "revenue~value": pa.array(values, type=pa.decimal128(38, scale)),
            "revenue~origins": pa.array(origins, type=pa.string()),
            ADMISSIBILITY_COLUMN: pa.array([stamp] * len(values), type=pa.string()),
        }
    )


def _total(values: list[Any], origins: list[Any], scale: int = 2) -> Any:
    from mloda.provider import FeatureSet
    from mloda.user import Feature

    fs = FeatureSet()
    fs.add(Feature("revenue__sources"))
    out = SourcesFeatureGroup.calculate_feature(_one_concept_table(values, origins, scale), fs)
    return (out.column("revenue__sources~value").to_pylist()[0], out.column("revenue__sources~origins").to_pylist()[0])


def test_a_priced_line_with_no_citation_is_refused() -> None:
    """The mirror of the unpriced line, and the one the title depends on.

    Keying contribution on the origin alone made a priced row with no citation vanish from
    its own total: values [10.00, 20.00] with origins ["GL.txt@abc:1", None] returned 10.00
    cited to one row. That is a smaller number wearing the authority of a complete one --
    the same failure as the known subtotal, arriving from the other side. A value the
    producer cannot cite is a broken contract, not a data condition, so it raises rather
    than nulling: nulling would let a producer bug pass as an unpriceable dossier.
    """
    from decimal import Decimal

    from mloda.community.feature_groups.german_ledger.sources import UncitedValue

    try:
        _total([Decimal("10.00"), Decimal("20.00")], ["GL.txt@abc:1", None])
    except UncitedValue as e:
        assert "20.00" in str(e), e
        return
    raise AssertionError("a priced line with no citation was totalled")


def test_a_blank_citation_is_not_a_citation() -> None:
    """ "" is not a source. Non-null was the whole test, so an empty string cited a total."""
    from decimal import Decimal

    from mloda.community.feature_groups.german_ledger.sources import UncitedValue

    for blank in ("", " ", "\n"):
        try:
            _total([Decimal("12.50")], [blank])
        except UncitedValue:
            continue
        raise AssertionError(f"a {blank!r} citation produced a total")


def test_a_blank_citation_does_not_make_a_row_contribute() -> None:
    """The same predicate one row over: a blank origin beside a null amount is not a line.

    Counted as contributing, it would read as an unpriced line and null a total that is
    perfectly stateable from the rows that do carry citations.
    """
    from decimal import Decimal

    value, origins = _total([Decimal("10.00"), None], ["GL.txt@abc:1", ""])
    assert str(value) == "10.00", value
    assert origins == ["GL.txt@abc:1"], origins


def test_the_total_is_summed_at_the_declared_precision() -> None:
    """Python's default Decimal context rounds at 28 significant digits.

    decimal128(38, 2) money is wider than that, so the default context silently rounded
    cents away on large ledgers: 123456789012345678901234567890.11 + 0.02 came back as
    ...900.00. A total that quietly loses cents is worse than one that refuses.
    """
    from decimal import Decimal

    value, _ = _total([Decimal("123456789012345678901234567890.11"), Decimal("0.02")], ["GL.txt@abc:1", "GL.txt@abc:2"])
    assert str(value) == "123456789012345678901234567890.13", value


def test_a_line_outside_the_closed_period_is_stamped_as_outside_it() -> None:
    """The guard must not stamp a period test onto a row it declined to apply it to.

    A 2026 booking is outside the cutoff's reach: the policy sees it and does not refuse
    it, but it never tested it against the closed period. Stamping it with the same
    verdict as a cleared 2025 line made the evidence claim an examination that never
    happened. The row stays admissible -- this guard has no grounds to refuse it -- and
    the stamp now says which of the two it is.
    """
    import datetime

    from mloda.community.feature_groups.german_ledger.reader import ADMISSIBILITY_COLUMN, is_admitted

    table = pa.table(
        {
            "Buchungsdatum": pa.array([datetime.date(2025, 10, 15), datetime.date(2026, 2, 3)]),
            "Erfassungsdatum": pa.array([datetime.date(2025, 10, 16), None]),
            GdpduReader.ORIGIN_COLUMN: pa.array(["GL.txt@abc:1", "GL.txt@abc:9"]),
        }
    )
    inside, outside = Closing2025.clear(table).column(ADMISSIBILITY_COLUMN).to_pylist()

    # A policy has three things to say, and only one of them is an admission.
    assert is_admitted(inside), inside
    assert "period_end=2025-12-31" in inside, inside
    assert not is_admitted(outside), "a row the cutoff never judged must not read as cleared"
    assert outside.startswith("outside-scope:late-entry-cutoff"), outside


def test_omitted_accuracy_and_encapsulator_take_the_standards_defaults() -> None:
    """The reader shipped two wrong GDPdU defaults, and both fixtures hid them.

    The standard's defaults are Accuracy 0 and TextEncapsulator '"'. This reader used 2
    and none, so a conforming dossier that omitted either element was misread: amounts
    divided by a hundred, and quoted fields containing the delimiter split down the
    middle. Every fixture declares both elements explicitly, which is exactly why 45
    tests passed over a reader that would misread a conforming file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "bare_defaults"
        shutil.copytree(FIX / "dossier_a", d)
        idx = d / "index.xml"
        idx.write_text(
            idx.read_text()
            .replace("<Numeric><Accuracy>2</Accuracy></Numeric>", "<Numeric/>")
            .replace('<TextEncapsulator>"</TextEncapsulator>', "")
        )

        spec = parse_descriptor(str(idx))
        assert spec.accuracy["Betrag"] == 0, spec.accuracy
        assert spec.encapsulator == '"', spec.encapsulator


def test_a_declared_range_is_refused_rather_than_disregarded() -> None:
    """<Range> declares which lines are data -- it is how a header is skipped.

    Reading the file while ignoring it turns a header into a booking, or fails a field
    count far from its cause. This reader does not implement it, so it refuses by name.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "declares_range"
        shutil.copytree(FIX / "dossier_a", d)
        idx = d / "index.xml"
        idx.write_text(idx.read_text().replace("<VariableLength>", "<VariableLength><Range><From>2</From></Range>"))
        try:
            parse_descriptor(str(idx))
        except ValueError as e:
            assert "<Range>" in str(e) and "does not " in str(e), e
            return
        raise AssertionError("a descriptor declaring <Range> was read anyway")


def test_a_sibling_subcolumn_cannot_become_the_total() -> None:
    """The worst defect found in review: a plausible total under a valid-looking citation.

    Column selection took the first name ending in `~value` from a SORTED list, so
    `revenue~other~value` -- which sorts before `revenue~value` -- silently became the
    answer, carrying the right concept's citations. That is precisely the failure this
    aggregation exists to prevent, arriving through its own column resolution. Exact
    names are now required and ambiguity is refused rather than resolved.
    """
    from decimal import Decimal

    from mloda.provider import FeatureSet
    from mloda.user import Feature

    from mloda.community.feature_groups.german_ledger.reader import ADMISSIBILITY_COLUMN, admissibility_verdict
    from mloda.community.feature_groups.german_ledger.sources import AmbiguousSourceColumns

    table = pa.table(
        {
            "revenue~value": pa.array([Decimal("10.00")], type=pa.decimal128(38, 2)),
            "revenue~other~value": pa.array([Decimal("999.00")], type=pa.decimal128(38, 2)),
            "revenue~origins": pa.array(["GL.txt@abc:1"], type=pa.string()),
            ADMISSIBILITY_COLUMN: pa.array([admissibility_verdict("test-policy")], type=pa.string()),
        }
    )
    fs = FeatureSet()
    fs.add(Feature("revenue__sources"))
    try:
        SourcesFeatureGroup.calculate_feature(table, fs)
    except AmbiguousSourceColumns as e:
        assert "revenue~other~value" in str(e), e
        return
    raise AssertionError("a sibling subcolumn was totalled as the concept's amount")


def test_a_binary_float_amount_is_refused_by_name() -> None:
    """`Decimal(0) + float` raised a TypeError from inside the sum, which says nothing.

    A stock CSV reader infers double, so this is the first thing a producer over an
    independent source hits. The refusal has to name the column and say who is
    responsible for the scale, which is the producer.
    """
    from mloda.provider import FeatureSet
    from mloda.user import Feature

    from mloda.community.feature_groups.german_ledger.reader import ADMISSIBILITY_COLUMN, admissibility_verdict

    table = pa.table(
        {
            "revenue~value": pa.array([10.0, 20.0], type=pa.float64()),
            "revenue~origins": pa.array(["a:1", "a:2"], type=pa.string()),
            ADMISSIBILITY_COLUMN: pa.array([admissibility_verdict("p")] * 2, type=pa.string()),
        }
    )
    fs = FeatureSet()
    fs.add(Feature("revenue__sources"))
    try:
        SourcesFeatureGroup.calculate_feature(table, fs)
    except ValueError as e:
        assert "binary float" in str(e) and "revenue~value" in str(e), e
        return
    raise AssertionError("a float amount column was summed into an audit total")


def test_both_exporter_shapes_resolve_through_one_class_in_one_process() -> None:
    """The architecture fix: profile identity is a property of the dossier, not the class.

    This is the test the old design could not have. Registering a second profile meant
    registering a second subclass, which shadowed the first for every later resolution in
    the process -- so the suite could only ever exercise one shape per process, and the
    second had to be proved in a subprocess. One class now serves both, and the two shapes
    reach the same total in the same interpreter.
    """
    from mloda.provider import FeatureSet
    from mloda.user import Feature

    fs = FeatureSet()
    fs.add(Feature("revenue"))
    totals = {}
    for dossier in ("dossier_a", "dossier_c"):
        table = GdpduReader.load_data(str(FIX / dossier), FeatureSet())
        mapped = SkrAccountFeatureGroup._map_accounts(table, fs)
        priced = [v for v in mapped.column("revenue~value").to_pylist() if v is not None]
        totals[dossier] = sum(priced)
    assert str(totals["dossier_a"]) == "45385.06", totals
    assert str(totals["dossier_c"]) == "45385.06", totals


def test_two_profiles_fitting_one_dossier_are_refused_not_ranked() -> None:
    """Where the old design guessed, this one stops.

    If two known shapes both fit the columns a dossier declares, nothing in the names says
    which exporter wrote it. Picking either is how a wrong mapping reaches a citation list
    that looks right, so selection refuses instead.
    """
    from mloda.provider import FeatureSet
    from mloda.user import Feature

    from mloda.community.feature_groups.german_ledger.skr import AmbiguousProfile, LedgerProfile

    table = GdpduReader.load_data(str(FIX / "dossier_a"), FeatureSet())
    fs = FeatureSet()
    fs.add(Feature("revenue"))
    original = SkrAccountFeatureGroup.PROFILES
    try:
        # both fit dossier_a: same columns, and nothing on the file says which is meant
        SkrAccountFeatureGroup.PROFILES = (
            LedgerProfile(account_column="Konto", amount_column="Betrag"),
            LedgerProfile(account_column="Konto", amount_column="Betrag"),
        )
        SkrAccountFeatureGroup._map_accounts(table, fs)
    except AmbiguousProfile as e:
        assert "more than one ledger profile fits" in str(e), e
        return
    finally:
        SkrAccountFeatureGroup.PROFILES = original
    raise AssertionError("two fitting profiles were ranked instead of refused")


def test_implied_accuracy_does_not_round_a_wide_amount_at_parse_time() -> None:
    """`scaleb` is arithmetic, so it obeys the context: 28 digits by default.

    That is narrower than the decimal128(38, s) this reader may produce, so a wide
    <ImpliedAccuracy> amount was rounded where it was parsed. Nothing downstream could
    recover it -- the aggregation's wider context only protects the sum, not the addend.
    """
    from decimal import Decimal

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "wide_implied"
        shutil.copytree(FIX / "dossier_c", d)
        wide = "12345678901234567890123456789012345678"  # 38 digits, legal decimal128
        journal = d / "JOURNAL.csv"
        rows = journal.read_text().splitlines()
        rows[0] = rows[0].replace("|1250000|", f"|{wide}|")
        journal.write_text("\n".join(rows) + "\n")

        table = GdpduReader.load_data(str(d), FeatureSet())
        got = table.column("Umsatz").to_pylist()[0]
        assert got == Decimal("123456789012345678901234567890123456.78"), got


def test_reader_answers_get_column_names_from_the_descriptor() -> None:
    """mloda 0.13.0 made `get_column_names` load-bearing for matching.

    `ReadFile._declines_unvalidated_separator_name` declines a chain- or column-separated
    feature name when a reader does NOT override this method, on the grounds that such a
    reader cannot confirm the name. This reader can: index.xml declares every column before
    a byte of data is read. Overriding it is the honest way to be exempt from that guard --
    bypassing it by omission, which is what a `match_subclass_data_access` override does,
    is the failure mode the 0.13.0 docstring warns about.
    """
    from mloda_plugins.feature_group.input_data.read_file import ReadFile

    assert GdpduReader._is_overridden(ReadFile, "get_column_names"), (
        "the 0.13.0 separator guard treats a non-overriding reader as unable to confirm names"
    )

    # Each dossier answers with ITS OWN declared names -- the second exporter's shape is not
    # the first one's, which is the whole point of reading the descriptor rather than sniffing.
    assert GdpduReader.get_column_names(str(FIX / "dossier_a")) == [
        "Satznr",
        "Konto",
        "Betrag",
        "Buchungsdatum",
        "Erfassungsdatum",
        "Belegtext",
        "__origin",
    ]
    assert GdpduReader.get_column_names(str(FIX / "dossier_c")) == [
        "Satz",
        "Sachkonto",
        "Umsatz",
        "Buchung",
        "Erfassung",
        "Text",
        "__origin",
    ]


def test_get_column_names_does_not_claim_the_admissibility_column() -> None:
    """A policy adds `__admissibility`, not the reader.

    Listing it here would claim a column this class does not produce -- the same silent
    overstatement the rest of this package exists to prevent, one layer down.
    """
    from mloda.community.feature_groups.german_ledger.reader import ADMISSIBILITY_COLUMN

    columns = GdpduReader.get_column_names(str(FIX / "dossier_a"))
    assert ADMISSIBILITY_COLUMN not in columns
    assert GdpduReader.ORIGIN_COLUMN in columns


def test_the_dossier_identity_names_its_fingerprint_not_a_local_path() -> None:
    """mloda 0.14.0 (#1577): the reader decides the data_access_identity extenders see.

    The default for a local path that exists is the path as given -- here an absolute
    path under the author's home directory, handed to every lineage and audit extender.
    The override names the dossier and the same descriptor-plus-data fingerprint the row
    citations carry, so an extender's record joins with the citations it vouches for.
    """
    ident = GdpduReader.data_access_identity(str(FIX / "dossier_a"))
    origin = GdpduReader.load_data(str(FIX / "dossier_a"), FeatureSet()).column("__origin")[0].as_py()
    fingerprint = origin.split("@", 1)[1].split(":", 1)[0]
    assert ident == f"gdpdu:dossier_a@{fingerprint}", (ident, origin)
    assert str(HERE) not in ident, "the identity leaks the local path"

    # a changed dossier is a different identity, as it is a different citation
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "dossier_a"
        shutil.copytree(FIX / "dossier_a", d)
        gl = d / "GL.txt"
        gl.write_text(gl.read_text().replace("Umsatz Oktober", "Umsatz Oktobers"))
        assert GdpduReader.data_access_identity(str(d)) != ident

        # a dossier that cannot be fingerprinted is named, not crashed on: core calls this
        # before the load, and the load is where the actual error belongs
        (d / "index.xml").write_text("<oops")
        assert GdpduReader.data_access_identity(str(d)) == "gdpdu:dossier_a"


def test_extenders_see_the_dossier_identity_on_input_data_load() -> None:
    """The override is only worth something if core actually hands it to extenders."""
    from importlib.metadata import version

    import pytest

    if tuple(int(p) for p in version("mloda").split(".")[:2]) < (0, 14):
        pytest.skip("reader-owned data_access_identity arrives in mloda 0.14.0 (#1577)")
    from mloda.steward import Extender, ExtenderHook, HookContext

    seen: list[str | None] = []

    class Recorder(Extender):
        def wraps(self) -> set[ExtenderHook]:
            return {ExtenderHook.INPUT_DATA_LOAD}

        def __call__(self, func: Any, *a: Any, **k: Any) -> Any:
            context = HookContext.current()
            assert context is not None
            seen.append(context.data_access_identity)
            return func(*a, **k)

    mloda.run_all(
        features=["revenue__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(FIX / "dossier_a")}),
        function_extender={Recorder()},
        plugin_collector=PLUGINS,
    )
    assert seen == [GdpduReader.data_access_identity(str(FIX / "dossier_a"))], seen


def test_a_declared_skip_num_bytes_is_refused_rather_than_disregarded() -> None:
    """<SkipNumBytes> declares a byte header in front of the data -- OrgaMon writes 154 bytes.

    It was neither honoured nor refused: the header was split into fields and, if its field
    count happened to match, read as a booking. That is a silent misread, the worst of the
    three reader defects in the corpus notes. The reader does not implement it, so it
    refuses by name at descriptor parse, the same way it refuses <Range>.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "declares_skip"
        shutil.copytree(FIX / "dossier_a", d)
        idx = d / "index.xml"
        idx.write_text(
            idx.read_text().replace("<VariableLength>", "<SkipNumBytes>154</SkipNumBytes>\n      <VariableLength>", 1)
        )
        try:
            parse_descriptor(str(idx))
        except ValueError as e:
            assert "<SkipNumBytes>" in str(e) and "does not " in str(e), e
            return
        raise AssertionError("a descriptor declaring <SkipNumBytes> was read anyway")


def test_a_data_file_named_in_url_but_absent_is_refused_by_name() -> None:
    """A bare FileNotFoundError names a path, not the declaration that promised the file."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "no_data"
        shutil.copytree(FIX / "dossier_a", d)
        (d / "GL.txt").unlink()
        try:
            GdpduReader.load_data(str(d), FeatureSet())
        except FileNotFoundError:
            raise AssertionError("an absent data file crashed instead of being refused by name")
        except ValueError as e:
            assert "<URL>" in str(e) and "GL.txt" in str(e) and "absent" in str(e), e
            return
        raise AssertionError("a dossier with no data file was read")


def test_bytes_invalid_in_the_declared_encoding_are_refused_with_their_offset() -> None:
    """A UnicodeDecodeError is a crash; the refusal names file, declared encoding and offset."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "bad_bytes"
        shutil.copytree(FIX / "dossier_a", d)
        idx = d / "index.xml"
        idx.write_text(idx.read_text().replace("<ANSI/>", "<UTF8/>"))
        gl = d / "GL.txt"
        good = gl.read_bytes()
        gl.write_bytes(good + b"9;4000;1,00;30.12.2025;02.01.2026;Gr\xfc\xdfe\r\n")
        offset = len(good) + len(b"9;4000;1,00;30.12.2025;02.01.2026;Gr")
        try:
            GdpduReader.load_data(str(d), FeatureSet())
        except UnicodeDecodeError:
            raise AssertionError("undecodable bytes crashed instead of being refused by name")
        except ValueError as e:
            assert "GL.txt" in str(e) and "utf-8" in str(e) and f"byte {offset}" in str(e), e
            return
        raise AssertionError("bytes invalid in the declared encoding were read")


def test_each_declared_code_page_decodes_its_own_bytes() -> None:
    """_CODEPAGES had no tests: UTF-16, OEM and Macintosh are pinned on self-made bytes.

    The same booking text is written in each code page and must come back identical, with
    the umlaut intact -- a wrong mapping reads it as mojibake, not as an error.
    """
    text_row = "9;4000;1,00;30.12.2025;02.01.2026;Müller Größe\r\n"
    for element, codec in (("UTF16", "utf-16"), ("OEM", "cp850"), ("Macintosh", "mac_roman")):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / element
            shutil.copytree(FIX / "dossier_a", d)
            idx = d / "index.xml"
            idx.write_text(idx.read_text().replace("<ANSI/>", f"<{element}/>"))
            gl = d / "GL.txt"
            gl.write_bytes((gl.read_bytes().decode("cp1252") + text_row).encode(codec))
            t = GdpduReader.load_data(str(d), FeatureSet())
            assert t.column("Belegtext").to_pylist()[-1] == "Müller Größe", (element, t.column("Belegtext")[-1])
            assert t.num_rows == 8, (element, t.num_rows)
