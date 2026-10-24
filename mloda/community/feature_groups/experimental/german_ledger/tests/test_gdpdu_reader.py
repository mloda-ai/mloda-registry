"""GdpduReader: descriptor parsing of untrusted index.xml files.

The end-to-end GDPdU claims are in test_gdpdu_claims.py; this file holds the descriptor
parser's own checks.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import ParseError  # nosec B405

import pytest
from mloda.provider import FeatureSet, SourceMatch
from mloda.user import Options

from mloda.community.feature_groups.experimental.german_ledger.reader import (
    GdpduReader,
    parse_citation,
    parse_descriptor_bytes,
)

HERE = Path(__file__).parent
DOSSIER_A = HERE / "fixtures" / "dossier_a"

BOMB = b"""<?xml version="1.0"?>
<!DOCTYPE DataSet [
  <!ENTITY a "aaaaaaaaaa">
  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
]>
<DataSet><Media><Table><URL>&b;</URL></Table></Media></DataSet>
"""

UNKNOWN_ENCODING = b"""<?xml version="1.0" encoding="bogus"?>
<DataSet><Media><Table><URL>GL.txt</URL></Table></Media></DataSet>
"""


def _doc(subset: str, body: str = "x") -> bytes:
    return (
        f'<?xml version="1.0"?>\n<!DOCTYPE DataSet [\n{subset}\n]>\n'
        f"<DataSet><Media><Table><URL>{body}</URL></Table></Media></DataSet>\n"
    ).encode()


ENTITY_DESCRIPTORS = {
    "billion_laughs": BOMB,
    "external_general": _doc('<!ENTITY x SYSTEM "file:///etc/passwd">', "&x;"),
    "parameter_declaration": _doc('<!ENTITY % p "x">'),
    "parameter_declared_and_referenced": _doc("<!ENTITY % p \"<!ENTITY y 'z'>\"> %p;"),
    "unparsed": _doc('<!NOTATION gif SYSTEM "image/gif">\n<!ENTITY logo SYSTEM "logo.gif" NDATA gif>'),
}


@pytest.mark.parametrize("raw", ENTITY_DESCRIPTORS.values(), ids=ENTITY_DESCRIPTORS.keys())
def test_a_descriptor_declaring_xml_entities_is_refused_by_name(raw: bytes) -> None:
    """A dossier is third-party input; entity expansion is how an XML bomb works."""
    with pytest.raises(ValueError, match=r"^index\.xml: the descriptor declares XML entities"):
        parse_descriptor_bytes(raw, "index.xml")


def test_a_malformed_descriptor_keeps_its_parse_error_type() -> None:
    """Malformed XML raises ElementTree.ParseError, not a ValueError."""
    with pytest.raises(ParseError):
        parse_descriptor_bytes(b"<DataSet><Media>", "index.xml")


@pytest.mark.parametrize("encoding", ["bogus", "utf-32", "idna"])
def test_an_undecodable_declared_encoding_is_refused_by_name(tmp_path: Path, encoding: str) -> None:
    """An encoding pyexpat cannot use raises a ValueError naming the file and the encoding."""
    (tmp_path / "index.xml").write_bytes(UNKNOWN_ENCODING.replace(b"bogus", encoding.encode()))
    with pytest.raises(ValueError) as exc_info:
        GdpduReader.load_data(str(tmp_path), FeatureSet())
    assert str(tmp_path / "index.xml") in str(exc_info.value)
    assert encoding in str(exc_info.value)


def test_a_descriptor_naming_the_gdpdu_dtd_still_parses() -> None:
    """GDPdU descriptors routinely carry <!DOCTYPE DataSet SYSTEM "gdpdu-01-09-2004.dtd">."""
    raw = (DOSSIER_A / "index.xml").read_bytes()
    assert b'<!DOCTYPE DataSet SYSTEM "gdpdu-01-09-2004.dtd">' in raw
    assert "Konto" in parse_descriptor_bytes(raw, "index.xml").columns


@pytest.mark.parametrize("raw", [BOMB, UNKNOWN_ENCODING], ids=["bomb", "unknown_encoding"])
def test_the_identity_of_a_refused_descriptor_is_still_a_name(tmp_path: Path, raw: bytes) -> None:
    """data_access_identity never raises: a refused descriptor still gets a name."""
    (tmp_path / "index.xml").write_bytes(raw)
    assert (
        GdpduReader.data_access_identity(SourceMatch(source=str(tmp_path), access=str(tmp_path)))
        == f"gdpdu:{tmp_path.name}"
    )


# --- reader overlap through the resolution chain ------------------------------------------------


def _resolve(folders: list[Path], name: str, options: dict[str, str] | None = None, journals: bool = False) -> Any:
    """Run `name` against `folders` with only CsvFG (or, with `journals`, the two journals) enabled."""
    from mloda.user import DataAccessCollection, Feature, PluginCollector, mloda
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
    from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG

    from mloda.community.feature_groups.experimental.german_ledger.policy import (
        DatevJournalFeatureGroup,
        JournalFeatureGroup,
    )

    groups: set[Any] = {DatevJournalFeatureGroup, JournalFeatureGroup} if journals else {CsvFG}
    feature = Feature(name, Options(options or {}))
    return mloda.run_all(
        features=[feature],
        compute_frameworks=[PyArrowTable],
        data_access_collection=DataAccessCollection(folders={str(f) for f in folders}),
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
    )


def test_a_gdpdu_folder_does_not_claim_a_csv_column_name(tmp_path: Path) -> None:
    """GdpduReader confirms only `gdpdu_journal`; any other name goes to the reader that has it."""
    folder = tmp_path / "dossier_c"
    shutil.copytree(HERE / "fixtures" / "dossier_c", folder)
    (folder / "plain.csv").write_text("Kontonummer,Bezeichnung\n4000,Erloese\n", encoding="utf-8")
    results = _resolve([folder], "Bezeichnung")
    assert [t.column("Bezeichnung").to_pylist() for t in results] == [["Erloese"]]
    assert _claimant([folder], "Bezeichnung") is None


def _claim(
    folders: list[Path], name: str, options: dict[str, str] | None = None, groups: set[Any] | None = None
) -> Any:
    """Resolve `name` over `folders` with only `groups` enabled (default: the GDPdU reader)."""
    from mloda.steward import resolve_feature
    from mloda.user import DataAccessCollection, Feature, PluginCollector

    return resolve_feature(
        Feature(name, Options(options or {})),
        plugin_collector=PluginCollector.enabled_feature_groups(groups or {GdpduReader}),
        data_access_collection=DataAccessCollection(folders={str(f) for f in folders}),
    )


def _claimant(folders: list[Path], name: str) -> Any:
    result = _claim(folders, name)
    return None if result.error else result.feature_group


def _twin_pair() -> list[Path]:
    return [HERE / "fixtures" / "twin_2025" / "gdpdu", HERE / "fixtures" / "twin_2025" / "datev"]


def _journal_origins(results: Any) -> list[str]:
    """The `__origin` of every journal row, across the result tables that carry one."""
    return [
        o
        for t in results
        if "gdpdu_journal~__origin" in t.column_names
        for o in t.column("gdpdu_journal~__origin").to_pylist()
    ]


@pytest.mark.parametrize(
    ("reader", "scheme"),
    [("GdpduReader", "GL.txt"), ("DatevExtfReader", "EXTF_Buchungsstapel.csv")],
    ids=["gdpdu", "datev"],
)
def test_pinning_a_reader_by_option_key_resolves_a_mixed_collection(reader: str, scheme: str) -> None:
    """The sibling reader declines when the other one is pinned, so only the pinned format answers."""
    pin = {reader: str(_twin_pair()[0 if reader == "GdpduReader" else 1])}
    origins = _journal_origins(_resolve(_twin_pair(), "gdpdu_journal", pin, journals=True))
    assert origins and all(o.startswith(scheme) for o in origins), origins


def _run_twin(options: dict[str, str]) -> Any:
    from mloda.user import DataAccessCollection, Feature, mloda
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

    from ._host import PLUGINS

    return mloda.run_all(
        features=[Feature("revenue__sources", Options(options))],
        compute_frameworks=[PyArrowTable],
        data_access_collection=DataAccessCollection(folders={str(f) for f in _twin_pair()}),
        plugin_collector=PLUGINS,
    )


def test_a_pinned_gdpdu_reader_resolves_the_total_over_the_twin_pair() -> None:
    results = _run_twin({"GdpduReader": str(_twin_pair()[0])})
    origins = [o for t in results for o in t.column("revenue__sources~origins").to_pylist()[0]]
    assert origins and all(o.startswith("GL.txt@") for o in origins), origins


def test_a_pinned_datev_reader_resolves_to_the_datev_journal_and_meets_the_cutoff() -> None:
    """Resolution reaches the DATEV journal; the GDPdU cutoff then fails closed (no Erfassungsdatum)."""
    from mloda.community.feature_groups.experimental.german_ledger.policy import LateEntryRefused

    from ._host import _cause

    with pytest.raises(Exception) as info:
        _run_twin({"DatevExtfReader": str(_twin_pair()[1])})
    assert _cause(info.value, LateEntryRefused) is not None, info.value


def test_pinning_both_readers_is_refused_as_ambiguous_not_as_unmatched() -> None:
    """Each reader must not decline for the other: resolution then reports both journal groups as ambiguous."""
    pin = {"GdpduReader": str(_twin_pair()[0]), "DatevExtfReader": str(_twin_pair()[1])}
    with pytest.raises(Exception) as info:
        _run_twin(pin)
    message = str(info.value)
    assert "Multiple feature groups found" in message, message
    assert "GdpduReader" in message, message
    assert "DatevExtfReader" in message, message
    assert "matched nothing" not in message, message


def test_a_host_subclass_of_the_gdpdu_reader_pinned_by_name_stands_the_datev_reader_down() -> None:
    """The subclass lives only inside this test, so no other resolution sees it."""
    import gc

    from mloda.community.feature_groups.experimental.german_ledger.datev import DatevExtfReader

    class _HostGdpdu(GdpduReader):
        pass

    try:
        folder = str(_twin_pair()[1])
        result = _claim([_twin_pair()[1]], "gdpdu_journal", {"_HostGdpdu": folder}, {DatevExtfReader})
        assert result.feature_group is not DatevExtfReader, result.feature_group
    finally:
        del _HostGdpdu
        gc.collect()


def test_two_dossiers_of_one_format_in_one_collection_are_refused_naming_the_pointer(tmp_path: Path) -> None:
    folders = [tmp_path / "a", tmp_path / "b"]
    for folder in folders:
        shutil.copytree(DOSSIER_A, folder)
    with pytest.raises(Exception) as info:
        _resolve(folders, "gdpdu_journal", journals=True)
    message = str(info.value)
    assert "several sources" in message, message
    assert 'options={"GdpduReader":' in message, message


# --- structured citations -----------------------------------------------------------------------


def test_a_citation_is_read_field_by_field() -> None:
    gdpdu = parse_citation("GL.txt@4564dc0deef2:7")
    assert (gdpdu.file, gdpdu.fingerprint, gdpdu.record, gdpdu.leg) == ("GL.txt", "4564dc0deef2", 7, None)
    datev = parse_citation("EXTF_Buchungsstapel.csv@6d85f9f5fa76:4/G")
    assert (datev.file, datev.record, datev.leg) == ("EXTF_Buchungsstapel.csv", 4, "G")


@pytest.mark.parametrize("bad", ["", "GL.txt", "GL.txt@xyz:1", "GL.txt@4564dc0deef2:", "GL.txt@4564dc0deef2:1/X"])
def test_a_malformed_citation_is_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_citation(bad)
