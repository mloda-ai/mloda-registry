"""GdpduReader: descriptor parsing of untrusted index.xml files.

The end-to-end GDPdU claims are in test_gdpdu_claims.py; this file holds the descriptor
parser's own checks.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from mloda.user import Options

from mloda.community.feature_groups.experimental.german_ledger.reader import GdpduReader, parse_descriptor_bytes

HERE = Path(__file__).parent
DOSSIER_A = HERE / "fixtures" / "dossier_a"

BOMB = b"""<?xml version="1.0"?>
<!DOCTYPE DataSet [
  <!ENTITY a "aaaaaaaaaa">
  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
]>
<DataSet><Media><Table><URL>&b;</URL></Table></Media></DataSet>
"""


def test_a_descriptor_declaring_xml_entities_is_refused_by_name() -> None:
    """A dossier is third-party input; entity expansion is how an XML bomb works."""
    with pytest.raises(ValueError, match="entit"):
        parse_descriptor_bytes(BOMB, "index.xml")


def test_a_descriptor_naming_the_gdpdu_dtd_still_parses() -> None:
    """GDPdU descriptors routinely carry <!DOCTYPE DataSet SYSTEM "gdpdu-01-09-2004.dtd">."""
    raw = (DOSSIER_A / "index.xml").read_bytes()
    assert b'<!DOCTYPE DataSet SYSTEM "gdpdu-01-09-2004.dtd">' in raw
    assert "Konto" in parse_descriptor_bytes(raw, "index.xml").columns


def test_the_identity_of_a_bomb_dossier_is_still_a_name(tmp_path: Path) -> None:
    """data_access_identity never raises: a refused descriptor still gets a name."""
    (tmp_path / "index.xml").write_bytes(BOMB)
    assert GdpduReader.data_access_identity(str(tmp_path)) == f"gdpdu:{tmp_path.name}"


# --- reader overlap through the resolution chain (mloda 0.15) -------------------------------


def _resolve(folders: list[Path], name: str, options: dict[str, str] | None = None, journals: bool = False) -> Any:
    """Run `name` against `folders` with only ReadFileFeature (or, with `journals`, the two journals) enabled."""
    from mloda.user import DataAccessCollection, Feature, PluginCollector, mloda
    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
    from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
    from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader  # noqa: F401  (registers the reader)

    from mloda.community.feature_groups.experimental.german_ledger.policy import (
        DatevJournalFeatureGroup,
        JournalFeatureGroup,
    )

    groups: set[Any] = {DatevJournalFeatureGroup, JournalFeatureGroup} if journals else {ReadFileFeature}
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
    assert not GdpduReader.match_subclass_data_access(str(folder), ["Bezeichnung"], Options())


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
