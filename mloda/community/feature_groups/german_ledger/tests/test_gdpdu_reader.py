"""GdpduReader: descriptor parsing of untrusted index.xml files.

The end-to-end GDPdU claims are in test_gdpdu_claims.py; this file holds the descriptor
parser's own checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mloda.community.feature_groups.german_ledger.reader import GdpduReader, parse_descriptor_bytes

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
