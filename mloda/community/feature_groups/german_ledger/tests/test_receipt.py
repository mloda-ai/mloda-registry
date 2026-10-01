"""The evidence receipt: a total names what it depended on, not only its rows.

The rows are cited already. What a citation cannot say is under which rules those rows became
this number: the policy that admitted them, the profile that read the columns, and the chart
and catalogue that defined the concept. The receipt says it, in our own versioned format for
now (`declared_attributes` later, mloda-registry #887).

The framework supplies the rest: `RunResult.plan`/`frames()` name the step and feature
groups that produced each frame, so `evidence_receipts` takes them from there instead of
inventing a carrier.
"""

import json
import subprocess  # nosec
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mloda.provider import FeatureSet
from mloda.steward import verified_context
from mloda.user import DataAccessCollection, Feature, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.german_ledger import skr
from mloda.community.feature_groups.german_ledger.reader import GdpduReader, parse_citation
from mloda.community.feature_groups.german_ledger.receipt import evidence_receipts, run_with_receipts
from mloda.community.feature_groups.german_ledger.skr import SkrAccountFeatureGroup
from mloda.community.feature_groups.german_ledger.sources import RECEIPT_VERSION

from ._host import PLUGINS

DOSSIER_A = Path(__file__).parent / "fixtures" / "dossier_a"


@pytest.fixture
def catalogue() -> Iterator[None]:
    """Restore the SKR04 catalogue after a test edits it; it is process-wide."""
    original = dict(skr.CHARTS)
    yield
    skr.CHARTS.clear()
    skr.CHARTS.update(original)


def _run(*features: str) -> Any:
    return mloda.run_all(
        features=list(features),
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(DOSSIER_A)}),
        plugin_collector=PLUGINS,
    )


def _basis(concept: str) -> dict[str, Any]:
    fs = FeatureSet()
    fs.add(Feature(concept))
    mapped = SkrAccountFeatureGroup._map_accounts(GdpduReader.load_data(str(DOSSIER_A), FeatureSet()), fs)
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


# --- the receipt on the total ------------------------------------------------------------------


def test_the_total_carries_a_receipt_beside_its_value_and_origins() -> None:
    [receipt] = evidence_receipts(_run("revenue__sources"))
    assert receipt["receipt"] == RECEIPT_VERSION
    assert receipt["feature"] == "revenue__sources"
    assert receipt["concept"] == "revenue"
    assert receipt["total"] == "45385.06"
    assert receipt["lines"] == 5 and receipt["unpriced"] == 0
    assert receipt["basis"]["catalogue"]["name"] == "SKR04_2025"


def test_the_receipt_names_the_policy_that_admitted_the_rows() -> None:
    [receipt] = evidence_receipts(_run("revenue__sources"))
    [policy] = receipt["policies"]
    assert policy.startswith("admitted:late-entry-cutoff;"), policy
    assert "lock=2026-01-15" in policy, policy
    assert receipt["verdicts"] == {"admitted": 7}, "every journal row was judged, not only the cited ones"


def test_the_receipt_names_the_steps_and_package_that_produced_it() -> None:
    [receipt] = evidence_receipts(_run("revenue__sources"))
    assert receipt["produced_by"] == "SourcesFeatureGroup"
    assert receipt["package"]["name"] == "mloda-community-german-ledger"
    groups = [s["feature_group"] for s in receipt["plan"]]
    assert "SkrAccountFeatureGroup" in groups and "TestClosing2025" in groups, groups


def test_a_changed_catalogue_changes_the_receipt(catalogue: None) -> None:
    """The receipt moves with the definition, not only with the rows."""
    [before] = evidence_receipts(_run("revenue__sources"))
    skr.CHARTS["04"] = {**skr.SKR04_2025, "revenue": range(4000, 4800)}
    [after] = evidence_receipts(_run("revenue__sources"))
    assert before["total"] == after["total"], "this dossier has no line in 4500-4799"
    assert after["basis"]["catalogue"]["fingerprint"] != before["basis"]["catalogue"]["fingerprint"]
    assert after != before


def test_two_totals_get_one_receipt_each() -> None:
    receipts = evidence_receipts(_run("revenue__sources", "receivables__sources"))
    assert sorted(r["concept"] for r in receipts) == ["receivables", "revenue"]


# --- who ran it: the verified principal -----------------------------------------------------------


def test_the_receipt_carries_the_run_and_the_verified_principal() -> None:
    """Set by the platform through verified_context, never through a feature's Options."""
    with verified_context(principal="pruefer@kanzlei.example", tenant_id="mandant-456"):
        _, [receipt] = run_with_receipts(
            ["revenue__sources"],
            compute_frameworks={PyArrowTable},
            data_access_collection=DataAccessCollection(folders={str(DOSSIER_A)}),
            plugin_collector=PLUGINS,
        )
    assert receipt["run"]["principal"] == "pruefer@kanzlei.example"
    assert receipt["run"]["tenant_id"] == "mandant-456"
    assert receipt["run"]["run_id"]


def test_without_a_verified_context_the_principal_is_stated_as_unknown() -> None:
    _, [receipt] = run_with_receipts(
        ["revenue__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(DOSSIER_A)}),
        plugin_collector=PLUGINS,
    )
    assert receipt["run"]["principal"] is None
    assert receipt["run"]["run_id"], "the run is still named"


def test_a_plain_run_all_cannot_see_the_run_and_says_so() -> None:
    """mloda builds no HookContext without an extender; the receipt states None, not a guess."""
    with verified_context(principal="pruefer@kanzlei.example"):
        [receipt] = evidence_receipts(_run("revenue__sources"))
    assert receipt["run"] == dict.fromkeys(receipt["run"])


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


def test_the_receipt_lists_its_citations_structured() -> None:
    [receipt] = evidence_receipts(_run("revenue__sources"))
    citations = receipt["citations"]
    assert len(citations) == receipt["lines"] == 5
    assert {c["file"] for c in citations} == {"GL.txt"}
    assert all(set(c) == {"file", "fingerprint", "record", "leg"} for c in citations)


# --- F1: why this producer, and what else could have answered -------------------------------------


def test_the_receipt_says_why_each_producer_answered() -> None:
    _, receipts = run_with_receipts(
        ["revenue__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(DOSSIER_A)}),
        plugin_collector=PLUGINS,
    )
    [receipt] = receipts
    resolution = {r["feature"]: r for r in receipt["resolution"]}
    assert set(resolution) == {"revenue__sources", "revenue", "gdpdu_journal__admitted", "gdpdu_journal"}
    assert resolution["revenue"]["chosen"] == ["SkrAccountFeatureGroup"]
    assert resolution["revenue"]["also_matched"] == []
    assert resolution["gdpdu_journal__admitted"]["chosen"] == ["TestClosing2025"]
    assert resolution["revenue__sources"]["requested"] is True


def test_a_diagnosis_of_another_request_is_refused() -> None:
    """The resolution must describe this run; a diagnosis of a different plan would lie."""
    other = mloda.diagnose(
        ["receivables__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(DOSSIER_A)}),
        plugin_collector=PLUGINS,
    )
    with pytest.raises(ValueError, match="does not describe this run"):
        evidence_receipts(_run("revenue__sources"), diagnosis=other)


def test_a_shadowing_subclass_is_named_in_the_receipt() -> None:
    """Its own process: the shadowing subclass registers process-wide."""
    script = str(Path(__file__).parent / "proofs" / "proof_resolution_receipt.py")
    proof = subprocess.run([sys.executable, script], capture_output=True, text=True)  # nosec B603
    assert proof.returncode == 0, proof.stdout + proof.stderr[-2000:]
    assert "chosen=['QuietRevision'] also_matched=['SkrAccountFeatureGroup']" in proof.stdout, proof.stdout
