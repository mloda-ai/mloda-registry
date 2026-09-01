"""Drift check: the binary-model contract doc must stay in sync with CONTRACT_VERSION."""

import re
from pathlib import Path

from mloda.testing.binary_model import CONTRACT_VERSION


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT_DOC = _REPO_ROOT / "docs" / "binary-model-contract.md"

_CONTRACT_VERSION_SENTENCE_PATTERN = re.compile(r"This is contract version (\d+)\.")


def test_binary_model_contract_doc_exists() -> None:
    """docs/binary-model-contract.md must exist so the binary-model contract has a public home."""
    assert _CONTRACT_DOC.is_file(), (
        f"{_CONTRACT_DOC.relative_to(_REPO_ROOT)} does not exist. Add the public binary-model "
        "contract document at that path (it must contain the literal sentence "
        f"'This is contract version {CONTRACT_VERSION}.' so this drift guard can verify it)."
    )


def test_binary_model_contract_doc_version_matches_contract_version() -> None:
    """The doc's 'This is contract version N.' sentence must match mloda.testing.binary_model.CONTRACT_VERSION."""
    if not _CONTRACT_DOC.is_file():
        raise AssertionError(
            f"{_CONTRACT_DOC.relative_to(_REPO_ROOT)} does not exist, so its contract version cannot be checked "
            "against mloda.testing.binary_model.CONTRACT_VERSION. Add the doc first "
            "(see test_binary_model_contract_doc_exists)."
        )

    doc_text = _CONTRACT_DOC.read_text(encoding="utf-8")
    match = _CONTRACT_VERSION_SENTENCE_PATTERN.search(doc_text)

    assert match is not None, (
        f"{_CONTRACT_DOC.relative_to(_REPO_ROOT)} does not contain the literal sentence "
        f"'This is contract version {CONTRACT_VERSION}.' (a plain sentence, not a code block or "
        "front-matter field). Add that sentence so this drift guard can extract the documented version."
    )

    documented_version = int(match.group(1))

    assert documented_version == CONTRACT_VERSION, (
        f"{_CONTRACT_DOC.relative_to(_REPO_ROOT)} says 'This is contract version {documented_version}.' but "
        f"mloda.testing.binary_model.CONTRACT_VERSION is {CONTRACT_VERSION}. Whenever one changes, the other must "
        f"be bumped too: update the doc's sentence to 'This is contract version {CONTRACT_VERSION}.' if "
        "CONTRACT_VERSION is correct, or bump CONTRACT_VERSION in mloda/testing/binary_model/__init__.py if the "
        "doc's version is the intended one."
    )
