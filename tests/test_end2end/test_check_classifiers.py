"""Classifier guard for scripts/check_classifiers.py: every generated classifier must be a known trove classifier."""

from __future__ import annotations

from pathlib import Path

from tests.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

chk = load_script("check_classifiers", _SCRIPTS / "check_classifiers.py")
gen = load_script("generate_pyproject", _SCRIPTS / "generate_pyproject.py")


def test_unknown_classifier_is_reported_with_package_and_value() -> None:
    known = {"Intended Audience :: Developers", "Typing :: Typed"}
    errors = chk.unknown_classifiers({"pkg-a": ["Intended Audience :: Developers", "Typing :: Typd"]}, known)
    assert len(errors) == 1
    assert "pkg-a" in errors[0]
    assert "Typing :: Typd" in errors[0]


def test_all_known_classifiers_yield_no_errors() -> None:
    known = {"Intended Audience :: Developers", "Typing :: Typed"}
    assert chk.unknown_classifiers({"pkg-a": sorted(known)}, known) == []


def test_generated_classifiers_cover_shared_and_generator_added() -> None:
    shared, packages_config = gen.load_configs()
    packages = packages_config["packages"]
    result = chk.generated_classifiers(shared, packages)
    assert set(result) == set(packages)
    for name, classifiers in result.items():
        for classifier in shared["project"]["classifiers"]:
            assert classifier in classifiers, f"{name} lacks shared classifier {classifier}"
    assert "Typing :: Typed" in result["mloda-community-aggregation"]
