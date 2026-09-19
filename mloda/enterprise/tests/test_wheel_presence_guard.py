"""Unit tests for the pure classification behind the two "wheel is absent" preconditions
(``binary_example/tests/test_binary_example_feature_group.py::TestManifest.
test_wheel_is_not_installed_precondition`` and
``test_licensing_invariants.py::TestEveryEnterpriseManifestImportsWithoutTheWheel.
test_no_binary_plugin_id_is_installed_as_a_wheel``).

An installed wheel is classified as a deliberate opt-in (``MLODA_REAL_WHEEL=1``, skip) or an
unexpected install (fail loudly) -- never silently either -- so a regression that makes the wheel
install by default (e.g. moved into the ``dev`` extra) is caught rather than skipped. Also checks
that every installed plugin_id is inspected, not just the first, and that ``MLODA_REAL_WHEEL`` is
in ``tox.ini``'s ``[testenv] passenv`` so the opt-in survives into every gate environment.
"""

from __future__ import annotations

import configparser
from pathlib import Path

from mloda.enterprise.tests.wheel_presence import classify_wheel_presence, unexpected_wheel_plugin_ids

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOX_INI = _REPO_ROOT / "tox.ini"


def _testenv_passenv() -> list[str]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(_TOX_INI)
    return parser["testenv"]["passenv"].split()


class TestClassifyWheelPresence:
    def test_absent_when_no_wheel_is_installed(self) -> None:
        assert classify_wheel_presence(spec_present=False, opt_in=False) == "absent"

    def test_absent_when_no_wheel_is_installed_even_with_opt_in_set(self) -> None:
        """The opt-in alone never fabricates a wheel install; absence always wins."""
        assert classify_wheel_presence(spec_present=False, opt_in=True) == "absent"

    def test_opted_in_when_wheel_installed_and_opt_in_set(self) -> None:
        assert classify_wheel_presence(spec_present=True, opt_in=True) == "opted_in"

    def test_unexpected_when_wheel_installed_without_opt_in(self) -> None:
        """An accidental install (opt-in not set) must be classified distinctly from a deliberate
        opt-in -- this is what makes the guard falsifiable instead of always skip-or-pass."""
        assert classify_wheel_presence(spec_present=True, opt_in=False) == "unexpected"


class TestUnexpectedWheelPluginIds:
    def test_checks_every_entry_not_just_the_first(self) -> None:
        """Regression guard for the loop bug: two installed-without-opt-in plugin_ids after a
        first, absent one must both be reported, not just the first installed one found."""
        entries = [("plugin_a", False), ("plugin_b", True), ("plugin_c", True)]
        assert unexpected_wheel_plugin_ids(entries, opt_in=False) == ["plugin_b", "plugin_c"]

    def test_empty_when_every_installed_entry_is_opted_in(self) -> None:
        entries = [("plugin_a", True), ("plugin_b", True)]
        assert unexpected_wheel_plugin_ids(entries, opt_in=True) == []

    def test_empty_when_nothing_is_installed(self) -> None:
        entries = [("plugin_a", False), ("plugin_b", False)]
        assert unexpected_wheel_plugin_ids(entries, opt_in=False) == []


class TestPassenvCarriesTheOptIn:
    def test_mloda_real_wheel_is_in_testenv_passenv(self) -> None:
        """[testenv:binary-model] and every pythonNNN env inherit [testenv], so this one entry
        covers every gate environment MLODA_REAL_WHEEL must survive into."""
        assert "MLODA_REAL_WHEEL" in _testenv_passenv()
