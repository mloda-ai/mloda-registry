"""Unit tests for the pure classification behind the two "wheel is absent" preconditions
(``binary_example/tests/test_binary_example_feature_group.py::TestManifest.
test_wheel_is_not_installed_precondition`` and
``test_licensing_invariants.py::TestEveryEnterpriseManifestImportsWithoutTheWheel.
test_no_binary_plugin_id_is_installed_as_a_wheel``).

Both currently ``pytest.skip()`` whenever the wheel happens to be importable, and otherwise
assert it is not -- so the check can only pass or skip, never fail, and a regression that makes
the wheel install by default (e.g. moved into the ``dev`` extra) would silently skip instead of
being caught.

``mloda.enterprise.tests.wheel_presence`` does not exist yet (Green phase adds it): every test
below fails with ``ModuleNotFoundError``/``NameError`` until it defines:

- ``classify_wheel_presence(spec_present: bool, opt_in: bool) -> str``, returning one of:
    - ``"absent"``     -- no wheel installed; the guard holds trivially.
    - ``"opted_in"``   -- a wheel is installed AND the ``MLODA_REAL_WHEEL=1`` opt-in was
      deliberately set (e.g. running ``tests/test_binary_model_real/`` on purpose); the caller
      should ``pytest.skip()``.
    - ``"unexpected"`` -- a wheel is installed WITHOUT the opt-in: an accidental install: the
      caller must fail loudly (``assert``), never skip.
- ``unexpected_wheel_plugin_ids(entries, opt_in) -> list[str]``, given an iterable of
  ``(plugin_id, spec_present)`` pairs, returning every ``plugin_id`` classified ``"unexpected"``.
  Must check every entry -- no early return -- so a regression that only inspects the first
  installed plugin_id is caught even though today's registry has a single BinaryModelMixin
  subclass.
"""

from __future__ import annotations

from mloda.enterprise.tests.wheel_presence import classify_wheel_presence, unexpected_wheel_plugin_ids


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
