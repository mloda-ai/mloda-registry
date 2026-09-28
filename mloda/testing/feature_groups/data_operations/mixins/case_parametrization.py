"""Shared base for pytest_generate_tests-based case mixins.

pytest calls only the first ``pytest_generate_tests`` hook it finds via MRO, so case
mixins (``NanPolicyTestMixin``, ``SingleValueStdVarTestMixin``, ...) must not each
define their own. Instead they inherit this base and declare their fixture in
``_case_fixtures``; this base parametrizes every declared fixture and chains to any
hook further down the MRO (e.g. a downstream base class).
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest


class CaseParametrizationTestMixin:
    """Base for case mixins: declare ``_case_fixtures``, never override ``pytest_generate_tests``."""

    _case_fixtures: ClassVar[dict[str, str]] = {}

    def pytest_generate_tests(self, metafunc: pytest.Metafunc) -> None:
        """Parametrize every fixture declared via ``_case_fixtures`` across the MRO, then chain."""
        merged: dict[str, tuple[str, type]] = {}
        for klass in type(self).__mro__:
            own = vars(klass).get("_case_fixtures")
            if not own:
                continue
            for fixture, getter in own.items():
                if fixture in merged and merged[fixture][1] is not klass:
                    _other_getter, other_klass = merged[fixture]
                    raise TypeError(f"fixture {fixture!r} declared by both {other_klass.__name__} and {klass.__name__}")
                merged[fixture] = (getter, klass)

        for fixture, (getter, _klass) in merged.items():
            if fixture not in metafunc.fixturenames:
                continue
            cases: list[Any] = sorted(getattr(self, getter)())
            metafunc.parametrize(fixture, cases, ids=cases)

        parent = getattr(super(), "pytest_generate_tests", None)
        if parent is not None:
            parent(metafunc)
