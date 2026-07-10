"""Reusable capability-hook test mixin for data-operations feature groups.

Each per-backend concrete test module mixes this in and declares the match-time
``supports_compute_framework`` behaviour it expects for that one backend. The
mixin drives the classmethod hook directly (``supports_compute_framework``) per
probe. The engine path is covered repo-locally by ``TestResolveFeatureIntegration``
in ``mloda/community/feature_groups/data_operations/tests/test_capability_hook.py``.

A backend declares its ``(feature_name, options)`` probes as ``supported`` /
``unsupported`` / ``conservative`` tuples:

- ``supported`` probes name operations the backend can compute; the hook must
  return ``True``.
- ``unsupported`` probes name operations the backend cannot compute; the hook
  must return ``False``. A backend that restricts nothing leaves this empty.
- ``conservative`` probes are unparsable feature names the hook must keep
  ``True`` by default; they are checked exactly like ``supported`` probes and
  default to a single ``totally_unrelated`` name.

Skipping when an optional dependency is missing is the concrete module's job via
``pytest.importorskip`` at module scope; this mixin never guards deps itself.

Requires the host class to provide:
- ``implementation_class()`` (from the family test base), and
- ``compute_framework_class()`` (from the framework adapter mixin).
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.options import Options


class CapabilityHookTestMixin:
    """Mixin providing shared match-time capability-hook tests.

    Requires the host class to provide:
    - ``implementation_class()`` (from the family test base)
    - ``compute_framework_class()`` (from the framework adapter mixin)
    """

    # -- Configuration methods (override per backend) --------------------------

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        """Probes the backend must accept (hook ``True``, framework kept supported)."""
        return ()

    @classmethod
    def capability_unsupported(cls) -> tuple[tuple[str, Options], ...]:
        """Probes the backend must reject (hook ``False``, framework moved to rejected)."""
        return ()

    @classmethod
    def capability_conservative(cls) -> tuple[tuple[str, Options], ...]:
        """Unparsable feature names the hook must keep ``True`` by default.

        Defaults to a single ``totally_unrelated`` probe so a backend that
        restricts nothing still exercises the accept path.
        """
        return (("totally_unrelated", Options()),)

    # -- Concrete test methods -------------------------------------------------

    def test_mixin_capability_hook_accepts(self) -> None:
        """Mixin: the hook accepts every supported and conservative probe.

        ``supports_compute_framework`` must return ``True`` for operations the
        backend can compute and for feature names it cannot parse (staying
        conservative), so those frameworks remain eligible at match time.
        """
        backend = self.implementation_class()  # type: ignore[attr-defined]
        framework = self.compute_framework_class()  # type: ignore[attr-defined]
        for name, opts in (*self.capability_supported(), *self.capability_conservative()):
            assert backend.supports_compute_framework(name, opts, framework) is True, (
                f"expected {name!r} accepted by {framework.__name__}"
            )

    def test_mixin_capability_hook_rejects(self) -> None:
        """Mixin: the hook rejects every unsupported probe.

        ``supports_compute_framework`` must return ``False`` for operations the
        backend cannot compute, so the framework is dropped at match time
        instead of failing later inside ``calculate_feature``. Backends that
        restrict no subtypes declare no unsupported probes and skip.
        """
        unsupported = self.capability_unsupported()
        backend = self.implementation_class()  # type: ignore[attr-defined]
        if not unsupported:
            # Guard the regression: a restriction with no probes would silently vanish.
            # supported_subtypes() probes an axis-keyed backend (_CAPABILITY_HAS_AXIS) on
            # its default axis, so a non-None return means the backend declares a
            # restriction on that axis and therefore must declare capability_unsupported
            # probes. None means unrestricted (nothing to probe).
            assert backend.supported_subtypes() is None, (
                f"{backend.__name__} declares a subtype restriction on its default axis via "
                "supported_subtypes() but ships no capability_unsupported probes; add them"
            )
            pytest.skip("backend restricts no subtypes")

        framework = self.compute_framework_class()  # type: ignore[attr-defined]
        for name, opts in unsupported:
            assert backend.supports_compute_framework(name, opts, framework) is False, (
                f"expected {name!r} rejected by {framework.__name__}"
            )

    def test_mixin_compute_framework_declared(self) -> None:
        """The mixin's framework is one the backend declares and that is installed."""
        backend = self.implementation_class()  # type: ignore[attr-defined]
        framework = self.compute_framework_class()  # type: ignore[attr-defined]
        assert framework in backend.compute_framework_definition()
        assert framework.is_available()
