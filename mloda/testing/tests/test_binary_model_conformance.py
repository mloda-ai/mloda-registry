"""Wires the class-based binary-model conformance kit to our own simulated CLI stub.

Every class attribute default in ``BinaryModelConformanceBase``/``HashOperationConformanceMixin``
already points at ``mloda.testing.binary_model.simulated_binary``, so nothing needs overriding
here; a future conformance run against a real binary reuses these classes unmodified by
subclassing with a different ``binary_cmd``."""

from __future__ import annotations

from mloda.testing.binary_model.conformance import BinaryModelConformanceBase, HashOperationConformanceMixin


class TestBinaryModelConformance(HashOperationConformanceMixin, BinaryModelConformanceBase):
    pass


def test_size_cap_constants_are_exported() -> None:
    """`MESSAGE_MAX_BYTES` and `STDERR_SOFT_CAP_BYTES` belong on the conformance kit's public
    surface, re-exported via `__all__` for external consumers (contract: Data handling)."""
    from mloda.testing.binary_model import conformance

    assert "MESSAGE_MAX_BYTES" in conformance.__all__
    assert "STDERR_SOFT_CAP_BYTES" in conformance.__all__
    assert "COLUMN_TYPES" in conformance.__all__
    assert [name for name in conformance.__all__ if not hasattr(conformance, name)] == []
