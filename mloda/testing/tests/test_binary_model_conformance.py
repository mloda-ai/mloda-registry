"""Wires the class-based binary-model conformance kit to our own simulated CLI stub.

Every class attribute default in ``BinaryModelConformanceBase``/``HashOperationConformanceMixin``
already points at ``mloda.testing.binary_model.simulated_binary``, so nothing needs overriding
here; a future conformance run against a real binary reuses these classes unmodified by
subclassing with a different ``binary_cmd``."""

from __future__ import annotations

from mloda.testing.binary_model.conformance import BinaryModelConformanceBase, HashOperationConformanceMixin


class TestBinaryModelConformance(HashOperationConformanceMixin, BinaryModelConformanceBase):
    pass
