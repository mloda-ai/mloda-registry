"""Wires the class-based binary-model conformance kit to our own simulated CLI stub.

``BinaryModelConformanceBase`` and ``HashOperationConformanceMixin`` (the reusable,
pip-installable conformance kit) live at ``mloda.testing.binary_model.conformance``; every class
attribute default there already points at ``mloda.testing.binary_model.simulated_binary``, so
nothing needs overriding here. A future conformance run against a real binary reuses those classes
unmodified by subclassing with a different ``binary_cmd`` (and, if needed, a different
``plugin_id``/``operations``/``column_types``) instead of writing a new test suite.
"""

from __future__ import annotations

from mloda.testing.binary_model.conformance import BinaryModelConformanceBase, HashOperationConformanceMixin


class TestBinaryModelConformance(HashOperationConformanceMixin, BinaryModelConformanceBase):
    pass
