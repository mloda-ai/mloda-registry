"""Both extender test_manifest.py files must share one eviction helper, not near-duplicate local copies."""

from __future__ import annotations

import importlib

from mloda.testing.import_isolation import block_root, evict_package

_MANIFEST_TEST_MODULES = [
    "mloda.community.extenders.openlineage.tests.test_manifest",
    "mloda.community.extenders.otel.tests.test_manifest",
]


def test_manifest_tests_share_one_eviction_helper() -> None:
    for module_name in _MANIFEST_TEST_MODULES:
        module = importlib.import_module(module_name)
        assert module.evict_package is evict_package, f"{module_name} must import evict_package from mloda.testing"
        assert module.block_root is block_root, f"{module_name} must import block_root from mloda.testing"
