"""Isolated: a reader subclass with a lower CLASSIFICATION replaces the stock reader process-wide.

A subprocess because the subclass registers process-wide. Exits 77 when the enterprise audit
package or defusedxml is absent.
"""

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import defusedxml  # noqa: F401

    from mloda.enterprise.extenders.audit import AuditExtender, ClassificationPolicy
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

from mloda.steward import verified_context
from mloda.user import DataAccessCollection, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.experimental.german_ledger.reader import GdpduReader
from mloda.community.feature_groups.experimental.german_ledger.tests._host import PLUGINS

TWIN = Path(__file__).parent.parent / "fixtures" / "twin_2025" / "gdpdu"


class InternalGdpduReader(GdpduReader):
    CLASSIFICATION = "internal"


class ListSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: Mapping[str, Any]) -> None:
        self.records.append(dict(record))


sink = ListSink()
gate = AuditExtender(
    sink=sink,
    fail_closed=True,
    required_identity=("tenant_id", "principal"),
    policy_version="proof",
    classification=ClassificationPolicy(clearance=lambda tenant, principal: "internal", undeclared="public"),
)
with verified_context(tenant_id="t", principal="svc"):
    result = mloda.run_all(
        ["revenue__sources"],
        compute_frameworks=[PyArrowTable],
        data_access_collection=DataAccessCollection(folders={str(TWIN)}),
        plugin_collector=PLUGINS,
        function_extender={gate},
    )

# The probed parent GdpduReader is denied at plan time and then dropped; only the subclass must pass.
gdpdu_denies = [
    r
    for r in sink.records
    if r["decision"] == "deny" and str(r.get("reader_class", "")).rsplit(".", 1)[-1] == "InternalGdpduReader"
]
calculate_levels = {r["classification"] for r in sink.records if r.get("hook") == "FEATURE_GROUP_CALCULATE_FEATURE"}
if gdpdu_denies or not result or calculate_levels != {"internal"}:
    print(f"UNEXPECTED: denies={gdpdu_denies} result={bool(result)} calculate={calculate_levels}")
    sys.exit(1)
print("REMEDY: a reader subclass declaring internal runs under clearance internal")
