"""A host's admissibility policy, for the end-to-end tests only.

Imported lazily by the tests that run the full chain: importing it is what makes
`gdpdu_journal__admitted` resolvable, and exactly one configured policy may be live per process.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from mloda.user import PluginCollector

from mloda.community.feature_groups.experimental.german_ledger.policy import (
    AdmissibilityPolicyGroup,
    DatevJournalFeatureGroup,
    JournalFeatureGroup,
)
from mloda.community.feature_groups.experimental.german_ledger.skr import SkrAccountFeatureGroup
from mloda.community.feature_groups.experimental.german_ledger.sources import SourcesFeatureGroup


class TestClosing2025(AdmissibilityPolicyGroup):
    """FY2025 closed on 31 Dec; entries keyed in after 15 Jan 2026 are refused."""

    __test__ = False  # a policy, not a pytest class
    LOCK_DATE = date(2026, 1, 15)
    PERIOD_END = date(2025, 12, 31)


# Every end-to-end run in the tests resolves against exactly this package's groups. The registry
# runs all packages' tests in shared processes, and an unscoped run would also see every other
# package's groups and test classes.
def plugins_with(policy: type[AdmissibilityPolicyGroup]) -> PluginCollector:
    """This package's groups with `policy` as the one live admissibility policy."""
    return PluginCollector.enabled_feature_groups(
        {JournalFeatureGroup, DatevJournalFeatureGroup, policy, SkrAccountFeatureGroup, SourcesFeatureGroup}
    )


PLUGINS = plugins_with(TestClosing2025)


def _cause(exc: BaseException, kind: type[BaseException]) -> BaseException | None:
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, kind):
            return seen
        seen = seen.__cause__ or seen.__context__
    return None


class ListSink:
    """An audit sink collecting every record."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: Mapping[str, Any]) -> None:
        self.records.append(dict(record))


def audit_gate(sink: ListSink, clearance: str) -> Any:
    """A fail-closed audit gate granting `clearance`; needs the enterprise audit package."""
    from mloda.enterprise.extenders.audit import AuditExtender, ClassificationPolicy

    return AuditExtender(
        sink=sink,
        fail_closed=True,
        required_identity=("tenant_id", "principal"),
        policy_version="german-ledger-test",
        classification=ClassificationPolicy(clearance=lambda tenant, principal: clearance, undeclared="public"),
    )
