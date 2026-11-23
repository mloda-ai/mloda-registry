"""A successful retry must not erase the preceding calculation's failed audit record."""

import json
from pathlib import Path
from typing import Any

import pytest
from mloda.provider import ComputeFramework, DataCreatorFeatureGroup, FeatureSet
from mloda.steward import Extender, ExtenderHook, verified_context
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda.user.pyarrow import PyArrowTable

from mloda.enterprise.extenders.audit import AuditExtender, NdjsonAuditSink


class _RetryOnce(Extender):
    def __init__(self) -> None:
        self.priority = 10

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ValueError:
            return func(*args, **kwargs)


@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
def test_successful_retry_retains_each_calculations_own_audit_outcome(
    tmp_path: Path, mode: ParallelizationMode
) -> None:
    attempts: list[None] = []

    class RetriedAuditFeature(DataCreatorFeatureGroup):
        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
            return {PyArrowTable}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> dict[str, list[int]]:
            attempts.append(None)
            if len(attempts) == 1:
                raise ValueError("first calculation failed")
            return {"RetriedAuditFeature": [7, 11]}

    path = tmp_path / "retry-audit.ndjson"
    audit = AuditExtender(sink=NdjsonAuditSink(path))
    with verified_context(tenant_id="retry-audit-tenant", principal="retry-audit-principal"):
        result = mloda.run_all(
            [Feature("RetriedAuditFeature")],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({RetriedAuditFeature}),
            function_extender={_RetryOnce(), audit},
            parallelization_modes={mode},
        )

    assert len(attempts) == 2
    assert result[0].to_pydict() == {"RetriedAuditFeature": [7, 11]}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert all(record["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name for record in records)
    assert records[0]["run_id"] == records[1]["run_id"] is not None
    assert records[0]["step_run_id"] == records[1]["step_run_id"] is not None
    assert [(record["status"], record["error_type"], record["cancel_type"]) for record in records] == [
        ("error", "builtins.ValueError", None),
        ("success", None, None),
    ]
