"""Real prepared runs seal saver credential reads as immutable audit records."""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest
from mloda.provider import (
    BaseArtifact,
    ComputeFramework,
    DataCreatorFeatureGroup,
    FeatureSet,
    credential_handle_spec,
    read_credential,
)
from mloda.steward import Extender, ExtenderHook, HookContext, LifecycleOutcome, RunContext, verified_context
from mloda.user import DataAccessCollection, Feature, ParallelizationMode, PluginCollector, mloda
from mloda.user.pyarrow import PyArrowTable

from mloda.enterprise.extenders.audit import (
    AuditExtender,
    HmacSha256Signer,
    NdjsonAuditSink,
    TeeAuditSink,
    verify_ndjson_log,
    verify_ndjson_log_coverage,
)
from mloda.enterprise.extenders.audit.tests.test_audit_extender import (
    _FailingSink,
    _hmac_signer,
    _ndjson,
    _sealing_config,
)

_COLUMN = "audit_artifact_saver_value"
_NO_ARTIFACT_COLUMN = "audit_without_artifact_value"
_CALCULATE = "audit-calculate-handle"
_SAVER = "audit-saver-handle"
_UNUSED = "audit-unused-handle"
_OPTIONS = {"calculate_handle": _CALCULATE, "saver_handle": _SAVER, "unused_handle": _UNUSED}
_CREDENTIALS = {handle: {"token": f"audit-saver-secret-{index}"} for index, handle in enumerate(_OPTIONS.values())}
_MODES = [ParallelizationMode.SYNC, ParallelizationMode.THREADING]


@dataclass
class _AuditState:
    events: list[str] = field(default_factory=list)
    calculations: int = 0
    saves: int = 0
    failure: RuntimeError | None = None


class _AuditArtifact(BaseArtifact):
    state = _AuditState()

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Any | None:
        for key in ("calculate_handle", "saver_handle", "saver_handle"):
            assert read_credential(features.get_options_key(key))["token"]
        cls.state.events.append("save")
        cls.state.saves += 1
        assert artifact == "audit-step-artifact"
        if cls.state.failure is not None:
            raise cls.state.failure
        return "audit-saved-artifact"


class _AuditArtifactFeatureGroup(DataCreatorFeatureGroup):
    state = _AuditState()
    PROPERTY_MAPPING = {key: credential_handle_spec("Artifact audit fixture handle.") for key in _OPTIONS}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {_COLUMN}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return _AuditArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.state.events.append("calculate")
        cls.state.calculations += 1
        for _ in range(2):
            assert read_credential(features.get_options_key("calculate_handle"))["token"]
        if features.artifact_to_save:
            features.save_artifact = "audit-step-artifact"
        return {name: [3, 5] for name in features.get_all_names()}


class _AuditWithoutArtifactFeatureGroup(_AuditArtifactFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {_NO_ARTIFACT_COLUMN}

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return None


class _FrozenSink:
    def __init__(self, state: _AuditState) -> None:
        self.state = state
        self.snapshots: list[str] = []

    def write(self, record: Mapping[str, Any]) -> None:
        self.state.events.append("write")
        self.snapshots.append(json.dumps(dict(record), sort_keys=True))


class _RunSnapshot(Extender):
    def __init__(self, state: _AuditState, audit_path: Path, manifest_path: Path) -> None:
        self.state = state
        self.audit_path = audit_path
        self.manifest_path = manifest_path
        self.priority = 200
        self.calculations: list[dict[str, Any]] = []
        self.seals: list[tuple[bytes, bytes]] = []
        self.outcomes: list[dict[str, Any]] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        finally:
            context = HookContext.current()
            assert context is not None
            self.calculations.append(
                {
                    "credentials_read": tuple(context.credentials_read),
                    "rows_out": context.rows_out,
                    "duration_seconds": context.duration_seconds,
                    "status": context.status,
                }
            )

    def on_run_complete(self, run: RunContext, outcome: LifecycleOutcome) -> None:
        self.state.events.append("sealed")
        self.seals.append(
            (
                self.audit_path.read_bytes() if self.audit_path.exists() else b"",
                self.manifest_path.read_bytes() if self.manifest_path.exists() else b"",
            )
        )
        self.outcomes.append(asdict(outcome))


@pytest.fixture
def state(monkeypatch: pytest.MonkeyPatch) -> _AuditState:
    current = _AuditState()
    monkeypatch.setattr(_AuditArtifactFeatureGroup, "state", current)
    monkeypatch.setattr(_AuditArtifact, "state", current)
    return current


def _prepare(
    tmp_path: Path,
    state: _AuditState,
    *,
    fail_closed: bool = False,
    saver_handle: str = _SAVER,
    group: type[_AuditArtifactFeatureGroup] = _AuditArtifactFeatureGroup,
) -> tuple[Any, AuditExtender, _FrozenSink, _RunSnapshot, HmacSha256Signer]:
    audit_path, manifest_path = _sealing_config(tmp_path)
    signer = _hmac_signer()
    frozen = _FrozenSink(state)
    audit = AuditExtender(
        sink=TeeAuditSink(frozen, NdjsonAuditSink(audit_path)),
        fail_closed=fail_closed,
        audit_path=audit_path,
        manifest_path=manifest_path,
        signer=signer,
        seal_failure_policy="raise",
    )
    snapshot = _RunSnapshot(state, audit_path, manifest_path)
    with verified_context(tenant_id="saver-tenant", project_id="saver-project", principal="saver-service"):
        session = mloda.prepare(
            [Feature(next(iter(group.feature_names_supported())), options={**_OPTIONS, "saver_handle": saver_handle})],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({group}),
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={audit, snapshot},
            data_access_collection=DataAccessCollection(credentials=_CREDENTIALS),
        )
    return session, audit, frozen, snapshot, signer


def _sealed_record(
    tmp_path: Path, audit: AuditExtender, frozen: _FrozenSink, snapshot: _RunSnapshot, signer: HmacSha256Signer
) -> dict[str, Any]:
    audit_path, manifest_path = _sealing_config(tmp_path)
    assert audit.seal_failures == 0
    assert len(snapshot.seals) == 1
    assert all(snapshot.seals[0]), (
        "the run-completion callback must already see the persisted audit and signed manifest"
    )
    assert snapshot.seals[0] == (audit_path.read_bytes(), manifest_path.read_bytes())
    coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=signer)
    assert coverage.head == verify_ndjson_log(audit_path, manifest_path, signer=signer)
    assert coverage.head is not None
    assert (coverage.sealed_runs, coverage.sealed_lines, coverage.unattributed_lines, coverage.unsealed_lines) == (
        1,
        1,
        0,
        {},
    )
    records = _ndjson(audit_path)
    assert len(records) == len(frozen.snapshots) == 1
    assert records == [json.loads(value) for value in frozen.snapshots]
    record = records[0]
    assert record["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name
    assert record["run_id"] and record["step_run_id"]
    seals = [seal for seal in _ndjson(manifest_path) if seal.get("run_id") == record["run_id"]]
    assert len(seals) == 1 and seals[0]["record_count"] == 1 and seals[0]["sealed_late"] is False
    return record


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("fail_closed", [False, True], ids=["observational", "fail-closed"])
@pytest.mark.parametrize("saver_fails", [False, True], ids=["saved", "save-failed"])
@pytest.mark.parametrize("same_handle", [False, True], ids=["separate-handles", "same-handle"])
def test_real_audit_records_include_saver_reads_before_automatic_sealing(
    tmp_path: Path,
    state: _AuditState,
    mode: ParallelizationMode,
    fail_closed: bool,
    saver_fails: bool,
    same_handle: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    saver_handle = _CALCULATE if same_handle else _SAVER
    if saver_fails:
        state.failure = RuntimeError(f"saver failed: {_CREDENTIALS[saver_handle]['token']}")
    session, audit, frozen, snapshot, signer = _prepare(
        tmp_path, state, fail_closed=fail_closed, saver_handle=saver_handle
    )

    if saver_fails:
        with pytest.raises(RuntimeError, match="saver failed") as error:
            session.run(parallelization_modes={mode})
        assert error.value is state.failure
    else:
        result = session.run(parallelization_modes={mode})
        assert result[0].to_pydict() == {_COLUMN: [3, 5]}
        assert result.artifacts == {_COLUMN: "audit-saved-artifact"}

    record = _sealed_record(tmp_path, audit, frozen, snapshot, signer)
    expected_reads = sorted({_CALCULATE, saver_handle})
    assert record["credentials_read"] == expected_reads, "the sealed snapshot must include every saver credential read"
    assert record["credential_handles"] == sorted({_CALCULATE, saver_handle, _UNUSED})
    assert record["status"] == ("error" if saver_fails else "success")
    assert record["error_type"] == ("builtins.RuntimeError" if saver_fails else None)
    assert record["cancel_type"] is None
    assert state.calculations == state.saves == 1
    assert state.events == ["calculate", "save", "write", "sealed"]
    assert snapshot.outcomes == [
        asdict(LifecycleOutcome("failed", "builtins.RuntimeError") if saver_fails else LifecycleOutcome("succeeded"))
    ]
    (raw,) = snapshot.calculations
    assert raw["credentials_read"] == (_CALCULATE,)
    assert raw["rows_out"] == record["rows_out"] == 2
    assert raw["duration_seconds"] is not None and raw["duration_seconds"] == record["duration_seconds"]
    assert raw["status"] == "success"
    for credential in _CREDENTIALS.values():
        assert credential["token"] not in "".join(frozen.snapshots) + caplog.text
        assert all(credential["token"].encode() not in content for content in snapshot.seals[0])


@pytest.mark.parametrize("mode", _MODES)
def test_a_real_run_without_an_artifact_keeps_its_audit_record_and_seal(
    tmp_path: Path, state: _AuditState, mode: ParallelizationMode
) -> None:
    session, audit, frozen, snapshot, signer = _prepare(tmp_path, state, group=_AuditWithoutArtifactFeatureGroup)

    result = session.run(parallelization_modes={mode})

    assert result[0].to_pydict() == {_NO_ARTIFACT_COLUMN: [3, 5]}
    assert result.artifacts == {}
    record = _sealed_record(tmp_path, audit, frozen, snapshot, signer)
    assert record["credentials_read"] == [_CALCULATE]
    assert record["status"] == "success" and record["error_type"] is None
    assert state.calculations == 1 and state.saves == 0
    assert state.events == ["calculate", "write", "sealed"]


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(
    ("raise_on_error", "fail_closed", "must_raise"),
    [(True, False, True), (False, False, False), (False, True, True)],
    ids=["default-strict", "warning-only", "fail-closed-overrides-warning-only"],
)
def test_a_real_successful_calculation_preserves_audit_sink_failure_policy(
    state: _AuditState, mode: ParallelizationMode, raise_on_error: bool, fail_closed: bool, must_raise: bool
) -> None:
    failure = OSError("audit sink unavailable")
    frozen = _FrozenSink(state)
    audit = AuditExtender(
        sink=TeeAuditSink(frozen, _FailingSink(failure)),
        raise_on_error=raise_on_error,
        fail_closed=fail_closed,
    )
    with verified_context(tenant_id="saver-tenant", project_id="saver-project", principal="saver-service"):
        session = mloda.prepare(
            [Feature(_COLUMN, options=_OPTIONS)],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({_AuditArtifactFeatureGroup}),
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={audit},
            data_access_collection=DataAccessCollection(credentials=_CREDENTIALS),
        )

    if must_raise:
        with pytest.raises(OSError, match="audit sink unavailable") as error:
            session.run(parallelization_modes={mode})
        assert error.value is failure
    else:
        result = session.run(parallelization_modes={mode})
        assert result[0].to_pydict() == {_COLUMN: [3, 5]}
        assert result.artifacts == {_COLUMN: "audit-saved-artifact"}
        assert state.saves == 1
    assert state.calculations == len(frozen.snapshots) == 1, "a sink failure must not rerun the successful calculation"
