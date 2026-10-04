"""Enterprise Audit Extender package."""

from mloda.enterprise.extenders.audit._quarantine import (
    QuarantinedLine,
    quarantine_damaged_lines,
    quarantine_from_rotation_entry,
    verify_quarantine_log,
)
from mloda.enterprise.extenders.audit._segments import rotate_ndjson_segment, verify_ndjson_segments
from mloda.enterprise.extenders.audit._signers import Ed25519Signer, HmacSha256Signer, ManifestSigner
from mloda.enterprise.extenders.audit._verify import (
    HeadAnchor,
    KeyAlreadyCurrentError,
    LogCoverage,
    ManifestVerificationError,
    NdjsonHeadAnchor,
    RunAlreadySealedError,
    RunNotPendingError,
    manifest_hash,
    seal_run,
    verify_manifest,
)
from mloda.enterprise.extenders.audit.audit_extender import (
    AuditExtender,
    AuditSink,
    IdentityRequiredError,
    NdjsonAuditSink,
    SealedRunRefusedError,
    TeeAuditSink,
)
from mloda.enterprise.extenders.audit.otel_log_sink import OtelLogAuditSink
from mloda.enterprise.extenders.audit.run_manifest import (
    rotate_manifest_key,
    seal_ndjson_runs,
    verify_ndjson_log,
    verify_ndjson_log_coverage,
)

__all__ = [
    "AuditExtender",
    "AuditSink",
    "Ed25519Signer",
    "HeadAnchor",
    "HmacSha256Signer",
    "IdentityRequiredError",
    "KeyAlreadyCurrentError",
    "LogCoverage",
    "ManifestSigner",
    "ManifestVerificationError",
    "NdjsonAuditSink",
    "NdjsonHeadAnchor",
    "OtelLogAuditSink",
    "QuarantinedLine",
    "RunAlreadySealedError",
    "RunNotPendingError",
    "SealedRunRefusedError",
    "TeeAuditSink",
    "manifest_hash",
    "quarantine_damaged_lines",
    "quarantine_from_rotation_entry",
    "rotate_manifest_key",
    "rotate_ndjson_segment",
    "seal_ndjson_runs",
    "seal_run",
    "verify_manifest",
    "verify_ndjson_log",
    "verify_ndjson_log_coverage",
    "verify_ndjson_segments",
    "verify_quarantine_log",
]
