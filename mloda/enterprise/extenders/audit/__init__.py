"""Enterprise Audit Extender package."""

from mloda.enterprise.extenders.audit.audit_extender import AuditExtender, AuditSink, NdjsonAuditSink
from mloda.enterprise.extenders.audit.run_manifest import (
    HmacSha256Signer,
    LogCoverage,
    ManifestSigner,
    ManifestVerificationError,
    RunAlreadySealedError,
    RunNotPendingError,
    manifest_hash,
    seal_ndjson_runs,
    seal_run,
    verify_manifest,
    verify_ndjson_log,
    verify_ndjson_log_coverage,
)

__all__ = [
    "AuditExtender",
    "AuditSink",
    "HmacSha256Signer",
    "LogCoverage",
    "ManifestSigner",
    "ManifestVerificationError",
    "NdjsonAuditSink",
    "RunAlreadySealedError",
    "RunNotPendingError",
    "manifest_hash",
    "seal_ndjson_runs",
    "seal_run",
    "verify_manifest",
    "verify_ndjson_log",
    "verify_ndjson_log_coverage",
]
