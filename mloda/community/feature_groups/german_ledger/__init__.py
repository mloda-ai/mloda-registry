"""mloda-community-german-ledger: cited, admissible totals from German ledger exports.

Reads GDPdU/GoBD dossiers and DATEV-Format EXTF booking batches, and answers concepts such as
``revenue__sources`` with a total AND the ledger lines behind it -- or refuses by name.

Importing this package registers its feature groups. Until mloda#1745 is fixed, register them
by import, not through ``PluginLoader.all()``: that call also loads the stock
``ReadFileFeature``, which then claims the same feature names and the request fails with
"Multiple feature groups found". See README.md.
"""

from __future__ import annotations

from mloda.community.feature_groups.german_ledger.datev import DatevExtfReader, DatevRefusal
from mloda.community.feature_groups.german_ledger.policy import (
    AdmissibilityPolicyGroup,
    AdmissibilityRefused,
    AdmissibilityRule,
    DatevJournalFeatureGroup,
    Festschreibung,
    JournalFeatureGroup,
    LateEntryCutoff,
    LateEntryRefused,
    PeriodBound,
)
from mloda.community.feature_groups.german_ledger.reader import GdpduReader
from mloda.community.feature_groups.german_ledger.receipt import (
    Citation,
    ReceiptContext,
    evidence_receipts,
    parse_citation,
    run_with_receipts,
)
from mloda.community.feature_groups.german_ledger.skr import (
    SKR03_2025,
    SKR04_2025,
    AmbiguousProfile,
    ChartConflict,
    GrossRevenueRefused,
    LedgerProfile,
    NoProfileForDossier,
    SkrAccountFeatureGroup,
)
from mloda.community.feature_groups.german_ledger.sources import (
    RECEIPT_VERSION,
    AmbiguousSourceColumns,
    InadmissibleTotal,
    SourcesFeatureGroup,
    UncitedValue,
)

__all__ = [
    "RECEIPT_VERSION",
    "SKR03_2025",
    "SKR04_2025",
    "AdmissibilityPolicyGroup",
    "AdmissibilityRefused",
    "AdmissibilityRule",
    "AmbiguousProfile",
    "AmbiguousSourceColumns",
    "Citation",
    "ChartConflict",
    "DatevExtfReader",
    "DatevJournalFeatureGroup",
    "DatevRefusal",
    "Festschreibung",
    "GdpduReader",
    "GrossRevenueRefused",
    "InadmissibleTotal",
    "JournalFeatureGroup",
    "LateEntryCutoff",
    "LateEntryRefused",
    "LedgerProfile",
    "NoProfileForDossier",
    "PeriodBound",
    "ReceiptContext",
    "SkrAccountFeatureGroup",
    "SourcesFeatureGroup",
    "UncitedValue",
    "evidence_receipts",
    "parse_citation",
    "run_with_receipts",
]
