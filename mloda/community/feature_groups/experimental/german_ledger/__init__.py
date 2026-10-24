"""mloda-community-german-ledger: cited, admissible totals from German ledger exports.

Reads GDPdU/GoBD dossiers and DATEV-Format EXTF booking batches, and answers concepts such as
``revenue__sources`` with a total AND the ledger lines behind it -- or refuses by name.

``PluginLoader.all()`` registers the feature groups through the entry point; importing the package registers them too.
"""

from __future__ import annotations

from mloda.community.feature_groups.experimental.german_ledger.datev import DatevExtfReader, DatevRefusal
from mloda.community.feature_groups.experimental.german_ledger.policy import (
    AdmissibilityPolicyGroup,
    AdmissibilityRefused,
    AdmissibilityRule,
    DatevJournalFeatureGroup,
    Festschreibung,
    ForFormat,
    JournalFeatureGroup,
    LateEntryCutoff,
    LateEntryRefused,
    PeriodBound,
)
from mloda.community.feature_groups.experimental.german_ledger.reader import Citation, GdpduReader, parse_citation
from mloda.community.feature_groups.experimental.german_ledger.skr import (
    SKR03_2025,
    SKR04_2025,
    AccountLengthUnsupported,
    AmbiguousProfile,
    ChartConflict,
    GrossRevenueRefused,
    LedgerProfile,
    NoProfileForDossier,
    SkrAccountFeatureGroup,
)
from mloda.community.feature_groups.experimental.german_ledger.sources import (
    AmbiguousSourceColumns,
    InadmissibleTotal,
    SourcesFeatureGroup,
    UncitedValue,
)

__all__ = [
    "SKR03_2025",
    "SKR04_2025",
    "AccountLengthUnsupported",
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
    "ForFormat",
    "GdpduReader",
    "GrossRevenueRefused",
    "InadmissibleTotal",
    "JournalFeatureGroup",
    "LateEntryCutoff",
    "LateEntryRefused",
    "LedgerProfile",
    "NoProfileForDossier",
    "PeriodBound",
    "SkrAccountFeatureGroup",
    "SourcesFeatureGroup",
    "UncitedValue",
    "parse_citation",
]
