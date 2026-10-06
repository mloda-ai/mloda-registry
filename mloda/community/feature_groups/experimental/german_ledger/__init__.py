"""mloda-community-german-ledger: cited, admissible totals from German ledger exports.

Reads GDPdU/GoBD dossiers and DATEV-Format EXTF booking batches, and answers concepts such as
``revenue__sources`` with a total AND the ledger lines behind it -- or refuses by name.

Importing this package registers its feature groups. Until mloda#1745 is fixed, register them
by import, not through ``PluginLoader.all()``: that call also loads the stock read feature groups
(``ReadDocumentFeature`` reads a dossier's ``GL.txt``), which then claim the same feature names
and the request fails with "Multiple feature groups found". See README.md.
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
