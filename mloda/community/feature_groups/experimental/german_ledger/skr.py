"""SkrAccountFeatureGroup: account numbers -> declarable concepts, with origins attached."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import pyarrow as pa
from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import DataAccessCollection, Feature, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from .datev import SOLL_POSITIVE, VORZEICHEN
from .policy import ADMITTED, unprefixed
from .reader import ADMISSIBILITY_COLUMN, GdpduReader, is_admitted

# Explicit, versioned, deliberately narrow -- and *stated*, not cited: these are two ranges
# chosen and written down here, not a DATEV catalogue reproduced under licence. "Which
# definition did you use?" is the auditor's next question, and this comment is the whole
# answer. Two ranges and str.isdigit() is a spike's chart of accounts, not a chart of accounts.
#
#   revenue      A published narrow slice: SKR04 Umsatzerloese 4000-4499. NOT all of class 4,
#                which also carries Sonderbetriebseinnahmen (4500-4505), Erloesschmaelerungen
#                (4700-4799, a contra that net revenue should include) and sonstige
#                betriebliche Ertraege (4830+). There is no single official "Umsatzerloese =
#                4000-4499"; other published BWA layouts cut it differently. This is a stated
#                choice, not the definition.
#   receivables  SKR04 Forderungen aus Lieferungen und Leistungen, 1200-1249.
#                1400 is *SKR03* receivables; in SKR04 it is abziehbare Vorsteuer, and mixing
#                the two charts is the classic error here.
#                Caveat for real dumps: 1200 is the Sammelkonto. DATEV posts FLL to Debitoren
#                (Personenkonten), so a real journal may carry no 1200-1249 lines at all, and
#                1246-1249 are value adjustments that an unsigned sum would add instead of net.
SKR04_2025: dict[str, range] = {
    "revenue": range(4000, 4500),
    "receivables": range(1200, 1250),
}

# The same two concepts in SKR03, stated the same way and just as narrow:
#
#   revenue      SKR03 Erloese 8000-8499, the counterpart of SKR04 4000-4499 (8400 here is
#                4400 there). Not all of class 8: Erloesschmaelerungen (8700-8799) and
#                sonstige Ertraege sit above it, as in SKR04.
#   receivables  SKR03 Forderungen aus Lieferungen und Leistungen, 1400-1449. 1200 in SKR03 is
#                Bank, so a host that reads SKR03 books under SKR04 counts its bank balance as
#                receivables. The same Sammelkonto caveat holds: DATEV posts FLL to Debitoren.
SKR03_2025: dict[str, range] = {
    "revenue": range(8000, 8500),
    "receivables": range(1400, 1450),
}

# Keyed the way a DATEV header writes field 27.
CHARTS: dict[str, dict[str, range]] = {"03": SKR03_2025, "04": SKR04_2025}

# The name each chart's catalogue is cited by. The fingerprint beside it is computed
# from the ranges actually in force, so an edited catalogue cannot pass under its old name.
CATALOGUE_NAMES: dict[str, str] = {"03": "SKR03_2025", "04": "SKR04_2025"}

# Automatikkonten inside the revenue slice: DATEV splits the VAT out of what is booked there,
# so the booked amount is GROSS. Source: Odoo's l10n_de chart templates (LGPL-3), branch 18.0
# at commit 925f8cbe6aeb, files account.account-de_skr0{3,4}.csv joined with
# account.tax-de_skr0{3,4}.csv: every account in the revenue slice whose default tax has a
# rate above 0 %. Tax-free and reverse-charge accounts (e.g. 8125, and 8130 despite its tax
# key's "19" in the name) default to 0 % and stay net. Odoo's defaults stand in for DATEV's
# own list, which is not reproduced here; a host whose books use further automatic revenue
# accounts must add them, or their gross amounts pass.
AUTOMATIC_REVENUE: dict[str, frozenset[int]] = {
    "03": frozenset({8196, 8300, 8310, 8315, 8400, 8410}),
    "04": frozenset({4186, 4200, 4300, 4310, 4315, 4400}),
}

# Each concept reports in its natural direction: revenue is a credit balance, receivables a
# debit one. A debit-positive journal (DATEV legs, +Soll/-Haben) is flipped for credit
# concepts; a journal that states no convention (GDPdU) is taken as declared, which is how a
# Sachkonten export writes its lines.
_CREDIT_CONCEPTS = frozenset({"revenue"})
AS_DECLARED = "as-declared"

# BU key 40 is "Aufhebung der Automatik": no tax split, so the booked amount stands as net.
_LIFTS_AUTOMATIC = "40"


@dataclass(frozen=True)
class LedgerProfile:
    """Which DECLARED columns carry the account and the amount, for one exporter.

    The descriptor says what the columns are called and typed; it does not say which of them
    is the account number and which is the money. That mapping is per exporter, and it is the
    one thing a second dossier actually costs. Host configuration, never sniffing: guessing
    which column is the amount is exactly what the descriptor exists to prevent.
    """

    account_column: str
    amount_column: str
    # How this exporter signs its amounts: "as-declared" (each line in its account's own
    # direction, as a Sachkonten export writes it) or "soll-positiv" (+Soll/-Haben). None
    # states nothing, and a journal that states nothing either is then taken as declared.
    sign: str | None = None

    def __post_init__(self) -> None:
        if self.sign not in (None, AS_DECLARED, SOLL_POSITIVE):
            raise ValueError(f"sign {self.sign!r} is not one of {AS_DECLARED!r}, {SOLL_POSITIVE!r} or None")


DATEV_LIKE = LedgerProfile(account_column="Konto", amount_column="Betrag")
SECOND_EXPORTER = LedgerProfile(account_column="Sachkonto", amount_column="Umsatz")


class NoProfileForDossier(Exception):
    """No known profile names columns this dossier declares."""


class AmbiguousProfile(Exception):
    """More than one profile fits this dossier, so which one is a guess."""


def select_profile(profiles: tuple[LedgerProfile, ...], declared: set[str]) -> LedgerProfile:
    """Pick the profile the DOSSIER fits, not the one class resolution happens to reach.

    Modelling one exporter as one FeatureGroup subclass made profile identity a property of
    the class graph, and mloda keeps a matching subclass over its parent: a second profile
    silently shadowed the first, and the run returned a plausible total under the wrong
    mapping (proof_profile_shadow.py). Three profiles collided outright. Both are the same
    mistake -- asking resolution order a question only the data can answer.

    Selection here is a property of the dossier: a profile fits when the dossier declares
    both of the columns it names. One fit is the answer, none is a configuration error that
    names what was declared, and more than one is refused rather than resolved -- two
    profiles that both fit are indistinguishable on names alone, and picking either is how
    a wrong mapping reaches a citation list that looks right.
    """
    fits = [p for p in profiles if p.account_column in declared and p.amount_column in declared]
    if not fits:
        wanted = "; ".join(f"{p.account_column!r}+{p.amount_column!r}" for p in profiles)
        raise NoProfileForDossier(
            f"no ledger profile fits this dossier. It declares "
            f"{', '.join(sorted(c for c in declared if not c.startswith('__')))}; "
            f"known profiles want {wanted}"
        )
    if len(fits) > 1:
        raise AmbiguousProfile(
            "more than one ledger profile fits this dossier ("
            + "; ".join(f"{p.account_column}+{p.amount_column}" for p in fits)
            + "). Column names alone cannot say which exporter wrote it, and the wrong "
            "mapping returns a plausible total with citations."
        )
    return fits[0]


class ChartConflict(Exception):
    """The books name one chart of accounts and the host another."""


class AccountLengthUnsupported(Exception):
    """A Sachkonto longer than the 4-digit accounts the catalogue states."""


class GrossRevenueRefused(Exception):
    """A revenue leg whose booked amount includes VAT, with no declared net derivation."""


def _chart(host: str, table: pa.Table) -> dict[str, range]:
    """The host's chart, checked against whatever the books say about themselves.

    A DATEV header may state its SKR (field 27; often empty). Empty leaves the host's word
    as the only one. A stated chart that differs from the host's is refused, not resolved:
    under the wrong chart, 1200 is a bank balance counted as receivables.
    """
    if host not in CHARTS:
        raise ValueError(f"host chart {host!r} is not one of {', '.join(repr(c) for c in sorted(CHARTS))}")
    if "SKR" in table.column_names:
        stated = sorted({str(v) for v in table.column("SKR").to_pylist() if v} - {host})
        if stated:
            raise ChartConflict(
                f"the books state SKR{', SKR'.join(stated)} and this host reads SKR{host}. "
                "The same account number means different things in the two charts, so one of "
                "them is wrong and neither is picked."
            )
    return CHARTS[host]


# The catalogue states 4-digit SKR accounts. A DATEV header may set a Sachkontenlänge of 5 to 8,
# and how such an account maps onto the 4-digit chart is not modelled here: no batch we hold
# uses one. Read as it is, a long Sachkonto matches no range and the concept would report an
# unattested null; guessed, it could report a wrong total under correct-looking citations.
_CATALOGUE_DIGITS = 4


def _refuse_long_sachkonten(table: pa.Table, profile: LedgerProfile) -> None:
    """Refuse a DATEV journal whose Sachkonten are longer than the catalogue's accounts.

    Only a DATEV journal says which accounts are Sachkonten (`Kontoart`); Personenkonten are
    longer by design and belong to no SKR range. A GDPdU journal carries no such column.
    """
    if "Kontoart" not in table.column_names:
        return
    accounts = table.column(profile.account_column).to_pylist()
    kinds = table.column("Kontoart").to_pylist()
    origins = table.column(GdpduReader.ORIGIN_COLUMN).to_pylist()
    long = [
        f"{a} ({o})" for a, k, o in zip(accounts, kinds, origins) if k == "Sachkonto" and len(a) > _CATALOGUE_DIGITS
    ]
    if long:
        raise AccountLengthUnsupported(
            f"{len(long)} leg(s) book on a Sachkonto longer than {_CATALOGUE_DIGITS} digits "
            f"(e.g. {long[0]}). The catalogue states {_CATALOGUE_DIGITS}-digit SKR accounts, and "
            "how a longer Sachkontenlänge maps onto them is not modelled, so it is refused, not guessed."
        )


def _gross_reason(account: int, bu: str, automatic: frozenset[int]) -> str | None:
    """Why a revenue leg's booked amount includes VAT, or None when it stands as booked."""
    if bu == _LIFTS_AUTOMATIC:
        return None
    if bu:
        return f"BU key {bu!r} on {account}"
    if account in automatic:
        return f"automatic account {account}"
    return None


def _fingerprint(chart: str) -> str:
    """Identifies the catalogue version in force: every concept's ranges and the automatic accounts."""
    whole = {
        "concepts": {c: [r.start, r.stop - 1] for c, r in sorted(CHARTS[chart].items())},
        "automatic_revenue": sorted(AUTOMATIC_REVENUE[chart]),
    }
    return hashlib.sha256(json.dumps(whole, sort_keys=True).encode()).hexdigest()[:12]


def _basis(concept: str, chart: str, profile: LedgerProfile, signed: bool) -> str:
    """What a concept's values were computed on, as one JSON document riding with the rows."""
    catalogue = CHARTS[chart]
    accounts = catalogue[concept]
    return json.dumps(
        {
            "concept": concept,
            "profile": asdict(profile),
            "chart": f"SKR{chart}",
            "catalogue": {
                "name": CATALOGUE_NAMES[chart],
                "fingerprint": _fingerprint(chart),
                "accounts": [[accounts.start, accounts.stop - 1]],
            },
            "sign": SOLL_POSITIVE if signed else AS_DECLARED,
        },
        sort_keys=True,
    )


class SkrAccountFeatureGroup(FeatureGroup):
    """Maps SKR accounts onto concepts, over the ADMITTED journal only.

    Its one input is `gdpdu_journal__admitted`, so a concept cannot be computed from rows
    no policy has seen: with no policy configured the input does not resolve.

    Each concept returns two columns at row-preserving line grain -- ``name~value`` and
    ``name~origins`` -- so a value and its source are one logical feature that nothing
    downstream can separate into different FeatureSets.

    PROFILES lists the exporter shapes this host knows. The one that applies is chosen from
    the columns the DOSSIER declares, not by subclassing this group per exporter -- see
    select_profile. The catalogue is unchanged either way, because a chart of accounts is a
    property of the books, not of the file layout.
    """

    PROFILES: tuple[LedgerProfile, ...] = (DATEV_LIKE, SECOND_EXPORTER)

    # The chart of accounts these books are kept in: "03" or "04". Host configuration, set
    # once at startup like PROFILES, never an agent-settable Option. It is a property of the
    # books, and a DATEV header that states a different one is refused (see _chart).
    CHART: str = "04"

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(SKR04_2025) | set(SKR03_2025)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        """Gate on the semantic catalog BEFORE the inherited rules.

        This group used to be the root, and the gate kept its reader from claiming every name
        (proof_collision.py). The root is now JournalFeatureGroup, which carries that gate;
        this one stays so the catalogue, not the inherited rules, decides what it answers.
        """
        if cls.get_column_base_feature(str(feature_name)) not in cls.feature_names_supported():
            return False
        return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    def input_features(self, options: Options, feature_name: Any) -> set[Feature] | None:
        return {Feature(ADMITTED)}

    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PyArrowTable}

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str | int | float | bool]:
        """The chart and catalogue in force, for the extender hooks; the profile stays in `~basis`."""
        return {
            "chart": f"SKR{cls.CHART}",
            "catalogue": CATALOGUE_NAMES[cls.CHART],
            "catalogue.fingerprint": _fingerprint(cls.CHART),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return cls._map_accounts(unprefixed(data, ADMITTED), features)

    @classmethod
    def _map_accounts(cls, table: Any, features: FeatureSet) -> Any:
        """The catalogue mapping, split out so it can be exercised on a table directly.

        One class serves every known exporter, so testing a second profile no longer means
        registering a second subclass -- which is what used to shadow this group for every
        later resolution in the process (proof_profile_shadow.py).
        """
        # A dossier no profile fits is a configuration error, and it has to say so. Reading
        # straight through gave a bare KeyError('Field "Konto" does not exist in schema')
        # from inside Arrow, naming neither the profile nor the dossier.
        declared = set(table.column_names)
        profile = select_profile(cls.PROFILES, declared)

        catalogue = _chart(cls.CHART, table)
        _refuse_long_sachkonten(table, profile)

        konto = table.column(profile.account_column).to_pylist()
        betrag = table.column(profile.amount_column).to_pylist()
        # Carry the descriptor's declared scale through; do not re-scale money in transit.
        amount_type = table.schema.field(profile.amount_column).type
        origins = table.column(GdpduReader.ORIGIN_COLUMN).to_pylist()

        # Common ledger grain: one row per ledger line for every requested concept, with nulls
        # where a line does not contribute. Filtering per concept would give each concept a
        # different length, and two concepts in one request could not share a table.
        # Only a DATEV journal carries a BU key; its revenue legs may be booked gross. A
        # GDPdU line is booked as declared, so the check has nothing to read there. Rows no
        # policy admitted are left to the aggregation, which refuses them first.
        n = len(konto)
        stated = table.column(VORZEICHEN).to_pylist() if VORZEICHEN in declared else [None] * n
        unknown = sorted({str(c) for c in stated if c not in (None, SOLL_POSITIVE)})
        if unknown:
            raise ValueError(
                f"sign convention {', '.join(repr(u) for u in unknown)} is not known; only "
                f"{SOLL_POSITIVE!r} or none (amounts as declared)"
            )
        # The journal's own statement wins where it makes one; the profile speaks for the rest.
        # Both speaking and disagreeing is refused, as with the chart: one of them is wrong.
        if profile.sign is not None:
            clash = sorted({str(c) for c in stated if c is not None and c != profile.sign})
            if clash:
                raise ValueError(
                    f"the profile states sign {profile.sign!r}, which contradicts the journal's "
                    f"{', '.join(clash)}; one of them is wrong and neither is picked"
                )
        conventions = [c if c is not None else (profile.sign or AS_DECLARED) for c in stated]
        bu_keys = table.column("BU-Schlüssel").to_pylist() if "BU-Schlüssel" in declared else None
        stamps = table.column(ADMISSIBILITY_COLUMN).to_pylist() if ADMISSIBILITY_COLUMN in declared else None
        judged = [True] * n if stamps is None else [is_admitted(s) for s in stamps]

        out: dict[str, pa.Array] = {}
        for name in sorted(features.get_all_names()):
            accounts = catalogue[str(name)]
            values: list[Any] = []
            srcs: list[Any] = []
            gross: list[str] = []
            for i, (account, amount, origin) in enumerate(zip(konto, betrag, origins)):
                member = account.isdigit() and int(account) in accounts
                if member and name == "revenue" and bu_keys is not None and judged[i]:
                    reason = _gross_reason(int(account), (bu_keys[i] or "").strip(), AUTOMATIC_REVENUE[cls.CHART])
                    if reason:
                        gross.append(f"{origin} ({reason})")
                if member and amount is not None and name in _CREDIT_CONCEPTS and conventions[i] == SOLL_POSITIVE:
                    amount = -amount
                values.append(amount if member else None)
                srcs.append(origin if member else None)
            if gross:
                raise GrossRevenueRefused(
                    f"revenue: {len(gross)} leg(s) are a gross automatic-account booking, so the "
                    f"booked amount includes VAT (e.g. {gross[0]}). No net derivation is declared "
                    "(that needs a tax-key table per year), so they are refused, not totalled."
                )
            out[f"{name}~value"] = pa.array(values, type=amount_type)
            out[f"{name}~origins"] = pa.array(srcs, type=pa.string())
            # The basis rides at line grain like the stamp, so nothing between here and the
            # total can drop the definition while keeping the numbers it defined.
            signed = SOLL_POSITIVE in conventions
            out[f"{name}~basis"] = pa.array([_basis(str(name), cls.CHART, profile, signed)] * n, type=pa.string())

        # The evidence rides along at the same grain. Dropping it here would let a total be
        # computed from rows whose admissibility nobody established -- the transform must not
        # be the place the verdict quietly disappears.
        if ADMISSIBILITY_COLUMN in declared:
            out[ADMISSIBILITY_COLUMN] = table.column(ADMISSIBILITY_COLUMN)
        return pa.table(out)
