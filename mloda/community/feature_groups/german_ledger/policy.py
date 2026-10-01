"""Admissibility as a chained feature group: journal -> journal__admitted -> concept.

mloda 0.14.0 (#1549) discards an extender's return value, so a policy can no longer stamp
rows from `INPUT_DATA_LOAD`: the old `LateEntryCutoffGuard` kept refusing but its verdict
never reached the aggregation, and every guarded request yielded no number. The verdict is
now a feature in its own right:

    gdpdu_journal            JournalFeatureGroup     the dossier's rows, every declared column
                             DatevJournalFeatureGroup  or a DATEV folder's legs
    gdpdu_journal__admitted  <host policy subclass>  the same rows, cleared and stamped
    revenue, receivables     SkrAccountFeatureGroup  concepts at line grain, stamp carried
    revenue__sources         SourcesFeatureGroup     totals only affirmatively stamped rows

The policy is a step in the resolution graph, so it is visible in `RunResult.plan`, and
removing it is a resolution failure rather than an unvetted total. Only one policy group can
be live per process, so a host that needs several checks lists them as rules in that one
group's RULES (LateEntryCutoff, Festschreibung, PeriodBound); they run in one step and write
one combined stamp.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, ClassVar, Optional

import pyarrow as pa
from mloda.provider import BaseInputData, FeatureGroup, FeatureSet
from mloda.user import DataAccessCollection, Feature, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from .datev import DatevExtfReader
from .reader import (
    ADMISSIBILITY_COLUMN,
    GdpduReader,
    admissibility_verdict,
    describe_verdicts,
    origin_format,
    outside_scope_verdict,
)

JOURNAL = "gdpdu_journal"
ADMITTED = f"{JOURNAL}__admitted"


class AdmissibilityRefused(Exception):
    """A ledger line an admissibility rule cannot clear, or a rule that cannot be evaluated.

    `verdicts` counts the journal's rows by verdict when the refusal comes from judging them
    (see apply_rules), and is None when it comes before any row was judged.
    """

    def __init__(self, message: str, verdicts: Optional[dict[str, int]] = None) -> None:
        super().__init__(message)
        self.verdicts = verdicts


class LateEntryRefused(AdmissibilityRefused):
    """A ledger line the late-entry cutoff cannot clear -- including one with no booking date."""


def _prefixed(table: pa.Table, feature: str) -> pa.Table:
    """Declared columns as subcolumns of `feature`; the stamp keeps its reserved name."""
    return pa.table({c if c == ADMISSIBILITY_COLUMN else f"{feature}~{c}": table.column(c) for c in table.column_names})


def unprefixed(table: pa.Table, feature: str) -> pa.Table:
    """The inverse of `_prefixed`: the dossier's own column names back, stamp included."""
    head = f"{feature}~"
    return pa.table(
        {
            c[len(head) :] if c.startswith(head) else c: table.column(c)
            for c in table.column_names
            if c.startswith(head) or c == ADMISSIBILITY_COLUMN
        }
    )


class _ReaderJournal(FeatureGroup):
    """Root: reads a dossier via READER and serves its rows as `gdpdu_journal~<column>`.

    Carries no meaning and no verdict. It exists so that the policy can sit between the
    reader and every concept, instead of inside the reader's load where 0.14.0 no longer
    lets it change anything.

    One sibling per reader, never a subclass of another: mloda keeps a matching subclass over
    its parent, so a DATEV journal subclassing the GDPdU one would silently win on a folder
    that holds both formats. As siblings, both match and resolution refuses. This base has
    no READER and matches nothing.
    """

    READER: Optional[type[BaseInputData]] = None

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        """Gate on the name BEFORE the inherited root/input-data rules.

        FeatureGroup.match_feature_group_criteria tries _is_root_and_matches_input_data first,
        so a reader that claims the dossier folder would make this group match *any* name --
        including revenue__sources -- and resolution would fail with "Multiple feature groups
        found" before anything loaded (proof_collision.py).
        """
        if cls.READER is None or str(feature_name) != JOURNAL:
            return False
        return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        # Returned directly: match_data_access takes the first matching subclass in the shared
        # pool, so a stock CsvReader could otherwise claim a dossier folder holding a .csv.
        return cls.READER() if cls.READER is not None else None

    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        reader = cls.input_data()
        assert reader is not None  # match_feature_group_criteria never admits a READER-less class
        return _prefixed(reader.load(features), JOURNAL)


class JournalFeatureGroup(_ReaderJournal):
    """The journal of a GDPdU dossier (index.xml + data file)."""

    READER = GdpduReader


class DatevJournalFeatureGroup(_ReaderJournal):
    """The journal of a folder of DATEV EXTF booking batches: one row per leg.

    DATEV carries no Erfassungsdatum, so the late-entry cutoff refuses every DATEV journal.
    That is the intended fail-closed outcome; a DATEV host runs the Festschreibung and
    PeriodBound rules instead.
    """

    READER = DatevExtfReader


# --- rules -----------------------------------------------------------------------------------
#
# A rule judges each row: ADMIT (examined and cleared), OUTSIDE (not this rule's to judge) or
# a refusal label naming why the row cannot be cleared. A column the rule needs but the journal
# lacks refuses the whole journal: an absent clock means the rule cannot be evaluated, not that
# it passes. Rules are frozen host configuration, never agent-settable Options.

ADMIT = "admit"
OUTSIDE = "outside"
# A rule scoped to one format (ForFormat) says this for the other format's rows. It neither
# admits nor excludes: the row is left to the rules that do apply, and a row no rule applies
# to is refused, because nothing vouched for it.
NOT_APPLICABLE = "not-applicable"
FORMATS = ("gdpdu", "datev")

# The combined verdict's policy name when a host runs more than one rule.
ALL_OF = "all-of"


class AdmissibilityRule(ABC):
    """One check a host's policy runs over the journal."""

    NAME: ClassVar[str]
    REFUSAL: ClassVar[type[AdmissibilityRefused]] = AdmissibilityRefused

    @abstractmethod
    def describe(self) -> str:
        """How a refusal names this rule, e.g. "the late-entry cutoff (lock date ...)"."""

    @abstractmethod
    def parameters(self) -> dict[str, object]:
        """What the stamp records about this rule's configuration."""

    @abstractmethod
    def judge(self, table: pa.Table) -> list[str]:
        """ADMIT, OUTSIDE or a refusal label per row; raise REFUSAL if it cannot be evaluated."""

    @property
    def name(self) -> str:
        """How the stamp and refusals name this rule; NAME unless a wrapper scopes it."""
        return self.NAME

    @property
    def refusal(self) -> type[AdmissibilityRefused]:
        """The exception this rule's per-row refusals raise."""
        return self.REFUSAL

    def _require(self, table: pa.Table, *names: str) -> None:
        missing = [n for n in names if n not in table.column_names]
        if missing:
            raise self.REFUSAL(
                f"journal carries no {', '.join(repr(m) for m in missing)}; {self.describe()} cannot be evaluated"
            )


@dataclass(frozen=True)
class LateEntryCutoff(AdmissibilityRule):
    """Lines booked into the closed period must have been keyed in by the lock date.

    This is a host policy, not a legal finding. It compares two configured dates against two
    declared columns and refuses the run when it cannot clear a line. It does not read a
    Festschreibekennzeichen, does not inspect change history or authorisation, and does not
    determine GoBD compliance: a legitimate year-end correction can carry a 31 December booking
    date and a January entry date, and this rule will refuse it. Refusing is the point -- the
    caller is told to look, not told a law was broken.

    What it does catch is the audit form of a training-set leak: the knowledge clock running
    ahead of the booking clock, where a number is answered from rows that were not knowable in
    the period the number claims to describe.
    """

    NAME: ClassVar[str] = "late-entry-cutoff"
    REFUSAL: ClassVar[type[AdmissibilityRefused]] = LateEntryRefused

    lock_date: date
    period_end: date
    booking_column: str = "Buchungsdatum"
    keying_column: str = "Erfassungsdatum"

    def describe(self) -> str:
        return f"the late-entry cutoff (lock date {self.lock_date}, period end {self.period_end})"

    def parameters(self) -> dict[str, object]:
        return {
            "lock": self.lock_date,
            "period_end": self.period_end,
            "booking": self.booking_column,
            "keying": self.keying_column,
        }

    def judge(self, table: pa.Table) -> list[str]:
        self._require(table, self.booking_column, self.keying_column)
        booked = table.column(self.booking_column).to_pylist()
        keyed = table.column(self.keying_column).to_pylist()

        # Both clocks fail closed per row, within the cutoff's reach. A null booking date
        # cannot be shown to fall outside the closed period, so it never passes. A null keying
        # date is refused on a line booked into that period, because it cannot be shown to
        # precede the cutoff -- but a line booked AFTER period_end is outside the rule's
        # reach entirely, and an empty Erfassungsdatum there is not this rule's business.
        # Two verdicts for the rows it does not refuse, because they were not examined alike:
        # a line booked into the closed period was tested and cleared (an admission); a line
        # booked after period_end was never held to the period, so it is OUTSIDE, and the
        # aggregation will not total it under a 2025 cutoff's authority.
        def one(book: Any, key: Any) -> str:
            if book is None:
                return "carrying no booking date"
            if book > self.period_end:
                return OUTSIDE
            if key is None:
                return "never keyed"
            if key > self.lock_date:
                return f"keyed in after {self.lock_date}"
            return ADMIT

        return [one(b, k) for b, k in zip(booked, keyed)]


@dataclass(frozen=True)
class Festschreibung(AdmissibilityRule):
    """A DATEV batch vouched for by its lock: Festschreibung 1 admits, 0 is outside the lock.

    An open batch is not wrong, so it is not refused -- but a lock rule cannot vouch for it,
    and the aggregation totals only what a rule vouched for. A missing flag fails closed.
    """

    NAME: ClassVar[str] = "festschreibung"

    column: str = "Festschreibung"

    def describe(self) -> str:
        return f"the Festschreibung rule (column {self.column!r})"

    def parameters(self) -> dict[str, object]:
        return {"column": self.column}

    def judge(self, table: pa.Table) -> list[str]:
        self._require(table, self.column)
        # `is`, not truthiness: a 0/1 string or an int is not the reader's boolean flag.
        return [
            ADMIT if flag is True else OUTSIDE if flag is False else "carrying no Festschreibung flag"
            for flag in table.column(self.column).to_pylist()
        ]


@dataclass(frozen=True)
class PeriodBound(AdmissibilityRule):
    """Only lines booked within [start, end] are vouched for; the rest are outside the period.

    The rule a DATEV host can run in place of the late-entry cutoff: DATEV carries no
    Erfassungsdatum, but every line carries its Belegdatum.
    """

    NAME: ClassVar[str] = "period-bound"

    start: date
    end: date
    booking_column: str = "Buchungsdatum"

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"period start {self.start} is after period end {self.end}")

    def describe(self) -> str:
        return f"the period bound ({self.start} to {self.end})"

    def parameters(self) -> dict[str, object]:
        return {"start": self.start, "end": self.end, "booking": self.booking_column}

    def judge(self, table: pa.Table) -> list[str]:
        self._require(table, self.booking_column)
        return [
            "carrying no booking date" if book is None else ADMIT if self.start <= book <= self.end else OUTSIDE
            for book in table.column(self.booking_column).to_pylist()
        ]


@dataclass(frozen=True)
class ForFormat(AdmissibilityRule):
    """`rule`, applied only to the rows of one format; NOT_APPLICABLE for the others.

    For a host that reads both GDPdU and DATEV: each carries what the other lacks (an
    Erfassungsdatum, a Festschreibung flag), so each is judged by the rules it can be judged
    by. A row's format is read from its citation -- a DATEV citation names a leg, a GDPdU one
    never does. A journal with none of this format's rows is not asked for the rule's columns.
    """

    NAME: ClassVar[str] = "for-format"

    format: str
    rule: AdmissibilityRule

    def __post_init__(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(f"format {self.format!r} is not one of {', '.join(repr(f) for f in FORMATS)}")

    @property
    def name(self) -> str:
        return f"{self.format}.{self.rule.name}"

    @property
    def refusal(self) -> type[AdmissibilityRefused]:
        return self.rule.refusal

    def describe(self) -> str:
        return f"{self.rule.describe()} for {self.format} rows"

    def parameters(self) -> dict[str, object]:
        return self.rule.parameters()

    def judge(self, table: pa.Table) -> list[str]:
        origins = (
            table.column(GdpduReader.ORIGIN_COLUMN).to_pylist()
            if GdpduReader.ORIGIN_COLUMN in table.column_names
            else [None] * table.num_rows
        )
        formats = [origin_format(o) for o in origins]
        out = [NOT_APPLICABLE if f is not None else "carrying no citation to tell its format" for f in formats]
        mine = [i for i, f in enumerate(formats) if f == self.format]
        if mine:
            for i, outcome in zip(mine, self.rule.judge(table.take(mine))):
                out[i] = outcome
        return out


def apply_rules(rules: Sequence[AdmissibilityRule], table: pa.Table) -> pa.Table:
    """Run every rule over the journal in one step, then stamp each row with one verdict.

    Refuses if any rule refuses, naming every refusing rule at once. Otherwise a row is
    admitted only if EVERY rule admitted it; one OUTSIDE makes the whole row outside-scope,
    and the stamp names which rule did not vouch for it. With a single rule the stamp is that
    rule's own verdict, so a one-rule host stamps exactly as before composites existed.
    """
    if not rules:
        raise AdmissibilityRefused("no admissibility rule is configured; nothing can be admitted")
    names = [r.name for r in rules]
    if len(set(names)) != len(names):
        # Their parameters would share keys in the stamp, and which configuration was meant
        # is the host's to say, not ours to guess.
        raise ValueError(f"each rule may appear once per policy; got {names}")
    # append_column permits duplicates, so a table stamped twice would carry two same-named
    # fields and a KeyError far downstream. Refuse here, where the cause is visible. (Two live
    # policy groups never get this far: resolution refuses them.)
    if ADMISSIBILITY_COLUMN in table.column_names:
        raise AdmissibilityRefused(
            f"{ADMISSIBILITY_COLUMN!r} is already present: another admissibility policy has "
            "stamped these rows. Several rules belong in one policy's RULES, not in two policies."
        )

    origin_col = GdpduReader.ORIGIN_COLUMN
    origins = table.column(origin_col).to_pylist() if origin_col in table.column_names else [None] * table.num_rows

    judged: list[list[str]] = []
    refusals: list[tuple[type[AdmissibilityRefused], str]] = []
    for rule in rules:
        try:
            outcomes = rule.judge(table)
        except AdmissibilityRefused as e:
            refusals.append((type(e), str(e)))
            continue
        judged.append(outcomes)
        # Grouped by label in first-seen order. Not "booked on/before {period_end}": an
        # unbooked line has no booking date at all, so a prefix naming one would be false.
        buckets: dict[str, list[Any]] = {}
        for outcome, origin in zip(outcomes, origins):
            if outcome not in (ADMIT, OUTSIDE, NOT_APPLICABLE):
                buckets.setdefault(outcome, []).append(origin)
        if buckets:
            parts = [f"{len(rows)} {label} ({', '.join(str(o) for o in rows)})" for label, rows in buckets.items()]
            refusals.append(
                (rule.refusal, f"{rule.describe()} found line(s) that cannot be cleared: " + "; ".join(parts))
            )
    # A row every judging rule passed over was vouched for by none of them.
    if len(judged) == len(rules):
        unjudged = [o for i, o in enumerate(origins) if all(outcomes[i] == NOT_APPLICABLE for outcomes in judged)]
        if unjudged:
            refusals.append(
                (
                    AdmissibilityRefused,
                    f"no rule applies to {len(unjudged)} line(s) ({', '.join(str(o) for o in unjudged)}); "
                    "nothing vouched for them",
                )
            )
    if refusals:
        # Count every row, so the refusal says how much of the journal it stopped: a row any
        # rule refused is refused; else a row a rule could not judge is unevaluated; else it
        # is outside-scope or admitted, as its stamp would have said.
        unevaluable = len(judged) < len(rules)
        counts: dict[str, int] = {}
        for i in range(table.num_rows):
            row = [outcomes[i] for outcomes in judged]
            if any(o not in (ADMIT, OUTSIDE, NOT_APPLICABLE) for o in row):
                kind = "refused"
            elif unevaluable or all(o == NOT_APPLICABLE for o in row):
                kind = "unevaluated"
            elif OUTSIDE in row:
                kind = "outside-scope"
            else:
                kind = "admitted"
            counts[kind] = counts.get(kind, 0) + 1
        kinds = {kind for kind, _ in refusals}
        message = " | ".join(m for _, m in refusals) + f" ({describe_verdicts(counts)})"
        raise (kinds.pop() if len(kinds) == 1 else AdmissibilityRefused)(message, verdicts=counts)

    # A total is worth what the admissibility of its rows is worth, so the verdict travels
    # WITH the data as evidence. SourcesFeatureGroup totals only affirmative verdicts, so a
    # row any rule did not vouch for yields no number rather than a total it silently joins.
    def stamp(outcomes: tuple[str, ...]) -> str:
        outside = [r.name for r, o in zip(rules, outcomes) if o == OUTSIDE]
        if len(rules) == 1:
            issue = outside_scope_verdict if outside else admissibility_verdict
            return issue(rules[0].name, **rules[0].parameters())
        params: dict[str, object] = {"rules": ",".join(names)}
        if outside:
            params["outside"] = ",".join(outside)
        params.update({f"{r.name}.{k}": v for r in rules for k, v in r.parameters().items()})
        return (outside_scope_verdict if outside else admissibility_verdict)(ALL_OF, **params)

    cache: dict[tuple[str, ...], str] = {}
    stamps = [cache[row] if row in cache else cache.setdefault(row, stamp(row)) for row in zip(*judged)]
    return table.append_column(ADMISSIBILITY_COLUMN, pa.array(stamps, type=pa.string()))


class AdmissibilityPolicyGroup(FeatureGroup):
    """`gdpdu_journal__admitted`: the journal's rows, cleared by the host's admissibility rules.

    Inert until a host subclass configures it -- the base matches nothing, so importing this
    module admits nothing. Exactly one configured subclass may be live in a process: none
    leaves `gdpdu_journal__admitted` unresolvable, and two are refused by resolution as
    "Multiple feature groups found". Either way there is no number. A host that needs several
    checks lists them in RULES; they run in this one step and write one combined stamp:

        class Closing2025(AdmissibilityPolicyGroup):           # one rule, the shorthand
            LOCK_DATE = date(2026, 1, 15)
            PERIOD_END = date(2025, 12, 31)

        class DatevClosing2025(AdmissibilityPolicyGroup):      # several rules
            RULES = (PeriodBound(date(2025, 1, 1), date(2025, 12, 31)), Festschreibung())

    The rules are class configuration written by the host -- never agent-settable Options,
    which a feature request could forge.
    """

    RULES: tuple[AdmissibilityRule, ...] = ()
    # Shorthand for RULES = (LateEntryCutoff(LOCK_DATE, PERIOD_END, BOOKING_COLUMN, KEYING_COLUMN),)
    LOCK_DATE: Optional[date] = None
    PERIOD_END: Optional[date] = None
    BOOKING_COLUMN = "Buchungsdatum"
    KEYING_COLUMN = "Erfassungsdatum"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.RULES and (cls.LOCK_DATE is not None or cls.PERIOD_END is not None):
            raise TypeError(
                f"{cls.__name__} sets both RULES and LOCK_DATE/PERIOD_END; put the "
                "LateEntryCutoff into RULES so the policy is written in one place"
            )

    @classmethod
    def rules(cls) -> tuple[AdmissibilityRule, ...]:
        if cls.RULES:
            return cls.RULES
        if cls.LOCK_DATE is not None and cls.PERIOD_END is not None:
            return (LateEntryCutoff(cls.LOCK_DATE, cls.PERIOD_END, cls.BOOKING_COLUMN, cls.KEYING_COLUMN),)
        return ()

    @classmethod
    def configured(cls) -> bool:
        return bool(cls.rules())

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return cls.configured() and str(feature_name) == ADMITTED

    def input_features(self, options: Options, feature_name: Any) -> Optional[set[Feature]]:
        return {Feature(JOURNAL)}

    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _prefixed(cls.clear(unprefixed(data, JOURNAL)), ADMITTED)

    @classmethod
    def clear(cls, table: pa.Table) -> pa.Table:
        """Refuse the lines the rules cannot clear; stamp every other line with its verdict."""
        if not cls.configured():
            raise AdmissibilityRefused(f"{cls.__name__} has no RULES (or LOCK_DATE/PERIOD_END) configured")
        return apply_rules(cls.rules(), table)
