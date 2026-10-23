"""Scenarios that register classes process-wide, so each runs in a fresh process and returns its facts."""

import json
import logging
import shutil
import subprocess  # nosec
import sys
import tempfile
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from mloda.provider import FeatureGroup, FeatureSet
from mloda.steward import verified_context
from mloda.user import DataAccessCollection, Feature, PluginCollector, PluginLoader, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.experimental.german_ledger.datev import DatevExtfReader
from mloda.community.feature_groups.experimental.german_ledger.policy import (
    AdmissibilityPolicyGroup,
    Festschreibung,
    ForFormat,
    JournalFeatureGroup,
    LateEntryCutoff,
    PeriodBound,
    unprefixed,
)
from mloda.community.feature_groups.experimental.german_ledger.reader import (
    ADMISSIBILITY_COLUMN,
    JOURNAL,
    GdpduReader,
    is_admitted,
)
from mloda.community.feature_groups.experimental.german_ledger.skr import (
    SECOND_EXPORTER,
    LedgerProfile,
    SkrAccountFeatureGroup,
)
from mloda.community.feature_groups.experimental.german_ledger.sources import (  # noqa: F401
    InadmissibleTotal,
    SourcesFeatureGroup,
)

FIX = Path(__file__).parent / "fixtures"


def run_isolated(name: str) -> dict[str, Any]:
    """Run the `name` scenario in a fresh interpreter and return the JSON it prints."""
    assert name.isidentifier(), name
    code = f"import json; from {__name__} import {name}; print(json.dumps({name}(), default=str))"
    # Runs our own scenario functions, not external input.
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)  # nosec B603
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    facts: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return facts


def _root(exc: BaseException) -> BaseException:
    while (deeper := exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)) is not None:
        exc = deeper
    return exc


def outcome(call: Callable[[], Any]) -> Any:
    """The call's value, or "<Exc>: <msg>" (or "<Exc> from <Root>: <msg>") of what it raised."""
    try:
        return call()
    except Exception as exc:
        root = _root(exc)
        named = type(root).__name__ if root is exc else f"{type(exc).__name__} from {type(root).__name__}"
        return f"{named}: {root}"


def _run(folder: Path, features: list[str], plugin_collector: Any = None) -> dict[str, Any]:
    results = mloda.run_all(
        features=features,
        compute_frameworks=[PyArrowTable],
        data_access_collection=DataAccessCollection(folders={str(folder)}),
        plugin_collector=plugin_collector,
    )
    return {k: v for t in results for k, v in t.to_pydict().items()}


def _revenue(folder: Path, plugin_collector: Any = None) -> str:
    """The one revenue total of the folder, as text (a refusing step that succeeds reads as this number)."""
    [value] = _run(folder, ["revenue__sources"], plugin_collector)["revenue__sources~value"]
    assert isinstance(value, Decimal), repr(value)
    return str(value)


def collision() -> dict[str, Any]:
    """What happens WITHOUT the match_feature_group_criteria gate."""

    # Registers by being defined; never referenced again.
    class UngatedJournal(JournalFeatureGroup):
        """Identical, except it does not gate on its own name first."""

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Any, data_access_collection: Any = None
        ) -> bool:
            unbound: Any = FeatureGroup.match_feature_group_criteria
            return bool(unbound.__func__(cls, feature_name, options, data_access_collection))

    return {"revenue": outcome(lambda: _revenue(FIX / "dossier_a"))}


def plugin_loader_all() -> dict[str, Any]:
    """The same request after `PluginLoader.all()` (mloda#1745): which groups claim the name."""
    from mloda.community.feature_groups.experimental.german_ledger.tests import _host  # noqa: F401

    PluginLoader.all()
    revenue = outcome(lambda: _revenue(FIX / "dossier_a"))
    # The refusal lists every claimant as a "- " line.
    return {"revenue": revenue, "claimants": [ln.strip() for ln in revenue.splitlines() if ln.strip().startswith("- ")]}


def dossier_c() -> dict[str, Any]:
    """dossier_c end to end, through its own profile, to a real total."""

    class Closing2025C(AdmissibilityPolicyGroup):
        """This host's policy: the clocks are declared under different names in dossier_c."""

        LOCK_DATE = date(2026, 1, 15)
        PERIOD_END = date(2025, 12, 31)
        BOOKING_COLUMN = "Buchung"
        KEYING_COLUMN = "Erfassung"

    got = {k: v[0] for k, v in _run(FIX / "dossier_c", ["revenue__sources", "receivables__sources"]).items()}
    return {
        "revenue": str(got["revenue__sources~value"]),
        "receivables": str(got["receivables__sources~value"]),
        "revenue_origins": got["revenue__sources~origins"],
        "receivables_origins": got["receivables__sources~origins"],
    }


def profile_shadow() -> dict[str, Any]:
    """A profile modelled as a SkrAccountFeatureGroup subclass: shadowing refuses, a third collides."""
    from mloda.community.feature_groups.experimental.german_ledger.tests import _host  # noqa: F401

    # ExporterB and ExporterD register by being defined; they are never referenced again.
    class ExporterB(SkrAccountFeatureGroup):
        """A second exporter modelled the rejected way: as a subclass carrying one profile."""

        PROFILES = (SECOND_EXPORTER,)

    def ask() -> str:
        return _revenue(FIX / "dossier_a")

    # The shadowing subclass still wins resolution (mloda's rule), so it must refuse, not map under the wrong profile.
    two_profile_classes = outcome(ask)

    class ExporterD(SkrAccountFeatureGroup):
        PROFILES = (LedgerProfile(account_column="AcctNo", amount_column="Value"),)

    return {"two_profile_classes": two_profile_classes, "three_profile_classes": outcome(ask)}


def reader_classification() -> dict[str, Any]:
    """A reader subclass with a lower CLASSIFICATION replaces the stock reader process-wide."""
    from mloda.community.feature_groups.experimental.german_ledger.tests._host import PLUGINS, ListSink, audit_gate

    class InternalGdpduReader(GdpduReader):
        CLASSIFICATION = "internal"

    sink = ListSink()
    gate = audit_gate(sink, "internal")
    with verified_context(tenant_id="t", principal="svc"):
        result = mloda.run_all(
            ["revenue__sources"],
            compute_frameworks=[PyArrowTable],
            data_access_collection=DataAccessCollection(folders={str(FIX / "twin_2025" / "gdpdu")}),
            plugin_collector=PLUGINS,
            function_extender={gate},
        )
    denies = [
        r
        for r in sink.records
        if r["decision"] == "deny" and str(r.get("reader_class", "")).rsplit(".", 1)[-1] == "InternalGdpduReader"
    ]
    levels = {r["classification"] for r in sink.records if r.get("hook") == "FEATURE_GROUP_CALCULATE_FEATURE"}
    return {"denies": [str(d) for d in denies], "has_result": bool(result), "calculate_levels": sorted(levels)}


def inseparable() -> dict[str, Any]:
    """Remove the admissibility policy, bypass it or double it, and the pipeline yields NO number."""
    logging.disable(logging.CRITICAL)  # mloda logs every failed step; the facts carry it

    def ask(plugin_collector: Any = None) -> str:
        return _revenue(FIX / "dossier_a", plugin_collector)

    # No policy class exists yet in this process, so the verdict feature does not resolve.
    without_policy = outcome(ask)

    from mloda.community.feature_groups.experimental.german_ledger.tests import _host  # noqa: F401

    got = {k: v[0] for k, v in _run(FIX / "dossier_a", ["revenue__sources"]).items()}

    # Reads the raw journal, wired past the policy. It shadows the concept group (mloda keeps a matching
    # subclass over its parent), which is why the later run disables it.
    class Bypass(SkrAccountFeatureGroup):
        def input_features(self, options: Any, feature_name: Any) -> Any:
            return {Feature(JOURNAL)}

        @classmethod
        def calculate_feature(cls, data: Any, features: Any) -> Any:
            return cls._map_accounts(unprefixed(data, JOURNAL), features)

    bypass = outcome(ask)
    without_bypass = PluginCollector.disabled_feature_groups(Bypass)

    # A second configured policy beside the host's: resolution refuses to pick one.
    class Closing2025Again(AdmissibilityPolicyGroup):
        LOCK_DATE = date(2026, 3, 31)
        PERIOD_END = date(2025, 12, 31)

    return {
        "without_policy": without_policy,
        "with_policy_revenue": str(got["revenue__sources~value"]),
        "with_policy_origins": got["revenue__sources~origins"],
        "bypass": bypass,
        "two_policies": outcome(lambda: ask(without_bypass)),
    }


def format_rules() -> dict[str, Any]:
    """One host reads GDPdU and DATEV, each judged by the rules its format can carry."""
    twin = FIX / "twin_2025"

    class MixedClosing2025(AdmissibilityPolicyGroup):
        """FY2025 for a host that receives both formats."""

        RULES = (
            ForFormat("gdpdu", LateEntryCutoff(lock_date=date(2026, 1, 15), period_end=date(2025, 12, 31))),
            ForFormat("datev", Festschreibung()),
            PeriodBound(date(2025, 1, 1), date(2025, 12, 31)),
        )

    SkrAccountFeatureGroup.CHART = "04"

    with tempfile.TemporaryDirectory() as tmp:
        locked = Path(tmp) / "datev"
        shutil.copytree(twin / "datev", locked)
        batch = locked / "EXTF_Buchungsstapel.csv"
        head, rest = batch.read_bytes().split(b"\r\n", 1)
        fields = head.split(b";")
        fields[20] = b"1"
        batch.write_bytes(b";".join(fields) + b"\r\n" + rest)

        return {
            "gdpdu_revenue": _revenue(twin / "gdpdu"),
            "locked_datev_revenue": _revenue(locked),
            # An open batch is outside the Festschreibung.
            "open_datev_revenue": outcome(lambda: _revenue(twin / "datev")),
        }


def composite_datev() -> dict[str, Any]:
    """A DATEV host's composite policy, end to end, through the one policy slot."""
    ledermann = FIX / "datev_ledermann"

    class DatevClosing2018(AdmissibilityPolicyGroup):
        """This host's policy: FY2018, and only what the Festschreibung vouches for."""

        RULES = (PeriodBound(date(2018, 1, 1), date(2018, 12, 31)), Festschreibung())

    SkrAccountFeatureGroup.CHART = "03"

    def copy(dst: Path, *, locked: bool, bu_8400: bytes | None = None) -> Path:
        shutil.copytree(ledermann, dst)
        batch = dst / "EXTF_Buchungsstapel.csv"
        lines = batch.read_bytes().split(b"\n")
        if locked:
            head = lines[0].split(b";")
            head[20] = b"1"
            lines[0] = b";".join(head)
        if bu_8400 is not None:
            row = lines[3].split(b";")
            row[8] = bu_8400
            lines[3] = b";".join(row)
        batch.write_bytes(b"\n".join(lines))
        return dst

    with tempfile.TemporaryDirectory() as tmp:
        shipped = copy(Path(tmp) / "shipped", locked=False)
        locked = copy(Path(tmp) / "locked", locked=True)
        net = copy(Path(tmp) / "net", locked=True, bu_8400=b'"40"')

        # Festschreibung 0: outside the lock.
        open_batch_revenue = outcome(lambda: _revenue(shipped))
        # Admitted by both rules, then the SKR03 concept refuses the gross automatic account 8400.
        locked_batch_revenue = outcome(lambda: _revenue(locked))
        # BU key 40 lifts the automatic: the booked amount is net, so a cited total.
        got = _run(net, ["revenue__sources"])
        origins = got["revenue__sources~origins"]
        # Core returns only the requested columns, so read the stamp from the policy's own step.
        stamps = DatevClosing2018.clear(DatevExtfReader.load_data(str(locked), FeatureSet())).column(
            ADMISSIBILITY_COLUMN
        )
        stamp_set = sorted(set(stamps.to_pylist()))
        return {
            "open_batch_revenue": open_batch_revenue,
            "locked_batch_revenue": locked_batch_revenue,
            "net_batch_revenue": [str(v) for v in got["revenue__sources~value"]],
            "net_batch_citations": (
                [str(c) for c in origins[0]] if len(origins) == 1 and isinstance(origins[0], list) else []
            ),
            "locked_batch_stamps": stamp_set,
            "locked_batch_stamps_admitted": [is_admitted(s) for s in stamp_set],
        }


def twin() -> dict[str, Any]:
    """The same books as a GDPdU dossier and a DATEV batch give the same totals."""
    twin_dir = FIX / "twin_2025"

    class TwinClosing2025(AdmissibilityPolicyGroup):
        """FY2025, judged by the one column both formats carry: the booking date."""

        RULES = (PeriodBound(date(2025, 1, 1), date(2025, 12, 31)),)

    SkrAccountFeatureGroup.CHART = "04"

    def total(folder: Path, concept: str) -> tuple[str, list[str]]:
        got = _run(folder, [f"{concept}__sources"])
        [value] = got[f"{concept}__sources~value"]
        assert isinstance(value, Decimal), repr(value)
        [origins] = got[f"{concept}__sources~origins"]
        return str(value), list(origins or [])

    facts: dict[str, Any] = {}
    for concept in ("revenue", "receivables"):
        gdpdu, gdpdu_cited = total(twin_dir / "gdpdu", concept)
        datev, datev_cited = total(twin_dir / "datev", concept)
        facts[concept] = {
            "gdpdu": gdpdu,
            "datev": datev,
            "gdpdu_cited": gdpdu_cited,
            "datev_cited": datev_cited,
        }
    return facts
