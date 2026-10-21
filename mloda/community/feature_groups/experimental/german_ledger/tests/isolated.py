"""Scenarios that register classes process-wide, so each runs in a fresh process and returns its facts."""

import json
import logging
import shutil
import subprocess  # nosec
import sys
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from mloda.provider import FeatureGroup, FeatureResolutionError, FeatureSet
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
    GrossRevenueRefused,
    LedgerProfile,
    NoProfileForDossier,
    SkrAccountFeatureGroup,
)
from mloda.community.feature_groups.experimental.german_ledger.sources import (  # noqa: F401
    InadmissibleTotal,
    SourcesFeatureGroup,
)

FIX = Path(__file__).parent / "fixtures"


def run_isolated(name: str) -> dict[str, Any]:
    """Run the `name` scenario in a fresh interpreter and return the JSON it prints."""
    code = f"import json; from {__name__} import {name}; print(json.dumps({name}(), default=str))"
    # Runs our own scenario functions, not external input.
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)  # nosec B603
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    facts: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return facts


def _root(exc: BaseException) -> BaseException:
    while (deeper := exc.__cause__ or exc.__context__) is not None:
        exc = deeper
    return exc


def _attempt(call: Callable[[], Any], wanted: type[BaseException] | None = None) -> dict[str, Any]:
    """Call it; report the exception (and its root) it raised, or the value it returned."""
    try:
        value = call()
    except Exception as exc:
        root = _root(exc)
        hit = next((e for e in (exc, root) if wanted is not None and isinstance(e, wanted)), None)
        return {
            "raised": True,
            "matched": wanted.__name__ if hit is not None and wanted is not None else None,
            "exc": type(exc).__name__,
            "root": type(root).__name__,
            "message": str(root),
            "first_line": str(exc).splitlines()[0] if str(exc) else "",
        }
    return {"raised": False, "matched": None, "value": value}


def _run(folder: Path, features: list[str], plugin_collector: Any = None) -> dict[str, Any]:
    results = mloda.run_all(
        features=features,
        compute_frameworks=[PyArrowTable],
        data_access_collection=DataAccessCollection(folders={str(folder)}),
        plugin_collector=plugin_collector,
    )
    return {k: v for t in results for k, v in t.to_pydict().items()}


def collision() -> dict[str, Any]:
    """What happens WITHOUT the match_feature_group_criteria gate."""

    class UngatedJournal(JournalFeatureGroup):
        """Identical, except it does not gate on its own name first."""

        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: Any, options: Any, data_access_collection: Any = None
        ) -> bool:
            unbound: Any = FeatureGroup.match_feature_group_criteria
            return bool(unbound.__func__(cls, feature_name, options, data_access_collection))

    return _attempt(lambda: _run(FIX / "dossier_a", ["revenue__sources"]), FeatureResolutionError)


def plugin_loader_all() -> dict[str, Any]:
    """The same request after `PluginLoader.all()` (mloda#1745): which groups claim the name."""
    from mloda.community.feature_groups.experimental.german_ledger.tests import _host  # noqa: F401

    PluginLoader.all()
    outcome = _attempt(lambda: _run(FIX / "dossier_a", ["revenue__sources"]), FeatureResolutionError)
    if outcome["raised"]:
        outcome["claimants"] = [
            line.strip() for line in outcome["message"].splitlines() if line.strip().startswith("- ")
        ]
        return outcome
    return {"raised": False, "matched": None, "value": [str(v) for v in outcome["value"]["revenue__sources~value"]]}


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

    class ExporterB(SkrAccountFeatureGroup):
        """A second exporter modelled the rejected way: as a subclass carrying one profile."""

        PROFILES = (SECOND_EXPORTER,)

    def ask() -> dict[str, Any]:
        return _run(FIX / "dossier_a", ["revenue__sources"])

    shadow = _attempt(ask, NoProfileForDossier)

    class ExporterD(SkrAccountFeatureGroup):
        PROFILES = (LedgerProfile(account_column="AcctNo", amount_column="Value"),)

    return {"shadow": shadow, "third": _attempt(ask, FeatureResolutionError)}


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

    def ask(plugin_collector: Any = None) -> dict[str, Any]:
        got = _run(FIX / "dossier_a", ["revenue__sources"], plugin_collector)
        return {k: v[0] for k, v in got.items()}

    def no_number(call: Callable[[], Any], wanted: type[BaseException]) -> dict[str, Any]:
        outcome = _attempt(call, wanted)
        if not outcome["raised"]:
            outcome["value"] = str(outcome["value"].get("revenue__sources~value"))
        return outcome

    without_policy = no_number(ask, FeatureResolutionError)

    from mloda.community.feature_groups.experimental.german_ledger.tests import _host  # noqa: F401

    with_policy = _attempt(ask)
    if not with_policy["raised"]:
        got = with_policy.pop("value")
        with_policy.update(value=str(got["revenue__sources~value"]), origins=got["revenue__sources~origins"])

    class Bypass(SkrAccountFeatureGroup):
        def input_features(self, options: Any, feature_name: Any) -> Any:
            return {Feature(JOURNAL)}

        @classmethod
        def calculate_feature(cls, data: Any, features: Any) -> Any:
            return cls._map_accounts(unprefixed(data, JOURNAL), features)

    bypass = no_number(ask, InadmissibleTotal)
    without_bypass = PluginCollector.disabled_feature_groups(Bypass)

    class Closing2025Again(AdmissibilityPolicyGroup):
        LOCK_DATE = date(2026, 3, 31)
        PERIOD_END = date(2025, 12, 31)

    two_policies = no_number(lambda: ask(without_bypass), FeatureResolutionError)
    return {
        "without_policy": without_policy,
        "with_policy": with_policy,
        "bypass": bypass,
        "two_policies": two_policies,
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

    def revenue(folder: Path) -> str:
        [value] = _run(folder, ["revenue__sources"])["revenue__sources~value"]
        return str(value)

    with tempfile.TemporaryDirectory() as tmp:
        locked = Path(tmp) / "datev"
        shutil.copytree(twin / "datev", locked)
        batch = locked / "EXTF_Buchungsstapel.csv"
        head, rest = batch.read_bytes().split(b"\r\n", 1)
        fields = head.split(b";")
        fields[20] = b"1"
        batch.write_bytes(b";".join(fields) + b"\r\n" + rest)

        return {
            "gdpdu": revenue(twin / "gdpdu"),
            "datev": revenue(locked),
            "open_datev": _attempt(lambda: revenue(twin / "datev"), InadmissibleTotal),
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

        open_batch = _attempt(lambda: _run(shipped, ["revenue__sources"]), InadmissibleTotal)
        locked_batch = _attempt(lambda: _run(locked, ["revenue__sources"]), GrossRevenueRefused)
        got = _run(net, ["revenue__sources"])
        origins = got["revenue__sources~origins"]
        stamps = DatevClosing2018.clear(DatevExtfReader.load_data(str(locked), FeatureSet())).column(
            ADMISSIBILITY_COLUMN
        )
        stamp_set = sorted(set(stamps.to_pylist()))
        return {
            "open_batch": open_batch,
            "locked_batch": locked_batch,
            "net_value": [str(v) for v in got["revenue__sources~value"]],
            "net_cited": [str(c) for c in origins[0]] if len(origins) == 1 and isinstance(origins[0], list) else [],
            "stamps": stamp_set,
            "stamps_admitted": [is_admitted(s) for s in stamp_set],
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
