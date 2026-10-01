"""Isolated: remove the admissibility policy and the pipeline yields NO number.

This is the claim that makes the four pieces one chain rather than four plugins shipped
together. Run as a subprocess: a policy is a FeatureGroup subclass, and defining one
registers it for the whole process, so "no policy" can only be shown before one exists.

The failure mode this forecloses: a host that simply omits the policy still gets a total,
cited and plausible, computed from rows whose admissibility nobody established. There are
three ways to leave the policy out, and each one must end in no number:

    WITHOUT POLICY  no policy class at all      -> gdpdu_journal__admitted does not resolve
    BYPASS          a concept wired past it     -> SourcesFeatureGroup: InadmissibleTotal
    TWO POLICIES    a second policy beside it   -> resolution refuses to pick one
"""

import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

from mloda.provider import FeatureResolutionError
from mloda.user import DataAccessCollection, Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.german_ledger.policy import JOURNAL, AdmissibilityPolicyGroup, unprefixed
from mloda.community.feature_groups.german_ledger.skr import SkrAccountFeatureGroup
from mloda.community.feature_groups.german_ledger.sources import InadmissibleTotal, SourcesFeatureGroup  # noqa: F401

logging.disable(logging.CRITICAL)  # mloda logs every failed step; this script reports it
FIX = Path(__file__).parent.parent / "fixtures"


def ask(plugin_collector: Any = None) -> dict[str, Any]:
    res = mloda.run_all(
        features=["revenue__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(folders={str(FIX / "dossier_a")}),
        plugin_collector=plugin_collector,
    )
    return {c: t.column(c).to_pylist()[0] for t in res for c in t.column_names}


def root_of(exc: BaseException) -> BaseException:
    while (deeper := exc.__cause__ or exc.__context__) is not None:
        exc = deeper
    return exc


def expect_no_number(label: str, plugin_collector: Any, wanted: type[BaseException]) -> None:
    """Not a different number -- no number."""
    try:
        got = ask(plugin_collector)
    except Exception as exc:
        hit = next((e for e in (exc, root_of(exc)) if isinstance(e, wanted)), None)
        if hit is None:
            print(f"UNEXPECTED: {label}: {type(root_of(exc)).__name__}: {str(root_of(exc))[:140]}")
            sys.exit(1)
        print(f"{label + ':':15} {type(hit).__name__} -- no number at all")
        return
    print(f"UNEXPECTED: {label} produced a total: {got.get('revenue__sources~value')}")
    sys.exit(1)


# 1. No policy class exists yet in this process.
expect_no_number("WITHOUT POLICY", None, FeatureResolutionError)

# 2. The host's policy: a cited total.
from mloda.community.feature_groups.german_ledger.tests import _host  # noqa: E402,F401  (the host policy)

got = ask()
if str(got["revenue__sources~value"]) != "45385.06" or len(got["revenue__sources~origins"]) != 5:
    print(f"UNEXPECTED: run under the policy gave {got['revenue__sources~value']}")
    sys.exit(1)
print(f"WITH POLICY:    {got['revenue__sources~value']} and {len(got['revenue__sources~origins'])} citations")


# 3. A concept that reads the raw journal instead of the admitted one. It shadows the real
#    concept group (mloda keeps a matching subclass over its parent), so later runs disable it.
class Bypass(SkrAccountFeatureGroup):
    def input_features(self, options: Any, feature_name: Any) -> Any:
        return {Feature(JOURNAL)}

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return cls._map_accounts(unprefixed(data, JOURNAL), features)


expect_no_number("BYPASS", None, InadmissibleTotal)
without_bypass = PluginCollector.disabled_feature_groups(Bypass)


# 4. A second configured policy beside the host's.
class Closing2025Again(AdmissibilityPolicyGroup):
    LOCK_DATE = date(2026, 3, 31)
    PERIOD_END = date(2025, 12, 31)


expect_no_number("TWO POLICIES", without_bypass, FeatureResolutionError)

print("CHAIN IS INSEPARABLE")
sys.exit(0)
