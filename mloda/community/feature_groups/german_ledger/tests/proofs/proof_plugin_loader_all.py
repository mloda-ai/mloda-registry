"""Isolated: the same request after `PluginLoader.all()` (mloda#1745, open).

GdpduReader subclasses ReadFile to reuse its code, which also enrols it in the stock
ReadFile family. `PluginLoader.all()` imports the stock `ReadFileFeature`, whose input data
is ReadFile() -- it asks every ReadFile subclass, GdpduReader included, and GdpduReader claims
any name once a dossier folder is present. The journal's own name gate cannot help: the
second claimant is a different feature group asking the same reader.

This matters for packaging, not just for tests: entry-point plugins are only discovered
through `PluginLoader.all()`, so a host that installs this from mloda-registry and loads it
the normal way gets no number. Kept as `ReadFile` subclass by decision (30 Sep); the
fix is upstream. When #1745 lands, this prints NO COLLISION and test_plugin_loader_all
fails on purpose -- flip it to assert the number.

Run as a subprocess -- `PluginLoader.all()` registers every stock plugin process-wide.
"""

import sys
from pathlib import Path

from mloda.provider import FeatureResolutionError
from mloda.user import DataAccessCollection, PluginLoader, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.german_ledger.skr import SkrAccountFeatureGroup  # noqa: F401
from mloda.community.feature_groups.german_ledger.sources import SourcesFeatureGroup  # noqa: F401
from mloda.community.feature_groups.german_ledger.tests import _host  # noqa: F401  (the host policy)

PluginLoader.all()

try:
    results = mloda.run_all(
        features=["revenue__sources"],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(
            folders={str(Path(__file__).parent.parent / "fixtures" / "dossier_a")}
        ),
    )
except FeatureResolutionError as exc:
    groups = [line.strip() for line in str(exc).splitlines() if line.strip().startswith("- ")]
    print(f"COLLISION REPRODUCED: {str(exc).splitlines()[0]}")
    for g in groups:
        print(f"  {g}")
    sys.exit(0)

value = next(
    t.column("revenue__sources~value").to_pylist()[0] for t in results if "revenue__sources~value" in t.column_names
)
print(f"NO COLLISION: revenue__sources~value {value}")
sys.exit(1)
