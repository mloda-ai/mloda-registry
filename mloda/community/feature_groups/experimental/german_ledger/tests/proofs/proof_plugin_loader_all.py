"""Isolated: the same request after `PluginLoader.all()` (mloda#1745, open).

`PluginLoader.all()` imports the stock read feature groups. GdpduReader now declines every
name but `gdpdu_journal`, but the stock `ReadDocumentFeature` still claims `revenue`: its
TextFileReader accepts the dossier's GL.txt. On mloda 0.15 the refusal reads "Multiple feature
groups found" with ReadDocumentFeature (source: TextFileReader on GL.txt) beside
SkrAccountFeatureGroup. The journal's own name gate cannot help: the second claimant is a
different feature group.

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

from mloda.community.feature_groups.experimental.german_ledger.skr import SkrAccountFeatureGroup  # noqa: F401
from mloda.community.feature_groups.experimental.german_ledger.sources import SourcesFeatureGroup  # noqa: F401
from mloda.community.feature_groups.experimental.german_ledger.tests import _host  # noqa: F401  (the host policy)

PluginLoader.all()

try:
    results = mloda.run_all(
        features=["revenue__sources"],
        compute_frameworks=[PyArrowTable],
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
