"""Isolated: the design that was rejected, and the framework rule that rejected it.

Modelling one exporter as one `SkrAccountFeatureGroup` subclass is the obvious way to
support a second dossier shape. It is also unsafe, and this script is why.

mloda keeps a matching *subclass* over its parent, so a second profile class silently
shadows the first for every later resolution in the process. Three sibling classes stop
being silent and collide outright with `FeatureResolutionError`. Two is the dangerous
count precisely because three is the loud one.

Profiles are therefore NOT subclasses here. `SkrAccountFeatureGroup.PROFILES` lists the
shapes the host knows and `select_profile` picks the one the *dossier* fits, so one class
serves every exporter and the resolution graph never has to answer a question only the
data can answer. `proof_dossier_c.py` runs a second shape through that single class with
no subclass at all.

What this script pins is that the framework rule is real and that the design no longer
turns it into a wrong number: with profile identity out of the class graph, a shadowing
subclass now produces a refusal naming the dossier's own columns instead of a plausible
total under the wrong mapping.

Run as a subprocess for the same reason as proof_collision.py -- defining these classes
registers them globally via subclass discovery and would poison every later resolution in
the test process.
"""

import sys
from pathlib import Path
from typing import Any

from mloda.provider import FeatureResolutionError
from mloda.user import DataAccessCollection, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.experimental.german_ledger.skr import (
    SECOND_EXPORTER,
    LedgerProfile,
    NoProfileForDossier,
    SkrAccountFeatureGroup,
)
from mloda.community.feature_groups.experimental.german_ledger.sources import SourcesFeatureGroup  # noqa: F401
from mloda.community.feature_groups.experimental.german_ledger.tests import _host  # noqa: F401  (the host policy)

FIX = Path(__file__).parent.parent / "fixtures"


class ExporterB(SkrAccountFeatureGroup):
    """A second exporter modelled the rejected way: as a subclass carrying one profile."""

    PROFILES = (SECOND_EXPORTER,)


def ask(folder: str) -> Any:
    return mloda.run_all(
        features=["revenue__sources"],
        compute_frameworks=[PyArrowTable],
        data_access_collection=DataAccessCollection(folders={str(FIX / folder)}),
    )


# dossier_a is the DATEV-like shape, which the BASE class handles. ExporterB shadows it.
try:
    ask("dossier_a")
    print("UNEXPECTED: the shadowing subclass did not take over")
    sys.exit(1)
except FeatureResolutionError as exc:
    print(f"UNEXPECTED: resolution refused instead of shadowing: {str(exc).splitlines()[0]}")
    sys.exit(1)
except Exception as exc:
    root: BaseException = exc
    while (deeper := root.__cause__ or root.__context__) is not None:
        root = deeper
    # The subclass still wins resolution -- that rule is mloda's, and it has not changed.
    # What changed is the consequence: selection is now driven by the dossier, so the
    # shadowing class cannot map dossier_a under the wrong profile. It refuses, and the
    # refusal names the columns the dossier actually declares.
    if not isinstance(root, NoProfileForDossier):
        print(f"UNEXPECTED: {type(root).__name__}: {str(root)[:140]}")
        sys.exit(1)
    if "Konto" not in str(root):
        print(f"UNEXPECTED: refusal does not name the dossier's columns: {str(root)[:140]}")
        sys.exit(1)
    print(f"SHADOWING REFUSES INSTEAD OF GUESSING: {str(root)[:120]}")


# A THIRD sibling stops being silent and becomes an outright resolution failure. Asserting
# this is the point: it is the same modelling mistake, one class further along.
class ExporterD(SkrAccountFeatureGroup):
    PROFILES = (LedgerProfile(account_column="AcctNo", amount_column="Value"),)


try:
    ask("dossier_a")
    print("UNEXPECTED: three profile classes resolved without complaint")
    sys.exit(1)
except FeatureResolutionError as exc:
    first = str(exc).splitlines()[0]
    if "Multiple feature groups found" not in first:
        print(f"UNEXPECTED: {first}")
        sys.exit(1)
    print(f"THIRD PROFILE COLLIDES: {first}")
    sys.exit(0)
except Exception as exc:
    print(f"UNEXPECTED: {type(exc).__name__}: {str(exc)[:120]}")
    sys.exit(1)
