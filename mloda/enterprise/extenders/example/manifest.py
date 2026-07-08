"""Entry-point manifest for mloda-enterprise-extenders-example.

Lists the concrete Extender classes that mloda discovers via the
``mloda.extenders`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.steward import Extender

from .enterprise_example_extender import EnterpriseExampleExtender

EXTENDERS: list[type[Extender]] = [
    EnterpriseExampleExtender,
]
