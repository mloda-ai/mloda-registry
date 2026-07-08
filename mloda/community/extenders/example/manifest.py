"""Entry-point manifest for mloda-community-extenders-example.

Lists the concrete Extender classes that mloda discovers via the
``mloda.extenders`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.steward import Extender

from .community_example_extender import CommunityExampleExtender

EXTENDERS: list[type[Extender]] = [
    CommunityExampleExtender,
]
