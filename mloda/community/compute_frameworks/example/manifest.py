"""Entry-point manifest for mloda-community-compute-frameworks-example.

Lists the concrete ComputeFramework classes that mloda discovers via the
``mloda.compute_frameworks`` entry point. See issue #271.
"""

from __future__ import annotations

from mloda.provider import ComputeFramework

from .community_example_compute_framework import CommunityExampleComputeFramework

COMPUTE_FRAMEWORKS: list[type[ComputeFramework]] = [
    CommunityExampleComputeFramework,
]
