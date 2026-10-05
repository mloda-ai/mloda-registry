"""Importing this module raises ImportError on a community OpenLineage major.minor mismatch."""

from __future__ import annotations

from .community_version import require_matching_community

require_matching_community()
