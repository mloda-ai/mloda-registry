"""Step identity helpers shared by the OpenLineage, OTel and audit extenders."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from typing import Any

from mloda.steward import Extender, HookContext


def owner_name(context: HookContext, func: Any) -> str:
    return context.feature_group_class or Extender.feature_group_name(func)


def step_run_id(
    run_id: str | None, job_name: str, feature_names: Iterable[str], compute_framework_name: str | None
) -> str | None:
    """Deterministic step id derived from the run id; None without a UUID run id. Known limit: two steps of one
    run with the same job, feature names and framework share an id."""
    if run_id is None:
        return None
    try:
        namespace = uuid.UUID(run_id)
    except ValueError:
        return None
    return str(uuid.uuid5(namespace, json.dumps([job_name, sorted(feature_names), compute_framework_name])))
