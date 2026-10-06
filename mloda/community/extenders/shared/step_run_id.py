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
    run_id: str | None,
    job_name: str,
    feature_names: Iterable[str],
    compute_framework_name: str | None,
    step_uuid: uuid.UUID | None = None,
) -> str | None:
    """Deterministic step id derived from the run id; None without a UUID run id. The step uuid keeps steps of
    one run apart; without it the legacy key is used."""
    if run_id is None:
        return None
    try:
        namespace = uuid.UUID(run_id)
    except ValueError:
        return None
    key: list[Any] = [job_name, sorted(feature_names), compute_framework_name]
    if step_uuid is not None:
        key.append(str(step_uuid))
    return str(uuid.uuid5(namespace, json.dumps(key)))
