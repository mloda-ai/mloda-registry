"""Builds a HookContext test fixture with sane FEATURE_GROUP_CALCULATE_FEATURE defaults."""

from __future__ import annotations

from mloda.steward import ExtenderHook, HookContext


def make_hook_context(
    *,
    hook: ExtenderHook = ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
    feature_group_class: str = "mloda.testing.DummyFeatureGroup",
    feature_group_version: str = "1",
    plugin_version: str | None = None,
    feature_names: tuple[str, ...] = ("value_int",),
    input_features: frozenset[str] | None = None,
    compute_framework_name: str = "PyArrowTable",
    rows_in: int | None = None,
    rows_out: int | None = None,
    duration_seconds: float | None = None,
    status: str | None = None,
    run_id: str | None = None,
    data_access_identity: str | None = None,
    tenant_id: str | None = None,
    project_id: str | None = None,
    principal: str | None = None,
    carrier: dict[str, str] | None = None,
    worker_index: int | None = None,
    data_access_format: str | None = None,
    data_access_dataset_version: str | None = None,
    join_type: str | None = None,
    join_keys: tuple[str, ...] | None = None,
    plan_feature_count: int | None = None,
    plan_node_count: int | None = None,
    plan_depth: int | None = None,
) -> HookContext:
    """Build a HookContext test fixture; every keyword lands unchanged on the result."""
    return HookContext(
        hook=hook,
        feature_group_class=feature_group_class,
        feature_group_version=feature_group_version,
        plugin_version=plugin_version,
        feature_names=feature_names,
        input_features=input_features,
        compute_framework_name=compute_framework_name,
        rows_in=rows_in,
        rows_out=rows_out,
        duration_seconds=duration_seconds,
        status=status,
        run_id=run_id,
        data_access_identity=data_access_identity,
        tenant_id=tenant_id,
        project_id=project_id,
        principal=principal,
        carrier=carrier,
        worker_index=worker_index,
        data_access_format=data_access_format,
        data_access_dataset_version=data_access_dataset_version,
        join_type=join_type,
        join_keys=join_keys,
        plan_feature_count=plan_feature_count,
        plan_node_count=plan_node_count,
        plan_depth=plan_depth,
    )
