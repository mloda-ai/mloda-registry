"""LineageFacetsExtender: the community OpenLineage emitter, used instead of it for enterprise lineage facets."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

import attr
from mloda.provider import FeatureGroup
from mloda.steward import Extender, ExtenderHook, HookContext
from openlineage.client.event_v2 import InputDataset, Job
from openlineage.client.facet_v2 import RunFacet, column_lineage_dataset, data_quality_assertions_dataset

from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender
from mloda.community.extenders.shared.bound_method import bound_method, class_attribute
from mloda.community.extenders.shared.data_access_identity import resolve_data_access_identity
from mloda.community.extenders.shared.open_invocations import OpenInvocationStack

if TYPE_CHECKING:
    from mloda.user import Feature

logger = logging.getLogger(__name__)

_PRODUCER = "https://github.com/mloda-ai/mloda-registry/tree/main/mloda/enterprise/extenders/lineage"
_SCHEMA_URL = (
    "https://github.com/mloda-ai/mloda-registry/blob/main/mloda/enterprise/extenders/lineage/lineage_extender.py"
)
_STEP_LEVEL_DESCRIPTION = "step-level declared inputs"
_MASKING = "masking"
_SOURCE_COLUMN = "lineage_source_column"
_VALIDATOR_METHODS: dict[ExtenderHook, str] = {
    ExtenderHook.VALIDATE_INPUT_FEATURE: "validate_input_features",
    ExtenderHook.VALIDATE_OUTPUT_FEATURE: "validate_output_features",
}

# Per open calculate call: identity -> described columns (None = undescribable), or None if nothing to verify.
_open_described_columns: OpenInvocationStack[dict[str, frozenset[str] | None] | None] = OpenInvocationStack(
    "lineage_open_described_columns"
)


@attr.define
class MlodaRunFacet(RunFacet):
    featureGroupVersion: str = attr.field()
    pluginVersion: str | None = attr.field()
    computeFramework: str = attr.field()
    maskedFeatures: list[str] = attr.field()
    structureHash: str = attr.field()

    @staticmethod
    def _get_schema() -> str:
        # The module in this repo, not a hosted JSON schema.
        return _SCHEMA_URL


class LineageFacetsExtender(OpenLineageExtender):
    """Used instead of OpenLineageExtender; adds column lineage, masking, validator outcomes and a structure hash."""

    producer: str = _PRODUCER

    def wraps(self) -> set[ExtenderHook]:
        return super().wraps() | set(_VALIDATOR_METHODS)

    def _dispatch(self, context: HookContext, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if context.hook in _VALIDATOR_METHODS:
            return self._call_validator(context, func, args, kwargs)
        if context.hook == ExtenderHook.INPUT_DATA_LOAD:
            result = super()._dispatch(context, func, args, kwargs)
            self._record_described_columns(context, func, args)
            return result
        # Opened around the whole calculate call, so loads recorded into it are visible when facets are built.
        described: dict[str, frozenset[str] | None] | None = {} if _source_columns(context, func, args) else None
        with _open_described_columns.open(self, described):
            return super()._dispatch(context, func, args, kwargs)

    def _record_described_columns(self, context: HookContext, func: Any, args: tuple[Any, ...]) -> None:
        described = _open_described_columns.find(self)
        if described is None:
            return
        identity = resolve_data_access_identity(args, context.data_access_identity)
        if identity is None:
            return
        _merge_described_columns(described, identity, _describe_columns(func, args))

    def _calculate_run_facets(self, context: HookContext, func: Any, args: tuple[Any, ...]) -> dict[str, Any]:
        facets = super()._calculate_run_facets(context, func, args)
        masked = _masked_features(context, func, args)
        facets["mloda"] = MlodaRunFacet(
            featureGroupVersion=context.feature_group_version,
            pluginVersion=context.plugin_version,
            computeFramework=context.compute_framework_name,
            maskedFeatures=masked,
            structureHash=_structure_hash(context, masked, _source_columns(context, func, args)),
            producer=self.producer,
        )
        return facets

    def _calculate_output_facets(
        self, context: HookContext, func: Any, args: tuple[Any, ...], name: str, inputs: list[InputDataset]
    ) -> dict[str, Any]:
        facets = super()._calculate_output_facets(context, func, args, name, inputs)
        if context.input_features:
            edges = [(self.dataset_namespace, input_name, input_name) for input_name in sorted(context.input_features)]
            description = _STEP_LEVEL_DESCRIPTION if len(context.feature_names) > 1 else None
        else:
            column = _source_column(func, args, name)
            # With several loaded datasets it is unknown which one holds the column.
            if column is None or len(inputs) != 1:
                return facets
            described = _open_described_columns.find(self)
            identity_columns = described.get(inputs[0].name) if described is not None else None
            if identity_columns is not None and column not in identity_columns:
                logger.warning(
                    "%s: output %r declares lineage_source_column %r, not found among the columns described for "
                    "dataset %r; no columnLineage edge is emitted",
                    type(self).__name__,
                    name,
                    column,
                    inputs[0].name,
                )
                return facets
            edges = [(inputs[0].namespace, inputs[0].name, column)]
            description = None
        masking = True if name in _masked_features(context, func, args) else None
        facets["columnLineage"] = self._column_lineage_facet(name, edges, masking, description)
        return facets

    def _column_lineage_facet(
        self, name: str, edges: list[tuple[str, str, str]], masking: bool | None, description: str | None
    ) -> column_lineage_dataset.ColumnLineageDatasetFacet:
        """`edges` are (namespace, dataset name, field) triples."""
        return column_lineage_dataset.ColumnLineageDatasetFacet(
            fields={
                name: column_lineage_dataset.Fields(
                    inputFields=[
                        column_lineage_dataset.InputField(
                            namespace=namespace,
                            name=dataset,
                            field=field,
                            transformations=[
                                column_lineage_dataset.Transformation(
                                    type="DIRECT", masking=masking, description=description
                                )
                            ],
                        )
                        for namespace, dataset, field in edges
                    ]
                )
            },
            producer=self.producer,
        )

    def _call_validator(self, context: HookContext, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        method = _VALIDATOR_METHODS[context.hook]
        if not _overrides_validator(func, method):
            return func(*args, **kwargs)

        if context.hook == ExtenderHook.VALIDATE_INPUT_FEATURE:
            validated = context.input_features or context.feature_names
        else:
            validated = context.feature_names

        def build_inputs(_: list[InputDataset], exc: BaseException | None) -> list[InputDataset]:
            assertion = data_quality_assertions_dataset.Assertion(assertion=method, success=exc is None)
            return [
                InputDataset(
                    namespace=self.dataset_namespace,
                    name=name,
                    inputFacets={
                        "dataQualityAssertions": data_quality_assertions_dataset.DataQualityAssertionsDatasetFacet(
                            assertions=[assertion], producer=self.producer
                        )
                    },
                )
                for name in sorted(validated)
            ]

        return self._run_with_events(
            func,
            args,
            kwargs,
            job=Job(namespace=self.job_namespace, name=f"{context.feature_group_class}.{method}"),
            run_facets=super()._calculate_run_facets(context, func, args),
            declared_inputs=[],
            build_inputs=build_inputs,
        )


def _overrides_validator(func: Any, method: str) -> bool:
    function = getattr(bound_method(func), "__func__", None)
    return function is not None and function is not getattr(FeatureGroup, method).__func__


def _own_option(feature: Feature | None, key: str) -> Any:
    # Own context key only: not inherited, not held with an equal value by the consumer (which core cannot tell apart).
    # Equality is checked for bool and str only: only `True` or a non-empty str can count, and it keeps `==` safe for
    # array-like or exotic values.
    if feature is None or key in feature.options.inherited_context_keys:
        return None
    value = feature.options.context.get(key)
    held = feature.child_options.context.get(key) if feature.child_options is not None else None
    if isinstance(value, (bool, str)) and type(held) is type(value) and held == value:
        return None
    return value


def _declares_class_masking(func: Any) -> bool:
    return class_attribute(func, _MASKING) is True


def _declares_masking(feature: Feature) -> bool:
    return _own_option(feature, _MASKING) is True


def _feature(args: tuple[Any, ...], name: str) -> Feature | None:
    features = Extender.feature_set(args)
    if features is None:
        return None
    return next((feature for feature in features.features if str(feature.name) == name), None)


def _source_column(func: Any, args: tuple[Any, ...], name: str) -> str | None:
    declared = _own_option(_feature(args, name), _SOURCE_COLUMN)
    if declared is True:
        return name
    if isinstance(declared, str) and declared:
        return declared
    declared = class_attribute(func, _SOURCE_COLUMN)
    if declared is True:
        return name
    column = declared.get(name) if isinstance(declared, dict) else None
    return column if isinstance(column, str) and column else None


def _source_columns(context: HookContext, func: Any, args: tuple[Any, ...]) -> list[list[str]]:
    # The declaration, not whether an edge was emitted: run facets are built before any data load fires.
    if context.input_features:
        return []
    columns = ((name, _source_column(func, args, name)) for name in sorted(context.feature_names))
    return [[name, column] for name, column in columns if column is not None]


def _describe_columns(func: Any, args: tuple[Any, ...]) -> frozenset[str] | None:
    """None when the load has no reader owning it, no positional data_access, or the describer raises."""
    describe = class_attribute(func, "describe_columns")
    if describe is None or not args:
        return None
    try:
        return frozenset(describe(args[0]))
    except Exception:
        return None


def _merge_described_columns(
    described: dict[str, frozenset[str] | None], identity: str, columns: frozenset[str] | None
) -> None:
    """Same identity loaded again: the union, or None if either load could not describe."""
    if identity in described:
        previous = described[identity]
        columns = None if previous is None or columns is None else previous | columns
    described[identity] = columns


def _masked_features(context: HookContext, func: Any, args: tuple[Any, ...]) -> list[str]:
    if _declares_class_masking(func):
        return sorted(context.feature_names)
    features = Extender.feature_set(args)
    if features is None:
        return []
    return sorted({str(feature.name) for feature in features.features if _declares_masking(feature)})


def _structure_hash(context: HookContext, masked: list[str], source_columns: list[list[str]]) -> str:
    structure: list[Any] = [
        context.feature_group_class,
        context.feature_group_version,
        context.plugin_version,
        context.compute_framework_name,
        sorted(context.feature_names),
        sorted(context.input_features or ()),
        masked,
    ]
    # Appended only when non-empty, so a step declaring no source column hashes as before.
    if source_columns:
        structure.append(source_columns)
    return hashlib.sha256(json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
