"""LineageFacetsExtender: the community OpenLineage emitter, used instead of it for enterprise lineage facets."""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import TYPE_CHECKING, Any

import attr
from mloda.provider import FeatureGroup
from mloda.steward import Extender, ExtenderHook, HookContext
from openlineage.client.event_v2 import InputDataset, Job
from openlineage.client.facet_v2 import RunFacet, column_lineage_dataset, data_quality_assertions_dataset

from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender

if TYPE_CHECKING:
    from mloda.user import Options

_PRODUCER = "https://github.com/mloda-ai/mloda-registry/tree/main/mloda/enterprise/extenders/lineage"
_SCHEMA_URL = (
    "https://github.com/mloda-ai/mloda-registry/blob/main/mloda/enterprise/extenders/lineage/lineage_extender.py"
)
_STEP_LEVEL_DESCRIPTION = "step-level declared inputs"
_MASKING = "masking"
_VALIDATOR_METHODS: dict[ExtenderHook, str] = {
    ExtenderHook.VALIDATE_INPUT_FEATURE: "validate_input_features",
    ExtenderHook.VALIDATE_OUTPUT_FEATURE: "validate_output_features",
}


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
        return super()._dispatch(context, func, args, kwargs)

    def _calculate_run_facets(self, context: HookContext, func: Any, args: tuple[Any, ...]) -> dict[str, Any]:
        facets = super()._calculate_run_facets(context, func, args)
        masked = _masked_features(context, func, args)
        facets["mloda"] = MlodaRunFacet(
            featureGroupVersion=context.feature_group_version,
            pluginVersion=context.plugin_version,
            computeFramework=context.compute_framework_name,
            maskedFeatures=masked,
            structureHash=_structure_hash(context, masked),
            producer=self.producer,
        )
        return facets

    def _calculate_output_facets(
        self, context: HookContext, func: Any, args: tuple[Any, ...], name: str
    ) -> dict[str, Any]:
        facets = super()._calculate_output_facets(context, func, args, name)
        if not context.input_features:
            return facets
        masking = True if name in _masked_features(context, func, args) else None
        description = _STEP_LEVEL_DESCRIPTION if len(context.feature_names) > 1 else None
        facets["columnLineage"] = column_lineage_dataset.ColumnLineageDatasetFacet(
            fields={
                name: column_lineage_dataset.Fields(
                    inputFields=[
                        column_lineage_dataset.InputField(
                            namespace=self.dataset_namespace,
                            name=input_name,
                            field=input_name,
                            transformations=[
                                column_lineage_dataset.Transformation(
                                    type="DIRECT", masking=masking, description=description
                                )
                            ],
                        )
                        for input_name in sorted(context.input_features)
                    ]
                )
            },
            producer=self.producer,
        )
        return facets

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


def _bound_method(func: Any) -> Any:
    """The bound method behind func; core's wrappers copy __self__ but not __func__, so stop at the method type."""
    return inspect.unwrap(func, stop=inspect.ismethod)


def _overrides_validator(func: Any, method: str) -> bool:
    function = getattr(_bound_method(func), "__func__", None)
    return function is not None and function is not getattr(FeatureGroup, method).__func__


def _declares_class_masking(func: Any) -> bool:
    owner = getattr(_bound_method(func), "__self__", None)
    feature_group = owner if isinstance(owner, type) else type(owner)
    return getattr(feature_group, _MASKING, None) is True


def _declares_masking(options: Options | None) -> bool:
    # Only the feature's own context key counts: group keys and forwarded (inherited) keys are another step's.
    return (
        options is not None and options.context.get(_MASKING) is True and _MASKING not in options.inherited_context_keys
    )


def _masked_features(context: HookContext, func: Any, args: tuple[Any, ...]) -> list[str]:
    if _declares_class_masking(func):
        return sorted(context.feature_names)
    features = Extender.feature_set(args)
    if features is None:
        return []
    return sorted({str(feature.name) for feature in features.features if _declares_masking(feature.options)})


def _structure_hash(context: HookContext, masked: list[str]) -> str:
    structure = [
        context.feature_group_class,
        context.feature_group_version,
        context.plugin_version,
        context.compute_framework_name,
        sorted(context.feature_names),
        sorted(context.input_features or ()),
        masked,
    ]
    return hashlib.sha256(json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
