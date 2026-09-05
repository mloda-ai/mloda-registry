"""OpenLineageExtender: emits OpenLineage RunEvents for mloda pipeline hooks."""

from __future__ import annotations

import contextvars
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mloda.steward import Extender, ExtenderHook, HookContext

from openlineage.client.client import OpenLineageClient
from openlineage.client.event_v2 import InputDataset, Job, OutputDataset, Run, RunEvent, RunState
from openlineage.client.facet_v2 import datasource_dataset, parent_run, schema_dataset

logger = logging.getLogger(__name__)

_PRODUCER = "https://github.com/mloda-ai/mloda-registry/tree/main/mloda/community/extenders/openlineage"


@dataclass
class _OpenCalculateInvocation:
    """Mutable state for one open FEATURE_GROUP_CALCULATE_FEATURE invocation, shared with any
    INPUT_DATA_LOAD nested inside it via the run_id/job identity."""

    run_id: str
    job: Job
    inputs: list[InputDataset] = field(default_factory=list)


_current_calculate_invocation: contextvars.ContextVar[_OpenCalculateInvocation | None] = contextvars.ContextVar(
    "openlineage_current_calculate_invocation", default=None
)


class OpenLineageExtender(Extender):
    """Emits one OpenLineage START/COMPLETE|FAIL RunEvent per calculate invocation, correlating
    nested INPUT_DATA_LOAD calls as inputs of the enclosing run."""

    def __init__(
        self,
        raise_on_error: bool = False,
        client: OpenLineageClient | None = None,
        job_namespace: str = "mloda",
        dataset_namespace: str = "mloda",
        root_job_name: str = "mloda.run_all",
    ) -> None:
        self.raise_on_error = raise_on_error
        self._client = client if client is not None else OpenLineageClient()
        self.job_namespace = job_namespace
        self.dataset_namespace = dataset_namespace
        self.root_job_name = root_job_name

    def wraps(self) -> set[ExtenderHook]:
        return {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
        }

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is None:
            return func(*args, **kwargs)

        if context.hook == ExtenderHook.INPUT_DATA_LOAD:
            return self._call_input_data_load(context, func, *args, **kwargs)
        return self._call_calculate_feature(context, func, *args, **kwargs)

    def _call_input_data_load(self, context: HookContext, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        invocation = _current_calculate_invocation.get()
        if invocation is None:
            logger.debug("OpenLineageExtender: INPUT_DATA_LOAD has no enclosing open calculate invocation to attach to")
            return result
        if context.data_access_identity is not None:
            invocation.inputs.append(
                InputDataset(
                    namespace=self.dataset_namespace,
                    name=context.data_access_identity,
                    facets={"dataSource": datasource_dataset.DatasourceDatasetFacet(name=context.data_access_identity)},
                )
            )
        return result

    def _call_calculate_feature(self, context: HookContext, func: Any, *args: Any, **kwargs: Any) -> Any:
        run_facets: dict[str, Any] = {}
        if context.run_id is not None:
            run_facets["parent"] = parent_run.ParentRunFacet(
                run=parent_run.Run(runId=context.run_id),
                job=parent_run.Job(namespace=self.job_namespace, name=self.root_job_name),
            )
        job = Job(namespace=self.job_namespace, name=context.feature_group_class)
        run = Run(runId=str(uuid.uuid4()), facets=run_facets)
        invocation = _OpenCalculateInvocation(run_id=run.runId, job=job)

        # Unguarded on purpose: this call must propagate naturally so _CompositeExtender's
        # raise_on_error fallback machinery sees the real failure and never double-invokes func.
        self._client.emit(
            RunEvent(
                eventType=RunState.START,
                eventTime=_now_iso(),
                run=run,
                job=job,
                producer=_PRODUCER,
                inputs=[],
                outputs=[],
            )
        )

        token = _current_calculate_invocation.set(invocation)
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            # Guarded: a broken transport on the FAIL path must never mask func's real exception.
            try:
                self._client.emit(
                    RunEvent(
                        eventType=RunState.FAIL,
                        eventTime=_now_iso(),
                        run=run,
                        job=job,
                        producer=_PRODUCER,
                        inputs=list(invocation.inputs),
                        outputs=[],
                    )
                )
            except Exception as emit_exc:
                logger.warning(
                    "OpenLineageExtender failed to emit FAIL event: %s: %s", type(emit_exc).__name__, emit_exc
                )
            logger.warning("OpenLineageExtender observed %s failure: %s: %s", job.name, type(exc).__name__, exc)
            raise
        finally:
            _current_calculate_invocation.reset(token)

        # Guarded: a bug in this post-success block must never corrupt func's already-computed result.
        try:
            fields = _infer_schema_fields(result)
            outputs = [_build_output_dataset(self.dataset_namespace, name, fields) for name in context.feature_names]
            self._client.emit(
                RunEvent(
                    eventType=RunState.COMPLETE,
                    eventTime=_now_iso(),
                    run=run,
                    job=job,
                    producer=_PRODUCER,
                    inputs=list(invocation.inputs),
                    outputs=outputs,
                )
            )
        except Exception as exc:
            logger.warning("OpenLineageExtender post-call instrumentation failed: %s: %s", type(exc).__name__, exc)

        return result


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_output_dataset(
    namespace: str, name: str, fields: list[schema_dataset.SchemaDatasetFacetFields] | None
) -> OutputDataset:
    facets: dict[str, Any] = {}
    if fields is not None:
        matching_fields = [f for f in fields if f.name == name]
        if matching_fields:
            facets["schema"] = schema_dataset.SchemaDatasetFacet(fields=matching_fields)
    return OutputDataset(namespace=namespace, name=name, facets=facets)


def _infer_schema_fields(result: Any) -> list[schema_dataset.SchemaDatasetFacetFields] | None:
    try:
        return _infer_schema_fields_unsafe(result)
    except Exception:
        return None


def _infer_schema_fields_unsafe(result: Any) -> list[schema_dataset.SchemaDatasetFacetFields] | None:
    schema = getattr(result, "schema", None)
    if schema and hasattr(schema, "items"):
        return [schema_dataset.SchemaDatasetFacetFields(name=str(n), type=str(t)) for n, t in schema.items()]

    if schema is not None and hasattr(schema, "names") and hasattr(schema, "types"):
        return [
            schema_dataset.SchemaDatasetFacetFields(name=str(n), type=str(t))
            for n, t in zip(schema.names, schema.types)
        ]

    # Spark StructType: schema.fields carries StructField entries with name and dataType.
    if schema is not None and hasattr(schema, "fields"):
        return [schema_dataset.SchemaDatasetFacetFields(name=str(f.name), type=str(f.dataType)) for f in schema.fields]

    columns = getattr(result, "columns", None)
    dtypes = getattr(result, "dtypes", None)
    if columns is not None and dtypes is not None:
        return [schema_dataset.SchemaDatasetFacetFields(name=str(c), type=str(t)) for c, t in zip(columns, dtypes)]

    column_names = getattr(result, "column_names", None)
    if column_names is not None:
        return [schema_dataset.SchemaDatasetFacetFields(name=str(n)) for n in column_names]

    return None
