"""OpenLineageExtender: emits OpenLineage RunEvents for mloda pipeline hooks."""

from __future__ import annotations

import atexit
import contextvars
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mloda.steward import Extender, ExtenderHook, HookContext

from mloda.community.extenders.openlineage import _process_local
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


_open_invocations: contextvars.ContextVar[tuple[tuple[int, "_OpenCalculateInvocation"], ...]] = contextvars.ContextVar(
    "openlineage_open_invocations", default=()
)


class OpenLineageExtender(Extender):
    """Emits one OpenLineage START/COMPLETE|FAIL|ABORT RunEvent per calculate invocation, correlating nested
    INPUT_DATA_LOAD calls as inputs. Sink resolution: injected client wins, else use_sdk_defaults, else inert.
    Emits happen synchronously on the calculation thread, so a blocking transport delays every wrapped calculation.
    close() flushes the client and is terminal; a self-built client also gets a bounded-timeout atexit flush
    (main process only, not MULTIPROCESSING workers). An injected client survives a pickle round trip
    in its own process; a copy unpickled elsewhere drops it and falls back to the resolution rule above."""

    _ATEXIT_CLOSE_TIMEOUT = 10.0

    def __init__(
        self,
        raise_on_error: bool = False,
        client: OpenLineageClient | None = None,
        job_namespace: str = "mloda",
        dataset_namespace: str = "mloda",
        root_job_name: str = "mloda.run_all",
        use_sdk_defaults: bool = False,
    ) -> None:
        self.raise_on_error = raise_on_error
        self._client = client
        self.job_namespace = job_namespace
        self.dataset_namespace = dataset_namespace
        self.root_job_name = root_job_name
        self.use_sdk_defaults = use_sdk_defaults
        self._client_lock = threading.Lock()
        self._closed = False
        self._logged_inert = False
        self._client_token = _process_local.register(self, client) if client is not None else None

    def _get_client(self) -> OpenLineageClient | None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} was closed; it can no longer be used to emit OpenLineage events")
        if self._client is not None:
            return self._client
        if not self.use_sdk_defaults:
            return None
        with self._client_lock:
            if self._client is None:
                self._client = OpenLineageClient()
                atexit.register(self.close, self._ATEXIT_CLOSE_TIMEOUT)
        return self._client

    def close(self, timeout: float = -1.0) -> bool:
        """Flush the underlying client. A no-op returning True if no client has been built yet, or if
        already closed. Idempotent: only the first call actually flushes; subsequent calls are no-ops."""
        with self._client_lock:
            if self._client is None or self._closed:
                return True
            self._closed = True
            atexit.unregister(self.close)
            client = self._client

        flushed = client.close(timeout)
        if not flushed:
            logger.warning("%s failed to flush all events within timeout", type(self).__name__)
        return flushed

    def _emit(self, event: RunEvent) -> None:
        client = self._get_client()
        if client is None:
            return
        client.emit(event)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_client"] = None
        state["_logged_inert"] = False
        del state["_client_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client_lock = threading.Lock()
        self._client = _process_local.resolve(self._client_token)

    def wraps(self) -> set[ExtenderHook]:
        return {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
        }

    def _log_inert_once(self) -> None:
        if self._logged_inert:
            return
        with self._client_lock:
            if not self._logged_inert:
                logger.warning(
                    "OpenLineageExtender is inert: no client injected and use_sdk_defaults is False; no "
                    "OpenLineage events will be emitted. Pass a client or use_sdk_defaults=True to enable emission."
                )
                self._logged_inert = True

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if self._client is None and not self.use_sdk_defaults:
            self._log_inert_once()
            return func(*args, **kwargs)

        context = HookContext.current()
        if context is None:
            return func(*args, **kwargs)

        if context.hook == ExtenderHook.INPUT_DATA_LOAD:
            return self._call_input_data_load(context, func, *args, **kwargs)
        return self._call_calculate_feature(context, func, *args, **kwargs)

    def _call_input_data_load(self, context: HookContext, func: Any, *args: Any, **kwargs: Any) -> Any:
        invocation = self._find_open_invocation()
        if invocation is not None and context.data_access_identity is not None:
            already_present = any(
                i.namespace == self.dataset_namespace and i.name == context.data_access_identity
                for i in invocation.inputs
            )
            if not already_present:
                invocation.inputs.append(
                    InputDataset(
                        namespace=self.dataset_namespace,
                        name=context.data_access_identity,
                        facets={
                            "dataSource": datasource_dataset.DatasourceDatasetFacet(
                                name=context.data_access_identity, producer=_PRODUCER
                            )
                        },
                    )
                )
        if invocation is None:
            logger.debug("OpenLineageExtender: INPUT_DATA_LOAD has no enclosing open calculate invocation to attach to")
        return func(*args, **kwargs)

    def _find_open_invocation(self) -> _OpenCalculateInvocation | None:
        """Return this instance's own last-opened invocation from the shared stack, else None."""
        my_id = id(self)
        for key, invocation in reversed(_open_invocations.get()):
            if key == my_id:
                return invocation
        return None

    def _call_calculate_feature(self, context: HookContext, func: Any, *args: Any, **kwargs: Any) -> Any:
        run_facets: dict[str, Any] = {}
        if context.run_id is not None:
            run_facets["parent"] = parent_run.ParentRunFacet(
                run=parent_run.Run(runId=context.run_id),
                job=parent_run.Job(namespace=self.job_namespace, name=self.root_job_name),
                producer=_PRODUCER,
            )
        job = Job(namespace=self.job_namespace, name=context.feature_group_class)
        run = Run(runId=str(uuid.uuid4()), facets=run_facets)
        invocation = _OpenCalculateInvocation(run_id=run.runId, job=job)

        # Unguarded on purpose: this call must propagate naturally so _CompositeExtender's
        # raise_on_error fallback machinery sees the real failure and never double-invokes func.
        self._emit(
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

        token = _open_invocations.set(_open_invocations.get() + ((id(self), invocation),))
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            event_state = RunState.FAIL if isinstance(exc, Exception) else RunState.ABORT
            # Guarded: a transport error on the FAIL/ABORT path must not mask the wrapped function's exception.
            try:
                self._emit(
                    RunEvent(
                        eventType=event_state,
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
                    "OpenLineageExtender failed to emit %s event: %s: %s",
                    event_state.name,
                    type(emit_exc).__name__,
                    emit_exc,
                )
            outcome = "failure" if event_state == RunState.FAIL else "abort"
            logger.warning("OpenLineageExtender observed %s %s: %s: %s", job.name, outcome, type(exc).__name__, exc)
            raise
        finally:
            _open_invocations.reset(token)

        # Guarded: a bug in this post-success block must never corrupt func's already-computed result.
        try:
            fields = _infer_schema_fields(result)
            outputs = [_build_output_dataset(self.dataset_namespace, name, fields) for name in context.feature_names]
            self._emit(
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
        first_match = next((f for f in fields if f.name == name), None)
        if first_match is not None:
            facets["schema"] = schema_dataset.SchemaDatasetFacet(fields=[first_match], producer=_PRODUCER)
    return OutputDataset(namespace=namespace, name=name, facets=facets)


def _infer_schema_fields(result: Any) -> list[schema_dataset.SchemaDatasetFacetFields] | None:
    try:
        return _infer_schema_fields_unsafe(result)
    except Exception:
        return None


def _infer_schema_fields_unsafe(result: Any) -> list[schema_dataset.SchemaDatasetFacetFields] | None:
    collect_schema = getattr(result, "collect_schema", None)
    if callable(collect_schema):
        schema = collect_schema()
    elif hasattr(type(result), "schema") or "schema" in getattr(result, "__dict__", {}):
        # `hasattr(type(result), ...)` alone catches a real class-level descriptor (pyarrow.Table,
        # pyspark.sql.DataFrame); the `__dict__` fallback catches a plain instance attribute. Neither
        # is true for pandas, whose unknown-attribute-as-column fallback lives only on the instance
        # via __getattr__, so a DataFrame with a "schema" column never enters this branch.
        schema = getattr(result, "schema", None)
    else:
        schema = None

    if schema is not None and hasattr(schema, "items"):
        return [schema_dataset.SchemaDatasetFacetFields(name=str(n), type=str(t)) for n, t in schema.items()]

    # Spark StructType: schema.fields carries StructField entries with name and dataType.
    if schema is not None and hasattr(schema, "fields"):
        return [schema_dataset.SchemaDatasetFacetFields(name=str(f.name), type=str(f.dataType)) for f in schema.fields]

    if schema is not None and hasattr(schema, "names") and hasattr(schema, "types"):
        return [
            schema_dataset.SchemaDatasetFacetFields(name=str(n), type=str(t))
            for n, t in zip(schema.names, schema.types)
        ]

    columns = getattr(result, "columns", None)
    dtypes = getattr(result, "dtypes", None)
    if columns is not None and dtypes is not None:
        return [schema_dataset.SchemaDatasetFacetFields(name=str(c), type=str(t)) for c, t in zip(columns, dtypes)]

    column_names = getattr(result, "column_names", None)
    if column_names is not None:
        return [schema_dataset.SchemaDatasetFacetFields(name=str(n)) for n in column_names]

    return None
