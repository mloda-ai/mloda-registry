"""OpenLineageExtender: emits OpenLineage RunEvents for mloda pipeline hooks."""

from __future__ import annotations

import atexit
import contextvars
import logging
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mloda.steward import Extender, ExtenderHook, HookContext, OutputSchema

from mloda.community.extenders.shared.pickle_safety import is_picklable
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


@dataclass
class _CloseState:
    """Close state for one client: shared by client identity for an injected client, private for a
    self-built one. `client` pins the id while a registry entry lives."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False
    result: bool | None = None
    client: OpenLineageClient | None = None


# Process-local, keyed by client identity; values are weak so a closed client is released with
# its last extender.
_shared_close_registry_lock = threading.Lock()
_shared_close_registry: weakref.WeakValueDictionary[int, _CloseState] = weakref.WeakValueDictionary()


def _get_or_create_close_state(client: OpenLineageClient) -> _CloseState:
    key = id(client)
    with _shared_close_registry_lock:
        state = _shared_close_registry.get(key)
        if state is None:
            state = _CloseState(client=client)
            _shared_close_registry[key] = state
        return state


class OpenLineageExtender(Extender):
    """Emits one OpenLineage START/COMPLETE|FAIL|ABORT RunEvent per calculate invocation, correlating nested
    INPUT_DATA_LOAD calls as inputs. Sink resolution: injected client wins, else use_sdk_defaults, else inert.
    Emits happen synchronously on the calculation thread, so a blocking transport delays every wrapped calculation.
    close() flushes the client and is terminal. A self-built client is always rebuilt per worker; an injected
    client that cannot survive pickling is dropped by a trial-pickle probe instead, falling back to the
    resolution rule above, while a genuinely picklable injected client is pickled as-is. Workers are
    terminated without a flush, so a synchronous transport is needed there."""

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
        self._logged_pickle_drop = False
        # Determined by whether a client was injected, not by when the lazy build happens to run.
        self._owns_client = client is None
        # Registry entry if injected, else a private state created upfront for the lazy build.
        self._close_state = _get_or_create_close_state(client) if client is not None else _CloseState()

    def _get_client(self) -> OpenLineageClient | None:
        if self._closed or self._close_state.closed:
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
        """Flush the underlying client; a no-op if none has been built yet, waiting out any build in
        flight. Otherwise every closer, including a sibling sharing an injected client, waits for one flush."""
        with self._client_lock:
            if self._client is None:
                return True
            client = self._client
            state = self._close_state
            if not self._closed:
                self._closed = True
                atexit.unregister(self.close)
            state.closed = True

        remaining = timeout
        if timeout < 0:
            acquired = state.lock.acquire(timeout=-1)
        else:
            # Probe first so an uncontended close passes the caller's timeout to the flush unchanged.
            acquired = state.lock.acquire(timeout=0)
            if not acquired:
                started = time.monotonic()
                acquired = state.lock.acquire(timeout=timeout)
                if acquired:
                    remaining = max(0.0, timeout - (time.monotonic() - started))
        if not acquired:
            return False
        try:
            result = state.result
            if result is None:
                result = client.close(remaining)
                if not result:
                    logger.warning("%s failed to flush all events within timeout", type(self).__name__)
                state.result = result
            return result
        finally:
            state.lock.release()

    def _emit(self, event: RunEvent) -> None:
        client = self._get_client()
        if client is None:
            return
        client.emit(event)

    def __getstate__(self) -> dict[str, Any]:
        client = self._client
        client_unpicklable = not self._owns_client and client is not None and not is_picklable(client)
        if client_unpicklable and not self._logged_pickle_drop:
            logger.warning(
                "OpenLineageExtender drops an injected client when pickled or copied because it could not be "
                "pickled; the copy is inert unless use_sdk_defaults=True, which lets it build its own client "
                "in its own process, e.g. under MULTIPROCESSING."
            )
            self._logged_pickle_drop = True
        state = dict(self.__dict__)
        if self._owns_client or client_unpicklable:
            # _owns_client keeps meaning "no client was injected" after unpickling too.
            state["_client"] = None
        state["_logged_inert"] = False
        state["_logged_pickle_drop"] = False
        del state["_client_lock"]
        del state["_close_state"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client_lock = threading.Lock()
        self._close_state = _get_or_create_close_state(self._client) if self._client is not None else _CloseState()

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

        # Unguarded on purpose: this call must propagate naturally so CompositeExtender's
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
            fields = _schema_dataset_fields(context.output_schema)
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
    namespace: str, name: str, fields: list[schema_dataset.SchemaDatasetFacetFields]
) -> OutputDataset:
    facets: dict[str, Any] = {}
    first_match = next((f for f in fields if f.name == name), None)
    if first_match is not None:
        facets["schema"] = schema_dataset.SchemaDatasetFacet(fields=[first_match], producer=_PRODUCER)
    return OutputDataset(namespace=namespace, name=name, facets=facets)


def _schema_dataset_fields(output_schema: OutputSchema | None) -> list[schema_dataset.SchemaDatasetFacetFields]:
    if output_schema is None:
        return []
    return [
        schema_dataset.SchemaDatasetFacetFields(name=name, type=str(type_) if type_ is not None else None)
        for name, type_ in output_schema
    ]
