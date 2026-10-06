"""OtelExtender: emits OpenTelemetry spans for mloda pipeline hooks."""

from __future__ import annotations

import logging
import os
import re
import reprlib
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from mloda.steward import (
    Extender,
    ExtenderHook,
    HookContext,
    LifecycleOutcome,
    PlanContext,
    RunContext,
    WarnOncePerInstance,
    pickle_failure_reason,
    scrub_credentials,
)
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    Span,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
    TracerProvider,
    set_span_in_context,
)

from mloda.community.extenders.otel.otel_multiprocessing import extract_carrier, trace_id_from_run_id
from mloda.community.extenders.shared.step_run_id import owner_name, step_run_id
from mloda.community.extenders.shared.teardown import (
    CLOSE_TIMEOUT,
    capped_close_timeout,
    force_flush,
    to_timeout_millis,
)

logger = logging.getLogger(__name__)

_TRACER_NAME = "mloda_community_otel"
_CONTENT_PREVIEW_MAX_LEN = 200
_TRUTHY_ENV_VALUES = {"true", "1"}

# Plan span contexts kept for parenting run roots (LRU eviction); read at call time.
_MAX_PLAN_SPANS = 1024

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

_SpanInts = tuple[int, int, int]  # (trace_id, span_id, trace_flags)

# Fixed, nonzero placeholder span id used as the parent span id when synthesizing a NonRecordingSpan
# from a run_id (no real parent span was ever created; only the deterministic trace_id matters here).
_RUN_ID_PARENT_SPAN_ID = 0x0000000000000001


# reprlib keeps only a short head and tail, so only the ends of a long text need scrubbing; the window
# is large enough to hold a presigned URL with a session token.
_SCRUB_WINDOW = 8192
_SECRET_KEY_TAIL = 64
# Authorization keys core's anchored pattern misses (X-Authorization, HTTP_AUTHORIZATION, ...).
_AUTHORIZATION_KEY = re.compile(r"(?:^|[-_])authorization(?:[-_]header)?\Z", re.IGNORECASE)


def _scrub_ends(text: str) -> str:
    if len(text) <= 2 * _SCRUB_WINDOW:
        return scrub_credentials(text)
    return scrub_credentials(text[:_SCRUB_WINDOW]) + " " + scrub_credentials(text[-_SCRUB_WINDOW:])


def _is_secret_key(key: object) -> bool:
    if isinstance(key, bytes):
        key = key.decode("latin-1")
    if not isinstance(key, str):
        return False
    if _AUTHORIZATION_KEY.search(key):
        return True
    # Token-shaped value so the Authorization pattern (scheme word or token only) fires.
    probe = f"{key[-_SECRET_KEY_TAIL:]}=x1"
    return scrub_credentials(probe) != probe


def _sorted_if_possible(keys: list[Any]) -> list[Any]:
    try:
        return sorted(keys)
    except TypeError:
        return keys


def _repr_shows_secret_value(x: Mapping[Any, Any], s: str) -> bool:
    if type(x).__repr__ is object.__repr__:
        return False
    # Values are rendered unbounded, as is repr(x) on this path.
    for key in x:
        if _is_secret_key(key) and (repr(key) in s or str(key) in s):
            v = x[key]
            if any(form and form in s for form in (repr(v), str(v))):
                return True
    return False


_CONTAINER_BASES = (dict, tuple, list, set, frozenset, deque)


def _has_stdlib_repr(x: object) -> bool:
    r = type(x).__repr__
    module = getattr(r, "__module__", None) or getattr(getattr(r, "__objclass__", None), "__module__", None)
    return module in ("builtins", "collections")


# Scrubs before reprlib's cut and redacts values under secret-named dict keys and secret-keyed pairs.
class _ScrubbingRepr(reprlib.Repr):
    def repr_str(self, x: str, level: int) -> str:
        return super().repr_str(_scrub_ends(x), level)

    def repr_dict(self, x: Mapping[Any, Any], level: int) -> str:
        if not x:
            return "{}"
        if level <= 0:
            return "{...}"
        pieces = []
        for key in _sorted_if_possible(list(x))[: self.maxdict]:
            value_repr = "'***'" if _is_secret_key(key) else self.repr1(x[key], level - 1)
            pieces.append("%s: %s" % (self.repr1(key, level - 1), value_repr))
        if len(x) > self.maxdict:
            pieces.append("...")
        return "{" + ", ".join(pieces) + "}"

    def repr_tuple(self, x: tuple[Any, ...], level: int) -> str:
        if level > 0 and len(x) == 2 and _is_secret_key(x[0]):
            return "(%s, '***')" % self.repr1(x[0], level - 1)
        return super().repr_tuple(x, level)

    def _repr_stdlib_container(self, x: object, level: int) -> str | None:
        if not _has_stdlib_repr(x):
            return None
        fields = getattr(type(x), "_fields", None) if isinstance(x, tuple) else None
        # Namedtuples render by field name on purpose so secret-named fields stay masked.
        if fields and isinstance(x, tuple) and not (len(x) == 2 and _is_secret_key(x[0])):
            return self.repr_dict(dict(zip(fields, x)), level)
        for base in _CONTAINER_BASES:
            if isinstance(x, base):
                renderer: Callable[[Any, int], str] = getattr(self, "repr_" + base.__name__)
                return renderer(x, level)
        return None

    def repr_instance(self, x: object, level: int) -> str:
        try:
            # reprlib dispatches on the type name, so builtin container subclasses land here.
            routed = self._repr_stdlib_container(x, level)
            if routed is not None:
                return routed
            raw = repr(x)
            if isinstance(x, Mapping) and _repr_shows_secret_value(x, raw):
                return self.repr_dict(x, level)
            s = _scrub_ends(raw)
        except Exception:
            return "<%s instance at %#x>" % (x.__class__.__name__, id(x))
        if len(s) > self.maxother:
            i = max(0, (self.maxother - 3) // 2)
            j = max(0, self.maxother - 3 - i)
            s = s[:i] + "..." + s[len(s) - j :]
        return s


# Bounded repr for content previews: only recurses into the first N elements of a container,
# so it never materializes a full repr/str of a huge result before truncation (see _content_preview).
_BOUNDED_REPR = _ScrubbingRepr()
_BOUNDED_REPR.maxlevel = 3
_BOUNDED_REPR.maxlist = 10
_BOUNDED_REPR.maxdict = 10
_BOUNDED_REPR.maxset = 10
_BOUNDED_REPR.maxfrozenset = 10
_BOUNDED_REPR.maxtuple = 10
_BOUNDED_REPR.maxstring = 30
_BOUNDED_REPR.maxother = 30

_NOOP_TRACER_PROVIDER = trace.NoOpTracerProvider()

# ProxyTracerProvider is returned while no global provider is set; NoOpTracerProvider only if installed deliberately.
_API_DEFAULT_PROVIDER_TYPES = (trace.ProxyTracerProvider, trace.NoOpTracerProvider)

_INERT_MESSAGE = (
    "OtelExtender is inert: no injected tracer_provider and use_sdk_defaults is False; no spans will be "
    "emitted. Inject a tracer_provider, or pass use_sdk_defaults=True and configure an OpenTelemetry SDK "
    "tracer provider, to enable emission."
)

_NO_SDK_PROVIDER_MESSAGE = (
    "OtelExtender found no OpenTelemetry SDK tracer provider while use_sdk_defaults is True; no spans will "
    "be exported. Configure an SDK TracerProvider with an exporter via opentelemetry.trace.set_tracer_provider "
    "(install opentelemetry-sdk if missing; under MULTIPROCESSING, in each worker via child_bootstrap), "
    "or inject a tracer_provider."
)

_SPAN_NAMES: dict[ExtenderHook, str] = {
    ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "calculate",
    ExtenderHook.VALIDATE_INPUT_FEATURE: "mloda.validate.input",
    ExtenderHook.VALIDATE_OUTPUT_FEATURE: "mloda.validate.output",
    ExtenderHook.INPUT_DATA_LOAD: "mloda.load",
    ExtenderHook.JOIN: "join",
}

_OPERATION_NAMES: dict[ExtenderHook, str] = {
    ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "calculate",
    ExtenderHook.VALIDATE_INPUT_FEATURE: "validate",
    ExtenderHook.VALIDATE_OUTPUT_FEATURE: "validate",
    ExtenderHook.INPUT_DATA_LOAD: "load",
    ExtenderHook.JOIN: "join",
}

# Hooks that record the context's declared attributes and rows.out after the call: calculate and load only.
_DECLARABLE_HOOKS = {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}

# Declared attribute values kept; other types are dropped silently.
_SCALAR_TYPES = (str, bool, int, float)

# Caps declared keys so the SDK's attribute limit can't evict core attributes.
_MAX_DECLARED_KEYS = 32


class OtelExtender(Extender):
    """Emits one OpenTelemetry span per wrapped hook invocation, populated from the ambient HookContext.
    Sink resolution: injected tracer_provider wins, else use_sdk_defaults (warns once if no SDK provider is set),
    else inert (no-op span).
    An injected tracer_provider that can't survive pickling (e.g. the real SDK TracerProvider, which
    holds locks) is dropped by a trial-pickle probe when a copy is made (worker processes under
    ParallelizationMode.MULTIPROCESSING), falling back to the resolution rule above; a picklable
    custom provider is kept as-is. close() flushes the resolved provider, capped at close_timeout (default 1s)
    and the worker's remaining close budget, and never calls shutdown() (core, not the extender, owns provider
    lifetime).
    on_run_start opens a `mloda.run` root span that parents the step spans of that run (parent: run carrier, else
    the caller's active span). trace_scope="plan" also opens a `mloda.plan` span in on_plan_start and parents
    run roots under it, linking the caller or carrier span; "run" (default) emits a plan span only for a failed
    plan. Calculate spans are named `calculate <FeatureGroup>`, join spans `join <join_type>`. Without a known
    root, spans fall back to the carrier or run_id trace."""

    close_timeout: float = CLOSE_TIMEOUT

    def __init__(
        self,
        raise_on_error: bool = False,
        capture_content: bool | None = None,
        mask: Callable[[Any], Any] | None = None,
        tracer_provider: TracerProvider | None = None,
        use_sdk_defaults: bool = False,
        trace_scope: Literal["run", "plan"] = "run",
    ) -> None:
        if trace_scope not in ("run", "plan"):
            raise ValueError(f"OtelExtender trace_scope must be 'run' or 'plan', got {trace_scope!r}")
        self.trace_scope = trace_scope
        if capture_content is True and mask is None:
            raise ValueError("OtelExtender capture_content=True requires a mask")
        self.raise_on_error = raise_on_error
        self.capture_content = capture_content
        self.mask = mask
        self._tracer_provider = tracer_provider
        self.use_sdk_defaults = use_sdk_defaults
        self._inert_warning = WarnOncePerInstance()  # shared by the inert and no-SDK warnings
        self._pickle_drop_warning = WarnOncePerInstance()
        self._no_mask_warning = WarnOncePerInstance()
        self._init_span_state()

    def _init_span_state(self) -> None:
        self._lock = threading.Lock()
        self._run_roots: dict[str, _SpanInts] = {}
        self._root_spans: dict[str, Span] = {}
        self._plan_ints: OrderedDict[str, _SpanInts] = OrderedDict()
        self._plan_spans: dict[str, Span] = {}

    def _tracer(self) -> trace.Tracer:
        return trace.get_tracer(_TRACER_NAME, tracer_provider=self._resolve_tracer_provider())

    def on_plan_start(self, plan: PlanContext) -> None:
        if self.trace_scope != "plan":
            return
        span = self._tracer().start_span("mloda.plan", attributes={"mloda.plan.id": plan.plan_id})
        span_context = span.get_span_context()
        if not span_context.is_valid:
            return
        with self._lock:
            self._plan_spans[plan.plan_id] = span
            self._plan_ints[plan.plan_id] = _ints(span_context)
            self._plan_ints.move_to_end(plan.plan_id)
            while len(self._plan_ints) > _MAX_PLAN_SPANS:
                self._plan_ints.popitem(last=False)

    def on_plan_complete(self, plan: PlanContext, outcome: LifecycleOutcome) -> None:
        failed = outcome.status == "failed"
        with self._lock:
            span = self._plan_spans.pop(plan.plan_id, None)
            if failed:
                self._plan_ints.pop(plan.plan_id, None)
        if span is None and failed and self.trace_scope == "run":
            start_ns = (plan.created_at - _EPOCH) // timedelta(microseconds=1) * 1000
            span = self._tracer().start_span(
                "mloda.plan", attributes={"mloda.plan.id": plan.plan_id}, start_time=start_ns
            )
        if span is not None:
            if plan.structure_hash is not None:
                span.set_attribute("mloda.plan.structure_hash", plan.structure_hash)
            _end_with_outcome(span, outcome, status_attribute=None)

    def on_run_start(self, run: RunContext, plan: PlanContext, steps: Any) -> None:
        run_id = run.run_id
        if run_id is None:
            return
        attributes: dict[str, str] = {"mloda.run.id": run_id}
        if plan.plan_id is not None:
            attributes["mloda.plan.id"] = plan.plan_id
        if plan.structure_hash is not None:
            attributes["mloda.plan.structure_hash"] = plan.structure_hash
        caller = _carrier_or_active_span_context(run.carrier)
        with self._lock:
            plan_ints = self._plan_ints.get(plan.plan_id) if self.trace_scope == "plan" else None
            if plan_ints is not None:
                self._plan_ints.move_to_end(plan.plan_id)
        if plan_ints is not None:
            parent: Context | None = _context_from_ints(plan_ints)
            links = [Link(caller)] if caller is not None else None
        else:
            parent = extract_carrier(run.carrier) if run.carrier else None
            links = None
        span = self._tracer().start_span("mloda.run", context=parent, links=links, attributes=attributes)
        span_context = span.get_span_context()
        if not span_context.is_valid:
            return
        with self._lock:
            self._run_roots[run_id] = _ints(span_context)
            self._root_spans[run_id] = span

    def on_run_complete(self, run: RunContext, outcome: LifecycleOutcome) -> None:
        if run.run_id is None:
            return
        with self._lock:
            self._run_roots.pop(run.run_id, None)
            span = self._root_spans.pop(run.run_id, None)
        if span is not None:
            _end_with_outcome(span, outcome, status_attribute="mloda.run.status")

    def _configured_tracer_provider(self) -> TracerProvider | None:
        """Injected provider wins, else the global SDK provider when use_sdk_defaults, else None."""
        if self._tracer_provider is not None:
            return self._tracer_provider
        if self.use_sdk_defaults:
            return trace.get_tracer_provider()
        return None

    def _resolve_tracer_provider(self) -> TracerProvider:
        provider = self._configured_tracer_provider()
        if provider is None:
            self._warn_once(_INERT_MESSAGE)
            return _NOOP_TRACER_PROVIDER
        if self.use_sdk_defaults and isinstance(provider, _API_DEFAULT_PROVIDER_TYPES):
            self._warn_once(_NO_SDK_PROVIDER_MESSAGE)
        return provider

    def _warn_once(self, message: str) -> None:
        self._inert_warning.warn_once(lambda: logger.warning(message))

    # Core calls close() with no args on graceful MULTIPROCESSING worker exit and ignores the result.
    def close(self) -> None:
        """Flush the resolved tracer_provider within close_timeout and the remaining close budget, best effort;
        never raises and never calls shutdown() (core, not the extender, owns provider lifetime). Inert (no
        injected provider, use_sdk_defaults False) touches no provider."""
        provider = self._configured_tracer_provider()
        if provider is None:
            return
        try:
            result = force_flush(provider, timeout_millis=to_timeout_millis(capped_close_timeout(self.close_timeout)))
        except Exception as exc:
            logger.warning("%s failed to flush tracer_provider: %s", type(self).__name__, type(exc).__name__)
            return
        if result is False:
            logger.warning("%s did not flush all spans within its close budget", type(self).__name__)

    def __getstate__(self) -> dict[str, Any]:
        provider = self._tracer_provider
        failure_reason = pickle_failure_reason(provider) if provider is not None else None
        if failure_reason is not None:
            self._pickle_drop_warning.warn_once(
                lambda: logger.warning(
                    "OtelExtender drops an injected tracer_provider when pickled or copied because it "
                    f"isn't picklable ({failure_reason}); the copy is inert unless use_sdk_defaults=True, "
                    "which lets it resolve a provider installed in its own process, e.g. via "
                    "child_bootstrap under MULTIPROCESSING."
                )
            )
        state = dict(self.__dict__)
        with self._lock:
            state["_run_roots"] = dict(self._run_roots)
        for key in ("_lock", "_root_spans", "_plan_ints", "_plan_spans"):
            state.pop(key, None)
        if failure_reason is not None:
            state["_tracer_provider"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        run_roots = self.__dict__.get("_run_roots", {})
        self._init_span_state()
        self._run_roots = run_roots

    def wraps(self) -> set[ExtenderHook]:
        return {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
            ExtenderHook.JOIN,
        }

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        span_name = _span_name(context)

        tracer_provider = self._resolve_tracer_provider()
        tracer = trace.get_tracer(_TRACER_NAME, tracer_provider=tracer_provider)
        with self._lock:
            root = self._run_roots.get(context.run_id) if context is not None and context.run_id else None
        parent_context = _parent_context(context, root)
        with tracer.start_as_current_span(
            span_name, record_exception=False, context=parent_context, set_status_on_exception=False
        ) as span:
            if context is not None:
                try:
                    _set_context_attributes(span, context)
                    _set_step_attributes(span, context, func)
                    if context.hook == ExtenderHook.INPUT_DATA_LOAD:
                        _set_load_attributes(span, context)
                    if context.hook == ExtenderHook.JOIN:
                        _set_join_attributes(span, context)
                except BaseException as exc:
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("error.type", f"{type(exc).__module__}.{type(exc).__qualname__}")
                    raise
                if context.hook in _DECLARABLE_HOOKS:
                    _set_declared_attributes(span, context)

            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", f"{type(exc).__module__}.{type(exc).__qualname__}")
                logger.warning("%s %s failed: %s", type(self).__name__, span_name, type(exc).__name__)
                raise

            try:
                if context is not None and context.hook in _DECLARABLE_HOOKS:
                    if context.rows_out is not None:
                        span.set_attribute("mloda.rows.out", context.rows_out)
                    if (
                        context.hook == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
                        and span.is_recording()
                        and self._content_capture_enabled()
                    ):
                        span.set_attribute("mloda.content.preview", self._content_preview(result))
            except Exception as exc:
                logger.warning("%s post-call instrumentation failed: %s", type(self).__name__, type(exc).__name__)

            return result

    def _content_capture_enabled(self) -> bool:
        if self.capture_content is not None:
            enabled = self.capture_content
        else:
            enabled = os.environ.get("MLODA_OTEL_TRACE_CONTENT", "").strip().lower() in _TRUTHY_ENV_VALUES
        if enabled and self.mask is None:
            self._no_mask_warning.warn_once(
                lambda: logger.warning(
                    "%s content capture needs a mask; no content preview is recorded", type(self).__name__
                )
            )
            return False
        return enabled

    def _content_preview(self, result: Any) -> str:
        assert self.mask is not None
        value = self.mask(result)
        return scrub_credentials(_BOUNDED_REPR.repr(value))[:_CONTENT_PREVIEW_MAX_LEN]


def _ints(span_context: SpanContext) -> _SpanInts:
    return (span_context.trace_id, span_context.span_id, int(span_context.trace_flags))


def _context_from_ints(ints: _SpanInts) -> Context:
    trace_id, span_id, flags = ints
    span_context = SpanContext(trace_id=trace_id, span_id=span_id, is_remote=True, trace_flags=TraceFlags(flags))
    return set_span_in_context(NonRecordingSpan(span_context))


def _carrier_or_active_span_context(carrier: Any) -> SpanContext | None:
    if carrier:
        span_context = trace.get_current_span(extract_carrier(carrier)).get_span_context()
    else:
        span_context = trace.get_current_span().get_span_context()
    return span_context if span_context.is_valid else None


def _end_with_outcome(span: Span, outcome: LifecycleOutcome, status_attribute: str | None) -> None:
    if status_attribute is not None:
        span.set_attribute(status_attribute, outcome.status)
    if outcome.status == "failed":
        span.set_status(Status(StatusCode.ERROR))
        if outcome.error_type is not None:
            span.set_attribute("error.type", outcome.error_type)
    span.end()


def _span_name(context: HookContext | None) -> str:
    if context is None:
        return "mloda.unknown"
    name = _SPAN_NAMES.get(context.hook, "mloda.unknown")
    if context.hook == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE and context.feature_group_class is not None:
        return f"{name} {context.feature_group_class.rsplit('.', 1)[-1]}"
    if context.hook == ExtenderHook.JOIN and context.join_type is not None:
        return f"{name} {context.join_type}"
    return name


def _parent_context(context: HookContext | None, root: _SpanInts | None = None) -> Context | None:
    """Pick the parent context for the span to be started, by priority (highest first):

    0. A run root known for context.run_id (see on_run_start): its context is the parent, winning over the
       carrier. For INPUT_DATA_LOAD an ambient span of the root's trace wins (None returned).
    1. INPUT_DATA_LOAD only: a valid ambient active span (e.g. the enclosing mloda.calculate span)
       wins and None is returned, making the load span its child. If a carrier or run_id is also set,
       the ambient span wins only when its trace id matches theirs; otherwise falls through to 2/3.
    2. context.carrier, if truthy: extracted into a real parent Context (propagated from another
       process via a W3C traceparent carrier).
    3. else context.run_id, if not None: a synthetic, non-recording parent Context whose trace_id is
       deterministically derived from run_id, so spans sharing a run_id correlate even when no carrier
       was ever exchanged.
    4. else None: today's existing default behavior (the ambient current context is used).
    """
    if context is None:
        return None

    if root is not None:
        ambient = trace.get_current_span().get_span_context()
        if context.hook == ExtenderHook.INPUT_DATA_LOAD and ambient.is_valid and ambient.trace_id == root[0]:
            return None
        return _context_from_ints(root)

    if context.hook == ExtenderHook.INPUT_DATA_LOAD:
        ambient_span_context = trace.get_current_span().get_span_context()
        if ambient_span_context.is_valid:
            if not context.carrier and context.run_id is None:
                return None
            expected_trace_id = _expected_trace_id(context)
            if expected_trace_id is None or expected_trace_id == ambient_span_context.trace_id:
                return None

    if context.carrier:
        return extract_carrier(context.carrier)

    run_trace_id = _run_trace_id(context)
    if run_trace_id is not None:
        span_context = SpanContext(
            trace_id=run_trace_id,
            span_id=_RUN_ID_PARENT_SPAN_ID,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        return set_span_in_context(NonRecordingSpan(span_context))

    return None


def _expected_trace_id(context: HookContext) -> int | None:
    """The trace id the carrier/run_id fallback rule would give (carrier wins), per _parent_context's priority."""
    if context.carrier:
        return trace.get_current_span(extract_carrier(context.carrier)).get_span_context().trace_id
    return _run_trace_id(context)


def _run_trace_id(context: HookContext) -> int | None:
    if context.run_id is None:
        return None
    try:
        return trace_id_from_run_id(context.run_id)
    except ValueError:
        return None


def _set_load_attributes(span: Span, context: HookContext) -> None:
    identity = context.data_access_identity
    if identity is not None:
        span.set_attribute("mloda.data_access.identity", identity)
    if context.data_access_format is not None:
        span.set_attribute("mloda.data_access.format", context.data_access_format)
    if context.data_access_identity_is_fallback is not None:
        span.set_attribute("mloda.data_access.identity_is_fallback", context.data_access_identity_is_fallback)


def _set_join_attributes(span: Span, context: HookContext) -> None:
    """mloda.join.* attributes from the join context; unset ones are omitted."""
    if context.join_type is not None:
        span.set_attribute("mloda.join.type", context.join_type)
    if context.join_keys is not None:
        span.set_attribute("mloda.join.keys", context.join_keys)
    if context.join_left_feature_group is not None:
        span.set_attribute("mloda.join.left_feature_group", context.join_left_feature_group)
    if context.join_right_feature_group is not None:
        span.set_attribute("mloda.join.right_feature_group", context.join_right_feature_group)
    asof = context.asof_config
    if asof is None:
        return
    span.set_attribute("mloda.join.asof.left_time_column", asof.left_time_column)
    span.set_attribute("mloda.join.asof.right_time_column", asof.right_time_column)
    span.set_attribute("mloda.join.asof.direction", asof.direction)
    span.set_attribute("mloda.join.asof.allow_exact_matches", asof.allow_exact_matches)
    tolerance = asof.tolerance
    if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool):
        span.set_attribute(
            "mloda.join.asof.tolerance", float(tolerance) if isinstance(tolerance, float) else int(tolerance)
        )
    elif isinstance(tolerance, timedelta):
        span.set_attribute("mloda.join.asof.tolerance_seconds", tolerance.total_seconds())


def _set_declared_attributes(span: Span, context: HookContext) -> None:
    """mloda.declared.<key> attributes from the hook context's declared_attributes (validated by core)."""
    if not span.is_recording() or not context.declared_attributes:
        return
    count = 0
    for key, value in context.declared_attributes.items():
        if count >= _MAX_DECLARED_KEYS:
            break
        if not isinstance(value, _SCALAR_TYPES):
            continue
        if isinstance(value, str):
            value = value[:_CONTENT_PREVIEW_MAX_LEN]
        span.set_attribute(f"mloda.declared.{key}", value)
        count += 1


def _set_step_attributes(span: Span, context: HookContext, func: Any) -> None:
    if context.hook != ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE:
        return
    step_id = step_run_id(
        context.run_id,
        owner_name(context, func),
        context.feature_names,
        context.compute_framework_name,
        context.step_uuid,
    )
    if step_id is not None:
        span.set_attribute("mloda.step.run_id", step_id)


def _set_context_attributes(span: Span, context: HookContext) -> None:
    span.set_attribute("mloda.operation.name", _OPERATION_NAMES.get(context.hook, "unknown"))
    if context.feature_group_class is not None:
        span.set_attribute("mloda.feature_group.name", context.feature_group_class)
    if context.feature_group_version is not None:
        span.set_attribute("mloda.feature_group.version", context.feature_group_version)
    if context.compute_framework_name is not None:
        span.set_attribute("mloda.compute_framework.name", context.compute_framework_name)

    if context.rows_in is not None:
        span.set_attribute("mloda.rows.in", context.rows_in)
    if len(context.feature_names) == 1:
        span.set_attribute("mloda.feature.name", context.feature_names[0])
    if context.plugin_version is not None:
        span.set_attribute("mloda.plugin.version", context.plugin_version)
    if context.run_id is not None:
        span.set_attribute("mloda.run.id", context.run_id)
    if context.step_uuid is not None:
        span.set_attribute("mloda.step.uuid", str(context.step_uuid))
    if context.plan_id is not None:
        span.set_attribute("mloda.plan.id", context.plan_id)
    if context.worker_index is not None:
        span.set_attribute("mloda.subprocess.worker_index", context.worker_index)
