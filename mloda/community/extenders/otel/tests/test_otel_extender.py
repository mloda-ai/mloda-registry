"""Tests for OtelExtender.

Every direct __call__ in this file is wrapped in a manually built HookContext.activate()
scope (mirroring core's _run_hook), so the "no ambient HookContext" case is never exercised
here. Each test gets its own isolated (TracerProvider, InMemorySpanExporter) pair via the
otel_capture fixture; the global tracer provider is never touched, since tests run under
pytest-xdist alongside other test files in this same process group.
"""

from __future__ import annotations

import ast
import contextlib
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from mloda.core.abstract_plugins.function_extender import _CompositeExtender  # no public equivalent yet
from mloda.core.abstract_plugins.hook_context import instrument  # no public equivalent yet
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.steward import Extender, ExtenderHook, HookContext
from mloda.user import PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from mloda.community.extenders.otel import OtelExtender
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator

# Canonical value_int column of the shared test dataset (mirrors the example extender's tests).
_VALUE_INT = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]

# The one attribute key that MUST carry content preview.
_CONTENT_ATTRIBUTE = "mloda.content.preview"


@pytest.fixture
def otel_capture() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    """A fresh, isolated (provider, exporter) pair per test; never touches the global provider."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    provider.shutdown()


def _make_context(
    *,
    hook: ExtenderHook = ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
    feature_group_class: str = "tests.otel.DummyFeatureGroup",
    feature_group_version: str = "1",
    plugin_version: str | None = None,
    feature_names: tuple[str, ...] = ("value_int",),
    input_features: frozenset[str] | None = None,
    compute_framework_name: str = "PyArrowTable",
    rows_in: int | None = None,
    rows_out: int | None = None,
    run_id: str | None = None,
) -> HookContext:
    """Build a HookContext with sane defaults; every field is overridable per test."""
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
        run_id=run_id,
    )


def _single_span_attributes(exporter: InMemorySpanExporter) -> Mapping[str, Any]:
    """The attributes of the one-and-only finished span; fails loudly if that's not the case."""
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, spans
    attributes = spans[0].attributes
    assert attributes is not None
    return attributes


class FailingOtelCalculateFeatureGroup(FeatureGroup):
    """Root feature group whose calculate_feature always raises, counting its invocations."""

    calls = 0

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"otel_boom_feature"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.calls += 1
        raise RuntimeError("inner boom")


def _run_value_int(*extenders: Extender, tracer_provider: TracerProvider | None = None) -> list[Any]:
    """Run the minimal ``value_int`` feature through run_all with the given extenders."""
    plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})

    results = mloda.run_all(
        ["value_int"],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
    )

    for table in results:
        if "value_int" in table.column_names:
            values: list[Any] = table.to_pydict()["value_int"]
            return values

    raise AssertionError("No result table with value_int found")


def _run_otel_boom_feature(*extenders: Extender) -> Any:
    """Run the always-failing ``otel_boom_feature`` through run_all with the given extenders."""
    plugin_collector = PluginCollector.enabled_feature_groups({FailingOtelCalculateFeatureGroup})

    return mloda.run_all(
        ["otel_boom_feature"],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
    )


class TestOtelExtenderImport:
    def test_import_from_package(self) -> None:
        from mloda.community.extenders.otel import OtelExtender

        assert OtelExtender is not None

    def test_class_is_accessible(self) -> None:
        from mloda.community.extenders.otel import OtelExtender

        assert isinstance(OtelExtender, type)


class TestOtelExtenderInheritance:
    def test_inherits_from_extender(self) -> None:
        assert issubclass(OtelExtender, Extender)

    def test_instance_is_extender(self) -> None:
        assert isinstance(OtelExtender(), Extender)


class TestOtelExtenderModuleImports:
    """opentelemetry-sdk is a dev-only extra of this package (only opentelemetry-api is a real runtime
    dependency), so the module must not import anything under opentelemetry.sdk at the top level."""

    def test_no_top_level_opentelemetry_sdk_import(self) -> None:
        from mloda.community.extenders.otel import otel_extender

        source = Path(otel_extender.__file__).read_text()
        tree = ast.parse(source)

        sdk_imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                sdk_imports.extend(alias.name for alias in node.names if alias.name.startswith("opentelemetry.sdk"))
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith("opentelemetry.sdk"):
                    sdk_imports.append(node.module)

        assert sdk_imports == [], (
            f"otel_extender.py has a top-level import of {sdk_imports} from opentelemetry.sdk, a dev-only "
            "extra (opentelemetry-sdk is declared only under this package's 'dev' extra, never as a "
            "runtime dependency); the module already has `from __future__ import annotations`, so any "
            "type reference needed purely for annotations must come from opentelemetry.trace (the "
            "API-only module) instead."
        )


class TestOtelExtenderErrorContract:
    """raise_on_error and wraps(): OtelExtender is observability-only, so it must default to warning-only."""

    def test_raise_on_error_defaults_to_false(self) -> None:
        """Observability must not break calculations by default."""
        assert OtelExtender().raise_on_error is False

    def test_raise_on_error_can_be_enabled(self) -> None:
        assert OtelExtender(raise_on_error=True).raise_on_error is True

    def test_raise_on_error_explicit_false(self) -> None:
        assert OtelExtender(raise_on_error=False).raise_on_error is False

    def test_wraps_returns_all_three_hooks(self) -> None:
        expected = {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
        }
        assert OtelExtender().wraps() == expected

    def test_wraps_is_unconditional_regardless_of_constructor_args(self) -> None:
        """opentelemetry-api is a hard dependency now: no "trace missing" degraded set exists anymore."""
        expected = {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
        }
        assert OtelExtender(raise_on_error=True, capture_content=True).wraps() == expected


class TestOtelExtenderConstructorOptions:
    """tracer_provider injection: the seam that keeps tests off the global OTel provider."""

    def test_tracer_provider_is_stored_on_construction(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, _ = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        assert otel._tracer_provider is provider

    def test_default_tracer_provider_is_none_and_call_still_works(self) -> None:
        otel = OtelExtender()
        context = _make_context()

        with context.activate():
            result = otel(lambda: 42)

        assert result == 42


class TestOtelExtenderSpanNaming:
    """One span per invocation, named by hook."""

    def test_calculate_hook_span_name(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = _make_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "mloda.calculate"

    def test_validate_input_hook_span_name(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = _make_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "mloda.validate.input"

    def test_validate_output_hook_span_name(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = _make_context(hook=ExtenderHook.VALIDATE_OUTPUT_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "mloda.validate.output"


class TestOtelExtenderSpanAttributes:
    """Span attributes under the mloda.* namespace, populated from the ambient HookContext."""

    def test_operation_name_for_calculate_hook(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = _make_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.operation.name"] == "calculate"

    def test_operation_name_for_validate_input_hook(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.operation.name"] == "validate"

    def test_operation_name_for_validate_output_hook(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context(hook=ExtenderHook.VALIDATE_OUTPUT_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.operation.name"] == "validate"

    def test_feature_group_name_and_version_attributes(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context(feature_group_class="pkg.mod.MyFeatureGroup", feature_group_version="7")
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        attrs = _single_span_attributes(exporter)
        assert attrs["mloda.feature_group.name"] == "pkg.mod.MyFeatureGroup"
        assert attrs["mloda.feature_group.version"] == "7"

    def test_compute_framework_name_attribute(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = _make_context(compute_framework_name="PyArrowTable")
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.compute_framework.name"] == "PyArrowTable"

    def test_rows_in_attribute(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = _make_context(rows_in=42)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.rows.in"] == 42

    def test_rows_out_attribute_present_after_successful_calculate(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """rows_out is set by instrument() DURING the func call; the extender must read it AFTER, not before."""
        provider, exporter = otel_capture
        context = _make_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            result = otel(instrument(context, func))

        assert result == [1, 2, 3]
        assert _single_span_attributes(exporter)["mloda.rows.out"] == 3

    def test_rows_out_absent_for_validate_hooks_even_when_context_carries_a_value(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """Gate on hook == calculate explicitly, not merely on rows_out being non-None."""
        provider, exporter = otel_capture
        context = _make_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE, rows_out=99)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.rows.out" not in _single_span_attributes(exporter)

    def test_feature_name_present_for_single_feature(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context(feature_names=("value_int",))
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.feature.name"] == "value_int"

    def test_feature_name_absent_for_zero_features(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context(feature_names=())
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.feature.name" not in _single_span_attributes(exporter)

    def test_feature_name_absent_for_multiple_features(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context(feature_names=("a", "b"))
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.feature.name" not in _single_span_attributes(exporter)

    def test_plugin_version_attribute_present_when_known(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context(plugin_version="1.2.3")
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.plugin.version"] == "1.2.3"

    def test_plugin_version_attribute_absent_when_unknown(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context(plugin_version=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.plugin.version" not in _single_span_attributes(exporter)

    def test_run_id_attribute_absent_when_none(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        """run_id is always None today (a future core release feature); guard against silent breakage."""
        provider, exporter = otel_capture
        context = _make_context(run_id=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.run.id" not in _single_span_attributes(exporter)


class TestOtelExtenderReturnValue:
    """__call__ must be a transparent wrapper around func's return value."""

    def test_call_returns_wrapped_function_result_unchanged(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, _ = otel_capture
        context = _make_context()
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            result = otel(lambda a, b: a + b, 3, 4)

        assert result == 7


class TestOtelExtenderFailureHandling:
    """When the WRAPPED FUNCTION raises: never swallow, always propagate, but observe first."""

    def test_call_propagates_func_failure_and_invokes_func_exactly_once(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, _ = otel_capture
        context = _make_context()
        otel = OtelExtender(tracer_provider=provider)
        calls = {"n": 0}

        def func() -> None:
            calls["n"] += 1
            raise RuntimeError("inner boom")

        with context.activate():
            with pytest.raises(RuntimeError, match="inner boom"):
                otel(func)

        assert calls["n"] == 1

    def test_call_sets_error_span_status_and_error_type_on_func_failure(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context()
        otel = OtelExtender(tracer_provider=provider)

        def func() -> None:
            raise RuntimeError("inner boom")

        with context.activate():
            with pytest.raises(RuntimeError):
                otel(func)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["error.type"] == "builtins.RuntimeError"

    def test_call_logs_warning_on_func_failure(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = otel_capture
        context = _make_context()
        otel = OtelExtender(tracer_provider=provider)

        def func() -> None:
            raise RuntimeError("inner boom")

        with caplog.at_level(logging.WARNING):
            with context.activate():
                with pytest.raises(RuntimeError):
                    otel(func)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OtelExtender" in r.message and "inner boom" in r.message for r in warnings), warnings

    def test_run_all_wrapped_function_failure_propagates_and_runs_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """A failure of the wrapped function propagates regardless of raise_on_error, without a re-run."""
        FailingOtelCalculateFeatureGroup.calls = 0

        with caplog.at_level(logging.WARNING):
            with pytest.raises(Exception, match="inner boom"):
                _run_otel_boom_feature(OtelExtender(raise_on_error=False))

        assert FailingOtelCalculateFeatureGroup.calls == 1

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("OtelExtender" in message and "inner boom" in message for message in warnings), warnings

    def test_call_logs_exception_type_and_span_name_for_a_message_less_exception(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        """`raise ValueError()` carries no message, so `logger.warning("OtelExtender %s", exc)` logs the
        literal string "OtelExtender " (str(exc) == ""): no exception type, no span/hook name, nothing
        actionable. The log must name both, independent of whether exc happens to carry a message."""
        provider, _ = otel_capture
        context = _make_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        def func() -> None:
            raise ValueError()

        with caplog.at_level(logging.WARNING):
            with context.activate():
                with pytest.raises(ValueError):
                    otel(func)

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("ValueError" in message and "mloda.calculate" in message for message in warnings), warnings

    def test_func_exception_message_does_not_leak_into_span_events_by_default(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """OTel's start_as_current_span defaults to record_exception=True: an exception propagating out of
        the with-block gets an auto-attached 'exception' span EVENT carrying its message and full stack
        trace (exception.message, exception.stacktrace), independent of capture_content /
        MLODA_OTEL_TRACE_CONTENT. When the func's exception message embeds raw data (a common real-world
        validation-error pattern), that value must not leak onto the span while capture_content stays at
        its metadata-only default."""
        provider, exporter = otel_capture
        context = _make_context()
        otel = OtelExtender(tracer_provider=provider)  # capture_content left at its metadata-only default
        marker = "SENSITIVE_ROW_VALUE_xyz123"

        def func() -> None:
            raise ValueError(f"invalid value found: {marker}")

        with context.activate():
            with pytest.raises(ValueError):
                otel(func)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1, spans
        span = spans[0]

        for event in span.events:
            event_attrs = event.attributes or {}
            for value in event_attrs.values():
                assert marker not in str(value), (
                    f"span event {event.name!r} attribute leaked the func exception's message ({marker!r}) "
                    "even though capture_content is False; OTel's default record_exception=True must be "
                    "disabled (or the message scrubbed) so metadata-only stays metadata-only"
                )


class TestOtelExtenderCompositeChaining:
    """OtelExtender chains via _CompositeExtender; faults are injected by patching `trace.get_tracer`
    to force a failure in the extender's own code, independent of any wrapped-function failure."""

    def test_own_failure_falls_back_when_raise_on_error_false(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = otel_capture
        context = _make_context()
        otel = OtelExtender(raise_on_error=False, tracer_provider=provider)
        composite = _CompositeExtender([otel])

        def func(x: int, y: int) -> int:
            return x + y

        with patch(
            "mloda.community.extenders.otel.otel_extender.trace.get_tracer",
            side_effect=RuntimeError("otel instrumentation boom"),
        ):
            with caplog.at_level(logging.WARNING):
                with context.activate():
                    result = composite(func, 3, 4)

        assert result == 7, "Failing OtelExtender must fall back to the wrapped function result"
        assert any(r.levelno == logging.WARNING and "OtelExtender" in r.message for r in caplog.records), caplog.records

    def test_own_failure_propagates_when_raise_on_error_true(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, _ = otel_capture
        context = _make_context()
        otel = OtelExtender(raise_on_error=True, tracer_provider=provider)
        composite = _CompositeExtender([otel])

        def func(x: int, y: int) -> int:
            return x + y

        with patch(
            "mloda.community.extenders.otel.otel_extender.trace.get_tracer",
            side_effect=RuntimeError("otel instrumentation boom"),
        ):
            with context.activate():
                with pytest.raises(RuntimeError, match="otel instrumentation boom"):
                    composite(func, 3, 4)


class TestOtelExtenderContentCapture:
    """Metadata-only by default; capture_content=True or MLODA_OTEL_TRACE_CONTENT opts in, mask redacts."""

    def test_no_content_attribute_by_default(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLODA_OTEL_TRACE_CONTENT", raising=False)
        provider, exporter = otel_capture
        context = _make_context()
        otel = OtelExtender(tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            otel(instrument(context, func))

        attrs = _single_span_attributes(exporter)
        assert not any("content" in key for key in attrs), attrs

    def test_content_attribute_present_when_capture_content_true(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLODA_OTEL_TRACE_CONTENT", raising=False)
        provider, exporter = otel_capture
        context = _make_context()
        otel = OtelExtender(capture_content=True, tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            otel(instrument(context, func))

        assert _CONTENT_ATTRIBUTE in _single_span_attributes(exporter)

    @pytest.mark.parametrize("value", ["true", "1"])
    def test_content_attribute_present_when_env_var_truthy(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        monkeypatch: pytest.MonkeyPatch,
        value: str,
    ) -> None:
        monkeypatch.setenv("MLODA_OTEL_TRACE_CONTENT", value)
        provider, exporter = otel_capture
        context = _make_context()
        otel = OtelExtender(tracer_provider=provider)  # capture_content constructor arg left at default False

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            otel(instrument(context, func))

        assert _CONTENT_ATTRIBUTE in _single_span_attributes(exporter)

    def test_content_attribute_absent_when_env_var_falsy(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLODA_OTEL_TRACE_CONTENT", "false")
        provider, exporter = otel_capture
        context = _make_context()
        otel = OtelExtender(tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            otel(instrument(context, func))

        assert _CONTENT_ATTRIBUTE not in _single_span_attributes(exporter)

    def test_content_attribute_absent_on_validate_hooks_even_when_capture_enabled(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE)
        otel = OtelExtender(capture_content=True, tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            otel(instrument(context, func))

        assert _CONTENT_ATTRIBUTE not in _single_span_attributes(exporter)

    def test_content_attribute_absent_on_func_failure(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context()
        otel = OtelExtender(capture_content=True, tracer_provider=provider)

        def func() -> None:
            raise RuntimeError("inner boom")

        with context.activate():
            with pytest.raises(RuntimeError):
                otel(instrument(context, func))

        assert _CONTENT_ATTRIBUTE not in _single_span_attributes(exporter)

    def test_content_attribute_is_bounded_for_large_results(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context()
        otel = OtelExtender(capture_content=True, tracer_provider=provider)
        raw = "x" * 10_000

        def func() -> str:
            return raw

        with context.activate():
            otel(instrument(context, func))

        preview = _single_span_attributes(exporter)[_CONTENT_ATTRIBUTE]
        assert len(str(preview)) < len(raw)

    def test_content_attribute_uses_mask_to_redact_raw_value(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context()
        secret = "SECRET_VALUE_12345"  # nosec
        masked = "***MASKED***"

        def mask(_value: Any) -> str:
            return masked

        otel = OtelExtender(capture_content=True, mask=mask, tracer_provider=provider)

        def func() -> str:
            return secret

        with context.activate():
            otel(instrument(context, func))

        attrs = _single_span_attributes(exporter)
        for value in attrs.values():
            assert secret not in str(value), attrs
        assert masked in str(attrs[_CONTENT_ATTRIBUTE])


class TestOtelExtenderPostCallInstrumentationFailure:
    """A bug in the extender's OWN post-call code (reading context.rows_out, then mask/str on a result
    that func already returned successfully) runs outside any try/except, inside the
    `start_as_current_span` block. If it raises, OTel's context manager marks the span ERROR and records
    an exception event, exactly as it would for a genuine func failure - even though func itself
    succeeded and the computed result is fine. That makes a successful run indistinguishable, from the
    trace backend's point of view, from a real pipeline failure."""

    def test_mask_failure_after_func_success_does_not_mark_span_error(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = _make_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)

        def broken_mask(_value: Any) -> Any:
            raise RuntimeError("mask boom")

        otel = OtelExtender(capture_content=True, mask=broken_mask, tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        # Called directly (not through _CompositeExtender): raise_on_error has no bearing on this path,
        # since it only governs how core's _invoke_extender reacts to a raise from ANYWHERE inside
        # __call__, not whether the extender's own post-call code corrupts the span it already built.
        # func already succeeded by the time broken_mask runs, so whatever escapes here is the
        # extender's own bug, not func's; let it propagate and inspect the span it leaves behind.
        with context.activate():
            with contextlib.suppress(Exception):
                otel(instrument(context, func))

        spans = exporter.get_finished_spans()
        assert len(spans) == 1, spans
        span = spans[0]
        assert span.status.status_code != StatusCode.ERROR, (
            "func succeeded, yet the span's own post-call attribute-setting code (mask/str on the "
            "result) raising made the span look identical to a genuine func failure; a broken mask must "
            "not be indistinguishable, from the trace backend's point of view, from func itself failing"
        )


class TestOtelExtenderContentPreviewCost:
    """_content_preview must bound the cost of previewing a large result, not materialize str(result) in
    full before slicing to 200 chars."""

    def test_content_preview_does_not_repr_every_element_of_a_large_result(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        class _CountingItem:
            calls = 0

            def __repr__(self) -> str:
                _CountingItem.calls += 1
                return "item"

        provider, exporter = otel_capture
        context = _make_context()
        otel = OtelExtender(capture_content=True, tracer_provider=provider)

        result = [_CountingItem() for _ in range(5000)]
        _CountingItem.calls = 0

        def func() -> list[_CountingItem]:
            return result

        with context.activate():
            otel(instrument(context, func))

        assert _CountingItem.calls < 50, (
            f"_content_preview repr'd {_CountingItem.calls} of 5000 elements while computing a 200-char "
            "preview; it must bound the cost of previewing a large result instead of materializing "
            "str(result) in full before truncating"
        )


class TestOtelExtenderRunAll:
    """End-to-end wiring through mloda.user.mloda.run_all: real spans, unmodified results."""

    def test_run_all_produces_expected_spans_and_leaves_result_unchanged(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})

        results = mloda.run_all(
            ["value_int"],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
            function_extender={OtelExtender(tracer_provider=provider)},
        )

        values = None
        for table in results:
            if "value_int" in table.column_names:
                values = table.to_pydict()["value_int"]
        assert values == _VALUE_INT

        spans = exporter.get_finished_spans()
        span_names = {span.name for span in spans}
        # A root DataCreator-based feature group never triggers VALIDATE_INPUT_FEATURE (no data
        # exists to validate before calculate runs), so only these two hooks are expected here.
        assert {"mloda.calculate", "mloda.validate.output"} <= span_names

        for span in spans:
            assert span.attributes is not None
            assert span.attributes.get("mloda.feature.name") == "value_int"
            assert span.attributes.get("mloda.compute_framework.name") == "PyArrowTable"
