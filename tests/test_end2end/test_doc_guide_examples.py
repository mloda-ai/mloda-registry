"""Validate the examples embedded in the FeatureConfig guide (issue #516) and the backend-families guide.

``scripts/lint_docs.py`` checks docs for broken links and raw-dict ``PROPERTY_MAPPING``
values, but never executes the config snippets themselves. A FeatureConfig example once
shipped with a feature name suffix (``_aggr``) that ``AggregationFeatureGroup.PREFIX_PATTERN``
never matched, and no ``partition_by``, so the example could never actually resolve; it took
two independent full reviews to catch it. This module extracts every FeatureConfig JSON block
from the guide, runs it through the real ``load_features_from_config`` parser, and for every
data-operation feature name (containing ``__``) checks that a real ``FeatureGroup`` subclass
actually accepts it via ``match_feature_group_criteria`` (the real runtime matcher).

``extract_config_blocks``, ``build_feature_group_registry`` and ``find_unresolved_features`` are
small, named, importable helpers so the regression tests below can feed the two historical-bug
variants through the exact same validation logic used against the real guide, instead of
re-implementing the check inline.

It also runs every discovered doc's python blocks in its own interpreter and calls its ``test_*`` functions,
except the docs in ``SKIPPED_DOCS``.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import re
import subprocess  # nosec
import sys
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, create_autospec, patch

import pytest
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator, FeatureSet, ImplementationPinError
from mloda.steward import resolve_feature
from mloda.user import DataAccessCollection, Options, PluginCollector, load_features_from_config
from mloda.user import mloda as real_mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

import mloda.community.feature_groups.data_operations as data_operations
from tests.test_end2end.test_binary_backed_features_guide import _section

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_CONFIG_GUIDE = REPO_ROOT / "docs" / "guides" / "feature-group-patterns" / "22-feature-config.md"
BACKEND_FAMILIES_GUIDE = REPO_ROOT / "docs" / "guides" / "feature-group-patterns" / "28-backend-families.md"
PANDAS_AGGREGATION_PIN = (
    "mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation:PandasAggregation"
)
WAY_2_HEADING = "## Way 2: Discriminator Option (Sibling FeatureGroups)"
WAY_2_MODULE = "guide_28_way_2"
WAY_3_HEADING = "## Way 3: Reader Delegation (Sibling Readers)"
WAY_3_MODULE = "guide_28_way_3"
TEST_HEADING = "\n## Test\n"
FRAMEWORK_SPECIFIC_GUIDE = REPO_ROOT / "docs" / "guides" / "feature-group-patterns" / "09-framework-specific.md"
TESTING_GUIDE = REPO_ROOT / "docs" / "guides" / "feature-group-patterns" / "10-testing-guide.md"
REFERENCE_CONTRACT_HEADING = "## Reference Contract and Divergence"
REFERENCE_CONTRACT_MODULE = "guide_09_reference_contract"
CONFORMANCE_HEADING = "## Operation Groups: Conformance to the Reference Contract"

_FENCE_RE = re.compile(r"```(json|python|py)\n(.*?)```", re.DOTALL)
_CONFIG_STRING_RE = re.compile(r'config\s*=\s*"""(.*?)"""', re.DOTALL)


def extract_config_blocks(markdown: str) -> list[list[dict[str, Any]]]:
    """FeatureConfig-shaped JSON lists from ``json``/``python``/``py`` fences in *markdown*.

    A ``json`` fence body is parsed directly; a ``python``/``py`` fence is scanned for
    ``config = \"\"\"...\"\"\"`` assignments (the guide's convention for embedding config JSON in
    executable prose). Anything that fails to parse as JSON is skipped. A bare dict is wrapped
    in a one-item list. Only lists containing at least one dict with a ``"name"`` key are kept,
    which filters out unrelated fences (e.g. plain Python snippets that aren't config blocks).
    """
    blocks: list[list[dict[str, Any]]] = []
    for fence_match in _FENCE_RE.finditer(markdown):
        lang, body = fence_match.group(1), fence_match.group(2)
        candidates = [body] if lang == "json" else [m.group(1) for m in _CONFIG_STRING_RE.finditer(body)]
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                parsed = [parsed]
            if isinstance(parsed, list) and any(isinstance(item, dict) and "name" in item for item in parsed):
                blocks.append(parsed)
    return blocks


@lru_cache(maxsize=1)
def build_feature_group_registry() -> dict[str, type[FeatureGroup]]:
    """Every concrete ``FeatureGroup`` subclass under ``data_operations``, keyed by class name.

    Every submodule is imported directly via ``pkgutil.walk_packages`` (skipping ``.tests``
    subpackages) before collecting subclasses, so the registry does not depend on which
    workspace members' entry points happen to be installed in the venv.
    """
    prefix = data_operations.__name__ + "."
    for module_info in pkgutil.walk_packages(data_operations.__path__, prefix=prefix):
        if ".tests" in module_info.name:
            continue
        importlib.import_module(module_info.name)
    return {cls.get_class_name(): cls for cls in get_all_subclasses(FeatureGroup)}


def find_unresolved_features(
    block: list[dict[str, Any]],
    registry: dict[str, type[FeatureGroup]],
    source: str,
) -> list[str]:
    """Diagnostic messages for every data-operation feature in *block* that can't be resolved.

    Loads *block* through the real ``load_features_from_config`` parser (which alone must not
    raise). For every non-string ``Feature`` whose name contains ``__``: if it claims a
    ``feature_group_scope``, that class must exist in *registry* and accept the name via
    ``match_feature_group_criteria``; otherwise some registered class must accept it. Each
    failure message names *source*, the feature name, and (if claimed) the feature_group, so a
    regression is diagnosable without re-deriving this check.
    """
    features = load_features_from_config(json.dumps(block))
    problems: list[str] = []
    for feature in features:
        if isinstance(feature, str):
            continue
        assert isinstance(feature, Feature)
        name = str(feature.name)
        if "__" not in name:
            continue
        scope = feature.feature_group_scope
        if scope:
            cls = registry.get(str(scope))
            if cls is None:
                problems.append(
                    f"{source}: feature {name!r} claims feature_group {scope!r}, which is not a "
                    "registered FeatureGroup subclass"
                )
            elif not cls.match_feature_group_criteria(feature.name, feature.options):
                problems.append(
                    f"{source}: feature {name!r} does not match feature_group {scope!r}'s "
                    "match_feature_group_criteria (PREFIX_PATTERN or required options mismatch)"
                )
        elif not any(cls.match_feature_group_criteria(feature.name, feature.options) for cls in registry.values()):
            problems.append(f"{source}: feature {name!r} is not matched by any registered FeatureGroup subclass")
    return problems


def test_feature_config_guide_blocks_are_resolvable() -> None:
    markdown = FEATURE_CONFIG_GUIDE.read_text(encoding="utf-8")
    blocks = extract_config_blocks(markdown)
    assert blocks, "no FeatureConfig blocks were extracted from the guide; the extraction regex may be stale"
    registry = build_feature_group_registry()
    problems = [
        problem
        for index, block in enumerate(blocks)
        for problem in find_unresolved_features(block, registry, f"{FEATURE_CONFIG_GUIDE.name}[block {index}]")
    ]
    assert problems == [], "\n".join(problems)


def test_feature_config_guide_scopes_aggregation_to_its_group_and_pins_an_implementation() -> None:
    """Per-framework aggregation classes are Implementations: scope to the group, pin with ``implementation``."""
    markdown = FEATURE_CONFIG_GUIDE.read_text(encoding="utf-8")
    assert "| `implementation` |" in markdown, "the field table must document `implementation`"
    items = [item for block in extract_config_blocks(markdown) for item in block if isinstance(item, dict)]
    scopes = {item["feature_group"] for item in items if "feature_group" in item}
    assert "AggregationFeatureGroup" in scopes
    assert "PandasAggregation" not in scopes, "PandasAggregation is an Implementation, not a feature_group scope"

    pinned = [item for item in items if item.get("implementation") == PANDAS_AGGREGATION_PIN]
    assert len(pinned) == 1, "the guide must show exactly one implementation pin example"
    assert pinned[0].get("feature_group") == "AggregationFeatureGroup"
    (feature,) = load_features_from_config(json.dumps([pinned[0]]))
    assert isinstance(feature, Feature)
    assert feature.implementation_pin == (PANDAS_AGGREGATION_PIN,)


def test_wrong_suffix_regression_is_rejected() -> None:
    """Historical bug: ``_aggr`` never matched ``AggregationFeatureGroup.PREFIX_PATTERN`` (``_agg``)."""
    block = [
        {
            "name": "sales__sum_aggr",
            "feature_group": "AggregationFeatureGroup",
            "context_options": {"partition_by": ["region"]},
        }
    ]
    registry = build_feature_group_registry()
    problems = find_unresolved_features(block, registry, "synthetic[wrong_suffix]")
    assert problems != [], "wrong-suffix regression was not caught; the validation logic is vacuous"
    assert "sales__sum_aggr" in problems[0]
    assert "AggregationFeatureGroup" in problems[0]


def test_missing_partition_by_regression_is_rejected() -> None:
    """Historical bug: no ``partition_by`` means AggregationFeatureGroup's required option is absent."""
    block = [{"name": "sales__sum_agg", "feature_group": "AggregationFeatureGroup"}]
    registry = build_feature_group_registry()
    problems = find_unresolved_features(block, registry, "synthetic[missing_partition_by]")
    assert problems != [], "missing-partition_by regression was not caught; the validation logic is vacuous"
    assert "sales__sum_agg" in problems[0]
    assert "AggregationFeatureGroup" in problems[0]


def _python_fences(heading: str, guide: Path = BACKEND_FAMILIES_GUIDE) -> list[str]:
    section = _section(guide.read_text(encoding="utf-8"), heading, ("\n## ",))
    return [m.group(2) for m in _FENCE_RE.finditer(section) if m.group(1) == "python"]


@lru_cache(maxsize=None)
def _load_namespace(heading: str, module: str, guide: Path = BACKEND_FAMILIES_GUIDE) -> dict[str, Any]:
    """Run the first python block of the guide section under *heading* once and return its namespace."""
    fences = _python_fences(heading, guide)
    assert fences, f"no python block found in the {heading!r} section of {guide.name}"
    code = compile(fences[0], f"<{guide.name} {module}>", "exec", dont_inherit=True)
    namespace: dict[str, Any] = {"__name__": module}
    exec(code, namespace)  # nosec
    return namespace


def load_way_2_usage_feature() -> Feature:
    """Evaluate the second python block of the Way 2 section (the usage expression) against its namespace."""
    fences = _python_fences(WAY_2_HEADING)
    assert len(fences) >= 2, "the Way 2 section no longer has a second python block with the usage expression"
    code = compile(fences[1], f"<{BACKEND_FAMILIES_GUIDE.name} Way 2 usage>", "eval", dont_inherit=True)
    feature = eval(code, {**_load_namespace(WAY_2_HEADING, WAY_2_MODULE), "Feature": Feature})  # nosec
    assert isinstance(feature, Feature), f"the Way 2 usage block no longer evaluates to a Feature: {feature!r}"
    return feature


def load_way_3_usage_call() -> tuple[Feature, DataAccessCollection, set[type[ComputeFramework]]]:
    """Run the second python block of the Way 3 section against a mock ``mloda`` and return what it passed."""
    fences = _python_fences(WAY_3_HEADING)
    assert len(fences) >= 2, "the Way 3 section no longer has a second python block with the run_all call"
    code = compile(fences[1], f"<{BACKEND_FAMILIES_GUIDE.name} Way 3 usage>", "exec", dont_inherit=True)
    mloda = Mock(spec_set=["run_all"])
    mloda.run_all = create_autospec(real_mloda.run_all)
    exec(code, {**_load_namespace(WAY_3_HEADING, WAY_3_MODULE), "Feature": Feature, "mloda": mloda})  # nosec
    assert mloda.run_all.call_count == 1, "the Way 3 usage block no longer calls mloda.run_all exactly once"
    args, kwargs = mloda.run_all.call_args
    assert len(args) == 1 and len(args[0]) == 1, "the Way 3 run_all call no longer passes a single feature"
    collection = kwargs.get("data_access_collection")
    assert isinstance(collection, DataAccessCollection), (
        "the Way 3 run_all call no longer passes data_access_collection"
    )
    frameworks = kwargs.get("compute_frameworks")
    assert frameworks, "the Way 3 run_all call no longer passes compute_frameworks"
    return args[0][0], collection, set(frameworks)


def _guide_feature_groups(namespace: dict[str, Any], module: str) -> set[type[FeatureGroup]]:
    found = {
        v for v in namespace.values() if isinstance(v, type) and issubclass(v, FeatureGroup) and v.__module__ == module
    }
    assert found, f"the guide block for {module} no longer defines any FeatureGroup"
    return found


def _resolve_way_2(feature: str | Feature) -> tuple[dict[str, Any], Any]:
    namespace = _load_namespace(WAY_2_HEADING, WAY_2_MODULE)
    for name in ("Bm25Retriever", "FaissRetriever"):
        assert name in namespace, f"the Way 2 block no longer defines {name}"
    result = resolve_feature(
        feature,
        plugin_collector=PluginCollector.enabled_feature_groups(_guide_feature_groups(namespace, WAY_2_MODULE)),
        compute_frameworks={PythonDictFramework},
    )
    return namespace, result


def _resolve_way_3(
    feature: Feature, data_access_collection: DataAccessCollection, compute_frameworks: set[type[ComputeFramework]]
) -> tuple[dict[str, Any], Any]:
    namespace = _load_namespace(WAY_3_HEADING, WAY_3_MODULE)
    assert "RdfLibSparqlFeatureGroup" in namespace, "the Way 3 block no longer defines RdfLibSparqlFeatureGroup"
    result = resolve_feature(
        feature,
        plugin_collector=PluginCollector.enabled_feature_groups(_guide_feature_groups(namespace, WAY_3_MODULE)),
        compute_frameworks=compute_frameworks,
        data_access_collection=data_access_collection,
    )
    return namespace, result


@pytest.mark.parametrize(
    "group, expected",
    [
        ({"retrieve_backend": "faiss"}, "FaissRetriever"),
        ({"retrieve_backend": "bm25"}, "Bm25Retriever"),
    ],
)
def test_way_2_discriminator_resolves_to_exactly_one_sibling(group: dict[str, str], expected: str) -> None:
    namespace, result = _resolve_way_2(Feature("retrieved_passages", options=Options(group=group)))
    assert result.feature_group is namespace[expected], result.error
    assert result.candidates == [namespace[expected]]


def test_way_2_guide_usage_line_resolves_to_exactly_one_sibling() -> None:
    namespace, result = _resolve_way_2(load_way_2_usage_feature())
    assert result.feature_group is namespace["FaissRetriever"], result.error
    assert result.candidates == [namespace["FaissRetriever"]]


def test_way_2_unknown_discriminator_value_resolves_to_nothing() -> None:
    _, result = _resolve_way_2(Feature("retrieved_passages", options=Options(group={"retrieve_backend": "fiass"})))
    assert result.feature_group is None
    assert result.candidates == []
    assert "No feature groups found" in str(result.error)


def test_way_3_guide_usage_resolves_to_the_reader_sibling() -> None:
    feature, collection, frameworks = load_way_3_usage_call()
    namespace, result = _resolve_way_3(feature, collection, frameworks)
    assert result.feature_group is namespace["RdfLibSparqlFeatureGroup"], result.error
    assert result.candidates == [namespace["RdfLibSparqlFeatureGroup"]]


def test_way_3_collection_without_the_slot_resolves_to_nothing() -> None:
    feature, _, frameworks = load_way_3_usage_call()
    collection = DataAccessCollection(credentials=[{"no_such_connector": {"locator": "graph.ttl"}}])
    _, result = _resolve_way_3(feature, collection, frameworks)
    assert result.feature_group is None, result.feature_group
    assert result.candidates == []
    assert "No feature groups found" in str(result.error)


def _exec_section_tests(fence: str, label: str, base_namespace: dict[str, Any], what: str) -> list[Callable[..., Any]]:
    """Exec *fence* in a copy of *base_namespace* and return the ``test_`` callables it newly defines."""
    namespace = dict(base_namespace)
    before = set(namespace)
    exec(compile(fence, f"<{label}>", "exec", dont_inherit=True), namespace)  # nosec
    tests = [v for k, v in namespace.items() if k not in before and k.startswith("test_") and callable(v)]
    assert tests, f"the {what} python block no longer defines a test_ function"
    return tests


def test_test_section_sibling_test_passes_against_the_way_2_example() -> None:
    fences = _python_fences(TEST_HEADING)
    assert fences, "the Test section of the guide no longer has a python block with a sibling test"
    with patch("mloda.steward.resolve_feature", wraps=resolve_feature) as spy:
        tests = _exec_section_tests(
            fences[0],
            f"{BACKEND_FAMILIES_GUIDE.name} Test",
            _load_namespace(WAY_2_HEADING, WAY_2_MODULE),
            "Test section",
        )
        for sibling_test in tests:
            sibling_test()
        assert spy.called, "the Test section sibling test no longer calls resolve_feature"
        for call in spy.call_args_list:
            assert "plugin_collector" in call.kwargs, "the Test section no longer passes plugin_collector"
            assert "compute_frameworks" in call.kwargs, "the Test section no longer passes compute_frameworks"
        spy.return_value = SimpleNamespace(feature_group=None, candidates=[], error=None)
        spy.side_effect = None
        for sibling_test in tests:
            with pytest.raises(AssertionError):
                sibling_test()


@lru_cache(maxsize=1)
def _price_source() -> type[FeatureGroup]:
    """Pandas-only source of a ``price`` column with a NaN, built on first use so nothing registers at import."""

    class PriceSource(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({"price"})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {"price": [1.0, float("nan"), 3.0]}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
            return {PandasDataFrame}

    return PriceSource


def _column_total_request(feature: str | Feature) -> tuple[dict[str, Any], dict[str, Any]]:
    """The guide 09 namespace and the shared ``explain``/``run_all`` keyword arguments for *feature*."""
    namespace = _load_namespace(REFERENCE_CONTRACT_HEADING, REFERENCE_CONTRACT_MODULE, FRAMEWORK_SPECIFIC_GUIDE)
    assert "ColumnTotal" in namespace, "the Reference Contract block no longer defines ColumnTotal"
    kwargs: dict[str, Any] = {
        "compute_frameworks": [PandasDataFrame, PyArrowTable],
        "plugin_collector": PluginCollector.enabled_feature_groups({_price_source(), namespace["ColumnTotal"]}),
    }
    return namespace, kwargs


def _column_total_step(feature: str | Feature) -> tuple[dict[str, Any], Any]:
    namespace, kwargs = _column_total_request(feature)
    steps = real_mloda.explain([feature], **kwargs)
    matching = [s for s in steps if s.feature_group is namespace["ColumnTotal"] and s.implementation is not None]
    assert len(matching) == 1, f"expected exactly one ColumnTotal compute step with an implementation: {steps!r}"
    return namespace, matching[0]


def _operation_groups_usage_feature() -> Feature:
    fences = _python_fences(REFERENCE_CONTRACT_HEADING, FRAMEWORK_SPECIFIC_GUIDE)
    assert len(fences) >= 2, (
        "the Reference Contract section no longer has a second python block with the pin expression"
    )
    code = compile(fences[1], f"<{FRAMEWORK_SPECIFIC_GUIDE.name} pin>", "eval", dont_inherit=True)
    feature = eval(code, {"Feature": Feature})  # nosec
    assert isinstance(feature, Feature), f"the pin block no longer evaluates to a Feature: {feature!r}"
    return feature


def test_operation_group_unpinned_prefers_the_matching_version() -> None:
    namespace, step = _column_total_step("price__sum_total")
    assert step.implementation is namespace["PyArrowColumnTotal"], step.implementation
    assert "prefers a matching version" in str(step.compute_framework_reason), step.compute_framework_reason


def test_operation_group_guide_pin_selects_the_divergent_version() -> None:
    namespace, step = _column_total_step(_operation_groups_usage_feature())
    reason = str(step.compute_framework_reason)
    assert step.implementation is namespace["PandasColumnTotal"], step.implementation
    assert "pinned" in reason, reason
    assert "runs a version that differs: skips NaN in sum" in reason, reason


def _column_total_values(feature: str | Feature, column: str) -> list[Any]:
    """Run *feature* for real and return *column* of the (pandas or pyarrow) result as a python list."""
    _, kwargs = _column_total_request(feature)
    results = real_mloda.run_all([feature], **kwargs)
    assert len(results) == 1, f"expected one result frame, got {len(results)}"
    return list(results[0][column].to_pylist() if hasattr(results[0][column], "to_pylist") else results[0][column])


def _is_missing(value: Any) -> bool:
    return value is None or value != value  # noqa: PLR0124


def test_operation_group_unpinned_sum_propagates_the_missing_value() -> None:
    values = _column_total_values("price__sum_total", "price__sum_total")
    assert len(values) == 3 and all(_is_missing(v) for v in values), values


def test_operation_group_guide_pin_sums_skipping_the_missing_value() -> None:
    feature = _operation_groups_usage_feature()
    assert _column_total_values(feature, "price__sum_total") == [4.0, 4.0, 4.0]


def test_operation_group_unpinned_max_skips_the_missing_value() -> None:
    assert _column_total_values("price__max_total", "price__max_total") == [3.0, 3.0, 3.0]


def test_operation_group_unknown_pin_raises() -> None:
    with pytest.raises(ImplementationPinError):
        _column_total_step(Feature("price__sum_total", implementation="Nope"))


def _parametrized_calls(fn: Any) -> list[dict[str, Any]]:
    """Every keyword combination of the ``pytest.mark.parametrize`` marks on *fn*."""
    calls: list[dict[str, Any]] = [{}]
    for mark in getattr(fn, "pytestmark", []):
        if mark.name != "parametrize":
            continue
        names = [n.strip() for n in mark.args[0].split(",")] if isinstance(mark.args[0], str) else list(mark.args[0])
        rows = [
            tuple(row.values) if isinstance(row, type(pytest.param())) else row if len(names) > 1 else (row,)
            for row in mark.args[1]
        ]
        calls = [{**call, **dict(zip(names, row, strict=True))} for call in calls for row in rows]
    return calls


def test_conformance_section_test_passes_against_the_operation_group_example() -> None:
    fences = _python_fences(CONFORMANCE_HEADING, TESTING_GUIDE)
    assert fences, "the Conformance section of the testing guide no longer has a python block"
    tests = _exec_section_tests(
        fences[0],
        f"{TESTING_GUIDE.name} Conformance",
        _load_namespace(REFERENCE_CONTRACT_HEADING, REFERENCE_CONTRACT_MODULE, FRAMEWORK_SPECIFIC_GUIDE),
        "Conformance section",
    )
    calls = 0
    for conformance_test in tests:
        for kwargs in _parametrized_calls(conformance_test):
            conformance_test(**kwargs)
            calls += 1
    assert calls > 0, "the Conformance section test was never called"


_DOC_GLOBS = ("*.md", "docs/**/*.md", "mloda/**/*.md")


def _has_python_fence(text: str) -> bool:
    return any(m.group(1) == "python" for m in _FENCE_RE.finditer(text))


def _discover_docs() -> list[str]:
    found = {path for pattern in _DOC_GLOBS for path in REPO_ROOT.glob(pattern) if path.is_file()}
    return sorted(
        path.relative_to(REPO_ROOT).as_posix() for path in found if _has_python_fence(path.read_text(encoding="utf-8"))
    )


DISCOVERED_DOCS = _discover_docs()

# Each doc runs in its own interpreter and must also pass on every supported Python.
SKIPPED_DOCS: dict[str, str] = {
    "docs/guides/feature-group-patterns/28-backend-families.md": (
        "rewrite for source feature groups pending; its examples and Test section are run by the tests in this module"
    ),
    "docs/guides/feature-group-patterns/29-binary-backed-features.md": (
        "needs a master.key; run by tests/test_end2end/test_binary_backed_features_guide.py"
    ),
    "docs/guides/12-export-telemetry.md": "needs the OpenTelemetry OTLP exporter, which is not a dependency",
    "mloda/enterprise/README.md": (
        "needs the PII key environment variable; run up to the license check by "
        "test_enterprise_readme_quick_start_runs_up_to_the_license_or_wheel_check in "
        "tests/test_end2end/test_binary_backed_features_guide.py"
    ),
}

_MARKER_PREFIX = "@@mloda-doc-runner-7f3a@@ "


def _marker_line(kind: str, text: str) -> str:
    return f"{_MARKER_PREFIX}{kind} {text}"


_DOC_CHECK_DRIVER = """\
import inspect
import sys
from pathlib import Path
import mktestdocs
from mloda.user import PluginLoader

_MARKER_PREFIX = "@@mloda-doc-runner-7f3a@@ "


def _marker_line(kind, text):
    return f"{_MARKER_PREFIX}{kind} {text}"


def _run_python(source):
    namespace = {"__name__": "__main__"}
    exec(source, namespace)
    tests = []
    for name, obj in namespace.items():
        if getattr(obj, "__module__", None) != "__main__":
            continue
        if isinstance(obj, type) and name.startswith("Test"):
            if any(m.startswith("test_") for m in dir(obj)):
                _reject(f"{name} is a test class the doc runner cannot call; use test_* functions")
        elif name.startswith("test_") and callable(obj):
            if inspect.iscoroutinefunction(obj) or inspect.isasyncgenfunction(obj) or inspect.isgeneratorfunction(obj):
                _reject(f"{name} is async or a generator, so the doc runner cannot call it")
            params = inspect.signature(obj).parameters.values()
            if any(p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
                _reject(f"{name} needs arguments, so the doc runner cannot call it")
            tests.append((name, obj))
    for name, func in tests:
        print(_marker_line("CALLING", name), flush=True)
        func()


def _reject(message):
    print(_marker_line("REJECTED", message), flush=True)
    sys.exit(1)


mktestdocs.register_executor("python", _run_python)
PluginLoader.all()
for path in sys.argv[1:]:
    print(_marker_line("CHECKING", path), flush=True)
    mktestdocs.check_md_file(fpath=Path(path), memory=True)
print(_marker_line("DONE", ""), flush=True)
""".replace("__MARKER_PREFIX__", repr(_MARKER_PREFIX))


def _display_path(path: str) -> str:
    if Path(path).is_relative_to(REPO_ROOT):
        return Path(path).relative_to(REPO_ROOT).as_posix()
    return path


def _parse_stdout(stdout: str | bytes | None) -> tuple[str, str, str, bool]:
    """Return the last marker kind and text, the last checked file (display form) and whether the run finished."""
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    kind, text, checked, done = "CHECKING", "", "<before the first file>", False
    for line in (stdout or "").splitlines():
        if not line.startswith(_MARKER_PREFIX):
            continue
        kind, _, text = line.removeprefix(_MARKER_PREFIX).partition(" ")
        if kind == "CHECKING":
            checked = _display_path(text)
        elif kind == "DONE":
            done = True
    return kind, text, checked, done


def run_md_files_isolated(paths: Sequence[Path], cwd: Path) -> None:
    """Run the python blocks of *paths* in one fresh interpreter and call their test_* functions."""
    cmd = [sys.executable, "-c", _DOC_CHECK_DRIVER, *map(str, paths)]
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)  # nosec
    except subprocess.TimeoutExpired as exc:
        kind, text, where, _ = _parse_stdout(exc.stdout)
        who = f"{text} " if kind == "CALLING" else ""
        raise AssertionError(f"{where}: {who}timed out after {exc.timeout}s") from exc
    kind, text, where, done = _parse_stdout(result.stdout)
    if result.returncode == 0:
        if done:
            return
        head = f"{where}: exited before the doc runner finished"
    elif kind == "REJECTED":
        head = f"{where}: {text}"
    elif kind == "CALLING":
        head = f"{where}: {text} failed"
    else:
        head = f"{where}: python block failed"
    tail = "\n".join(result.stdout.splitlines()[-50:])
    raise AssertionError(f"{head}\nstdout (tail):\n{tail}\nstderr:\n{result.stderr}")


def _doc_params() -> list[Any]:
    return [
        pytest.param(doc, id=doc, marks=pytest.mark.xfail(reason=SKIPPED_DOCS[doc], strict=True))
        if doc in SKIPPED_DOCS
        else pytest.param(doc, id=doc)
        for doc in DISCOVERED_DOCS
    ]


@pytest.mark.parametrize("doc", _doc_params())
def test_doc_python_blocks_execute(doc: str, tmp_path: Path) -> None:
    run_md_files_isolated([REPO_ROOT / doc], cwd=tmp_path)


def test_skipped_docs_are_discovered_docs_with_reasons() -> None:
    assert set(SKIPPED_DOCS) <= set(DISCOVERED_DOCS), sorted(set(SKIPPED_DOCS) - set(DISCOVERED_DOCS))
    assert all(reason.strip() for reason in SKIPPED_DOCS.values())


_FIXED_TMP = "/tmp/"  # nosec B108


def test_doc_python_blocks_do_not_write_to_fixed_tmp_paths() -> None:
    offenders = [
        doc
        for doc in DISCOVERED_DOCS
        for m in _FENCE_RE.finditer((REPO_ROOT / doc).read_text(encoding="utf-8"))
        if m.group(1) in ("python", "py") and _FIXED_TMP in m.group(2)
    ]
    assert offenders == [], f"use a temp directory instead of a /tmp/ literal in: {sorted(set(offenders))}"


def test_failing_python_block_names_the_file(tmp_path: Path) -> None:
    good = tmp_path / "good.md"
    bad = tmp_path / "bad.md"
    good.write_text("```python\nassert True\n```\n", encoding="utf-8")
    bad.write_text(
        "```py\nraise RuntimeError('not run')\n```\n\n```python\nraise RuntimeError('boom')\n```\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError) as excinfo:
        run_md_files_isolated([good, bad], cwd=tmp_path)
    message = str(excinfo.value)
    assert message.startswith(str(bad)), message
    assert "boom" in message
    assert "not run" not in message


def test_doc_test_functions_are_called(tmp_path: Path) -> None:
    doc = tmp_path / "broken_test.md"
    doc.write_text(
        "```python\ndef test_ok():\n    assert True\n\ndef test_broken():\n    raise RuntimeError('boom')\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError) as excinfo:
        run_md_files_isolated([doc], cwd=tmp_path)
    message = str(excinfo.value)
    assert message.startswith(f"{doc}: test_broken failed"), message
    assert "boom" in message


def test_imported_test_functions_are_not_called(tmp_path: Path) -> None:
    (tmp_path / "helper.py").write_text("def test_imported():\n    raise RuntimeError('imported')\n", encoding="utf-8")
    doc = tmp_path / "imports_test.md"
    doc.write_text(
        "```python\nimport sys\nsys.path.insert(0, '.')\nfrom helper import test_imported\n```\n",
        encoding="utf-8",
    )
    run_md_files_isolated([doc], cwd=tmp_path)


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("def test_needs_arg(x):\n    pass\n", "test_needs_arg"),
        ("class TestThing:\n    def test_method(self):\n        pass\n", "TestThing"),
        (
            "class Base:\n    def test_method(self):\n        pass\n\nclass TestFromMixin(Base):\n    pass\n",
            "TestFromMixin",
        ),
        ("async def test_async():\n    pass\n", "test_async"),
        ("def test_gen():\n    yield\n", "test_gen"),
    ],
)
def test_uncallable_doc_tests_fail_loudly(tmp_path: Path, source: str, name: str) -> None:
    doc = tmp_path / "uncallable.md"
    doc.write_text(f"```python\n{source}```\n", encoding="utf-8")
    with pytest.raises(AssertionError) as excinfo:
        run_md_files_isolated([doc], cwd=tmp_path)
    message = str(excinfo.value)
    assert message.startswith(f"{doc}: "), message
    assert name in message.splitlines()[0], message


def test_doc_output_cannot_fake_a_marker(tmp_path: Path) -> None:
    doc = tmp_path / "fake_marker.md"
    doc.write_text('```python\nprint("CALLING not_me")\nraise RuntimeError("boom")\n```\n', encoding="utf-8")
    with pytest.raises(AssertionError) as excinfo:
        run_md_files_isolated([doc], cwd=tmp_path)
    message = str(excinfo.value)
    assert message.startswith(f"{doc}: python block failed"), message
    assert "not_me" not in message.splitlines()[0], message


def test_doc_cannot_pass_by_exiting_early(tmp_path: Path) -> None:
    doc = tmp_path / "early_exit.md"
    doc.write_text(
        '```python\nimport sys\ndef test_never():\n    raise RuntimeError("x")\nsys.exit(0)\n```\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError) as excinfo:
        run_md_files_isolated([doc], cwd=tmp_path)
    assert str(excinfo.value).startswith(f"{doc}: "), str(excinfo.value)


def test_timeout_in_a_doc_test_names_the_function(tmp_path: Path) -> None:
    doc = tmp_path / "slow.md"
    doc.write_text("```python\ndef test_slow():\n    pass\n```\n", encoding="utf-8")
    output = _marker_line("CHECKING", str(doc)) + "\n" + _marker_line("CALLING", "test_slow") + "\n"
    expired = subprocess.TimeoutExpired(cmd="x", timeout=300, output=output)
    with patch("tests.test_end2end.test_doc_guide_examples.subprocess.run", side_effect=expired):
        with pytest.raises(AssertionError) as excinfo:
            run_md_files_isolated([doc], cwd=tmp_path)
    assert str(excinfo.value).startswith(f"{doc}: test_slow timed out"), str(excinfo.value)
