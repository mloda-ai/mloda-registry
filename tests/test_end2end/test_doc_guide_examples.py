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
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.steward import resolve_feature
from mloda.user import DataAccessCollection, Options, PluginCollector, load_features_from_config
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

import mloda.community.feature_groups.data_operations as data_operations
from tests.test_end2end.test_binary_backed_features_guide import _section

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_CONFIG_GUIDE = REPO_ROOT / "docs" / "guides" / "feature-group-patterns" / "22-feature-config.md"
BACKEND_FAMILIES_GUIDE = REPO_ROOT / "docs" / "guides" / "feature-group-patterns" / "28-backend-families.md"
WAY_2_HEADING = "## Way 2: Discriminator Option (Sibling FeatureGroups)"
WAY_2_MODULE = "guide_28_way_2"
WAY_3_HEADING = "## Way 3: Reader Delegation (Sibling Readers)"
WAY_3_MODULE = "guide_28_way_3"

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


def test_wrong_suffix_regression_is_rejected() -> None:
    """Historical bug: ``_aggr`` never matched ``AggregationFeatureGroup.PREFIX_PATTERN`` (``_agg``)."""
    block = [
        {
            "name": "sales__sum_aggr",
            "feature_group": "PandasAggregation",
            "context_options": {"partition_by": ["region"]},
        }
    ]
    registry = build_feature_group_registry()
    problems = find_unresolved_features(block, registry, "synthetic[wrong_suffix]")
    assert problems != [], "wrong-suffix regression was not caught; the validation logic is vacuous"
    assert "sales__sum_aggr" in problems[0]
    assert "PandasAggregation" in problems[0]


def test_missing_partition_by_regression_is_rejected() -> None:
    """Historical bug: no ``partition_by`` means AggregationFeatureGroup's required option is absent."""
    block = [{"name": "sales__sum_agg", "feature_group": "PandasAggregation"}]
    registry = build_feature_group_registry()
    problems = find_unresolved_features(block, registry, "synthetic[missing_partition_by]")
    assert problems != [], "missing-partition_by regression was not caught; the validation logic is vacuous"
    assert "sales__sum_agg" in problems[0]
    assert "PandasAggregation" in problems[0]


def _python_fences(heading: str) -> list[str]:
    section = _section(BACKEND_FAMILIES_GUIDE.read_text(encoding="utf-8"), heading, ("\n## ",))
    return [m.group(2) for m in _FENCE_RE.finditer(section) if m.group(1) == "python"]


@lru_cache(maxsize=None)
def _load_namespace(heading: str, module: str) -> dict[str, Any]:
    """Run the first python block of the guide section under *heading* once and return its namespace."""
    fences = _python_fences(heading)
    assert fences, f"no python block found in the {heading!r} section of the guide"
    code = compile(fences[0], f"<{BACKEND_FAMILIES_GUIDE.name} {module}>", "exec", dont_inherit=True)
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
    mloda = Mock()
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
    collection = DataAccessCollection(credentials=[{"oxigraph_sparql": {"locator": "graph.ttl"}}])
    _, result = _resolve_way_3(feature, collection, frameworks)
    assert result.feature_group is None
    assert result.candidates == []
    assert "No feature groups found" in str(result.error)
