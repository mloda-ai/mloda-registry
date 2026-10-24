# Backend Families: Picking One Sibling

How to ship several interchangeable implementations of one feature and let exactly one of them claim each request.

**What**: Three shapes for a family of sibling FeatureGroups (or source groups) where one sibling must win.
**When**: One feature, many backends (BM25 vs. FAISS retrieval, rdflib vs. Oxigraph SPARQL), or one operation with many methods.
**Why**: mloda asks every FeatureGroup whether it matches. Zero matches fails with `No feature groups found`, more than one with `Multiple feature groups found`. The family has to make the answer unique.
**Where**: Connector plugins, e.g. [rag_integration](https://github.com/mloda-ai/rag_integration) (way 2) and [open-kgo](https://github.com/mloda-ai/open-kgo) (way 3).

## Pick a Shape

| | Way 1: name-encoded | Way 2: discriminator option | Way 3: reader delegation |
|---|---|---|---|
| Variation lives in | the feature name | sibling FeatureGroups | sibling source groups |
| Selector | `PREFIX_PATTERN` capture, or the same key in options | `options[<discriminator>]` | the credential slot the source group finds |
| mloda calls | mixin matcher + `FeatureChainParser` | your `match_feature_group_criteria` | `find_sources` per claim route |
| Base kept inert by | abstract base | empty `BACKENDS` dict | abstract base, empty `CONNECTOR_ID` |
| Discriminator value validated | yes (`allowed_values`, `element_validator`) | no, an unknown value matches nothing | no, an unknown slot matches nothing |

Choose by who decides:

```text
Does the user name the variant in the feature name (income__median_imputed)?
    YES → Way 1
Is the variant a free choice with the same inputs and output (retrieve_backend="faiss")?
    YES → Way 2
Is the variant fixed by the data source the user connects (rdflib_sparql credentials)?
    YES → Way 3
```

## Way 1: Name-Encoded (FeatureChainParserMixin)

One class, many names. The method is captured from the name or read from options, and `PROPERTY_MAPPING` validates it:

```python
from mloda.provider import DefaultOptionKeys, FeatureChainParserMixin, FeatureGroup, property_spec


class MeanImputedFeature(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__(?P<imputation_method>[\w]+)_imputed$"
    PROPERTY_MAPPING = {
        "imputation_method": property_spec("Method", strict=True, allowed_values={"mean": "Mean", "median": "Median"}),
        DefaultOptionKeys.in_features: property_spec("Source feature"),
    }
```

This is the only shape where `PROPERTY_MAPPING` rejects a bad value with a reason in the error. Full pattern: [Chained Features](03-chained-features.md), [Discriminator Keys](14-feature-matching.md#discriminator-keys-for-configuration-based-matching).

## Way 2: Discriminator Option (Sibling FeatureGroups)

All siblings share one feature name. A discriminator option picks the sibling; each sibling claims a disjoint set of values:

```python
from typing import Any, ClassVar

from mloda.provider import DataCreator, FeatureGroup, FeatureSet, property_spec
from mloda.user import DataAccessCollection, Feature, FeatureName, Options


class BaseRetriever(FeatureGroup):
    ROOT_FEATURE_NAME = "retrieved_passages"
    BACKEND = "retrieve_backend"
    BACKENDS: ClassVar[dict[str, str]] = {}  # empty: the base never matches

    PROPERTY_MAPPING = {
        BACKEND: property_spec("Retrieval backend", context=False),
        "query_text": property_spec("Query", context=False, default=None),
    }

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({cls.ROOT_FEATURE_NAME})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) != cls.ROOT_FEATURE_NAME:
            return False
        return options.get(cls.BACKEND) in cls.BACKENDS


class Bm25Retriever(BaseRetriever):
    BACKENDS = {"bm25": "BM25 lexical ranking"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any: ...  # rank, return [{name: passages}]


class FaissRetriever(BaseRetriever):
    BACKENDS = {"faiss": "Dense FAISS ranking"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any: ...
```

```python
Feature("retrieved_passages", options=Options(group={"retrieve_backend": "faiss", "query_text": "cat"}))
```

Rules:

- **Keep the value sets disjoint.** Two siblings claiming `"faiss"` fail with `Multiple feature groups found`. Pin this with a test that enables every sibling at once.
- **Only the discriminator stays required.** Since mloda 0.15.0 a key without a default must be present for a sibling to match, through the overridden matcher too: omitting it matches no sibling, and the error names the absent key. Declare every other key with `default=None` ([Options](11-options.md#validation-and-conditional-requirements)).
- **Know what `PROPERTY_MAPPING` still does.** An override replaces the matcher, so `strict` / `allowed_values` / `element_validator` and `match_guard` only run if it calls `super().match_feature_group_criteria()` or `cls.passes_option_declarations(options)` (mloda 0.15.1+): here `retrieve_backend="fiass"` ends in a bare `No feature groups found`. Required presence, defaults, `required_when`, and option attribution still apply (see the note in [Custom Matching Override](14-feature-matching.md#custom-matching-override)).
- **Record why a sibling declined.** End the matcher like this so the error lists each sibling's reason (a matching sibling hides them). `NAME_STAGE` and `record_match_rejection` come from `mloda.provider` (`mloda>=0.14.0`):

```py
backend = options.get(cls.BACKEND)
if backend in cls.BACKENDS:
    return True
if cls.BACKENDS:  # the inert base stays silent
    reason = f"{cls.BACKEND} {backend!r} not one of: {', '.join(cls.BACKENDS)}"
    record_match_rejection(cls.__name__, reason, stage=NAME_STAGE)
return False
```

```text
  - Bm25Retriever (feature name): retrieve_backend 'fiass' not one of: bm25
  - FaissRetriever (feature name): retrieve_backend 'fiass' not one of: faiss
```

- **Declare the discriminator as group** (`context=False`) and pass it in `group`: it changes which group computes the feature, so it is part of the feature's identity ([Options](11-options.md)).

Reference: rag_integration's seven connector families (retrieve, rerank, generate, graph_rag, knowledge_graph, structured, orchestrator), e.g. [`connectors/retrieve/base.py`](https://github.com/mloda-ai/rag_integration/blob/main/rag_integration/feature_groups/connectors/retrieve/base.py).

## Way 3: Reader Delegation (Sibling Readers)

The siblings are source FeatureGroups (`SourceFeatureGroup` subclasses) that claim a request by the credential slot the `DataAccessCollection` holds. The base owns matching and calls `find_sources` on each claim route; override `match_feature_group_criteria` only to narrow a declared prefix (below). An abstract family base declines; each concrete leaf matches itself.

```python
from typing import Any, ClassVar

from mloda.provider import ClaimRoute, ComputeFramework, FeatureSet, NamePolicy, SourceFeatureGroup, SourceMatch
from mloda.user import DataAccessCollection, Feature, Options, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class KgSourceFG(SourceFeatureGroup):
    CONNECTOR_ID: ClassVar[str] = ""  # empty: the base never matches
    # Pointed-only (searched=False) unless query_text is present, which unlocks the route.
    CLAIM_ROUTES = (ClaimRoute("credentials", NamePolicy.OPEN, False, ("query_text",)),)

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        if not cls.CONNECTOR_ID or data_access_collection is None:
            return []
        slot = data_access_collection.resolve(
            "credentials",
            predicate=lambda creds: isinstance(creds.get(cls.CONNECTOR_ID), dict),
            hint=options.get("data_access_handle"),
        )
        # source is the credential-free label; the access travels in SourceMatch.access
        return [] if slot is None else [SourceMatch(source=cls.CONNECTOR_ID, access=slot[cls.CONNECTOR_ID])]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}


class RdfLibSparqlFeatureGroup(KgSourceFG):
    CONNECTOR_ID = "rdflib_sparql"

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: FeatureSet) -> Any:
        return [{"knows": "alice"}]  # a real leaf queries the graph in match.access
```

```python
mloda.run_all(
    [Feature("knows", options=Options(context={"query_text": "SELECT ..."}))],
    compute_frameworks=[PythonDictFramework],
    data_access_collection=DataAccessCollection(credentials=[{"rdflib_sparql": {"locator": "graph.ttl"}}]),
)
```

Rules:

- **Leaves never compete.** `KgSourceFG` is abstract (it leaves `load_neutral` unimplemented), so it declines. Each leaf finds only its own `CONNECTOR_ID` slot; two leaves whose slots are both in the collection are ordinary feature-group ambiguity, resolved by `data_access_handle` or `feature_group=`.
- **Choose the claim route deliberately.** An `OPEN` route must be `searched=False` (core raises `TypeError` otherwise): it claims any name only when the user points at the group (`options={"RdfLibSparqlFeatureGroup": ...}` or `feature_group=`) or when a `required_options` key such as `query_text` is present, as above. To claim by name instead, use `NamePolicy.CHECKED` with `columns`/`has_column`, or `NamePolicy.DECLARED` with `declared_names()` or `declared_name_prefixes()`.
- **A declared prefix also claims chains on it.** `declared_name_prefixes()` returning `("rdflib_sparql__",)` claims `rdflib_sparql__x__sum_aggr` too. Until core handles it, override `match_feature_group_criteria` to decline names with a further `__` before calling `super()`.
- **Database-shaped families.** Derive from `ReadDBFG` when the source is a table catalog: it brings credential discovery, per-run catalog caching, `login_key` and the optional `ReadDBFG.QUERY_ROUTE`. Implement `is_valid_credentials` (the slot check), `database_identity`, `connect`, `list_tables`, `table_columns` and `produce_rows`.
- **Decline what you cannot serve.** Ownership established but content failing is a recorded decline ([Decline Names You Cannot Confirm](27-input-data-readers.md#decline-names-you-cannot-confirm)); a plain non-match stays silent.
- **Pick one route per family.** A class-name pointer (`options={"RdfLibSparqlFeatureGroup": value}`) is final against other groups and the collection, and the leaf above has no route for it without `query_text`: it declines with `was pointed at but found no source`. Decide whether the family serves pointers, credentials or both, and test that route.
- **Extra matcher I/O goes through `gated_probe`.** If a leaf overrides `match_feature_group_criteria` to do I/O, call `super()` and wrap the I/O in `cls.gated_probe(feature_name, options, data_access_collection, probe)` so an `INPUT_DATA_MATCH` gate can refuse it ([Reader match hook](../11-create-extender.md#reader-match-hook)).

Reference: open-kgo's [`kg/feature_group.py`](https://github.com/mloda-ai/open-kgo/blob/main/open_kgo/feature_groups/kg/feature_group.py) and its nine connector families, e.g. [`kg/rdf/rdflib_sparql.py`](https://github.com/mloda-ai/open-kgo/blob/main/open_kgo/feature_groups/kg/rdf/rdflib_sparql.py):

- The family base stays abstract through an abstract `READER_CLASS`; each concrete only pins `READER_CLASS`.
- One `NamePolicy.DECLARED` route with `declared_name_prefixes()` (`<CONNECTOR_ID>__`), plus the chain-declining override above.
- `query_text` is a required option only on query families (rdf, network_pg, agent_memory).
- `find_sources` narrows the collection by `data_access_handle`; two credentials for one source get `@handle` sources, so they abort as ambiguous instead of picking the first.

## Keep the Advertised Surface Honest

In ways 2 and 3 nothing validates `PROPERTY_MAPPING` values against what the backend actually reads, so a sibling can advertise a key it silently ignores. Put the guarantee back under test, as open-kgo does in [`kg/tests/contract_surface.py`](https://github.com/mloda-ai/open-kgo/blob/main/open_kgo/feature_groups/kg/tests/contract_surface.py):

- `test_strict_enum_honored_or_waived`: every strict key is either narrowed to the values this backend supports or explicitly waived.
- `test_no_unconsumed_advertised_keys`: every other advertised key is read by the reader or explicitly waived.

Run both over every concrete sibling. A new backend that forgets to trim its surface then fails CI instead of misleading callers. The rule is stated in open-kgo's [`kg/reader_base.py`](https://github.com/mloda-ai/open-kgo/blob/main/open_kgo/feature_groups/kg/reader_base.py) module docstring.

## Chaining a Family Member onto an Upstream Family

A sibling that consumes another family's feature (graph RAG over a `knowledge_graph` source) takes the upstream's name as an option and forwards everything except its own keys:

```python
FAMILY_OPTION_KEYS = frozenset({"graph_backend", "graph_source", "query_text", "top_k"})


def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
    source = options.get("graph_source")
    if source is None:
        return None
    return {Feature(str(source), forward_group_exclude=FAMILY_OPTION_KEYS)}
```

The upstream then selects its own sibling from the forwarded discriminator (`kg_backend`). Context keys do not forward by default. See [Input-feature forwarding](26-input-feature-forwarding.md).

## Test

- Enable every sibling at once and resolve each value: exactly one match per value.
- An unknown value or slot resolves to nothing.
- Way 3: enable only your leaves with `PluginCollector.enabled_feature_groups`, so stock source groups such as `CsvFG` do not claim the same request.

Resolve each value with `resolve_feature` instead of running it ([Debugging Matching](14-feature-matching.md#debugging-matching)). For the way 2 siblings:

```python
from mloda.steward import resolve_feature
from mloda.user import Feature, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


def test_faiss_resolves_to_exactly_one_sibling() -> None:
    result = resolve_feature(
        Feature("retrieved_passages", options=Options(group={"retrieve_backend": "faiss"})),
        plugin_collector=PluginCollector.enabled_feature_groups({Bm25Retriever, FaissRetriever}),
        compute_frameworks={PythonDictFramework},
    )
    assert result.feature_group is FaissRetriever, result.error
    assert result.candidates == [FaissRetriever]
```

Pass both arguments: the pinned `plugin_collector` keeps other loaded plugins out of the result, and `compute_frameworks` keeps it independent of which compute framework plugins the process has imported (with none loaded, every matching sibling is reported as eliminated: `none of its compute frameworks are enabled for this run`).

## Combines With

- **Pattern 3 (Chained features)**: way 1
- **Pattern 14 (Feature matching)**: the match loop and override caveats behind way 2
- **Pattern 26 (Input-feature forwarding)**: chaining family members
- **Pattern 27 (Source FeatureGroups)**: pointers, claim routes and name declines behind way 3
