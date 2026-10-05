# Backend Families: Picking One Sibling

How to ship several interchangeable implementations of one feature and let exactly one of them claim each request.

**What**: Three shapes for a family of sibling FeatureGroups (or readers) where one sibling must win.
**When**: One feature, many backends (BM25 vs. FAISS retrieval, rdflib vs. Oxigraph SPARQL), or one operation with many methods.
**Why**: mloda asks every FeatureGroup whether it matches. Zero matches fails with `No feature groups found`, more than one with `Multiple feature groups found`. The family has to make the answer unique.
**Where**: Connector plugins, e.g. [rag_integration](https://github.com/mloda-ai/rag_integration) (way 2) and [open-kgo](https://github.com/mloda-ai/open-kgo) (way 3).

## Pick a Shape

| | Way 1: name-encoded | Way 2: discriminator option | Way 3: reader delegation |
|---|---|---|---|
| Variation lives in | the feature name | sibling FeatureGroups | sibling readers |
| Selector | `PREFIX_PATTERN` capture, or the same key in options | `options[<discriminator>]` | the data access (credential slot) the reader accepts |
| mloda calls | mixin matcher + `FeatureChainParser` | your `match_feature_group_criteria` | `input_data().matches(...)` |
| Base kept inert by | abstract base | empty `BACKENDS` dict | empty `CONNECTOR_ID` |
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
from mloda.user import DataAccessCollection, FeatureName, Options


class BaseRetriever(FeatureGroup):
    ROOT_FEATURE_NAME = "retrieved_passages"
    BACKEND = "retrieve_backend"
    BACKENDS: ClassVar[dict[str, str]] = {}  # empty: the base never matches

    PROPERTY_MAPPING = {
        BACKEND: property_spec("Retrieval backend", context=False),
        "query_text": property_spec("Query", context=False),
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
- **Know what `PROPERTY_MAPPING` still does.** An override bypasses the parser, so `strict` / `allowed_values` / `element_validator` never run: `retrieve_backend="fiass"` ends in a bare `No feature groups found`. Defaults, `required_when`, and option attribution still apply (see the note in [Custom Matching Override](14-feature-matching.md#custom-matching-override)).
- **Record why a sibling declined.** End the matcher like this so the error lists each sibling's reason (a matching sibling hides them). `NAME_STAGE` and `record_match_rejection` come from `mloda.provider` (`mloda>=0.14.0`):

```python
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

The FeatureGroup never decides. Two parallel trees pair 1:1: each leaf FeatureGroup returns its own reader from `input_data()`, and the reader claims the request by the data access it accepts. No `match_feature_group_criteria` override anywhere.

```python
from typing import Any, ClassVar

from mloda.provider import BaseInputData, ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import DataAccessCollection, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class KgReader(BaseInputData):
    CONNECTOR_ID: ClassVar[str] = ""  # empty: the base never matches

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if not cls.CONNECTOR_ID or not isinstance(data_access, DataAccessCollection):
            return None
        return data_access.resolve(
            "credentials",
            predicate=lambda creds: isinstance(creds.get(cls.CONNECTOR_ID), dict),
            hint=options.get("data_access_handle"),
        )


class RdfLibSparqlReader(KgReader):
    CONNECTOR_ID = "rdflib_sparql"

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any: ...  # data_access[CONNECTOR_ID] is the slot


class KgFeatureGroup(FeatureGroup):
    READER_CLASS: ClassVar[type[KgReader] | None] = None

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return cls.READER_CLASS() if cls.READER_CLASS else None

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return cls.READER_CLASS().load(features)  # type: ignore[union-attr]


class RdfLibSparqlFeatureGroup(KgFeatureGroup):
    READER_CLASS = RdfLibSparqlReader
```

```python
mloda.run_all(
    [Feature("knows", options=Options(context={"query_text": "SELECT ..."}))],
    compute_frameworks=[PythonDictFramework],
    data_access_collection=DataAccessCollection(credentials=[{"rdflib_sparql": {"locator": "graph.ttl"}}]),
)
```

Rules:

- **A FeatureGroup only probes its own reader subtree.** `input_data()` returning `RdfLibSparqlReader()` limits the probe to that class and its final subclasses, so a leaf never competes with its siblings. Keep the family base's `READER_CLASS` at `None`: a base returning the family reader probes every leaf and takes whichever accepts first.
- **Derive the reader family from `BaseInputData`, not `ReadDB` or `ReadFile`.** The stock `ReadDBFeature` probes every `ReadDB` subclass, so a `ReadDB`-based family also matches it and fails with `Multiple feature groups found` unless the user disables `ReadDBFeature` or scopes with `feature_group=`. Same rule as [Your Own Root FeatureGroup](27-input-data-readers.md#your-own-root-featuregroup).
- **The feature name is free.** Nothing parses it, so decline names the reader cannot serve ([Decline Names You Cannot Confirm](27-input-data-readers.md#decline-names-you-cannot-confirm)).
- **Pick one route per family.** The reader above serves the credential route only. The class-name option key of [Input-data readers](27-input-data-readers.md) hands the reader the option value instead of the collection; this reader declines it, and resolution falls through to whichever sibling owns a slot in the credentials.

Reference: open-kgo's nine connector families, e.g. [`kg/rdf/rdflib_sparql.py`](https://github.com/mloda-ai/open-kgo/blob/main/open_kgo/feature_groups/kg/rdf/rdflib_sparql.py). open-kgo builds its readers on `ReadDB`, so it resolves cleanly only while `ReadDBFeature` stays unloaded (importing `read_db` does not load it; `PluginLoader.all()` does).

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
- Way 3: run with `ReadDBFeature` / `ReadFileFeature` enabled if your readers derive from a stock base.

Use `resolve_feature` to see the candidates ([Debugging Matching](14-feature-matching.md#debugging-matching)).

## Combines With

- **Pattern 3 (Chained features)**: way 1
- **Pattern 14 (Feature matching)**: the match loop and override caveats behind way 2
- **Pattern 26 (Input-feature forwarding)**: chaining family members
- **Pattern 27 (Input-data readers)**: reader subtrees, class-name selection, and name declines behind way 3
