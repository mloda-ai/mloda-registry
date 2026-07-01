# Pattern 26: Consuming Another Feature Group (Selector Option Forwarding)

A feature group can depend on **another group's** root/source feature by naming
it in `input_features`. When the upstream group has a selector (a discriminator
that decides which backend or variant answers the request), the consuming group
often has to forward that selector to the upstream itself. This page covers the
**plugin-author side** of that hand-off, which the caller-side context rules in
[Options](11-options.md) and the name-convention chains in
[Chained features](03-chained-features.md) do not.

**What**: A feature group that lists another group's feature as a dependency and
forwards the upstream's selector options to it.
**When**: A connector/consumer needs data from a separate source group whose
resolution depends on a selector (backend, corpus, index variant).
**Why**: Parent **group** options auto-merge into a child feature, but parent
**context** options do not. Selectors that callers pass as context must be
forwarded by hand.
**Where**: A retriever consuming a corpus source, a graph connector consuming a
knowledge-graph source, any group that sits on top of a pluggable source.
**How**: In `input_features`, build the child `Feature` and copy the selector
into its options explicitly; protect your own group tuning keys from merging onto
the child.

## The forwarding rule

When your `input_features` returns `{Feature(name, options=...)}`, mloda merges
the consuming feature's options down into that child before the upstream group
resolves it. The merge is asymmetric:

| Parent option kind | Reaches the child automatically? | Notes |
|--------------------|----------------------------------|-------|
| `group` | Yes | Group options merge down (they affect upstream resolution). |
| `context` | No | Context stays local unless `propagate_context_keys` lists the key. |

So a selector that a caller places in `group` reaches the upstream for free. A
selector a caller places in `context` (the common case for RAG-style metadata)
does **not** reach the upstream. The consuming group must read it from its own
options and set it explicitly on the child feature. Read the value with
`options.get(key)` (which searches group then context) and put it in the child's
`group` so it participates in upstream resolution.

> **Why not just tell callers to use group options?** The per-call selector is
> request metadata the caller attaches as context (next to the query text), so
> it stays out of the consumer's own hash and batching. Forwarding lets the
> caller keep it as context while still steering the upstream. See
> [Options: Group vs Context](11-options.md#group-vs-context).

## Keeping your own keys off the child

The concern is the mirror image: the consumer's own **group** keys *do*
auto-merge onto the child, so a group-level tuning key (say `top_k`) would land
on the corpus source and pollute its option surface for other, group-style
callers. Two protections keep such parent-only group keys off the child:

- `in_features` is **always** protected and never merged down.
- Any key listed in the child feature's `feature_chainer_parser_key` option is
  treated as merge-protected, so the parent keeps its value and the child does
  not inherit it.

Note the direction: the protected-keys list is read from the **child** feature
you build in `input_features` (it is the `self` side of the merge), not from the
consumer. Context keys never need protecting because context does not merge in
the first place. This merge-protected-keys mechanism is the current way to keep
group keys local; the core discussion is in
[mloda-ai/mloda#542](https://github.com/mloda-ai/mloda/issues/542).

## Complete Example

A connector that answers over a pluggable **corpus source**. The caller attaches
the upstream backend selector `corpus_backend` and the `query_text` as context
(request metadata), and keeps `top_k` as a group tuning key (it changes the
consumer's output). The connector forwards `corpus_backend` to the corpus source
(context does not auto-merge) and protects `top_k` so that group key does not
leak onto the corpus source.

```python
from typing import Any
from mloda.provider import DefaultOptionKeys, FeatureGroup, FeatureSet
from mloda.user import Feature, Options, FeatureName


class CorpusConnector(FeatureGroup):
    """Retrieve over a corpus source, forwarding the backend selector upstream."""

    QUERY_TEXT = "query_text"
    TOP_K = "top_k"
    CORPUS_BACKEND = "corpus_backend"  # selector for the upstream corpus source
    CORPUS_FEATURE = "corpus"          # the upstream group's root feature

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        child_options = Options()

        # Forward the upstream selector: the caller passed it as context, which
        # does not auto-merge, so set it on the child's group by hand.
        backend = options.get(self.CORPUS_BACKEND)
        if backend is not None:
            child_options.add_to_group(self.CORPUS_BACKEND, backend)

        # Protect our own group tuning key so it does not merge onto the corpus
        # source. The list is read from this child feature (the merge's `self`).
        child_options.add_to_group(
            DefaultOptionKeys.feature_chainer_parser_key,
            frozenset({self.TOP_K}),
        )

        return {Feature(self.CORPUS_FEATURE, child_options)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            options = feature.options
            query = options.get(cls.QUERY_TEXT)
            top_k = int(options.get(cls.TOP_K) or 5)
            # data[cls.CORPUS_FEATURE] holds the upstream corpus rows.
            corpus = data[cls.CORPUS_FEATURE]
            return {"answer": [cls._retrieve(corpus, query, top_k)]}
        return {"answer": []}

    @classmethod
    def _retrieve(cls, corpus: Any, query: Any, top_k: int) -> Any:
        raise NotImplementedError
```

The caller keeps the selector and query as context and only `top_k` is a group
option:

```python
Feature(
    "CorpusConnector",
    Options(
        group={"top_k": 3},  # tuning key: stays off the corpus source via protection
        context={"corpus_backend": "faiss", "query_text": "who wrote it?"},  # forwarded / local
    ),
)
```

## Test

This drives the same merge the engine runs (`update_with_protected_keys`) so the
forwarding and protection are actually exercised, not assumed.

```python
def test_forwarding_and_protection():
    parent = Options(
        group={"top_k": 3},  # would auto-merge onto the child if unprotected
        context={"corpus_backend": "faiss", "query_text": "q"},  # context: does not merge
    )
    child = next(iter(CorpusConnector().input_features(parent, FeatureName("CorpusConnector"))))

    # The connector forwarded the context selector into the child's group.
    assert child.options.get("corpus_backend") == "faiss"

    # Simulate the framework merging the consumer's options down into the source.
    child.options.update_with_protected_keys(parent)

    assert child.options.get("corpus_backend") == "faiss"  # forwarded value survives
    assert child.options.get("top_k") is None              # protected group key stayed off
    assert child.options.get("query_text") is None         # context never propagates
```

## Real Implementations

The pattern surfaced in a graph_rag connector consuming a `knowledge_graph`
source feature and forwarding its graph backend selector. The core-side
discussion of the merge-protected-keys mechanism, and why context does not
propagate to input features, lives in
[mloda-ai/mloda#542](https://github.com/mloda-ai/mloda/issues/542).

## Combines With

- **Pattern 1 (Root features)**: The upstream you consume is typically a root/source feature.
- **Pattern 4 (Multi-input)**: Consume several upstream sources at once.
- **Pattern 11 (Options)**: Group vs context and `propagate_context_keys` decide what flows without manual forwarding.
