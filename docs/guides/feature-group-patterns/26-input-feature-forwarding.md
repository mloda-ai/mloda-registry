# Pattern 26: Consuming Another Group's Root Feature (Input-Feature Option Forwarding)

When a feature group declares **another group's** feature in `input_features()`, the consumer hands its own group options down to that upstream feature. This page covers how that forwarding works and how to keep a consumer's query-specific options (a selector, a `top_k`, a local threshold) off an upstream whose option surface should stay clean.

**What**: Controlling which of a consumer's options flow to an input feature it does not own.
**When**: A feature group consumes a source/corpus/connector feature published by a different group (e.g. a RAG connector consuming a `knowledge_graph` source).
**Why**: Options set on the consumer forward to input features by default; an upstream that does not recognize a forwarded key rejects the feature.
**Where**: `input_features()` return values, on the `Feature(...)` objects you construct there.
**How**: Leave children at the default to inherit everything, or use the typed opt-outs `forward_group`, `forward_group_exclude`, and the context pull/push pair `inherit_context_keys` / `propagate_context_keys`.

> Requires `mloda>=0.9.0`. Earlier releases used a `feature_chainer_parser_key` denylist shield that is removed in 0.9.0; do not use it. The API below is the replacement.

## The Default: Group Options Forward

A consumer's **group** options flow onto every input feature by default. **Context** options never flow through this forwarding merge (the one exception is a bare-string child, which shares the consumer's `Options` outright: see the caveat below). `in_features` itself never flows.

| Consumer option category | Flows to input features by default? | How to change it |
|--------------------------|-------------------------------------|------------------|
| `group` | Yes (all keys) | `forward_group`, `forward_group_exclude` on the child |
| `context` | No | `inherit_context_keys` (child pull) or `propagate_context_keys` (consumer push) |
| `in_features` | Never | not configurable |

Forward-by-default means a consumer configured once at the top ("use backend X") transparently configures the upstream too. The cost: a consumer-local key you never meant for the upstream also forwards, and if the upstream does not accept it, resolution fails with:

```text
Feature group(s) [...] match the name 'knowledge_graph' but reject it because of
extra group option(s) {'top_k'}. Group options flow onto input features from the
consumer by default; ... Keep them off 'knowledge_graph' by setting
forward_group_exclude={...}, an allowlist, or forward_group=False on the child in
the consumer's input_features.
```

That is the signal to carve the key out.

## The Directives

Set these on the `Feature(...)` you return from `input_features()`:

| Directive | Effect |
|-----------|--------|
| `forward_group=None` (default) | Inherit **all** consumer group keys. Leave children here unless you have a reason not to. |
| `forward_group=True` | Same as the default. Do **not** stamp this on children (see below). |
| `forward_group=False` | Inherit **nothing**. Isolates the child from the consumer's group options. |
| `forward_group={"a", "b"}` | Allowlist: inherit only these group keys. |
| `forward_group_exclude={"top_k"}` | Forward everything **except** these keys. The fit for a consumer-local selector. |
| `inherit_context_keys={"tenant"}` | Pull these **context** keys from the consumer into the child's context. |

`forward_group=False` combined with a non-empty `forward_group_exclude` is contradictory and raises `ValueError`.

## Worked Example: A Connector Consuming a Source Feature

`GraphRagConnector` runs a query against a knowledge graph. It consumes the `knowledge_graph` root feature published by a different group. The connector accepts a `backend` selector (which the upstream also understands, so it *should* forward) and a `top_k` (purely consumer-local, which the upstream rejects, so it must **not** forward).

```python
from typing import Any, Optional

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options


class GraphRagConnector(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        # `backend` forwards by default (the upstream understands it).
        # `top_k` is consumer-local; carve it off so the upstream is not asked to match it.
        # `tenant` lives in context, so it does not flow implicitly: pull it explicitly.
        return {
            Feature(
                "knowledge_graph",
                forward_group_exclude={"top_k"},
                inherit_context_keys={"tenant"},
            )
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        ...
```

Caller side, the consumer is configured once:

```python
Feature(
    "graph_rag_answer",
    Options(
        group={"backend": "neo4j", "top_k": 5},
        context={"tenant": "acme"},
    ),
)
```

`backend="neo4j"` reaches `knowledge_graph`; `top_k=5` does not; `tenant="acme"` reaches it because the child pulled it. If the consumer would rather **push** context down (so every input feature in the chain sees it without each one opting in), it sets `propagate_context_keys` on its own `Options` instead:

```python
Options(context={"tenant": "acme"}, propagate_context_keys=frozenset({"tenant"}))
```

`inherit_context_keys` (child pull) and `propagate_context_keys` (consumer push) are symmetric; use whichever side owns the decision. See [Options: Context Propagation](11-options.md#context-propagation).

## Leave Children at the Default

The engine forwards by default, so a child left at `forward_group=None` already inherits everything. **Do not** stamp `forward_group=True` on children to "enable" forwarding: it is redundant, and it destroys the `None` sentinel that distinguishes "author said nothing" from "author made a choice". Only set `forward_group` / `forward_group_exclude` when you are actively opting a key **out**.

## Conflicts Raise, Equal Values Are a No-Op

If a forwarded group key already exists on the child with a **different** value, forwarding raises `ValueError` (naming the key, both values, and the opt-out remedies). If the child already holds the **same** value, forwarding is a silent no-op. So a child may safely pre-set a key to the value the consumer would forward; it may not silently override it.

## Caveat: Bare-String Children Share the Consumer's Options

The directives above only take effect on children returned as explicit `Feature(...)` objects. A child returned as a **bare string** from `input_features()` (e.g. `return {"knowledge_graph"}`) is constructed with the consumer's own `Options` instance, so it already carries the consumer's **entire** group and context, and the forwarding directives (`forward_group`, `forward_group_exclude`, `inherit_context_keys`) are no-ops on it (mloda logs a warning if you pass them). Because everything is already shared, there is no way to carve a consumer-local key off a bare-string child. To control what the upstream sees, whether that means excluding a key, isolating the child, or selecting which context reaches it, return an explicit `Feature("knowledge_graph", forward_group_exclude={...})`, not the string.

## Test

```python
from mloda.user import FeatureName, Options


def test_connector_carves_local_key_and_pulls_context():
    child = next(iter(GraphRagConnector().input_features(Options(), FeatureName("graph_rag_answer"))))
    assert str(child.name) == "knowledge_graph"
    assert child.forward_group_exclude == frozenset({"top_k"})
    assert child.inherit_context_keys == frozenset({"tenant"})
    # Children are left at the default sentinel, not stamped True.
    assert child.forward_group is None
```

## Real Implementations

| File | Description |
|------|-------------|
| [test_input_features_forward_group_defaults.py](https://github.com/mloda-ai/mloda/blob/0.9.0/tests/test_plugins/feature_group/experimental/test_input_features_forward_group_defaults.py) | Plugin `input_features()` leaving children at the default sentinel; explicit directives preserved |
| [test_feature_collection_forwarding.py](https://github.com/mloda-ai/mloda/blob/0.9.0/tests/test_core/test_abstract_plugins/test_components/test_feature_collection_forwarding.py) | `forward_group` / `forward_group_exclude` allowlist and carve-out semantics |
| [Feature Configuration](https://mloda-ai.github.io/mloda/in_depth/feature-config/) | Group vs context, propagation |

## Combines With

- **Pattern 1 (Root features)**: the upstream you consume is typically a root/source feature group.
- **Pattern 8 (Links joins)**: an alternative when the two groups relate by a join key rather than a direct input dependency.
- **[Options](11-options.md)**: group vs context, and `propagate_context_keys`.
