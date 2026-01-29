# Feature Matching

How mloda determines which FeatureGroup handles a requested feature name.

**What**: The process of finding which FeatureGroup handles a requested feature name.
**When**: Every feature request triggers matching to find the responsible FeatureGroup.
**Why**: Exactly one FeatureGroup must handle each feature name; matching resolves this.
**Where**: `match_feature_group_criteria()` method, checked against all registered FeatureGroups.

## How It Works

When you request a feature (e.g., `Feature.not_typed("price__scaled")`), mloda checks each FeatureGroup's `match_feature_group_criteria()` method. Exactly one must return `True`.

---

## Default Matching Priority

The default `match_feature_group_criteria()` checks in order:

```
1. Input data match     → Root features with matching input_data()
2. Data access match    → MatchData mixin with data connection
3. Exact class name     → "MyFeature" == class MyFeature
4. Prefix match         → "MyFeature_x".startswith("MyFeature_")
5. Explicit names       → name in feature_names_supported()
```

First `True` wins. If FeatureChainParserMixin is used, pattern matching is also applied.

---

## Custom Matching Override

Override `match_feature_group_criteria()` for custom logic:

```python
@classmethod
def match_feature_group_criteria(
    cls,
    feature_name: str,
    options: Options,
    data_access_collection: Optional[DataAccessCollection] = None,
) -> bool:
    return feature_name.endswith("_score")
```

**Note:** This only controls MATCHING. It doesn't define discoverable names - users must know to request matching names.

---

## Matching vs Naming

| Concept | Method | Purpose |
|---------|--------|---------|
| **Naming** | `feature_names_supported()`, class name | Define what names exist (discoverable) |
| **Matching** | `match_feature_group_criteria()` | Check if a requested name is handled |

See [Feature Naming](13-feature-naming.md) for defining names.
