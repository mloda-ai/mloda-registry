# Pattern 8: Links and Joins

Join feature groups to combine data from different sources.

**What**: Features that join data from multiple feature groups.
**When**: Combining data from different sources, cross-source feature combinations, self-joins.
**Why**: Real-world features often require data from multiple tables/sources.
**Where**: Order + Customer joins, user + activity joins, enrichment lookups.
**How**: Attach `Link` to features via `input_features()`, Feature parameter, or `mloda.run_all()`.

## Key Characteristic

| Aspect | Value |
|--------|-------|
| Class | `Link` for defining joins |
| Class | `JoinSpec` for specifying join sides |
| Usage | Attach to Feature objects or pass to `mloda.run_all(links=...)` |

## Complete Example

```python
from typing import Any, Optional, Set
from mloda.user import Link, JoinSpec, Index, Feature, FeatureName, Options
from mloda.provider import FeatureGroup, FeatureSet


class OrderWithCustomer(FeatureGroup):
    """Join orders with customer data."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        link = Link.inner_on(OrderFeatureGroup, CustomerFeatureGroup)
        return {
            Feature(name="order_value", link=link),
            Feature(name="customer_name"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Data is already joined by the framework
        return data
```

## Test

```python
def test_input_features_with_link():
    fg = OrderWithCustomer()
    features = fg.input_features(Options({}), FeatureName("test"))
    assert features is not None
    assert any(f.link is not None for f in features)
```

## Join Types

| Type | SQL Equivalent | Method |
|------|----------------|--------|
| `Link.inner()` | INNER JOIN | `inner_on()` |
| `Link.left()` | LEFT JOIN | `left_on()` |
| `Link.right()` | RIGHT JOIN | `right_on()` |
| `Link.outer()` | FULL OUTER JOIN | `outer_on()` |
| `Link.append()` | UNION ALL | `append_on()` |
| `Link.union()` | UNION | `union_on()` |

## Using JoinSpec (Explicit Control)

```python
link = Link.inner(
    left=JoinSpec(FeatureGroupA, "id"),
    right=JoinSpec(FeatureGroupB, "ref_id")
)
```

## Using _on Methods (Convenience)

```python
# Auto-derives index from index_columns()
link = Link.inner_on(UserFeatureGroup, OrderFeatureGroup)

# Select specific index position
link = Link.inner_on(UserFG, OrderFG, left_index=0, right_index=1)
```

## Self-Joins with Aliases

```python
link = Link.inner_on(UserFeatureGroup, UserFeatureGroup,
                     self_left_alias={"side": "left"},
                     self_right_alias={"side": "right"})

features = {
    Feature("age", options={"side": "left"}),
    Feature("age", options={"side": "right"}),
}
```

## Feature-Level Links

Links can also be set directly on Feature objects.

## Via Feature Parameter

```python
# Pass link when creating a Feature
link = Link.left(
    JoinSpec(OrderFeatureGroup, Index(("order_id",))),
    JoinSpec(CustomerFeatureGroup, Index(("customer_id",)))
)
feature = Feature(name="order_value", link=link, index=Index(("order_id",)))
```

## Via input_features()

```python
def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
    link = Link.left(
        JoinSpec(OrderFeatureGroup, Index(("order_id",))),
        JoinSpec(CustomerFeatureGroup, Index(("customer_id",)))
    )
    return {
        Feature(name="order_value", link=link, index=Index(("order_id",))),
        Feature(name="customer_name", index=Index(("customer_id",))),
    }
```

## When to Use Each Approach

| Approach | Use When |
|----------|----------|
| `input_features()` with link | Joins for derived feature dependencies |
| Feature `link` parameter | Dynamic joins at feature creation |
| `mloda.run_all(links=...)` | Global joins for entire computation |

## Real Implementations

| File | Description |
|------|-------------|
| [join_data.md](https://github.com/mloda-ai/mloda/blob/main/docs/docs/in_depth/join_data.md) | In-depth join documentation |

## Combines With

- **Index Features** (Pattern 7): Links require `index_columns()` defined on joined feature groups
- **Framework-specific** (Pattern 9): Different merge engines per framework
