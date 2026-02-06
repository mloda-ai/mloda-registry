# Domain

How to use domains to match features with specific FeatureGroups.

**What**: A filter mechanism to match features with their corresponding FeatureGroups.
**When**: Multiple FeatureGroups could handle the same feature name.
**Why**: Disambiguate feature-to-FeatureGroup matching; separate business/logical contexts.
**Where**: Different data sources for same feature, testing vs production, business domains.

## How It Works

- A feature with a domain only matches FeatureGroups with the same domain
- A feature without a domain can match any FeatureGroup
- FeatureGroups default to `"default_domain"` if not overridden

## Setting Domains

On features:

```python
from mloda.user import Feature

Feature("Revenue", domain="Sales")
```

On FeatureGroups:

```python
from mloda.user import Domain
from mloda.provider import FeatureGroup

class SalesRevenueGroup(FeatureGroup):
    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("Sales")
```

## Domain Propagation

When a feature depends on others via `input_features()`, the parent's domain propagates automatically to children. Override with an explicit domain.

| Child Definition | Parent Domain | Result |
|------------------|---------------|--------|
| `"child"` (string) | `"Sales"` | Inherits "Sales" |
| `Feature("child")` | `"Sales"` | Inherits "Sales" |
| `Feature("child", domain="Finance")` | `"Sales"` | Keeps "Finance" |
| Any | None | No domain |

### Cross-Domain Dependency

```python
class SalesRevenueGroup(FeatureGroup):
    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("Sales")

    def input_features(self, options, feature_name):
        return {
            "base_amount",                              # Inherits "Sales"
            Feature("exchange_rate", domain="Finance"), # Uses "Finance"
        }
```

### Shared Utilities

```python
def input_features(self, options, feature_name):
    return {
        "domain_specific_feature",                    # Inherits parent domain
        Feature("date_parser", domain="Common"),      # Always use Common
    }
```

### Test Isolation

```python
# In test code — all children inherit "Test" and use mock implementations
Feature("ProcessPayment", domain="Test")
```

## Troubleshooting

**"Multiple feature groups found"**: Child feature has no domain and matches multiple FeatureGroups. Set domain on the parent feature or explicitly on the child.

**Unexpected FeatureGroup resolution**: Child inherited parent's domain but you wanted a different one. Set explicit domain: `Feature("child", domain="CorrectDomain")`.

**Domain not propagating**: Parent feature doesn't have a domain set. Ensure the top-level feature has an explicit domain.

## Full Documentation

See [Domain](https://mloda-ai.github.io/mloda/in_depth/domain/) for detailed patterns and best practices.
