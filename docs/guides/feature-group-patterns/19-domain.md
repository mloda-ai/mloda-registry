# Domain

How to use domains to match features with specific FeatureGroups.

**What**: A filter mechanism to match features with their corresponding FeatureGroups.
**When**: Multiple FeatureGroups could handle the same feature name.
**Why**: Disambiguate feature-to-FeatureGroup matching; separate business/logical contexts.
**Where**: Different data sources for same feature, testing vs production, business domains.

## How It Works

When you request a feature, mloda searches for a matching FeatureGroup. If multiple FeatureGroups could handle the same feature name, domains help resolve the ambiguity.

- A feature with a domain only matches FeatureGroups with the same domain
- A feature without a domain can match any FeatureGroup
- FeatureGroups default to `"default_domain"` if not overridden

## When to Use Domains

Domains are useful when you have the same logical feature coming from different sources or contexts:

- **Business domains**: Sales revenue vs Fraud revenue calculations
- **Data sources**: Same feature from different databases or APIs
- **Environments**: Testing data vs production data
- **Regions**: EU data vs US data with different schemas

## Setting Domains

On features: Use the `domain` parameter or include `"domain"` in options.

On FeatureGroups: Override the `get_domain()` classmethod to return a `Domain` object.

## Full Documentation

See [Domain](https://mloda-ai.github.io/mloda/in_depth/domain/) for detailed patterns.
