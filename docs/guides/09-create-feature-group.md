# Create a Feature Group

Use this guide to find the right pattern for your feature group.

## Decision Tree

```
Q1: Does it load external data (file, DB, external API)?
    YES → Pattern 1: Root Feature (if data passed at runtime, also see Q18)
    NO  → Continue

Q2: How many input features?
    0 → Pattern 1: Root Feature (or DataCreator, see 01-root-features)
    1+ → Pattern 2: Derived (see Q3)

Q3: Should it be reusable via naming pattern (input__operation)?
    YES → Add Pattern 3: FeatureChainParserMixin (if 2+ inputs, use Pattern 4)
    NO  → Continue

Q4: How many output columns?
    1 → Standard
    N → Pattern 5: Multi-output (feature~0, feature~1)

Q5: Does it need fitted/trained state between runs?
    YES → Add Pattern 6: Artifact

Q6: Does it need time ordering or group-by?
    YES → Add Pattern 7: Index

Q7: Does it join data from multiple feature groups?
    YES → Add Pattern 8: Links/Joins

Q8: Framework-specific API needed?
    YES → Pattern 9: Framework-specific

Q9: Does the feature name come from configuration vs naming pattern?
    Configuration → Pattern 3 with PROPERTY_MAPPING

Q10: Does it need custom matching logic beyond class name?
    YES → See 14-feature-matching

Q11: Does it need to filter data (time ranges, categories)?
    YES → See 15-filter-concepts

Q12: Does it need input/output validation?
    YES → See 16-validators

Q13: Does it need to match against a data connection?
    YES → See 17-data-connection-matching

Q14: Does it need explicit data type declaration?
    YES → See 18-datatypes

Q15: Could multiple FeatureGroups handle the same feature name?
    YES → See 19-domain

Q16: Do you need to track changes or ensure reproducibility?
    YES → See 20-versioning

Q17: Need to understand calculate_feature() and runtime context?
    YES → See 12-calculate-feature

Q18: Does it receive data programmatically at runtime (e.g. via API call)?
    YES → See ApiInputData docs: https://mloda-ai.github.io/mloda/in_depth/api-input-data/

Q19: Need to create feature groups dynamically or simplify complex input sources?
    YES → See 21-experimental-shortcuts

Q20: Need to define features via JSON config (e.g., for AI agents or config-driven pipelines)?
    YES → See 22-feature-config

Q21: Need to consume results incrementally as each feature group completes?
    YES → See 23-streaming

Q22: Need to build the execution plan once at startup and reuse it for repeated calls with fresh data?
    YES → See 24-realtime

Q23: Need both plan reuse and incremental per-group delivery?
    YES → Use session.stream_run() — see 23-streaming and 24-realtime
```

## Pattern Guides

| Pattern | When to Use |
|---------|-------------|
| [01-root-features](feature-group-patterns/01-root-features.md) | Loading data from files, APIs, databases, or generating synthetic data |
| [02-derived-features](feature-group-patterns/02-derived-features.md) | Simple transformation with static output name |
| [03-chained-features](feature-group-patterns/03-chained-features.md) | Reusable transforms via naming pattern (`price__scaled`) |
| [04-multi-input-features](feature-group-patterns/04-multi-input-features.md) | Combining multiple inputs (`a&b__distance`) |
| [05-multi-output-features](feature-group-patterns/05-multi-output-features.md) | Multiple output columns (`emb~0`, `emb~1`) |
| [06-artifact-features](feature-group-patterns/06-artifact-features.md) | Fitted models, cached computations |
| [07-index-features](feature-group-patterns/07-index-features.md) | Time series, group-by, window functions |
| [08-links-joins](feature-group-patterns/08-links-joins.md) | Joining data from multiple feature groups |
| [09-framework-specific](feature-group-patterns/09-framework-specific.md) | Pandas-only, Polars-only features |
| [10-testing-guide](feature-group-patterns/10-testing-guide.md) | 3-level testing approach |

## Concepts

| Guide | What It Covers |
|-------|----------------|
| [11-options](feature-group-patterns/11-options.md) | Group vs context options for feature configuration |
| [12-calculate-feature](feature-group-patterns/12-calculate-feature.md) | Implementing the core computation method |
| [13-feature-naming](feature-group-patterns/13-feature-naming.md) | How to define feature names |
| [14-feature-matching](feature-group-patterns/14-feature-matching.md) | How mloda finds the right FeatureGroup |
| [15-filter-concepts](feature-group-patterns/15-filter-concepts.md) | Filtering data during computation |
| [16-validators](feature-group-patterns/16-validators.md) | Input/output validation |
| [17-data-connection-matching](feature-group-patterns/17-data-connection-matching.md) | Matching against data connections |
| [18-datatypes](feature-group-patterns/18-datatypes.md) | Data type declaration and validation |
| [19-domain](feature-group-patterns/19-domain.md) | Disambiguate feature-to-FeatureGroup matching |
| [20-versioning](feature-group-patterns/20-versioning.md) | Version tracking and reproducibility |
| [21-experimental-shortcuts](feature-group-patterns/21-experimental-shortcuts.md) | Dynamic creation and input source helpers |
| [22-feature-config](feature-group-patterns/22-feature-config.md) | JSON-based feature definition for AI agents and config pipelines |
| [23-streaming](feature-group-patterns/23-streaming.md) | Consume results incrementally with `stream_all` |
| [24-realtime](feature-group-patterns/24-realtime.md) | Reuse execution plans with `prepare` + `run` |
