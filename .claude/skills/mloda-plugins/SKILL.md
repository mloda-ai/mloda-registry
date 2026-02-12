---
name: mloda-plugins
description: mloda plugin development guides. Use when creating FeatureGroups, ComputeFrameworks, Extenders, building plugins, sharing plugins, or understanding the mloda plugin system.
---

# mloda Plugin Development Guides

Location: `docs/guides/`

## Quick Start Guides (01-08)

| # | Guide | Description |
|---|-------|-------------|
| 01 | `01-use-existing-plugin.md` | Install and use a community plugin |
| 02 | `02-discover-plugins.md` | Find available plugins and explore installed ones |
| 03 | `03-create-plugin-in-project.md` | Add feature groups inline without separate package |
| 04 | `04-create-plugin-package.md` | Create standalone installable plugin from template |
| 05 | `05-share-with-team.md` | Distribute via private git repo |
| 06 | `06-publish-to-community.md` | Submit to mloda-registry |
| 07 | `07-contribute-to-official.md` | Improve existing plugins |
| 08 | `08-become-official.md` | Get plugin merged into registry |

---

## Feature Group Decision Tree (Guide 09)

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

### Feature Group Pattern Guides

Location: `docs/guides/feature-group-patterns/`

| # | Guide | Description |
|---|-------|-------------|
| 01 | `01-root-features.md` | Data sources (CSV, DB, API, synthetic) |
| 02 | `02-derived-features.md` | Transform inputs with static output name |
| 03 | `03-chained-features.md` | Reusable transforms via `input__op` naming |
| 04 | `04-multi-input-features.md` | Combine inputs using `&` separator |
| 05 | `05-multi-output-features.md` | Multiple outputs using `~` separator |
| 06 | `06-artifact-features.md` | Save/load fitted state (scalers, models) |
| 07 | `07-index-features.md` | Ordering, grouping, joining columns |
| 08 | `08-links-joins.md` | Join data from multiple feature groups |
| 09 | `09-framework-specific.md` | Restrict to certain frameworks |
| 10 | `10-testing-guide.md` | Unit, framework, integration testing |
| 11 | `11-options.md` | Group vs context configuration |
| 12 | `12-calculate-feature.md` | Core computation method |
| 13 | `13-feature-naming.md` | Define feature names |
| 14 | `14-feature-matching.md` | Resolve name to FeatureGroup |
| 15 | `15-filter-concepts.md` | Filter data by conditions |
| 16 | `16-validators.md` | Input/output validation |
| 17 | `17-data-connection-matching.md` | Match against data sources |
| 18 | `18-datatypes.md` | Arrow-based type system |
| 19 | `19-domain.md` | Disambiguate matching |
| 20 | `20-versioning.md` | Version tracking |
| 21 | `21-experimental-shortcuts.md` | Dynamic creation helpers |
| 22 | `22-feature-config.md` | JSON-based feature definition |
| 23 | `23-streaming.md` | Incremental results with `stream_all` |
| 24 | `24-realtime.md` | Reuse execution plans with `prepare` + `run` |

---

## Compute Framework Decision Tree (Guide 10)

```
Q1: Does your framework require a connection/session object?
    YES → Q2
    NO  → Q3

Q2: Is it a data lake table format (Iceberg, Delta, Hudi)?
    YES → Category 5: Data Lake
    NO  → Category 3: Stateful Connection

Q3: Does your framework use lazy evaluation?
    YES → Category 2: Stateless Lazy
    NO  → Q4

Q4: Does your framework have external dependencies?
    YES → Category 1: Stateless Eager
    NO  → Category 4: Zero Dependency

Q5: Do you need cross-framework conversion?
    YES → See 08-framework-transformer

Q6: Does your library have built-in PyArrow conversion?
    YES → Simplifies transformer (use .to_arrow()/.from_arrow())
    NO  → Manual conversion in 08-framework-transformer

Q7: Need to understand merge/join operations?
    YES → See 06-merge-engine

Q8: Do you need multi-column index support for joins?
    YES → See 06-merge-engine (Index with tuple)

Q9: Need to understand filter operations?
    YES → See 07-filter-engine

Q10: Should connections be auto-created or user-provided?
    AUTO  → Add fallback in set_framework_connection_object()
    USER  → Require via data_connections parameter

Q11: Ready to test your implementation?
    YES → See 09-testing-guide
```

### Compute Framework Pattern Guides

Location: `docs/guides/compute-framework-patterns/`

| # | Guide | Description |
|---|-------|-------------|
| 01 | `01-stateless-eager.md` | In-memory immediate (Pandas, PyArrow) |
| 02 | `02-stateless-lazy.md` | Deferred execution (Polars Lazy, Ibis) |
| 03 | `03-stateful-connection.md` | Connection required (DuckDB, Spark) |
| 04 | `04-zero-dependency.md` | Pure Python (List[Dict]) |
| 05 | `05-data-lake.md` | Catalog-based (Iceberg, Delta, Hudi) |
| 06 | `06-merge-engine.md` | JOIN and UNION operations |
| 07 | `07-filter-engine.md` | Filter operations |
| 08 | `08-framework-transformer.md` | Cross-framework conversion |
| 09 | `09-testing-guide.md` | Testing your implementation |

---

## Extender Decision Tree (Guide 11)

```
Q1: What do you want to wrap?
    Feature calculation → FEATURE_GROUP_CALCULATE_FEATURE
    Input validation   → VALIDATE_INPUT_FEATURE
    Output validation  → VALIDATE_OUTPUT_FEATURE

Q2: Need execution order control?
    YES → Set custom priority (lower runs first, default 100)

Q3: Need state with ParallelizationMode.MULTIPROCESSING?
    YES → Use class-level storage (pickle-safe)
```

Full guide: `docs/guides/11-create-extender.md`
