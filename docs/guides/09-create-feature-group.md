# Create a Feature Group

Use this guide to find the right pattern for your feature group.

## Decision Tree

```text
Q1: Does it load external data (file, DB, external API)?
    YES → Pattern 1: Root Feature (if data passed at runtime, also see Q18)
    NO  → Continue

Q2: How many input features?
    0 → Pattern 1: Root Feature (or DataCreatorFeatureGroup, see 01-root-features)
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
                    (declare the value space via allowed_values / property_spec; see Pattern 11)

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
    YES → See ApiDataFG docs: https://mloda-ai.github.io/mloda/in_depth/access-feature-data/#apidatafg

Q19: Need to create feature groups dynamically or simplify complex input sources?
    YES → See 21-experimental-shortcuts

Q20: Need to define features via JSON config (e.g., for AI agents or config-driven pipelines)?
    YES → See 22-feature-config

Q21: Need to consume results incrementally as each feature group completes?
    YES → See 23-streaming

Q22: Need to build the execution plan once at startup and reuse it for repeated calls with fresh data?
    YES → See 24-realtime

Q23: Need both plan reuse and incremental per-group delivery?
    YES → Use session.stream_run(), see 23-streaming and 24-realtime

Q24: Need conditional aggregation (include only rows matching a predicate)?
    YES → See 25-masking

Q25: Reading a file format, database or API with a source FeatureGroup (pointing, custom format, non-file / HTTP source)?
    YES → See 27-input-data-readers

Q26: Does it declare another feature group's feature (source, connector) in input_features(), and need to control which of its own options forward to that input?
    YES → See 26-input-feature-forwarding

Q27: Several interchangeable backends for one feature (e.g. BM25 vs. FAISS)?
    YES → See 28-backend-families

Q28: Ready to test your implementation?
    YES → See 10-testing-guide

Q29: Need to decide whether an option affects feature identity/resolution (group) vs is runtime-only metadata (context)?
    YES → See 11-options

Q30: Need to define what feature names your FeatureGroup provides (beyond the default class name)?
    YES → See 13-feature-naming

Q31: Is the model a compiled binary (shipped as a wheel, possibly license-gated) that must stay a black box?
    YES → See 29-binary-backed-features
```

Every pattern, with links: [Feature group patterns index](feature-group-patterns/index.md).
