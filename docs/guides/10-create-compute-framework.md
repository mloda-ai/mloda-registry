# Create a Compute Framework Plugin

Use this guide to add support for a new data processing library to mloda.

## Decision Tree

```text
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
    AUTO  → Add fallback in set_framework_connection_object(); connection_requirement() returns SELF_MANAGED
    USER  → Require via data_access_collection parameter; connection_requirement() returns REQUIRED
    (see 03-stateful-connection, "Declare the Connection Requirement")

Q11: Ready to test your implementation?
    YES → See 09-testing-guide

Q12: Need to map a native column type (e.g. pandas dtype, Arrow type) to mloda's DataType?
    YES → See 10-data-type-extraction
```

Every pattern, with links: [Compute framework patterns index](compute-framework-patterns/index.md).

## Timezone Validation (Opt-In)

Timezone guard for merge and filter engines: see [06-merge-engine](compute-framework-patterns/06-merge-engine.md#timezone-validation-opt-in).
