# Versioning

How mloda tracks feature group versions.

**What**: Automatic version identifier for each FeatureGroup.
**When**: Tracking changes, managing compatibility, debugging.
**Why**: Detect when the code a feature group can run changes; ensure reproducibility.
**Where**: Extender records (lineage, audit, tracing) via `HookContext.feature_group_version`, logging, model lineage.

## How It Works

Each FeatureGroup has a `version()` method that returns a composite identifier combining:

1. **mloda package version** - The installed mloda version
2. **Module name** - Where the feature group is defined
3. **Implementation hash** - SHA-256 hash of the code the feature group can run

The hash covers the class, its first-party base classes, and the helper functions, classes and module-level constants they reference, across first-party modules (the feature group's top-level package, and for plugins the whole `mloda.*` namespace except mloda itself). It ignores docstrings, comments, formatting, and unreferenced functions. Third-party code is not hashed; by default (`ThirdPartyVersionMode.INCLUDE`) the name and version of each referenced third-party package count instead.

This means the version changes automatically when:
- mloda is upgraded
- The feature group or code it reaches is modified (not just reformatted or re-documented)
- A referenced third-party package is upgraded (unless excluded)
- The feature group is moved to a different module

Code reached only at runtime (registries, reflection, `importlib`) and data or config files are not covered. Reference such a class in the class body (e.g. `VERSION_INCLUDES = (Scaler,)`) to make it count.

## Usage

```python
from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import PyArrowAggregation

# Get version of any FeatureGroup
version = PyArrowAggregation.version()
# e.g., "0.17.0-mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation-5f61d8cf..."
```

## Third-Party Dependencies

Override `version_third_party_mode()` to leave dependency versions out. A shared base class sets it for all its subclasses:

```python
from mloda.provider import FeatureGroup, ThirdPartyVersionMode


class DependencyAgnostic(FeatureGroup):
    @classmethod
    def version_third_party_mode(cls) -> ThirdPartyVersionMode:
        return ThirdPartyVersionMode.EXCLUDE
```

## Custom Versioning

`FeatureGroup.version()` calls `BaseFeatureGroupVersion.version(cls)` directly, so subclassing `BaseFeatureGroupVersion` alone changes nothing. Override `version()` on your FeatureGroup instead:

```python
class MyFeatureGroup(FeatureGroup):
    @classmethod
    def version(cls) -> str:
        return "1.0.0"  # Custom version logic
```

## Side Stores

A store a feature group keeps beside its features (for example a SQLite file whose path comes from a context option) is shared by every installed release pointed at it, and `version()` does not cover it. Adding a column with `ALTER TABLE ... ADD COLUMN` breaks an older release that inserts positionally, and only at run time. In SQLite or any similar store:

- Add tables, not columns.
- Never rename or drop a table or column an older release reads or writes.
- Name the columns in every insert: `INSERT INTO t (a, b) VALUES (?, ?)`.

## Description Method

FeatureGroups also have a `description()` method that returns the class docstring or class name if no docstring is provided.

## Full Documentation

See [Feature Group Versioning](https://mloda-ai.github.io/mloda/in_depth/feature-group-version/) for what the hash covers in detail and its blind spots.
