# Versioning

How mloda tracks feature group versions.

**What**: Automatic version identifier for each FeatureGroup.
**When**: Tracking changes, managing compatibility, debugging.
**Why**: Detect when feature group implementation changes; ensure reproducibility.
**Where**: Logging, auditing, model lineage, cache invalidation.

## How It Works

Each FeatureGroup has a `version()` method that returns a composite identifier combining:

1. **mloda package version** - The installed mloda version
2. **Module name** - Where the feature group is defined
3. **Source code hash** - SHA-256 hash of the class source code

This means the version changes automatically when:
- mloda is upgraded
- The feature group code is modified
- The feature group is moved to a different module

## Usage

```python
# Get version of any FeatureGroup
version = MyFeatureGroup.version()
# e.g., "0.2.6|my_package.features|a1b2c3d4..."
```

## Custom Versioning

Override by subclassing `BaseFeatureGroupVersion`:

```python
from mloda.provider import BaseFeatureGroupVersion

class MyVersioning(BaseFeatureGroupVersion):
    @classmethod
    def version(cls, feature_group_class) -> str:
        return "1.0.0"  # Custom version logic
```

## Description Method

FeatureGroups also have a `description()` method that returns the class docstring or class name if no docstring is provided.

## Full Documentation

See [Feature Group Versioning](https://mloda-ai.github.io/mloda/in_depth/feature-group-version/) for details.
