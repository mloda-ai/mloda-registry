# Discover Plugins

How to find what plugins are available and what's installed in your environment.

## Finding Available Plugins

### Documentation

The simplest way to discover available plugins:

1. **This repository**: Browse the `plugins/` directory for community and official plugins
2. **Plugin READMEs**: Each plugin package has a README describing its feature groups

## Discovering Installed Plugins

mloda provides a discovery API to explore loaded plugins at runtime.

### Setup

```python
from mloda.user import PluginLoader

# Load all plugins first
PluginLoader.all()
```

### List Feature Groups

```python
from mloda.steward import get_feature_group_docs

# Get all feature groups
for fg in get_feature_group_docs():
    print(f"{fg.name}: {fg.description}")
    print(f"  Version: {fg.version}")
    print(f"  Frameworks: {fg.compute_frameworks}")
```

### Filter Feature Groups

```python
from mloda.steward import get_feature_group_docs

# Search by name (case-insensitive partial match)
docs = get_feature_group_docs(name="Customer")

# Search in description
docs = get_feature_group_docs(search="aggregation")

# Filter by compute framework
docs = get_feature_group_docs(compute_framework="PandasDataframe")

# Filter by version
docs = get_feature_group_docs(version_contains="1.0")
```

### List Compute Frameworks

```python
from mloda.steward import get_compute_framework_docs

# Get available frameworks
for cfw in get_compute_framework_docs():
    print(f"{cfw.name}: {cfw.description}")
    print(f"  Data framework: {cfw.expected_data_framework}")
    print(f"  Has merge engine: {cfw.has_merge_engine}")

# Include unavailable frameworks
all_frameworks = get_compute_framework_docs(available_only=False)
```

### List Extenders

```python
from mloda.steward import get_extender_docs

# Get all extenders
for ext in get_extender_docs():
    print(f"{ext.name}: {ext.description}")
    print(f"  Wraps: {ext.wraps}")

# Filter by what they wrap
docs = get_extender_docs(wraps="feature_group")
```

## Debugging Feature Resolution

When a feature doesn't resolve as expected, use `resolve_feature()` to understand which FeatureGroup handles it:

```python
from mloda.steward import resolve_feature

# Check which FeatureGroup handles a feature name
result = resolve_feature("my_feature_name")

if result.feature_group:
    print(f"Resolved to: {result.feature_group.__name__}")
else:
    print(f"Error: {result.error}")

# See all candidates before subclass filtering
print(f"Candidates: {[fg.__name__ for fg in result.candidates]}")
```

### ResolvedFeature Fields

| Field | Type | Description |
|-------|------|-------------|
| `feature_name` | `str` | The input feature name |
| `feature_group` | `Type[FeatureGroup] \| None` | The resolved FeatureGroup class |
| `candidates` | `List[Type[FeatureGroup]]` | All matching FeatureGroups before filtering |
| `error` | `str \| None` | Error message if resolution failed |

### Common Issues

**Multiple candidates**: When `candidates` has multiple entries but `feature_group` is `None`, you likely have conflicting FeatureGroups. The most specific subclass should win.

**No candidates**: No FeatureGroup's `match_feature_group_criteria()` returned `True` for the name. Check your naming pattern.

## Next Steps

- [Create a plugin in your project](03-create-plugin-in-project.md) - Build your own feature group
- [Create a plugin package](04-create-plugin-package.md) - Package for distribution
