# Use an Existing Plugin

Install and use a community plugin in your project.

## Install

```bash
pip install mloda-community
```

## Use

```python
from mloda.user import PluginLoader, mloda, Feature

# Auto-discover installed plugins
PluginLoader.all()

# Use features from the plugin
result = mloda.run_all([Feature("example_feature")])
```

## Direct Import

You can also import plugin classes directly:

```python
from mloda.community.feature_groups.example import CommunityExampleFeatureGroup
```
