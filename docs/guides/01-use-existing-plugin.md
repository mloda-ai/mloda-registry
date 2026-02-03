# Use an Existing Plugin

Install and use a community plugin in your project.

## Install

```bash
pip install mloda-community
```

## Use

```python
from mloda.user import PluginLoader

# Auto-discover installed plugins
PluginLoader.load()

# Use features from the plugin
from mloda import mlodaAPI, Feature
result = mlodaAPI.run_all([Feature("example_feature")])
```

## Direct Import

You can also import plugin classes directly:

```python
from mloda.community.feature_groups.example import CommunityExampleFeatureGroup
```
