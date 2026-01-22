# Create a Plugin Inline

Add feature groups directly to your existing project without creating a separate package.

## When to Use This

- You want to keep plugins in your main project
- You don't need to share the plugin as a separate package
- Quick prototyping or project-specific features

## Step 1: Add Plugin Folder

Create a folder structure in your project:

```
my-project/
├── src/
│   └── my_app/
├── my_features/                  # Your mloda plugins
│   ├── __init__.py
│   └── scoring/
│       ├── __init__.py
│       ├── customer.py
│       └── tests/
│           └── test_customer.py
├── pyproject.toml
└── tests/
```

## Step 2: Implement FeatureGroup

```python
# my_features/scoring/customer.py
from mloda.provider import FeatureGroup

class CustomerScoring(FeatureGroup):
    """Customer scoring calculations."""

    @classmethod
    def calculate_feature(cls, data, features):
        return {"customer_score": 100}
```

## Step 3: Use Your Plugin

### Option A: Direct Import

```python
from mloda import mlodaAPI, Feature
from my_features.scoring import CustomerScoring

result = mlodaAPI.run_all([Feature("CustomerScoring")])
```

### Option B: Via PluginLoader

```python
from mloda.user import PluginLoader

PluginLoader.load()

result = mlodaAPI.run_all([Feature("CustomerScoring")])
```

## Next Steps

- [Create a feature group](09-create-feature-group.md) - Detailed feature group guide
- [Create a plugin package](04-create-plugin-package.md) - Package for distribution
- [Share with your team](05-share-with-team.md) - Distribute via private git repo
