# Data Connection Matching

How to match feature groups against data connections.

**What**: Match features to specific data sources via `MatchData` mixin.
**When**: Feature availability depends on the data connection (database, API, file).
**Why**: Same feature name can have different implementations per data source.
**Where**: Database connectors, API clients, file readers with specific schemas.

## How It Works

`MatchData` checks if a feature group can handle a request based on the available data connection.

```python
from typing import Any, Optional
from mloda.provider import FeatureGroup
from mloda.core.abstract_plugins.components.match_data.match_data import MatchData
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.options import Options

class DatabaseFeature(MatchData, FeatureGroup):
    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        if data_access_collection and data_access_collection.has_connection("my_database"):
            return data_access_collection.get("my_database")
        return None
```

## Matching Priority

In `match_feature_group_criteria()`, data access matching is checked early:

```
1. Input data match     → Root features
2. Data access match    → ConnectionMatcherMixin ← HERE
3. Exact class name     → Default matching
...
```

## Full Documentation

See [Data Access Patterns](https://mloda-ai.github.io/mloda/in_depth/data-access-patterns/) for detailed patterns.
