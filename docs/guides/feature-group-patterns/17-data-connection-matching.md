# Data Connection Matching

How to match feature groups against data connections.

**What**: Match features to specific data sources via the `MatchData` mixin.
**When**: Feature availability depends on the data connection (database, API, file).
**Why**: Same feature name can have different implementations per data source.
**Where**: Database connectors, API clients, file readers with specific schemas.

## How It Works

The `MatchData` mixin checks if a feature group can handle a request based on the available data connection. A feature group opts in by mixing it in and overriding `match_data_access`, which returns the resource to bind (or `None` to decline).

```python
from typing import Any
from mloda.provider import FeatureGroup, MatchData
from mloda.user import DataAccessCollection, Options

class DatabaseFeature(MatchData, FeatureGroup):
    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
        framework_connection_object: Any | None = None,
    ) -> Any:
        if data_access_collection is None:
            return None
        # resolve(...) returns the single connection, None if none match, and
        # raises if several match without a disambiguating handle (see
        # "Named Handles" below).
        return data_access_collection.resolve(
            "connection", hint=options.get("data_access_handle")
        )
```

## Matching Priority

In `match_feature_group_criteria()`, data access matching is checked early:

```text
1. Input data match     → Root features
2. Data access match    → MatchData mixin        ← HERE
3. Exact class name     → Default matching
...
```

## Folder-Based Matching

```python
class CsvFeature(FeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: str, options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if data_access_collection is None or not data_access_collection.folders:
            return False
        # `folders` is a dict[handle, path] in mloda 0.7.0+; iterate `.values()`
        # for the path (iterating the dict directly yields handle names).
        options.set("_data_folder", next(iter(data_access_collection.folders.values())))
        return True
```

## Subclass Data Access Matching

```python
@classmethod
def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
    ...
```

The `options` parameter is required.

## Named Handles and Multi-Source Disambiguation

Since mloda 0.7.0, `DataAccessCollection` registers each resource under a stable
**handle**. Pass a `dict[handle, value]` to name resources explicitly; a `set`/`list`
still works and auto-assigns internal handles.

```python
from mloda.user import DataAccessCollection

dac = DataAccessCollection(
    connections={"warehouse": warehouse_conn, "reporting": reporting_conn},
    files={"tx": "/data/transactions.parquet"},
)
dac.handles()  # {'warehouse': 'connection', 'reporting': 'connection', 'tx': 'file'}
```

A feature group resolves one resource with `resolve(kind, predicate=..., hint=...)`:

- 0 matches → returns `None`
- 1 match → returns the resource
- more than one match and no `hint` → raises `ValueError` rather than letting
  iteration order decide

When several resources of the same kind could match a consumer, the user picks one
by name via the `data_access_handle` key in `Options`. The feature group forwards it
as the `hint`:

```python
from mloda.user import Options

# User side: pin which named connection this feature should bind to.
options = Options(context={"data_access_handle": "warehouse"})

# Feature group side (inside match_data_access):
conn = data_access_collection.resolve(
    "connection", hint=options.get("data_access_handle")
)  # -> warehouse_conn
```

If the handle does not exist, names a resource of the wrong kind, or fails the
predicate, `resolve` raises a `ValueError` that names the offending handle and the
available handles of that kind, so misconfiguration fails fast instead of silently
binding the wrong source.

## Credentials

Credentials are a resource kind alongside connections, files, and folders, but with
one twist: a credential *is* a dict, so a bare `{connector_id: slot}` dict collides
with the named `{handle: value}` form. Since mloda 0.9.0, wrap each credential slot
in the typed `Credential` class so the meaning comes from the type, not the nesting
depth:

```python
from mloda.user import Credential, DataAccessCollection

# kwargs form and dict form are equivalent
DataAccessCollection(credentials=Credential(sqlite="/data/app.db"))
DataAccessCollection(credentials=Credential({"sqlite": "/data/app.db"}))

# a list registers each entry as its own auto-named slot
DataAccessCollection(credentials=[Credential(host="h"), {"host": "h2"}])
# a dict names each slot by handle; the value must itself be a mapping
DataAccessCollection(credentials={"pg-prod": Credential(host="h")})
```

`Credential` is unwrapped to a plain dict at registration, so feature groups and
`is_valid_credentials` implementations keep receiving plain dicts. Nothing changes
downstream.

Two shapes now fail fast at construction instead of silently mis-matching later:

- A named-form value that is not a mapping (for example
  `credentials={"prod": "dsn-string"}`) raises `ValueError`. Keep the handle and
  make the value a mapping: `credentials={"prod": {"dsn": "dsn-string"}}` or
  `credentials={"prod": Credential(dsn="dsn-string")}`. A bare
  `credentials=Credential(dsn="dsn-string")` also constructs, but auto-names the
  slot, so a later `data_access_handle="prod"` lookup would no longer find it.
- A `HashableDict` credential value raises `ValueError`. Pass `Credential(...)` or a
  plain dict instead.

Resolver ambiguity errors redact credential values (keys stay visible, values render
as `***`), so secrets stay out of logs and tracebacks.

## Full Documentation

See [Data Access Patterns](https://mloda-ai.github.io/mloda/in_depth/data-access-patterns/) for detailed patterns.
