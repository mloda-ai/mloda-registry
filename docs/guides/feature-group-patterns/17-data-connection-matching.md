# Data Connection Matching

How to match feature groups against data connections.

**What**: Bind a feature to one connection of the `DataAccessCollection`.
**When**: Feature availability depends on the data connection (database, API, file).
**Why**: Same feature name can have different implementations per data source.
**Where**: Database connectors, API clients, file source groups with specific schemas.

## How It Works

Connections live in the `DataAccessCollection`; the framework binds the single connection it accepts.

- Pin one with the `connection_handle` option, or `Feature(name, connection=conn)` (sets a handle for you).
- A source group that reads a connection declares `ClaimRoute("connection", ...)`.
- It chooses among the connection's content in `find_sources` and reports the connection a match reads with `match_connection`.

```python
from typing import Any, ClassVar
from mloda.provider import ClaimRoute, FeatureGroup, NamePolicy, PropertySpec, SourceFeatureGroup, SourceMatch
from mloda.user import DataAccessCollection, Options


class DatabaseSourceFG(SourceFeatureGroup):
    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (
        ClaimRoute("connection", NamePolicy.CHECKED, True, ("connection_handle",)),
    )
    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        "connection_handle": PropertySpec("Name of the connection to read.", default=None)
    }

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        connections = {} if data_access_collection is None else data_access_collection.connections
        handle = cls.read_option("connection_handle", options)
        if handle not in connections:
            return []  # record_match_rejection(...) says why, see the Source FeatureGroups guide
        return [SourceMatch(f"db::{handle}", connections[handle])]

    @classmethod
    def match_connection(cls, match: SourceMatch) -> Any | None:
        return match.access
```

`DuckdbFG` in `mloda_plugins` is the complete model (catalog read, `columns`, `load_neutral`, an `Implementation` per framework). The in-repo `mloda/community/feature_groups/experimental/german_ledger` reader shows a source group with its own claim route.

## Matching Priority

```text
1. Input data match     → Source FeatureGroups (claim routes, find_sources)
2. Exact class name     → Default matching
...
```

## Folder-Based Matching

```python
class CsvFeature(FeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if data_access_collection is None or not data_access_collection.folders:
            return False
        # `folders` is a dict[handle, path]; iterate `.values()` for the path (iterating
        # the dict directly yields handle names).
        options.set("_data_folder", next(iter(data_access_collection.folders.values())))
        return True
```

## Source Group Matching

Source FeatureGroups find their data in the collection through `find_sources(route, feature_name, options, data_access_collection)`, which runs behind the `INPUT_DATA_MATCH` gate. For the source-selection contract (class-name pointers, claim routes, non-file / HTTP sources), see
[Source FeatureGroups](27-input-data-readers.md).

## Named Handles and Multi-Source Disambiguation

`DataAccessCollection` registers each resource under a stable **handle**. Pass a
`dict[handle, value]` to name resources explicitly; a `set`/`list` still works and
auto-assigns internal handles.

```python
from mloda.user import DataAccessCollection

warehouse_conn, reporting_conn = object(), object()  # stand-ins for real connection objects

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

# Feature group side (inside find_sources, where the collection is data_access_collection):
conn = dac.resolve("connection", hint=options.get("data_access_handle"))
assert conn is warehouse_conn
```

If the handle does not exist, names a resource of the wrong kind, or fails the
predicate, `resolve` raises a `ValueError` that names the offending handle and the
available handles of that kind, so misconfiguration fails fast instead of silently
binding the wrong source.

## Credentials

Credentials are a resource kind alongside connections, files, and folders, but with one
twist: a credential *is* a dict, so a bare `{connector_id: slot}` dict collides with the
named `{handle: value}` form. Wrap each credential slot in the typed `Credential` class
so the meaning comes from the type, not the nesting depth:

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

A named-form value that is not a mapping (for example
`credentials={"prod": "dsn-string"}`) now raises `ValueError` at construction
instead of silently mis-matching later. Keep the handle and make the value a
mapping: `credentials={"prod": {"dsn": "dsn-string"}}` or
`credentials={"prod": Credential(dsn="dsn-string")}`. A bare
`credentials=Credential(dsn="dsn-string")` also constructs, but auto-names the
slot, so a later `data_access_handle="prod"` lookup would no longer find it.

Resolver ambiguity errors redact credential values (keys stay visible, values render
as `***`), so secrets stay out of logs and tracebacks.

To pick one sibling source group by its credential slot key, see
[Backend Families: reader delegation](28-backend-families.md#way-3-reader-delegation-sibling-readers).

## Full Documentation

See [Data Access Patterns](https://mloda-ai.github.io/mloda/in_depth/data-access-patterns/) for detailed patterns.
