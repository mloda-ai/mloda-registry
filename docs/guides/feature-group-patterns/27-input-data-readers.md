# Source FeatureGroups: Pointing, Custom Formats and Non-File Sources

How to point a feature at a source FeatureGroup and how to write your own for a file format or a non-file (HTTP/API) source.

**What**: One FeatureGroup per kind of source (`CsvFG`, `ParquetFG`, `SqliteFG`, your own), selected per feature by an Options key equal to the group's class name.
**When**: A plugin reads a file format, a database or an API that no stock group covers, or a request must be routed to one of several sources.
**Why**: Source selection is contract-driven, not guesswork; knowing the contract avoids reading mloda core source.
**Where**: Connector plugins (a CKAN CSV group, a direct-URL group, a REST JSON group side by side).

Core describes the model (claim routes, resolution, errors): [Data Access Patterns](https://github.com/mloda-ai/mloda/blob/0.18.0/docs/docs/in_depth/data-access-patterns.md). Request data (`api_data=`) is served by the source group `ApiDataFG`; synthetic data by `DataCreatorFeatureGroup`.

## Pointing a Feature at a Group

A feature selects a source group with an Option whose **key equals the group's class name**. The value is the source the group receives (a path, URL, `Credential`, or any object the group understands). The class object works as the key too:

```python
from mloda.user import Feature
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG

by_name = Feature("customer_id", options={"CsvFG": "data/customers.csv"})
by_class = Feature("customer_id", options={CsvFG: "data/customers.csv"})
scoped = Feature("customer_id", feature_group=CsvFG)
```

A pointer makes the other source groups skip the collection; `feature_group=` scopes resolution to the group and its subclasses without choosing a source; `data_access_handle` only narrows among the sources the `DataAccessCollection` holds. A subclass also answers to its parent's name. Details: core's "Pointing a feature at a source".

`DataAccessCollection(files={"a": "a.csv", "b": "b.csv"}, column_to_file={"customer_id": "b"})` pins a column to one file handle; a pinned file that cannot serve the column aborts or declines, never falling back to another file.

### Several Accepting Readers

Without a pointer, every source group searches the `DataAccessCollection`. One group finding a name in several sources aborts naming them; two groups finding the same column fail as ordinary feature-group ambiguity. Resolve an overlap, with your own groups or with a stock one owning the same suffix, by one of:

- **Subclass the group you extend.** A subclass of `CsvFG` answers to `CsvFG` pointers and takes over its parent.
- **Narrow `find_sources`** (suffix, scheme, host) so the groups find disjoint sources.
- **Pin per feature** with the class-name pointer, or `feature_group=`, or `data_access_handle`.

### Overriding the selection

Customize discovery through `CLAIM_ROUTES` and `find_sources`. Put source I/O in `find_sources`, `columns` or `has_column`: they run behind the `INPUT_DATA_MATCH` extender hook, so a gate can refuse a source before anything is read (see [Reader match hook](../11-create-extender.md#reader-match-hook)). Overriding `match_feature_group_criteria` is rarely needed: call `super()` and run any extra I/O through `cls.gated_probe(feature_name, options, data_access_collection, probe)`. A probe run any other way is never gated.

`gated_probe` returns a `SourceProbeOutcome` (from `mloda.provider`), which is always truthy: `refused` is the refusal reason or `None`, `result` the probe's value. Decline when `refused` is set:

```py
outcome = cls.gated_probe(feature_name, options, data_access_collection, lambda: cls.catalog_has(feature_name))
return outcome.refused is None and bool(outcome.result)
```

## Writing a ReadFileFG Subclass

Implement `suffixes()` and `column_names(path)`. The base discovers files and folders, applies the pinned-file rule, caches per run and hands a `FileSource` to the compute framework. A format whose parsing differs from the stock transformers must also override `load_neutral`, otherwise the parent format is parsed. `load_neutral` must not call `column_names`, which would list the file a second time.

```python
from collections.abc import Collection
from typing import Any

from mloda.provider import FeatureSet, ReadFileFG


class SensorCsvFG(ReadFileFG):
    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (".sensorcsv",)

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        with open(path, encoding="utf-8") as handle:
            header = handle.readline()
        if "#sensor-schema" not in header:
            raise ValueError("its header lacks the #sensor-schema marker")
        return header.replace("#sensor-schema", "").strip().split(",")

    @classmethod
    def load_neutral(cls, match: Any, features: FeatureSet) -> Any: ...  # parse match.access, return the table
```

`suffixes()` runs at class definition and must return a non-empty tuple of non-empty strings. `column_names` raises `OSError` or `ValueError` for an unreadable file (declined with a recorded reason) and `ImportError` for a missing backend; any other exception ends matching for that candidate. A subclass that raises `NotImplementedError` from `column_names` is eliminated for every name.

### Reader Options

An option is a `PROPERTY_MAPPING` key, read with `cls.read_option(key, options)` (an undeclared key raises). Extend the parent's mapping, since it declares `data_access_handle`. A key that must reject some values at match time, such as a collection, takes a `match_guard`, as core's `data_access_handle` does: a value the guard refuses declines the group with a recorded reason instead of reaching the reader.

```python
from mloda.provider import property_spec


class UnitSensorCsvFG(SensorCsvFG):
    PROPERTY_MAPPING = {
        **SensorCsvFG.PROPERTY_MAPPING,
        "sensor_unit": property_spec(
            "Unit of the sensor values", default="si", match_guard=lambda v: isinstance(v, str), expected="a str unit"
        ),
    }
```

### Document Groups

A `ReadDocumentFG` subclass implements `suffixes()` (optionally `read_text` and `handover_suffixes`) and claims three names, `<Group>`, `<Group>~source` and `<Group>~file_type`, one row per document. A document group skips the suffixes its `handover_suffixes()` lists (none by default). Listing a suffix in the `document_suffixes` context option gives it to document groups and makes `ReadFileFG` groups decline it.

### Decline Names You Cannot Confirm

Record why a group owns a source but cannot serve the request; the reason lands in the near-miss block of the "No feature groups found" error, labeled `(input data)`. `ValueError` from `column_names` already does this for file groups, and `ReadFileFG` and `ReadDBFG` record missing columns automatically. A custom group records its own decline points with `record_match_rejection`, then returns a falsy value:

```python
from mloda.provider import INPUT_DATA_STAGE, record_match_rejection

record_match_rejection(
    "SensorCsvFG",
    "SensorCsvFG matched /data/run1.sensorcsv but could not read its columns: its header lacks the marker",
    stage=INPUT_DATA_STAGE,
)
```

- Record only when ownership is established but the content fails (right suffix, missing column; valid credentials, declined feature). Plain non-matches stay silent.
- Never raise to decline. A raise marked with `escalate_match_abort` propagates; any other raise ends matching for that candidate. Record, then return a falsy value.
- The first recording per owner wins, and recorded reasons are discarded when the group ultimately matches. Name the group and the concrete source in the reason.
- Recording outside an engine-opened window is a no-op; in tests use `collect_match_rejections`.
- A chain-shaped name (`value__rebased`) is not a column of a source, so a `NamePolicy.CHECKED` route never claims it. A group on `NamePolicy.DECLARED` claims exactly `declared_names()` and `declared_name_prefixes()`: keep them to names the group serves. A derived name pointed at an open route goes to the group that computes it.

## Your Own Root FeatureGroup

A source FeatureGroup is the root group: write the behavior you want (its own `compute_framework_rule()`, domain, columns) on the source group itself. A `DataCreatorFeatureGroup` builds its data in `calculate_feature`. The reserved reader option key is `"mloda_source_reader"` (`RESERVED_READER_OPTION_KEY` in `mloda.core.abstract_plugins.components.options`).

## Non-File / HTTP Sources

`ReadFileFG` is file-suffix and directory shaped. For a non-file source write a `SourceFeatureGroup` subclass with a pointed-only open route: `ClaimRoute(kind, NamePolicy.OPEN, False)`. It claims a name only when pointed at (`options={"UbaAirFG": url}` or `feature_group=UbaAirFG`). An open route that is searched raises `TypeError` at class definition. A plain `FeatureGroup` is not a pointer target, and `ApiDataFG` serves in-memory data passed through the API request, it is not an HTTP client.

```python
from typing import Any

from mloda.provider import ClaimRoute, FeatureSet, NamePolicy, SourceFeatureGroup, SourceMatch
from mloda.user import DataAccessCollection, Feature, Options


class UbaAirFG(SourceFeatureGroup):
    CLAIM_ROUTES = (ClaimRoute("url", NamePolicy.OPEN, False),)

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        url = options.get("UbaAirFG")
        if not (isinstance(url, str) and url.startswith("https://api.")):
            return []
        return [SourceMatch(source=url.split("?")[0], access=url)]

    @classmethod
    def load_neutral(
        cls, match: SourceMatch, features: FeatureSet
    ) -> Any: ...  # HTTP GET match.access, return the table


features = [Feature("pm10_value", options={"UbaAirFG": "https://api.example.org/airdata/v4"})]
```

`SourceMatch.source` is the credential-free label (equality and hash use it, never `access`). Several sources found for one name in one group abort, so keep `source` unique per access. A per-framework fast path is an `Implementation[UbaAirFG]` from `mloda.provider`, defined at module level; without one `load_neutral` runs and a transformer converts.

### Audit Identity

Extenders record the source through `data_access_identity(match)`, a classmethod whose default returns `match.source`. Override it for a finer identity (it names the dataset, not the login). Its value is recorded as given, with no stripping in the registry, and a sealed audit log cannot be redacted, so never return a credential. Scrub your own error text with `mloda.provider.scrub_credentials` before logging.

## Required Declarations

A consumer that needs a property of its input's source (a unit, a scale) requires it on the input feature instead of having the source group inspect consumer options and record a rejection. A source group declares scalars with its own `declared_attributes(features)` classmethod (the same declarations extenders record, on output datasets and audit records too, see [Declared attributes](../11-create-extender.md#declared-attributes)). The source group declares for its own step; there is no separate reader declaration.

```python
from collections.abc import Mapping
from typing import Any

from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options


class Pm10Rebased(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("pm10_value", required_declarations={"scale": None})}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any: ...


class ScaledUbaAirFG(UbaAirFG):  # UbaAirFG from above
    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        return {"scale": 0.001}  # unknown scale: omit the key, never declare a placeholder
```

- `None` requires the key with any value, so a declarer that does not know the value omits the key: a placeholder such as `""` or `"unknown"` satisfies the requirement. A concrete value (`{"scale": 0.001}`) must be equal and of the same type.
- Core checks the candidate group's declarations at plan time, calling `declared_attributes` with `features=None`, so a declaration must not depend on the `FeatureSet`. A `declared_attributes` that raises refuses the candidate.
- A refused run fails at plan time, before loading any data, with a reason such as `Pm10Rebased requires declared 'scale'; ScaledUbaAirFG declares none` (when it declares nothing; `declares no such key` when it declares other keys), or `requires declared 'scale' == 5; ScaledUbaAirFG declares 0.001` for a value.
- To keep the check opt-in, add the requirement only when a consumer option asks for it. A chained group whose `input_features` builds plain features from the parsed name overrides `input_features` to attach it. A top-level request can carry it too (`Feature("pm10_value", required_declarations={"scale": None})`; its reason starts with `request for 'pm10_value'`).
- Declarations are also recorded on extender spans, OpenLineage events and audit records (untruncated in the last two), and a sealed audit log cannot be redacted, so never echo an option value or a credential.

## Column Discovery

`describe_columns(match) -> dict[str, DataType | None]` maps each column name to its `DataType` (`None` where unknown). The `SourceFeatureGroup` default maps every name from `columns(match)` to `None` and raises `NotImplementedError` when `columns` returns `None`. `ReadFileFG` lists names from `column_names`; the stock `ParquetFG`, `FeatherFG` and `OrcFG` report the file's stored schema, `JsonFG` pyarrow's inferred types and `SqliteFG` SQLite's declared column types. A non-file group like `UbaAirFG` above enumerates nothing until it overrides `columns` (or `describe_columns`). `count_rows(match, compute_framework)` returns a row count without loading, or `None` when only a read can tell. From a prepared session use `session.describe_columns(step)` and `session.count_rows(step)`.

## Test

Core's contract mixins for source groups ([tests/mixins/reader_feature_groups/](https://github.com/mloda-ai/mloda/tree/0.17.0/tests/mixins/reader_feature_groups) in the mloda repository) are not shipped in the wheel and cannot be imported; read them for what a group must keep. Test end to end: run the feature through `mloda.prepare` or `mloda.run_all` with `PluginCollector.enabled_feature_groups({YourFG})`, so stock groups such as `CsvFG` do not claim the same name:

```python
from mloda.steward import resolve_feature
from mloda.user import Feature, PluginCollector


def test_pointer_reaches_the_group() -> None:
    result = resolve_feature(
        Feature("pm10_value", options={"UbaAirFG": "https://api.example.org/airdata/v4"}),
        plugin_collector=PluginCollector.enabled_feature_groups({UbaAirFG}),
    )
    assert result.feature_group is UbaAirFG, result.error
```

A group that overrides `match_feature_group_criteria` or probes through `gated_probe` also needs an end-to-end gate test: plan with `mloda.prepare` and an `INPUT_DATA_MATCH` extender that sets `never_fall_back = True` and raises for the group, then assert the refusal is on `FeatureResolutionError.refusals` (or that a sibling serves the feature). To make a gate current without a run, wrap a matcher call in `mloda.steward.reader_probe_scope(ReaderProbeScope(extender, None, None, None, None))`. Core's [test_input_data_match_hook.py](https://github.com/mloda-ai/mloda/blob/0.17.0/tests/test_plugins/extender/test_input_data_match_hook.py) shows the pattern.

## Real Implementations

| File | Description |
|------|-------------|
| [source_feature_group.py](https://github.com/mloda-ai/mloda/blob/0.17.0/mloda/core/abstract_plugins/components/input_data/source_feature_group.py) | `SourceFeatureGroup`: claim routes, `find_sources`, `load_neutral`, `gated_probe` |
| [claim_route.py](https://github.com/mloda-ai/mloda/blob/0.17.0/mloda/core/abstract_plugins/components/input_data/claim_route.py) | `ClaimRoute`, `NamePolicy`, `SourceMatch` |
| [read_file_fg.py](https://github.com/mloda-ai/mloda/blob/0.17.0/mloda/core/abstract_plugins/components/input_data/read_file_fg.py) | `ReadFileFG`: `suffixes`, `column_names` |
| [read_db_fg.py](https://github.com/mloda-ai/mloda/blob/0.17.0/mloda/core/abstract_plugins/components/input_data/read_db_fg.py) | `ReadDBFG`: credentials, catalogs, optional query route |
| [csv_fg.py](https://github.com/mloda-ai/mloda/blob/0.17.0/mloda_plugins/feature_group/input_data/file_formats/csv_fg.py) | `CsvFG` |

## Combines With

- **Pattern 1 (Root features)**: source groups are the root features that read files, documents and databases
- **Pattern 17 (Data connection matching)**: `DataAccessCollection` and handles for connection-shaped sources
- **Pattern 11 (Options)**: option keys and context vs group semantics
- **Pattern 26 (Input-feature option forwarding)**: a forwarded pointer binds every input it reaches
- **Pattern 28 (Backend families)**: selecting a sibling source group by the credential slot it finds instead of a class-name key
