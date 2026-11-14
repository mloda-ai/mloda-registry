# Source FeatureGroups: Pointing, Custom Formats and Non-File Sources

How to point a feature at a source FeatureGroup and how to write your own for a file format or a non-file (HTTP/API) source.

**What**: One FeatureGroup per kind of source (`CsvFG`, `ParquetFG`, `SqliteFG`, your own), selected per feature by an Options key equal to the group's class name.
**When**: A plugin reads a file format, a database or an API that no stock group covers, or a request must be routed to one of several sources.
**Why**: Source selection is contract-driven, not guesswork; knowing the contract avoids reading mloda core source.
**Where**: Connector plugins (a CKAN CSV group, a direct-URL group, a REST JSON group side by side).

Core describes the model (claim routes, resolution, errors): [Data Access Patterns](https://github.com/mloda-ai/mloda/blob/0.20.0/docs/docs/in_depth/data-access-patterns.md). Request data (`api_data=`) is served by the source group `ApiDataFG`; synthetic data by `DataCreatorFeatureGroup`.

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
- When a request reaches a `NamePolicy.DECLARED` route (searched, or unlocked by its `required_options`) for a name it declares and `find_sources` finds no source, core records a reason: `<Group> found no source serving '<name>'` unpointed, `<Group> was pointed at but found no source for '<name>'` under an explicit pointer (a class-name options key, or `feature_group=` naming the group). A `feature_group=` scope naming a concrete family base records nothing. So record in `find_sources` only for a more specific, domain-worded reason, or under a base scope. A derived group that is the only match for its name skips the input-chain check (see [Feature Matching](14-feature-matching.md#how-it-works)) and fails at this input, whose recorded reason names the cause.
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

from mloda.provider import ClaimRoute, FeatureSet, NamePolicy, SourceFeatureGroup, SourceMatch, scrub_credentials
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
        return [SourceMatch(source=scrub_credentials(url).split("?")[0].split("#")[0], access=url)]

    @classmethod
    def load_neutral(
        cls, match: SourceMatch, features: FeatureSet
    ) -> Any: ...  # HTTP GET match.access, return a columnar dict

    @classmethod
    def neutral_type(cls, match: SourceMatch) -> type | None:
        return dict


features = [Feature("pm10_value", options={"UbaAirFG": "https://api.example.org/airdata/v4"})]
```

`SourceMatch.source` is the credential-free label (equality and hash use it, never `access`). Several sources found for one name in one group abort, so keep `source` unique per access. A per-framework fast path is an `Implementation[UbaAirFG]` from `mloda.provider`, defined at module level; without one `load_neutral` runs and a transformer converts.

`neutral_type` names the type `load_neutral` returns (default `None`). A run without `compute_frameworks` then prefers a framework that takes it as is (here `PythonDictFramework`), and the plan step says `loads natively`. Without it the first framework in default order is used, and its conversion can fail on mixed or nested cells. Runs that pass `compute_frameworks` are unaffected. `ApiDataFG` reports its own value type but returns `None` once a subclass overrides `load_neutral` or `api_values`, so such a subclass declares `neutral_type` itself.

### Binding a Collection Credential

A subclass can bind a credential from a `DataAccessCollection` slot at match time and report the collection entry itself:

```python
from typing import Any, NamedTuple

from mloda.provider import INPUT_DATA_STAGE, ClaimRoute, FeatureSet, SourceMatch, record_match_rejection
from mloda.user import DataAccessCollection, Options


class UbaAccess(NamedTuple):
    url: str
    credentials: Any  # the collection entry itself, never a copy


class KeyedUbaAirFG(UbaAirFG):  # UbaAirFG from above
    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        found = super().find_sources(route, feature_name, options, data_access_collection)
        slots = data_access_collection.credentials if data_access_collection else {}
        handle = options.get("data_access_handle")
        entries = [
            entry for name, entry in slots.items() if name == handle or (handle is None and "uba_token" in entry)
        ]
        if found and len(entries) != 1:
            reason = f"{cls.get_class_name()} needs one credential slot holding uba_token; data_access_handle picks one"
            record_match_rejection(cls.get_class_name(), reason, stage=INPUT_DATA_STAGE)
            return []
        return [SourceMatch(match.source, UbaAccess(match.access, entries[0])) for match in found]

    @classmethod
    def match_credentials(cls, match: SourceMatch) -> Any | None:
        return match.access.credentials

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: FeatureSet) -> Any: ...  # GET match.access.url with its token
```

`data_access_handle` picks the slot when several hold a token; see [Credentials](#credentials-bound-at-match-time-or-read-at-run-time) for why `match_credentials` returns the entry, not a copy.

### Reading a Rotating Credential

A credential that must change between runs is read inside `load_neutral`, which runs in the same read window as `calculate_feature`. There is no `match_credentials` override, so the handle is declared, not bound, and `session.run(credentials=...)` rotates it per run. Point the feature at it with `Feature("pm10_value", options={"UbaAirFG": url, "token_handle": "api"})`.

```python
from typing import Any, ClassVar

from mloda.provider import FeatureSet, SourceMatch, credential_handle_spec, read_credential


class RotatingUbaAirFG(UbaAirFG):  # UbaAirFG from above
    TOKEN_HANDLE = "token_handle"
    PROPERTY_MAPPING: ClassVar = {
        TOKEN_HANDLE: credential_handle_spec(
            "Credential handle whose 'token' field authorizes the GET", required_fields=("token",)
        )
    }

    @classmethod
    def fetch(cls, url: str, token: str) -> Any: ...  # HTTP GET url with the token

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: FeatureSet) -> Any:
        token = read_credential(features.get_options_key(cls.TOKEN_HANDLE))["token"]
        return cls.fetch(match.access, token)
```

### Audit Identity

Extenders record the source through `data_access_identity(match)`, a classmethod whose default returns `match.source`. Override it for a finer identity (it names the dataset, not the login). Its value is recorded as given, with no stripping in the registry, and a sealed audit log cannot be redacted, so never return a credential. Scrub your own error text with `mloda.provider.scrub_credentials` before logging. Core's file groups cut a URI path parameter region (`;...`) to `;***` in `match.source`; a group that builds its own `source` from a URI, or an override that returns one, cuts it the same way.

## Credentials: Bound at Match Time or Read at Run Time

Since mloda 0.20.0 a source group has two paths: bind what opens or selects the source at match time; read what is used on loaded rows, or must rotate, at run time.

- **Bind at match time** ([example](#binding-a-collection-credential)): `find_sources` takes the credential from a collection slot (narrowed by `data_access_handle`) and carries it in `SourceMatch.access`. The value is fixed at `prepare`. Override `match_credentials(match)` to return the collection entry itself, not a field or a copy (`ReadDBFG` already does): only then do the plan, lock and hook contexts name the handle and `session.run(credentials=...)` refuses a per-run value for it ("bound by a source step"). Without the override nothing is recorded, and a per-run value for that handle reaches only another step that declares it, while the source keeps its prepare-time value. A copy or a pointer value records `<pointer>`, and a bare `Credential(...)` records `<unnamed>`, so register under a name. Binding grants no `read_credential`.
- **Read at run time** ([example](#reading-a-rotating-credential)): declare a `credential_handle_spec` option and call `read_credential` in `load_neutral` or `calculate_feature`. The plan lists the handle in `credential_handles` with `bound_credential_handle` None (declared, not bound), and `session.run(credentials=...)` passes that run a fresh value.

A `Credential` may sit in options only as a source group's class-name pointer value (`{"SqliteFG": Credential(...)}`); core refuses it on any other group, and it is the one exception to "never put a secret in options". Prefer a named slot.

A group never reads `os.environ` and no option or slot names an environment variable: the caller reads it (`Credential(token=os.environ["API_TOKEN"])`). A source that needs no credential binds nothing (`match_credentials` returns `None`), and one that needs it only sometimes, such as on a cache miss, reads it at run time. See [Secrets](11-options.md#secrets) and core's [Runtime access versus match-time binding](https://github.com/mloda-ai/mloda/blob/0.20.0/docs/docs/in_depth/named-data-access-handles.md#runtime-access-versus-match-time-binding).

## Families of Sibling Sources

A base that implements `find_sources` and `load_neutral` is concrete, so its class name is a pointer key and a `feature_group=` scope for every subclass; an abstract base drops out of both, so `feature_group="Base"` covers the family only when the base is concrete.

A class-name pointer on a derived request forwards to every input (see [Input-feature forwarding](26-input-feature-forwarding.md#caveat-a-forwarded-source-pointer-reaches-every-input)). Under that pointer a `NamePolicy.CHECKED` sibling whose source lacks the column fails the request, even when another sibling has it; a `NamePolicy.DECLARED` sibling records `'x' is not a name <Group> declares` and declines. A family serving a fixed set of columns uses `DECLARED`.

For picking a sibling by the credential slot it finds, and an abstract base whose leaves never compete, see [Way 3](28-backend-families.md#way-3-reader-delegation-sibling-readers).

```python
from typing import Any, ClassVar

from mloda.provider import ClaimRoute, FeatureSet, NamePolicy, SourceFeatureGroup, SourceMatch
from mloda.steward import resolve_feature
from mloda.user import DataAccessCollection, Feature, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class StationFG(SourceFeatureGroup):
    COLUMNS: ClassVar[frozenset[str]] = frozenset()
    CLAIM_ROUTES = (ClaimRoute("station", NamePolicy.DECLARED, False),)

    @classmethod
    def declared_names(cls) -> frozenset[str]:
        return cls.COLUMNS

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        return [SourceMatch(source=cls.__name__)]

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: FeatureSet) -> Any:
        return {name: [1] for name in features.get_all_names()}


class TemperatureFG(StationFG):
    COLUMNS = frozenset({"temperature"})


class WindFG(StationFG):
    COLUMNS = frozenset({"wind_speed"})


def test_family_pointer_reaches_the_sibling_that_declares_the_name() -> None:
    family = PluginCollector.enabled_feature_groups({StationFG, TemperatureFG, WindFG})
    scoped = Feature("wind_speed", feature_group=StationFG)
    pointed = Feature("wind_speed", options={"StationFG": "station-7"})
    for feature in (scoped, pointed):
        result = resolve_feature(feature, plugin_collector=family, compute_frameworks={PythonDictFramework})
        assert result.feature_group is WindFG, result.error
```

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

Conformance suites for source groups live in `mloda.testing.source`. Mix in the one that fits the base class: `SourceFeatureGroupTestMixin` (`source_suite`), `ReadFileFGTestMixin` (`read_file_suite`), `ReadDBFGTestMixin` (`read_db_suite`) or `ReadDocumentFGTestMixin` (`read_document_suite`). `loads_into_framework_suite` and `column_discovery_suite` add further checks. A host for `SensorCsvFG` (its `load_neutral` must really parse the file; the sketch above stubs it):

```py
from pathlib import Path
from typing import Any

from mloda.testing.source.read_file_suite import ReadFileFGTestMixin


class TestSensorCsvFG(ReadFileFGTestMixin):
    feature_group_class = SensorCsvFG
    present_column = "pm10_value"
    missing_column = "no_such_column"

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        names = list(columns)
        rows = zip(*(columns[name] for name in names))
        lines = ["#sensor-schema " + ",".join(names), *(",".join(str(v) for v in row) for row in rows)]
        path.write_text("\n".join(lines) + "\n")
```

Test end to end: run the feature through `mloda.prepare` or `mloda.run_all` with `PluginCollector.enabled_feature_groups({YourFG})`, so stock groups such as `CsvFG` do not claim the same name:

```python
from typing import Any

import pytest

from mloda.steward import resolve_feature
from mloda.user import Credential, DataAccessCollection, Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


def test_pointer_reaches_the_group() -> None:
    result = resolve_feature(
        Feature("pm10_value", options={"UbaAirFG": "https://api.example.org/airdata/v4"}),
        plugin_collector=PluginCollector.enabled_feature_groups({UbaAirFG}),
    )
    assert result.feature_group is UbaAirFG, result.error


def test_keyed_reader_records_the_bound_handle() -> None:
    plan = mloda.explain(
        [Feature("pm10_value", options={"UbaAirFG": "https://api.example.org/airdata/v4"})],
        data_access_collection=DataAccessCollection(credentials={"uba": Credential(uba_token="placeholder-token")}),
        plugin_collector=PluginCollector.enabled_feature_groups({KeyedUbaAirFG}),
        compute_frameworks=[PythonDictFramework],
    )
    assert [step.bound_credential_handle for step in plan] == ["uba"]
    assert [step.credential_handles for step in plan] == [("uba",)]


def test_keyed_reader_refuses_a_per_run_value_for_its_bound_handle() -> None:
    session = mloda.prepare(
        [Feature("pm10_value", options={"UbaAirFG": "https://api.example.org/airdata/v4"})],
        data_access_collection=DataAccessCollection(credentials={"uba": Credential(uba_token="placeholder-token")}),
        plugin_collector=PluginCollector.enabled_feature_groups({KeyedUbaAirFG}),
        compute_frameworks=[PythonDictFramework],
    )
    with pytest.raises(ValueError, match="bound by a source step"):
        session.run(credentials={"uba": Credential(uba_token="rotated-placeholder-token")})


def test_rotating_reader_reads_each_runs_credential() -> None:
    tokens: list[str] = []

    class RecordingRotatingUbaAirFG(RotatingUbaAirFG):
        @classmethod
        def fetch(cls, url: str, token: str) -> Any:
            tokens.append(token)
            return {"pm10_value": [1]}

    kwargs: dict[str, Any] = {
        "plugin_collector": PluginCollector.enabled_feature_groups({RecordingRotatingUbaAirFG}),
        "compute_frameworks": [PythonDictFramework],
        "data_access_collection": DataAccessCollection(credentials={"api": Credential(token="first-api-token")}),
    }
    features = [
        Feature("pm10_value", options={"UbaAirFG": "https://api.example.org/airdata/v4", "token_handle": "api"})
    ]
    plan = mloda.explain(features, **kwargs)
    assert [step.credential_handles for step in plan] == [("api",)]
    assert [step.bound_credential_handle for step in plan] == [None]
    session = mloda.prepare(features, **kwargs)
    session.run()
    session.run(credentials={"api": Credential(token="rotated-api-token")})
    session.run()
    assert tokens == ["first-api-token", "rotated-api-token", "first-api-token"]
```

A group that overrides `match_feature_group_criteria` or probes through `gated_probe` also needs an end-to-end gate test: plan with `mloda.prepare` and an `INPUT_DATA_MATCH` extender that sets `never_fall_back = True` and raises for the group, then assert the refusal is on `FeatureResolutionError.refusals` (or that a sibling serves the feature). To make a gate current without a run, wrap a matcher call in `mloda.steward.reader_probe_scope(ReaderProbeScope(extender, None, None, None, None))`. Core's [test_input_data_match_hook.py](https://github.com/mloda-ai/mloda/blob/0.20.0/tests/test_plugins/extender/test_input_data_match_hook.py) shows the pattern.

## Real Implementations

| File | Description |
|------|-------------|
| [source_feature_group.py](https://github.com/mloda-ai/mloda/blob/0.20.0/mloda/core/abstract_plugins/components/input_data/source_feature_group.py) | `SourceFeatureGroup`: claim routes, `find_sources`, `load_neutral`, `gated_probe` |
| [claim_route.py](https://github.com/mloda-ai/mloda/blob/0.20.0/mloda/core/abstract_plugins/components/input_data/claim_route.py) | `ClaimRoute`, `NamePolicy`, `SourceMatch` |
| [read_file_fg.py](https://github.com/mloda-ai/mloda/blob/0.20.0/mloda/core/abstract_plugins/components/input_data/read_file_fg.py) | `ReadFileFG`: `suffixes`, `column_names` |
| [read_db_fg.py](https://github.com/mloda-ai/mloda/blob/0.20.0/mloda/core/abstract_plugins/components/input_data/read_db_fg.py) | `ReadDBFG`: credentials, catalogs, optional query route |
| [csv_fg.py](https://github.com/mloda-ai/mloda/blob/0.20.0/mloda_plugins/feature_group/input_data/file_formats/csv_fg.py) | `CsvFG` |

## Combines With

- **Pattern 1 (Root features)**: source groups are the root features that read files, documents and databases
- **[Pattern 17 (Data connection matching)](17-data-connection-matching.md#credentials)**: `DataAccessCollection`, credential slots and handles for connection-shaped sources
- **Pattern 11 (Options)**: option keys, context vs group semantics, and credential handles a group reads at run time
- **Pattern 26 (Input-feature option forwarding)**: a forwarded pointer binds every input it reaches
- **Pattern 28 (Backend families)**: selecting a sibling source group by the credential slot it finds instead of a class-name key
