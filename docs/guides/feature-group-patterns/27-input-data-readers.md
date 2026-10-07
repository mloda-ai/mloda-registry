# Input-Data Readers: Sibling Selection and Non-File Sources

How to run several `BaseInputData` readers under one root FeatureGroup and point each feature at the right one, including readers for non-file (HTTP/API) sources.

**What**: Select a specific reader per feature via an Options key equal to the reader's class name; build non-file readers by subclassing `ReadFile`.
**When**: One root FeatureGroup fronts multiple data sources (CSV file, CSV URL, JSON REST endpoint), each with its own reader.
**Why**: Reader selection is contract-driven, not guesswork; knowing the contract avoids reading mloda core source.
**Where**: Connector plugins with several readers (e.g. a CKAN CSV reader, a direct-URL CSV reader, and a REST JSON reader side by side).

## The Selection Contract

A feature selects a reader with an Option whose **key equals the reader's class name** (`BaseInputData.data_access_name()`, i.e. `cls.__name__`). The value is the data access the reader receives (a path, URL, or any object the reader understands):

```python
from mloda.user import Feature

Feature("pm10_value", options={UbaAirReader.__name__: "https://api.example.org/airdata/v4"})
```

The reader class itself also works as the key. It is normalized to the class-name string when the `Options` object is constructed, so both spellings are one identity (equal, same hash, same lookup):

```python
Feature("pm10_value", options={UbaAirReader: "https://api.example.org/airdata/v4"})
```

As long as sibling readers have distinct class names, they cannot collide: each feature's option key routes to exactly one reader. Two readers sharing a `__name__` across modules are unsupported. A key that names no known reader simply matches nothing.

### Several Accepting Readers

Distinct names only protect pinned features. Without a pin, every reader in the family probes the `DataAccessCollection`, and when more than one accepts, the feature fails to resolve (`No feature groups found`) with the root group's match-hook reason `ValueError: Several readers accept the data access for feature(s) ...`; no reader wins by order. A feature pinning several readers that all accept fails with `pins several readers that accept it`. Resolve an overlap, with your own readers or with a stock one owning the same suffix, by one of:

- **Subclass the reader you extend.** A subclass that accepts an equal data access replaces its parent instead of conflicting (a custom `.csv` reader subclasses `CsvReader`).
- **Narrow `match_subclass_data_access`** (suffix, scheme, host) so the accepts are disjoint.
- **Pin per feature** by the reader's option key, as above.

### The reserved "BaseInputData" key

While matching, the `(ReaderClass, data_access)` pair is written under the reserved `"BaseInputData"` options key, then moved onto `Feature.input_data_match` and removed from options (it may carry credentials). `BaseInputData.init_reader` takes that pair at load time. The class-name option key is the normal way to select a reader; tests may call `feature_scope_data_access` directly, which writes the key into the given `Options`. Setting it twice with different values raises `ValueError`. Two candidates that both match raise `Multiple feature groups found`, naming each candidate and its source.

## Sibling Readers Under One Root FeatureGroup

The option key selects the reader; the selected reader then confirms the data access via `match_subclass_data_access`. Return the data access to claim it, `None` to decline:

```python
from typing import Any
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class GovDataReader(ReadFile):
    """CKAN-hosted CSV."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if isinstance(data_access, str) and "ckan" in data_access:
            return data_access
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any: ...  # fetch and parse, return the table


class UbaAirReader(ReadFile):
    """REST JSON endpoint."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if isinstance(data_access, str) and data_access.startswith("https://api."):
            return data_access
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any: ...  # HTTP GET, normalize JSON, return the table
```

Both are siblings under the stock `ReadFileFeature` root group; no extra FeatureGroup is needed. The user routes per feature (snippets on this page build on each other):

```python
features = [
    Feature("population", options={GovDataReader.__name__: ckan_url}),
    Feature("pm10_value", options={UbaAirReader: api_url}),
]
```

Both `GovDataReader` and `UbaAirReader` above accept any `feature_names` unconditionally; see [Decline Names You Cannot Confirm](#decline-names-you-cannot-confirm) below before shipping either as written.

## Your Own Root FeatureGroup

`ReadFileFeature`, `ReadDBFeature` and `ReadDocumentFeature` front every subclass of `ReadFile`, `ReadDB` and `ReadDocument`, HTTP readers built on `ReadFile` included (see [Non-File / HTTP Sources](#non-file--http-sources)). Write your own root group only when the group needs behavior of its own, such as its own `compute_framework_rule()` or domain. Its reader family must derive from `BaseInputData` directly: a family built on a stock base under a second root group also matches the stock root and fails with `Multiple feature groups found`. The root group loads the data in `calculate_feature`, as `ReadFileFeature` does:

```python
from typing import Any
from mloda.provider import BaseInputData, FeatureGroup, FeatureSet
from mloda.user import Options


class WarehouseReader(BaseInputData):
    """Family base; a feature picks a subclass by its class-name option key."""


class SalesTableReader(WarehouseReader):
    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if isinstance(data_access, str) and data_access.startswith("warehouse://sales"):
            return data_access
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any: ...  # query, return the table


class WarehouseFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return WarehouseReader()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return WarehouseReader().load(features)
```

Core hands a root group no loaded data, so returning `data` instead fails at run time with `ValueError: Data <class 'NoneType'> is not supported by PandasDataFrame` (named after the compute framework in use). A `DataCreator` group is different: it builds its data in `calculate_feature`.

## Non-File / HTTP Sources

`ReadFile`'s default matching (`match_read_file_data_access`) is file-suffix and directory shaped. For a non-file source (an HTTP endpoint returning JSON), the sanctioned recipe is: subclass `ReadFile`, override `match_subclass_data_access` and `load_data` wholesale. On that path `suffix()` is never consulted (it is inert), so you do not implement it.

`ApiInputData` is not the tool for this despite its name: it injects in-memory data passed through the API request and is not an HTTP client.

A reader that overrides `load_data` wholesale is classified as a final reader structurally; no reader code runs during classification.

Extenders record the data access through the reader's `data_access_identity(data_access)` classmethod. Core's default keeps a URL's scheme, host and path and drops the query (a URL it cannot parse, such as one with `@` in the query, becomes `str`), so endpoints that differ only in their query share one identity in audit records, lineage datasets and spans. In lineage, a well-formed `s3://bucket/key`, `gs://bucket/key`, `abfss://container@account.dfs.core.windows.net/path` or `file:///path` identity (or an absolute path) becomes an OpenLineage namespace and name, while malformed ones (a port, an empty key or bucket, another host) do not; any other identity keeps `dataset_namespace` as its namespace, and the `dataSource` facet name is always the identity. Override it for a finer identity: its value is recorded as given, with no stripping in the registry, and a sealed audit log cannot be redacted, so never return a credential. Core flags a fallback identity (`data_access_identity_is_fallback`); audit records and load spans carry the flag and lineage marks the dataset.

## Reader Options

Declare the options a reader reads in `READER_OPTIONS`, one `PropertySpec` per key (declarations merge along the class hierarchy, the subclass winning), and read them with `cls.reader_option(key, options)`. It reads the group or context value as given, falls back to the declared default, and raises `ValueError` for an undeclared key, or for an absent key with no default (such a key is required at match time):

```python
from typing import Any, ClassVar
from mloda.provider import FeatureSet, PropertySpec


class UbaAirReader(ReadFile):  # added to UbaAirReader above
    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        "uba_station": PropertySpec(
            "One station id",
            default=None,
            strict_validation=True,
            scalar_only=True,
            element_validator=lambda value: isinstance(value, str),
        ),
    }

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        station = cls.reader_option("uba_station", features.options)  # None when unset
        ...
```

`strict_validation=True` (which needs `allowed_values` or an `element_validator`) checks a list, tuple or set value element by element, so `["DEBE010", "DEBE034"]` passes. `scalar_only=True` rejects the collection outright; it is reader-only (a `PROPERTY_MAPPING` spec with it is rejected), exists only on `PropertySpec` (not the `property_spec` builder) and requires `strict_validation=True`. Both checks run at match time, before `match_subclass_data_access` is called: a value they refuse makes the reader a non-match, and the resolution error names the reason, for example `reader option 'uba_station' value is a list of 2 elements, but the declaration of UbaAirReader marks it scalar_only and rejects a collection outright`. See [PROPERTY_MAPPING Configuration](https://mloda-ai.github.io/mloda/in_depth/property-mapping/) for the spec fields.

## Required Declarations

A consumer that needs a property of its input's source (a unit, a scale) requires it on the input feature instead of having the reader inspect consumer options and call `record_match_rejection`. Readers and feature groups declare scalars with a `declared_attributes(features)` classmethod (the same declarations extenders record, see [Declared attributes](../11-create-extender.md#declared-attributes)):

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


class UbaAirReader(ReadFile):  # added to UbaAirReader above
    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        return {"scale": 0.001}  # unknown scale: omit the key, never declare a placeholder
```

- `None` requires the key with any value, so a declarer that does not know the value omits the key: a placeholder such as `""` or `"unknown"` satisfies the requirement. A concrete value (`{"scale": 0.001}`) must be equal and of the same type.
- Core checks the candidate group's declarations merged with the selected reader's (the reader wins on a shared key) at plan time, calling `declared_attributes` with `features=None`, so a declaration must not depend on the `FeatureSet`. An unpinned reader that misses is skipped, so a sibling that declares the key can match; a pinned reader is final. A `declared_attributes` that raises refuses the candidate.
- A refused run fails at plan time, before loading any data, with a reason such as `Pm10Rebased requires declared 'scale'; UbaAirReader declares none` (when it declares nothing; `declares no such key` when it declares other keys), or `requires declared 'scale' == 5; UbaAirReader declares 0.001` for a value.
- To keep the check opt-in, add the requirement only when a consumer option asks for it. A chained group whose `input_features` builds plain features from the parsed name overrides `input_features` to attach it. A top-level request can carry it too (`Feature("pm10_value", required_declarations={"scale": None})`; its reason starts with `request for 'pm10_value'`).
- Declarations are also recorded on extender spans, so never put a credential in one.

## Decline Names You Cannot Confirm

A wholesale `match_subclass_data_access` override replaces `ReadFile`'s own column check entirely, so nothing stops it from claiming a feature name it has no way to verify. If a consumer's own name is chain-shaped (`value__rebased`) and it forwards this reader's option key to its upstream source feature (Pattern 26), an unconditional accept collides with the root group using the same reader, and resolution fails with `Multiple feature groups found`, pointing at neither reader as the cause.

Once the data access is confirmed yours, decline a name that carries the chain separator before returning it (core strips only a trailing digits-only `~N` multi-output suffix, so a custom non-digit part suffix, Pattern 5, still carries `COLUMN_SEPARATOR` on the matching path; check for both). Record why: the recorded rejection is not just diagnostic, it gates the name-based resolution rules for this owner, so a silent decline can still leave the collision in place.

```python
from mloda.provider import CHAIN_SEPARATOR, COLUMN_SEPARATOR, INPUT_DATA_STAGE, record_match_rejection


# UbaAirReader.match_subclass_data_access, extended to decline after confirming ownership:
@classmethod
def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
    if not (isinstance(data_access, str) and data_access.startswith("https://api.")):
        return None
    if any(CHAIN_SEPARATOR in name or COLUMN_SEPARATOR in name for name in feature_names):
        record_match_rejection(
            cls.get_class_name(),
            f"{cls.get_class_name()} cannot confirm a chain/column-separated feature name",
            stage=INPUT_DATA_STAGE,
        )
        return None
    return data_access
```

A stock `ReadFile` subclass that never implements `get_column_names`, and a stock `ReadDB` subclass that never implements `check_feature_in_data_access`, already decline a chain-separated name for free from mloda core, recorded rejection included, unless the name is pinned via `column_to_file`. Only a wholesale `match_subclass_data_access` override needs the decline written out like this: it bypasses the built-in check entirely, so a subclass that declares `get_column_names` but raises `NotImplementedError` gets neither the free decline nor a wholesale one. If your reader can list its columns, prefer that over a wholesale override: implement `get_column_names` (or `check_feature_in_data_access` for `ReadDB`) instead, and let the built-in check do the work for you.

If your wholesale override still matches by file suffix (i.e. you do implement `suffix()`, unlike the HTTP case above), delegate to `cls._file_matches(path, feature_names, document_suffixes)` rather than hand-writing this check: it returns a bool, so use `return path if cls._file_matches(...) else None`, and it keeps the `document_suffixes` exclusion (from `cls.reader_option("document_suffixes", options)`, see [Reader Options](#reader-options)), `validate_columns`, and the recorded decline intact.

## Column Discovery

The `get_column_names` seam used above to decline chain-separated names also backs `describe_columns(data_access) -> dict[str, DataType | None]` on `BaseInputData`: it maps each column name to its `DataType` (`None` where unknown), and raises `NotImplementedError` by default. `ReadFile` wraps it around `get_column_names` for free (every name mapped to `None`) once a subclass implements that; a reader that only overrides `match_subclass_data_access`/`load_data` wholesale, like `UbaAirReader` and `GovDataReader` above, never implements `get_column_names`, so the inherited default still raises `NotImplementedError` unless the reader overrides `describe_columns` directly. `ParquetReader`, `FeatherReader`, and `OrcReader` override it with the file's stored schema, `JsonReader` with pyarrow's inferred types, and `ReadDB`'s `SQLITEReader` with SQLite's declared column types.

## Test

Follow the contract at unit level, then end to end:

```python
from mloda.provider import BaseInputData
from mloda.user import Options


def test_option_key_routes_to_reader() -> None:
    options = Options({UbaAirReader.__name__: "https://api.example.org/airdata/v4"})
    assert BaseInputData.feature_scope_data_access(options, "pm10_value") is True
    assert options.get("BaseInputData")[0] is UbaAirReader
```

End to end, run the feature through `mloda.run_all` with `PluginCollector.enabled_feature_groups({ReadFileFeature})`. mloda core pins the full contract in [test_sibling_reader_selection.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/feature_group/input_data/test_sibling_reader_selection.py); mirror its isolation trick (a unique marker option key checked in `match_subclass_data_access`) so test readers never hijack matching in unrelated tests.

## Real Implementations

| File | Description |
|------|-------------|
| [base_input_data.py](https://github.com/mloda-ai/mloda/blob/main/mloda/core/abstract_plugins/components/input_data/base_input_data.py) | `feature_scope_data_access`, class-or-string key helper, `init_reader` (takes the matched `(ReaderClass, data_access)` pair) |
| [read_file.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/input_data/read_file.py) | `ReadFile` base, `match_subclass_data_access` seam |
| [test_sibling_reader_selection.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/feature_group/input_data/test_sibling_reader_selection.py) | The pinned selection contract |
| [test_reader_declines_chain_separated_names.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/feature_group/input_data/test_reader_declines_chain_separated_names.py) | Pins the chain/column-separated-name decline contract for `ReadFile`/`ReadDB` |

See also [Data Access Patterns](https://mloda-ai.github.io/mloda/in_depth/data-access-patterns/) for the underlying model.

## Combines With

- **Pattern 1 (Root features)**: readers are the `input_data()` of root features
- **Pattern 17 (Data connection matching)**: `DataAccessCollection` and handles for connection-shaped sources
- **Pattern 11 (Options)**: option keys and context vs group semantics
- **Pattern 26 (Input-feature option forwarding)**: a forwarded reader key can collide with a root reader that doesn't decline names it cannot confirm
- **Pattern 28 (Backend families)**: selecting a sibling reader by the data access it accepts (a credential slot) instead of a class-name key
