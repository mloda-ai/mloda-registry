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

### The reserved "BaseInputData" key

When a reader matches, the `(ReaderClass, data_access)` pair is stored under the reserved `"BaseInputData"` options key and consumed by `init_reader` at load time. The class-name option key is the normal way to select a reader; advanced callers and tests may preseed the reserved key directly. Setting it twice with different values raises `ValueError`.

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

## Non-File / HTTP Sources

`ReadFile`'s default matching (`match_read_file_data_access`) is file-suffix and directory shaped. For a non-file source (an HTTP endpoint returning JSON), the sanctioned recipe is: subclass `ReadFile`, override `match_subclass_data_access` and `load_data` wholesale. On that path `suffix()` is never consulted (it is inert), so you do not implement it.

`ApiInputData` is not the tool for this despite its name: it injects in-memory data passed through the API request and is not an HTTP client.

A reader that overrides `load_data` wholesale is classified as a final reader structurally; no reader code runs during classification.

## Decline Names You Cannot Confirm

A wholesale `match_subclass_data_access` override replaces `ReadFile`'s own column check entirely, so nothing stops it from claiming a feature name it has no way to verify. If a chained group elsewhere forwards this reader's option key for a name like `value__rebased` (Pattern 26), an unconditional accept collides with that group and resolution fails with `Multiple feature groups found`, pointing at neither reader as the cause.

Decline a name that carries the chain separator before accepting the data access (core only strips a multi-output `~N` suffix, so a wholesale override never sees `COLUMN_SEPARATOR` on the matching path; check for it anyway if you call `match_subclass_data_access` directly, e.g. from a test), and record why so the rejection is attributable instead of silent:

```python
from mloda.provider import CHAIN_SEPARATOR, COLUMN_SEPARATOR, INPUT_DATA_STAGE, record_match_rejection


# UbaAirReader.match_subclass_data_access, extended to decline first:
@classmethod
def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
    if any(CHAIN_SEPARATOR in name or COLUMN_SEPARATOR in name for name in feature_names):
        record_match_rejection(
            cls.get_class_name(),
            f"{cls.get_class_name()} cannot confirm a chain/column-separated feature name",
            stage=INPUT_DATA_STAGE,
        )
        return None
    if isinstance(data_access, str) and data_access.startswith("https://api."):
        return data_access
    return None
```

A stock `ReadFile` subclass that never implements `get_column_names`, and a stock `ReadDB` subclass that never implements `check_feature_in_data_access`, already decline a chain-separated name for free from mloda core, recorded rejection included. Only a wholesale `match_subclass_data_access` override needs the decline written out like this. If your reader can list its columns, confirm the name instead of declining it outright: implement `get_column_names` (or `check_feature_in_data_access` for `ReadDB`) and let the built-in check do the work.

If your wholesale override is still file-shaped, call `cls._file_matches(path, feature_names, document_suffixes)` rather than hand-writing this check: it keeps the suffix check, the `document_suffixes` exclusion, `validate_columns`, and the recorded decline intact.

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
| [base_input_data.py](https://github.com/mloda-ai/mloda/blob/main/mloda/core/abstract_plugins/components/input_data/base_input_data.py) | `feature_scope_data_access`, class-or-string key helper, `init_reader` |
| [read_file.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/input_data/read_file.py) | `ReadFile` base, `match_subclass_data_access` seam |
| [test_sibling_reader_selection.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/feature_group/input_data/test_sibling_reader_selection.py) | The pinned selection contract |
| [test_reader_declines_chain_separated_names.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/feature_group/input_data/test_reader_declines_chain_separated_names.py) | Pins the chain/column-separated-name decline contract for `ReadFile`/`ReadDB` |

See also [Data Access Patterns](https://mloda-ai.github.io/mloda/in_depth/data-access-patterns/) for the underlying model.

## Combines With

- **Pattern 1 (Root features)**: readers are the `input_data()` of root features
- **Pattern 17 (Data connection matching)**: `DataAccessCollection` and handles for connection-shaped sources
- **Pattern 11 (Options)**: option keys and context vs group semantics
