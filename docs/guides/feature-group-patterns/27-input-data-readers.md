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

Because class names are unique per class, sibling readers never collide: each feature's option key routes to exactly one reader. A key that names no known reader simply matches nothing.

### The reserved "BaseInputData" key

When a reader matches, the `(ReaderClass, data_access)` pair is stored under the reserved `"BaseInputData"` options key and consumed by `init_reader` at load time. Do not set this key yourself; setting it twice with different values raises `ValueError`.

## Sibling Readers Under One Root FeatureGroup

Each reader decides for itself whether a given data access belongs to it by overriding `match_subclass_data_access`. Return the data access to claim it, `None` to decline:

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
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        ...  # fetch and parse, return the table


class UbaAirReader(ReadFile):
    """REST JSON endpoint."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if isinstance(data_access, str) and data_access.startswith("https://api."):
            return data_access
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        ...  # HTTP GET, normalize JSON, return the table
```

Both are siblings under the stock `ReadFileFeature` root group; no extra FeatureGroup is needed. The user routes per feature:

```python
features = [
    Feature("population", options={GovDataReader.__name__: ckan_url}),
    Feature("pm10_value", options={UbaAirReader: api_url}),
]
```

## Non-File / HTTP Sources

`ReadFile`'s default matching (`match_read_file_data_access`) is file-suffix and directory shaped. For a non-file source (an HTTP endpoint returning JSON), the sanctioned recipe is: subclass `ReadFile`, override `match_subclass_data_access` and `load_data` wholesale. On that path `suffix()` is never consulted (it is inert), so you do not implement it.

`ApiInputData` is not the tool for this despite its name: it injects in-memory data passed through the API request and is not an HTTP client.

A reader that overrides `load_data` wholesale is classified as a final reader structurally; no reader code runs during classification.

## Test

Follow the contract at unit level, then end to end:

```python
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
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
| [base_input_data.py](https://github.com/mloda-ai/mloda/blob/main/mloda/core/abstract_plugins/components/input_data/base_input_data.py) | `feature_scope_data_access`, key normalization, `init_reader` |
| [read_file.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/input_data/read_file.py) | `ReadFile` base, `match_subclass_data_access` seam |
| [test_sibling_reader_selection.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/feature_group/input_data/test_sibling_reader_selection.py) | The pinned selection contract |

See also [Data Access Patterns](https://mloda-ai.github.io/mloda/in_depth/data-access-patterns/) for the underlying model.

## Combines With

- **Pattern 1 (Root features)**: readers are the `input_data()` of root features
- **Pattern 17 (Data connection matching)**: `DataAccessCollection` and handles for connection-shaped sources
- **Pattern 11 (Options)**: option keys and context vs group semantics
