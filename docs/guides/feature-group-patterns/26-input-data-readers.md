# Input Data Readers

How to write an input-data reader and how a feature picks one when several are available.

**What**: `BaseInputData` subclasses that load raw data for a root feature group.
**When**: Your root feature group reads from files, databases, documents, or an HTTP endpoint.
**Why**: Readers keep the loading logic out of `calculate_feature` and let one root feature group serve several sources.
**Where**: CSV/Parquet readers, database connectors, document loaders, API-backed connectors.
**How**: Subclass a reader family, override its hooks, and select the reader per feature via an option key.

## Reader Families and Their Hooks

Each family exposes a recommended hook seam. Overriding `load_data` wholesale stays supported in every family.

| Family | Override | Notes |
|--------|----------|-------|
| `ReadFile` | `load_data`, `suffix` | `CsvReader` returns a `FileSource` descriptor that the target compute framework materializes into its native table type. Returning a concrete table directly is equally supported. |
| `ReadDB` | `produce_rows`, `connect`, `is_valid_credentials` | Optionally `prepare_credentials` and `build_query`. |
| `ReadDocument` | `produce_document`, `suffix` | Optionally `document_file_type`. |

Readers are classified structurally: `is_final_reader()` is decided from what a class declares, no reader code is executed. A class is a final reader when it overrides `load_data` wholesale, or when it overrides every hook named by its family's `_final_reader_requires()` (`("produce_rows", "connect")` for `ReadDB`, `("produce_document", "suffix")` for `ReadDocument`). Family bases are never discovered as final readers themselves.

**Warning**: an intermediate base that re-declares a hook with a bare `raise NotImplementedError` body counts as an override and enters discovery. Re-anchor such a base by declaring `_final_reader_requires` instead.

## Sibling Readers Under One Root Feature Group

Several readers can sit behind a single root feature group. A feature selects one with an option whose key is the reader's class name (`BaseInputData.data_access_name()`, i.e. `cls.__name__`, unique per class, so siblings cannot collide):

```python
from mloda.user import Feature

Feature("population", options={GovDataReader.__name__: ckan_url})
Feature("turnout", options={ElectionsReader.__name__: csv_url})
```

The reader class itself is accepted as the key and normalized to the same string, so both forms are one identity:

```python
Feature("population", options={GovDataReader: ckan_url})
```

The matched `(ReaderClass, data_access)` pair is stored under the reserved `"BaseInputData"` option key and consumed by `init_reader` at load time. Do not set that key yourself.

## Non-File Sources (HTTP, JSON APIs)

`ReadFile`'s default matching is suffix and directory based, but a subclass that overrides `match_subclass_data_access` bypasses it. That is the sanctioned recipe for a URL-backed reader; on this path `suffix()` is never consulted (it is inert).

```python
from typing import Any
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class UbaAirReader(ReadFile):
    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return ()  # inert on the overridden matching path

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if isinstance(data_access, str) and data_access.startswith("https://"):
            return data_access
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        ...  # fetch the endpoint, return a table or a FileSource
```

`ApiInputData` is not the tool for this: it injects in-memory data passed through the API request and is not an HTTP client.

## Real Implementations

| File | Description |
|------|-------------|
| [read_file.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/input_data/read_file.py) | Base file reader |
| [read_db.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/input_data/read_db.py) | Base database reader |
| [read_document.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/input_data/read_document.py) | Base document reader |
| [csv.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/input_data/read_files/csv.py) | CSV reader returning a `FileSource` |

## Full Documentation

See [Data Access Patterns](https://mloda-ai.github.io/mloda/in_depth/data-access-patterns/).

## Combines With

- **Pattern 1 (Root features)**: readers are how a root feature group gets its data
- **Pattern 11 (Options)**: the reader-selection key is an ordinary feature option
