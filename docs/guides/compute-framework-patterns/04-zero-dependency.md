# Category 4: Zero Dependency Frameworks

Zero dependency frameworks use pure Python with no external libraries.

**What**: Frameworks using native Python structures (list, dict).
**When**: Minimal environments, serverless, testing without deps.
**Why**: Always available; no installation required.
**Where**: Serverless functions, embedded systems, educational use.
**How**: Same as Category 1, but skip `is_available()` and use a columnar `dict[str, list[Any]]`.

## Key Difference from Category 1

| Aspect | Category 1 (Eager) | Category 4 (Zero Dep) |
|--------|-------------------|----------------------|
| Dependencies | External library | **None** |
| `is_available()` | Check import | Not needed (always True) |
| Data format | Library-specific | `dict[str, list[Any]]` (columnar) |
| Column access | `data.columns` | `data.keys()` |

## Data Format

```python
# Columnar: dict[str, list[Any]] - one key per column, all lists share a length
{"col1": [1, 2], "col2": ["a", "b"]}
```

`{"a": []}` is a valid zero-row frame (one known column). `{}` is the only schema-less value.

## What's Different

```python
class MyZeroDependencyFramework(ComputeFramework):
    # No is_available() needed - always available

    @classmethod
    def expected_data_framework(cls) -> Any:
        return dict  # Native Python dict, columnar

    def _extract_column_names(self, data: Any) -> set[str]:
        if isinstance(data, dict):
            return set(data.keys())
        # Row-wise list[dict] is still an accepted pre-transform shape.
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return set(data[0].keys())
        return set()

    def transform(self, data: Any, feature_names: set[str]) -> dict[str, list[Any]]:
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            # Row-based list to columnar dict
            if not data:
                return {}
            return {k: [row[k] for row in data] for k in data[0].keys()}
        raise ValueError(f"Data type {type(data)} not supported")
```

## Real Implementations

| File | Description |
|------|-------------|
| [python_dict/python_dict_framework.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/python_dict/python_dict_framework.py) | Python Dict |
| [python_dict/python_dict_merge_engine.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/python_dict/python_dict_merge_engine.py) | Pure Python merge |

## Combines With

- **Category 1**: Inherits base structure (minus dependency check)
- **Merge Engine** (Concept 6): Pure Python joins
- **Filter Engine** (Concept 7): List comprehension filters
