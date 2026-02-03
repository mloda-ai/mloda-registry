# Category 4: Zero Dependency Frameworks

Zero dependency frameworks use pure Python with no external libraries.

**What**: Frameworks using native Python structures (list, dict).
**When**: Minimal environments, serverless, testing without deps.
**Why**: Always available; no installation required.
**Where**: Serverless functions, embedded systems, educational use.
**How**: Same as Category 1, but skip `is_available()` and use `List[Dict]`.

## Key Difference from Category 1

| Aspect | Category 1 (Eager) | Category 4 (Zero Dep) |
|--------|-------------------|----------------------|
| Dependencies | External library | **None** |
| `is_available()` | Check import | Not needed (always True) |
| Data format | Library-specific | `List[Dict[str, Any]]` |
| Column access | `data.columns` | Iterate rows for keys |

## Data Format

```python
# Row-based: List[Dict[str, Any]]
[{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]
```

## What's Different

```python
class MyZeroDependencyFramework(ComputeFramework):
    # No is_available() needed - always available

    @classmethod
    def expected_data_framework(cls) -> Any:
        return list  # Native Python list

    def set_column_names(self) -> None:
        # Must iterate rows to get all keys
        all_columns: Set[str] = set()
        for row in self.data:
            if isinstance(row, dict):
                all_columns.update(row.keys())
        self.column_names = all_columns

    def transform(self, data: Any, feature_names: Set[str]) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            # Columnar dict to row-based list
            if all(isinstance(v, list) for v in data.values()):
                length = len(next(iter(data.values())))
                return [{k: data[k][i] for k in data.keys()} for i in range(length)]
            return [data]  # Single row
        if isinstance(data, list):
            return data
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
