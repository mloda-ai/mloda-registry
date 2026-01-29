# Input/Output Validation

How to validate input and output data in feature groups.

**What**: Validate data before and after feature computation.
**When**: You need to enforce data quality constraints (ranges, types, missing values).
**Why**: Catch invalid data early; provide clear error messages.
**Where**: `validate_input_features()` and `validate_output_features()` methods.

## Input Validation

```python
@classmethod
def validate_input_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
    if data["my_input"].isnull().any():
        raise ValueError("Input contains null values")
    return True
```

## Output Validation

```python
@classmethod
def validate_output_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
    if len(data[cls.get_class_name()]) != 3:
        raise ValueError("Output should have 3 elements")
    return True
```

## Using BaseValidator

For reusable validation logic (e.g., with Pandera):

```python
from mloda.provider import BaseValidator

class MyValidator(BaseValidator):
    def validate(self, data) -> Optional[bool]:
        # Custom validation logic
        return True
```

## Full Documentation

See [Data Quality](https://mloda-ai.github.io/mloda/in_depth/data-quality/) for detailed patterns including Pandera integration.
