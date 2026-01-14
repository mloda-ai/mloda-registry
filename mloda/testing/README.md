# mloda-testing

Test utilities for mloda plugin development.

## Installation

```bash
pip install mloda-testing
```

## Usage

```python
from mloda.testing import FeatureGroupTestBase

class TestMyFeatureGroup(FeatureGroupTestBase):
    feature_group_class = MyFeatureGroup
```
