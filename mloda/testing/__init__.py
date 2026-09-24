import pkgutil

import pytest

# Rewrite asserts in the shipped mixins so a failure shows the compared values.
pytest.register_assert_rewrite(*(f"{__name__}.{module.name}" for module in pkgutil.iter_modules(__path__)))
