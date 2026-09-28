import pkgutil
import sys

# Rewrite asserts in the shipped mixins so a failure shows the compared values. Registers the children, not
# __name__: this package is mid-import here, so registering it would only warn.
# Skip pytest unless it is already loaded; only a pytest run installs the rewrite hook.
if "pytest" in sys.modules:
    import pytest

    pytest.register_assert_rewrite(*(f"{__name__}.{module.name}" for module in pkgutil.iter_modules(__path__)))
