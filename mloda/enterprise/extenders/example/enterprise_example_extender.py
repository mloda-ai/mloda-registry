"""Enterprise Example Extender implementation."""

from typing import Any, Set

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook


class EnterpriseExampleExtender(Extender):
    """An example enterprise extender that demonstrates the Extender pattern."""

    def wraps(self) -> Set[ExtenderHook]:
        """Return the hooks this extender wraps."""
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute the wrapped function."""
        return func(*args, **kwargs)
