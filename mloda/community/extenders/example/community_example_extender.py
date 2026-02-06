"""Community Example Extender implementation."""

from typing import Any, Set

from mloda.steward import Extender, ExtenderHook


class CommunityExampleExtender(Extender):
    """Community Example Extender for demonstrating plugin structure."""

    def wraps(self) -> Set[ExtenderHook]:
        """Return the set of hooks this extender wraps."""
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute the wrapped function without modification."""
        return func(*args, **kwargs)
