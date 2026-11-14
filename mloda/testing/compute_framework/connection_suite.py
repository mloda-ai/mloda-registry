"""Re-export of mloda._testing.compute_framework.connection_suite."""

from mloda._testing.compute_framework.connection_suite import (
    ConnectionEndToEndSuite,
    ConnectionPickSuite,
    root_binding_groups,
)

__all__ = ["ConnectionEndToEndSuite", "ConnectionPickSuite", "root_binding_groups"]
