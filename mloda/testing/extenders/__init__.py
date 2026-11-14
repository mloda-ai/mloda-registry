"""Skips the extender test helpers unless pyarrow is installed."""

from mloda.testing.import_isolation import require_pyarrow

require_pyarrow()
