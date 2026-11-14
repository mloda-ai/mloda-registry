"""Everything below needs Arrow directly or through the shared runners and bases, so the guard is package-wide."""

from mloda.testing.import_isolation import require_pyarrow

require_pyarrow()
