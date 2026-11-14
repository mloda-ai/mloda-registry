"""Re-export of mloda._testing.compute_framework_shape."""

from mloda._testing.compute_framework_shape import (
    applicable_aspects,
    assert_compute_framework_shape,
    check_compute_framework_shape,
    serves_asof_merge,
    serves_connection,
)

__all__ = [
    "applicable_aspects",
    "assert_compute_framework_shape",
    "check_compute_framework_shape",
    "serves_asof_merge",
    "serves_connection",
]
