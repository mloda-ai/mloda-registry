"""Characterization tests for the shared ``unique_helper_name`` collision helper.

These lock in the current, intentional behavior of the suffix contract and the
``Container[str]`` acceptance for ``taken`` (plain ``set``, pandas
``DataFrame.columns``, pyarrow ``Table.column_names``). They assert existing
behavior only and must not drive any production change.
"""

from __future__ import annotations

import pytest

from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name


class TestUniqueHelperName:
    def test_returns_base_when_absent(self) -> None:
        """An unused base name is returned verbatim, with no suffix."""
        assert unique_helper_name("__mloda_x", set()) == "__mloda_x"
        assert unique_helper_name("__mloda_x", {"other"}) == "__mloda_x"

    def test_first_collision_appends_suffix_1(self) -> None:
        """A taken base yields ``base_1``; a base ending in ``_`` keeps the double underscore."""
        assert unique_helper_name("base", {"base"}) == "base_1"
        # A base already ending in "_" produces an intentional double underscore.
        assert unique_helper_name("__mloda_x_", {"__mloda_x_"}) == "__mloda_x__1"

    def test_skips_occupied_suffixes_to_lowest_free(self) -> None:
        """The lowest free ``base_N`` (N>=1) is chosen, skipping occupied suffixes."""
        assert unique_helper_name("base", {"base", "base_1"}) == "base_2"
        assert unique_helper_name("base", {"base", "base_1", "base_2"}) == "base_3"

    def test_accepts_plain_set(self) -> None:
        """A plain ``set[str]`` is a valid ``taken`` container for both branches."""
        taken: set[str] = {"col", "__mloda_idx"}
        assert unique_helper_name("__mloda_idx", taken) == "__mloda_idx_1"
        assert unique_helper_name("__mloda_other", taken) == "__mloda_other"

    def test_accepts_pandas_columns_index(self) -> None:
        """A pandas ``DataFrame.columns`` (Index) supports ``in`` and works as ``taken``."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1], "__mloda_idx": [2]})
        assert unique_helper_name("__mloda_idx", df.columns) == "__mloda_idx_1"
        assert unique_helper_name("__mloda_absent", df.columns) == "__mloda_absent"

    def test_accepts_pyarrow_column_names(self) -> None:
        """A pyarrow ``Table.column_names`` (list[str]) works as ``taken``."""
        pa = pytest.importorskip("pyarrow")
        table = pa.table({"a": [1], "__mloda_idx": [2]})
        assert unique_helper_name("__mloda_idx", table.column_names) == "__mloda_idx_1"
        assert unique_helper_name("__mloda_absent", table.column_names) == "__mloda_absent"

    def test_collision_check_is_case_sensitive(self) -> None:
        """Different case is not a collision: ``__mloda_x`` vs ``__MLODA_X`` are distinct."""
        assert unique_helper_name("__mloda_x", {"__MLODA_X"}) == "__mloda_x"
