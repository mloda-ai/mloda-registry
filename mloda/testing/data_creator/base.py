"""Base test data creator FeatureGroup for data operations plugins.

Provides a fixed 12-row dataset with deliberate edge cases (nulls, duplicates,
gaps, empty strings, all-null columns) that data-operations plugins can use as
a shared test fixture. Follows the ATestDataCreator pattern from mloda core:
a FeatureGroup with input_data() -> DataCreator(...) that serves as a root
data source in mloda's pipeline.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class DataOperationsTestDataCreator(FeatureGroup):
    """Test data source FeatureGroup for data-operations plugins.

    Call ``get_raw_data()`` for a plain dict of lists (12 rows).
    Call ``create()`` on framework subclasses for framework-native data.
    Use directly in PluginCollector sets for integration tests.
    """

    compute_framework: type[ComputeFramework] = PyArrowTable

    NULL_POSITIONS: dict[str, set[int]] = {
        "region": {11},
        "category": {6},
        "timestamp": {10},
        "event_date": {2},
        "value_int": {4},
        "value_float": {2, 11},
        "amount": {1, 7},
        "name": {2, 11},
        "is_active": {3, 9},
        "score": set(range(12)),
    }

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(set(cls.get_raw_data().keys()))

    @classmethod
    def get_raw_data(cls) -> dict[str, list[Any]]:
        """Return the canonical 12-row test dataset as a column-oriented dict."""
        return {
            # Partition / group keys
            "region": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", None],
            "category": ["X", "Y", "X", "Y", "X", "Y", None, "X", "Y", "X", "Y", "X"],
            # Order keys
            "timestamp": [
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 1, 2, tzinfo=timezone.utc),
                datetime(2023, 1, 3, tzinfo=timezone.utc),
                datetime(2023, 1, 5, tzinfo=timezone.utc),  # gap: Jan 4 skipped
                datetime(2023, 1, 6, tzinfo=timezone.utc),
                datetime(2023, 1, 6, tzinfo=timezone.utc),  # duplicate timestamp
                datetime(2023, 1, 7, tzinfo=timezone.utc),
                datetime(2023, 1, 8, tzinfo=timezone.utc),
                datetime(2023, 1, 9, tzinfo=timezone.utc),
                datetime(2023, 1, 10, tzinfo=timezone.utc),
                None,  # null timestamp (row 10)
                datetime(2023, 1, 12, tzinfo=timezone.utc),
            ],
            "event_date": [
                date(2023, 1, 1),
                date(2023, 1, 3),
                None,
                date(2023, 1, 5),
                date(2023, 1, 6),
                date(2023, 1, 7),
                date(2023, 1, 7),
                date(2023, 1, 9),
                date(2023, 1, 10),
                date(2023, 1, 10),
                date(2023, 1, 11),
                date(2023, 1, 12),
            ],
            # Numeric columns
            "value_int": [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10],
            "value_float": [1.5, 2.5, None, 0.0, -3.14, 5.5, 6.5, 7.5, 1e-15, 100.0, 0.0, None],
            "amount": [100.0, None, 250.0, 75.0, 300.0, 0.0, 150.0, None, 50.0, 200.0, 125.0, 80.0],
            # String column
            "name": [
                "Alice",
                "bob",
                None,
                "",
                " Eve ",
                "FRANK",
                "Grace",
                "alice",
                "  ",
                "Bob",
                "h\u00e9llo",
                None,
            ],
            # Boolean column
            "is_active": [True, False, True, None, True, False, True, True, False, None, True, False],
            # All-null column (aggregation edge case)
            "score": [None, None, None, None, None, None, None, None, None, None, None, None],
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return cls.get_raw_data()

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {cls.compute_framework}

    @classmethod
    def create(cls) -> Any:
        """Return the test dataset in framework-native format. Subclasses must override."""
        raise NotImplementedError
