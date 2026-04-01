"""Percentile-specific security tests: SQL injection rejection in feature names
and partition_by values, plus percentile range validation.

Generic match-validation tests live in ``test_base.py`` via
``MatchValidationTestBase``.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options

from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)


class TestSqlInjectionInFeatureNames:
    """Verify that SQL injection attempts in feature names are rejected.

    Feature names that do not match the ``{source}__p{N}_percentile`` pattern
    are rejected outright. Names that do match the pattern syntactically will
    pass match_feature_group_criteria (the source column portion is opaque at
    match time); actual SQL safety is enforced at the backend level via
    identifier quoting and parameterized queries.
    """

    @pytest.mark.parametrize(
        "malicious_name",
        [
            "'; DROP TABLE users--",
            "value_int; DELETE FROM data",
            "UNION SELECT * FROM secrets",
            "1=1; --",
            "__p50_percentile",
        ],
    )
    def test_sql_injection_without_valid_pattern_rejected(self, malicious_name: str) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria(malicious_name, options, None)
        assert result is False, f"Malformed feature name should be rejected: {malicious_name}"

    @pytest.mark.parametrize(
        "malicious_name",
        [
            "value_int__p50_grouped",
            "value_int__p50",
            "p50_percentile",
            "value_int__percentile",
        ],
    )
    def test_wrong_suffix_pattern_rejected(self, malicious_name: str) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria(malicious_name, options, None)
        assert result is False, f"Wrong suffix pattern should be rejected: {malicious_name}"


class TestSqlInjectionInPartitionBy:
    """Verify that non-string partition_by values are rejected by type validation."""

    @pytest.mark.parametrize(
        "malicious_partition",
        [
            [42],
            [None],
            [True],
            [3.14],
        ],
    )
    def test_non_string_partition_by_rejected(self, malicious_partition: list[Any]) -> None:
        options = Options(context={"partition_by": malicious_partition})
        result = PercentileFeatureGroup.match_feature_group_criteria("value_int__p50_percentile", options, None)
        assert result is False, f"Non-string partition_by should be rejected: {malicious_partition}"

    def test_non_list_partition_by_rejected(self) -> None:
        options = Options(context={"partition_by": "not_a_list"})
        result = PercentileFeatureGroup.match_feature_group_criteria("value_int__p50_percentile", options, None)
        assert result is False

    def test_empty_partition_by_rejected(self) -> None:
        options = Options(context={"partition_by": []})
        result = PercentileFeatureGroup.match_feature_group_criteria("value_int__p50_percentile", options, None)
        assert result is False

    def test_missing_partition_by_rejected(self) -> None:
        options = Options(context={})
        result = PercentileFeatureGroup.match_feature_group_criteria("value_int__p50_percentile", options, None)
        assert result is False


class TestPercentileRangeValidation:
    """Verify that percentile values outside valid range are rejected."""

    @pytest.mark.parametrize(
        "valid_name",
        [
            "value_int__p0_percentile",
            "value_int__p1_percentile",
            "value_int__p25_percentile",
            "value_int__p50_percentile",
            "value_int__p75_percentile",
            "value_int__p99_percentile",
            "value_int__p100_percentile",
        ],
    )
    def test_valid_percentile_values_accepted(self, valid_name: str) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria(valid_name, options, None)
        assert result is True, f"Valid percentile should be accepted: {valid_name}"

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "value_int__p101_percentile",
            "value_int__p200_percentile",
            "value_int__p999_percentile",
            "value_int__p-1_percentile",
            "value_int__p-50_percentile",
        ],
    )
    def test_invalid_percentile_values_rejected(self, invalid_name: str) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria(invalid_name, options, None)
        assert result is False, f"Invalid percentile should be rejected: {invalid_name}"

    @pytest.mark.parametrize(
        "bad_name",
        [
            "value_int__pfoo_percentile",
            "value_int__p_percentile",
            "value_int__pabc_percentile",
            "value_int__p1.5_percentile",
        ],
    )
    def test_non_integer_percentile_in_name_rejected(self, bad_name: str) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria(bad_name, options, None)
        assert result is False, f"Non-integer percentile should be rejected: {bad_name}"


class TestConfigBasedPercentileValidation:
    """Verify that config-based percentile values are validated."""

    @pytest.mark.parametrize(
        "percentile_value",
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )
    def test_valid_config_percentile_accepted(self, percentile_value: float) -> None:
        options = Options(
            context={
                "percentile": percentile_value,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True, f"Valid config percentile {percentile_value} should be accepted"

    @pytest.mark.parametrize(
        "percentile_value",
        [1.5, -0.1, 2.0, -1.0, 100.0],
    )
    def test_invalid_config_percentile_rejected(self, percentile_value: float) -> None:
        options = Options(
            context={
                "percentile": percentile_value,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False, f"Invalid config percentile {percentile_value} should be rejected"

    @pytest.mark.parametrize(
        "bad_type",
        ["fifty", [0.5], None],
    )
    def test_wrong_type_config_percentile_rejected(self, bad_type: object) -> None:
        options = Options(
            context={
                "percentile": bad_type,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False, f"Wrong-type percentile {bad_type!r} should be rejected"
