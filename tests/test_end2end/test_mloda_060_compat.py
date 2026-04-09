"""Verify mloda 0.6.0 API contracts that this registry depends on.

These tests guard against regressions if the minimum mloda version
is ever accidentally lowered, or if a future mloda release changes
these contracts again.
"""

import inspect


def test_feature_name_is_str_subclass() -> None:
    """FeatureName became a str subclass in 0.6.0."""
    from mloda.user import FeatureName

    fn = FeatureName("my_feature")
    assert isinstance(fn, str)
    assert fn == "my_feature"
    assert str(fn) == "my_feature"


def test_filter_type_uses_upper_case_members() -> None:
    """FilterType enum members were renamed to UPPER_CASE in 0.6.0."""
    from mloda.user import FilterType

    expected = {"MIN", "MAX", "EQUAL", "RANGE", "REGEX", "CATEGORICAL_INCLUSION"}
    actual = {m.name for m in FilterType}
    assert expected == actual


def test_stream_all_is_mloda_staticmethod() -> None:
    """stream_all moved to mlodaAPI as a staticmethod in 0.6.0."""
    from mloda.user import mloda

    assert hasattr(mloda, "stream_all")
    assert callable(mloda.stream_all)


def test_validate_input_features_returns_none() -> None:
    """validate_input_features contract changed to raise-on-failure in 0.6.0."""
    from mloda.provider import FeatureGroup

    sig = inspect.signature(FeatureGroup.validate_input_features)
    assert sig.return_annotation == "None"


def test_validate_output_features_returns_none() -> None:
    """validate_output_features contract changed to raise-on-failure in 0.6.0."""
    from mloda.provider import FeatureGroup

    sig = inspect.signature(FeatureGroup.validate_output_features)
    assert sig.return_annotation == "None"


def test_options_has_add_to_group() -> None:
    """Options.add_to_group is the supported write API in 0.6.0."""
    from mloda.user import Options

    assert hasattr(Options, "add_to_group")
    assert callable(Options.add_to_group)


def test_duckdb_relation_has_order_method() -> None:
    """DuckdbRelation.order() was added as a public method in 0.6.0."""
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

    assert hasattr(DuckdbRelation, "order")
    assert callable(DuckdbRelation.order)
