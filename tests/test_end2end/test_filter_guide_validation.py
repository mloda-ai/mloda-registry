"""Validate filter-related API references documented in guides against the actual mloda 0.6.x API.

These tests verify that:
1. FilterType enum members and values match what our guides document
2. GlobalFilter.add_filter() accepts both string and FilterType enum values
3. FeatureGroup.final_filters() exists with the expected signature and default
4. BaseFilterEngine.final_filters() and applicable_filters() exist with expected defaults
5. SingleFilter creation works with string and enum filter types
6. FilterParameterImpl properties work as documented
7. FeatureGroup.final_filters() can be overridden by subclasses
"""

import inspect


def test_filter_type_enum_has_upper_case_members() -> None:
    """Verify FilterType enum members exist with expected lowercase string values."""
    from mloda.core.filter.filter_type_enum import FilterType

    expected = {
        "MIN": "min",
        "MAX": "max",
        "EQUAL": "equal",
        "RANGE": "range",
        "REGEX": "regex",
        "CATEGORICAL_INCLUSION": "categorical_inclusion",
    }

    for member_name, expected_value in expected.items():
        member = FilterType[member_name]
        assert member.value == expected_value, (
            f"FilterType.{member_name} should have value '{expected_value}', got '{member.value}'"
        )


def test_filter_type_enum_members_accessible_as_attributes() -> None:
    """Verify FilterType members are accessible via dot notation."""
    from mloda.core.filter.filter_type_enum import FilterType

    assert FilterType.MIN is not None
    assert FilterType.MAX is not None
    assert FilterType.EQUAL is not None
    assert FilterType.RANGE is not None
    assert FilterType.REGEX is not None
    assert FilterType.CATEGORICAL_INCLUSION is not None


def test_global_filter_add_filter_accepts_string_filter_type() -> None:
    """Verify GlobalFilter.add_filter() works with a string filter type."""
    from mloda.core.filter.global_filter import GlobalFilter

    gf = GlobalFilter()
    gf.add_filter("feature_a", "min", {"value": 10})
    assert len(gf.filters) == 1


def test_global_filter_add_filter_accepts_enum_filter_type() -> None:
    """Verify GlobalFilter.add_filter() works with a FilterType enum value."""
    from mloda.core.filter.filter_type_enum import FilterType
    from mloda.core.filter.global_filter import GlobalFilter

    gf = GlobalFilter()
    gf.add_filter("feature_b", FilterType.MAX, {"value": 100})
    assert len(gf.filters) == 1


def test_global_filter_add_filter_accepts_both_types_together() -> None:
    """Verify string and enum filter types can coexist in the same GlobalFilter."""
    from mloda.core.filter.filter_type_enum import FilterType
    from mloda.core.filter.global_filter import GlobalFilter

    gf = GlobalFilter()
    gf.add_filter("feature_a", "min", {"value": 10})
    gf.add_filter("feature_b", FilterType.MAX, {"value": 100})
    assert len(gf.filters) == 2


def test_feature_group_final_filters_exists_as_classmethod() -> None:
    """Verify FeatureGroup.final_filters() exists and is a classmethod."""
    from mloda.core.abstract_plugins.feature_group import FeatureGroup

    assert hasattr(FeatureGroup, "final_filters")
    raw = inspect.getattr_static(FeatureGroup, "final_filters")
    assert isinstance(raw, classmethod), "final_filters should be a classmethod"


def test_feature_group_final_filters_return_annotation() -> None:
    """Verify FeatureGroup.final_filters() has return type 'bool | None'."""
    from mloda.core.abstract_plugins.feature_group import FeatureGroup

    sig = inspect.signature(FeatureGroup.final_filters)
    assert sig.return_annotation == "bool | None", (
        f"Expected return annotation 'bool | None', got '{sig.return_annotation}'"
    )


def test_feature_group_final_filters_returns_none_by_default() -> None:
    """Verify FeatureGroup.final_filters() returns None by default."""
    from mloda.core.abstract_plugins.feature_group import FeatureGroup

    result = FeatureGroup.final_filters()
    assert result is None, f"Expected None, got {result}"


def test_base_filter_engine_final_filters_exists_as_classmethod() -> None:
    """Verify BaseFilterEngine.final_filters() exists and is a classmethod."""
    from mloda.core.filter.filter_engine import BaseFilterEngine

    assert hasattr(BaseFilterEngine, "final_filters")
    raw = inspect.getattr_static(BaseFilterEngine, "final_filters")
    assert isinstance(raw, classmethod), "final_filters should be a classmethod"


def test_base_filter_engine_final_filters_returns_false_by_default() -> None:
    """Verify BaseFilterEngine.final_filters() returns False by default."""
    from mloda.core.filter.filter_engine import BaseFilterEngine

    result = BaseFilterEngine.final_filters()
    assert result is False, f"Expected False, got {result}"


def test_base_filter_engine_applicable_filters_exists_as_classmethod() -> None:
    """Verify BaseFilterEngine.applicable_filters() exists and is a classmethod."""
    from mloda.core.filter.filter_engine import BaseFilterEngine

    assert hasattr(BaseFilterEngine, "applicable_filters")
    raw = inspect.getattr_static(BaseFilterEngine, "applicable_filters")
    assert isinstance(raw, classmethod), "applicable_filters should be a classmethod"


def test_single_filter_creation_with_string_filter_type() -> None:
    """Verify SingleFilter can be created with a string filter type."""
    from mloda.core.filter.single_filter import SingleFilter

    sf = SingleFilter("my_feature", "min", {"value": 10})
    assert sf.filter_type == "min"
    assert isinstance(sf.filter_type, str)


def test_single_filter_creation_with_enum_filter_type() -> None:
    """Verify SingleFilter can be created with a FilterType enum value."""
    from mloda.core.filter.filter_type_enum import FilterType
    from mloda.core.filter.single_filter import SingleFilter

    sf = SingleFilter("my_feature", FilterType.MIN, {"value": 10})
    assert sf.filter_type == "min"
    assert isinstance(sf.filter_type, str), "filter_type should be stored as a string even when created with enum"


def test_single_filter_stores_filter_type_as_string() -> None:
    """Verify filter_type is always stored as a string regardless of input type."""
    from mloda.core.filter.filter_type_enum import FilterType
    from mloda.core.filter.single_filter import SingleFilter

    sf_str = SingleFilter("feat", "equal", {"value": "x"})
    sf_enum = SingleFilter("feat", FilterType.EQUAL, {"value": "x"})
    assert type(sf_str.filter_type) is str
    assert type(sf_enum.filter_type) is str
    assert sf_str.filter_type == sf_enum.filter_type


def test_filter_parameter_impl_from_dict_with_value() -> None:
    """Verify FilterParameterImpl.from_dict() works and value property returns the value."""
    from mloda.core.filter.filter_parameter import FilterParameterImpl

    fp = FilterParameterImpl.from_dict({"value": 42})
    assert fp.value == 42


def test_filter_parameter_impl_from_dict_with_values_list() -> None:
    """Verify FilterParameterImpl values property returns a list."""
    from mloda.core.filter.filter_parameter import FilterParameterImpl

    fp = FilterParameterImpl.from_dict({"values": ["a", "b", "c"]})
    assert fp.values == ["a", "b", "c"]


def test_filter_parameter_impl_from_dict_with_range() -> None:
    """Verify FilterParameterImpl min_value and max_value properties work with 'min'/'max' keys."""
    from mloda.core.filter.filter_parameter import FilterParameterImpl

    fp = FilterParameterImpl.from_dict({"min": 10, "max": 100})
    assert fp.min_value == 10
    assert fp.max_value == 100


def test_filter_parameter_impl_max_exclusive_default() -> None:
    """Verify max_exclusive defaults to False."""
    from mloda.core.filter.filter_parameter import FilterParameterImpl

    fp = FilterParameterImpl.from_dict({"value": 1})
    assert fp.max_exclusive is False


def test_filter_parameter_impl_max_exclusive_true() -> None:
    """Verify max_exclusive can be set to True."""
    from mloda.core.filter.filter_parameter import FilterParameterImpl

    fp = FilterParameterImpl.from_dict({"min": 0, "max": 50, "max_exclusive": True})
    assert fp.max_exclusive is True


def test_filter_parameter_impl_properties_return_none_when_absent() -> None:
    """Verify properties return None when their keys are not in the dict."""
    from mloda.core.filter.filter_parameter import FilterParameterImpl

    fp = FilterParameterImpl.from_dict({"value": 42})
    assert fp.values is None
    assert fp.min_value is None
    assert fp.max_value is None


def test_feature_group_final_filters_override_returns_false() -> None:
    """Verify a FeatureGroup subclass can override final_filters to return False."""
    from mloda.core.abstract_plugins.feature_group import FeatureGroup

    class StrictFilterGroup(FeatureGroup):
        @classmethod
        def final_filters(cls) -> bool | None:
            return False

    assert StrictFilterGroup.final_filters() is False


def test_feature_group_final_filters_override_returns_true() -> None:
    """Verify a FeatureGroup subclass can override final_filters to return True."""
    from mloda.core.abstract_plugins.feature_group import FeatureGroup

    class FinalFilterGroup(FeatureGroup):
        @classmethod
        def final_filters(cls) -> bool | None:
            return True

    assert FinalFilterGroup.final_filters() is True


def test_feature_group_final_filters_base_unaffected_by_override() -> None:
    """Verify overriding in a subclass does not affect the base class default."""
    from mloda.core.abstract_plugins.feature_group import FeatureGroup

    class OverriddenGroup(FeatureGroup):
        @classmethod
        def final_filters(cls) -> bool | None:
            return True

    assert OverriddenGroup.final_filters() is True
    assert FeatureGroup.final_filters() is None
