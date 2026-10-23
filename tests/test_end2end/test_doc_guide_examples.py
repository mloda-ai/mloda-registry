def test_derived_feature_isolated():
    """
    Verify that a derived feature correctly uses a mocked dependency.

    The example in the guide creates a ``source`` feature that is mocked to
    return the constant value ``5``.  The ``derived`` feature simply doubles the
    value of its dependency.  By mocking the dependency we can assert the exact
    output of the derived feature without needing any real data source.
    """
    from unittest.mock import Mock

    # Mock the source feature – it pretends to be a Feature instance that
    # returns the constant value 5 when evaluated.
    source = Mock()
    source.name = "source"
    source.compute.return_value = 5  # type: ignore[attr-defined]

    # Define the derived feature that doubles the value of its dependency.
    # In the guide this is expressed as:
    #   derived = Feature("derived", options=Options(...), dependencies=[source])
    # For the purpose of the test we only need the name and the dependency.
    derived = Feature(
        "derived",
        options=Options(),
        dependencies=[source],
    )

    # Run the feature through mloda.  The framework will call ``source.compute``
    # (returning 5) and then apply the derived‑feature logic (doubling it).
    result = mloda.run_all([derived])

    # ``result`` is a list of ``FeatureResult`` objects.  The first (and only)
    # element corresponds to the ``derived`` feature.
    assert len(result) == 1, "expected exactly one result from the derived feature"
    derived_result = result[0]

    # The derived feature should output ``10`` (2 * 5).
    assert derived_result.value == 10, (
        f"derived feature produced {derived_result.value!r}, expected 10 "
        "(double the mocked source value)"
    )