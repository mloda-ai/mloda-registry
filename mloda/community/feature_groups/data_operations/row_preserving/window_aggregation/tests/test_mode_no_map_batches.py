"""Regression test: Polars Lazy window aggregation must not use map_batches for mode."""

from __future__ import annotations

from pathlib import Path


def test_polars_lazy_window_aggregation_does_not_use_map_batches() -> None:
    """Ensure the mode path uses pure Polars expressions (not per-group Python callbacks)."""
    target = Path(__file__).resolve().parents[1] / "polars_lazy_window_aggregation.py"
    content = target.read_text(encoding="utf-8")
    assert "map_batches" not in content, (
        "polars_lazy_window_aggregation.py must not use map_batches (stay on the lazy/vectorised path)"
    )
    assert "_mode_with_insertion_order" not in content, (
        "polars_lazy_window_aggregation.py must not define or call the per-group Python mode helper"
    )
