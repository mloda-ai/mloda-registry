"""Hand-computed pinning rows of the rank reference."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

NAN = float("nan")

RANK_RULES: Mapping[str, str] = MappingProxyType(
    {
        "order": "Each partition is ordered by order_by ascending, stably: input order breaks ties.",
        "order_tiers": "In order_by real values come first, then NaN, then None; NaN and None are separate tiers.",
        "ties": "rank and percent_rank give a tie run the rank of its first row; dense_rank has no gaps.",
        "top_bottom": "top_N and bottom_N place NaN order values before None order values, both after every real value.",
        "partition_keys": "NaN partition keys merge into one group; a None partition key is its own group.",
        "all_none": "A partition whose order values are all None ties on every row.",
        "single_row": "percent_rank of a single-row partition is 0.0.",
        "dtype": "row_number, rank, dense_rank and ntile_N are int64; percent_rank is float64; top_N and bottom_N are bool.",
        "empty": "An empty input gives an empty column of the declared type.",
    }
)

_I = DataType.INT64
_D = DataType.DOUBLE
_S = DataType.STRING
_B = DataType.BOOLEAN
_CTX = {"partition_by": ["p"], "order_by": "o"}

RANK_PINNING = PinningTable(
    operation="rank",
    rules=RANK_RULES,
    rows=(
        PinningRow(
            case="order",
            feature="o__row_number_ranked",
            inputs={"p": (_S, ("a", "a", "a")), "o": (_D, (3.0, None, 1.0))},
            expected={"o__row_number_ranked": (_I, (2, 3, 1))},
            context=_CTX,
        ),
        PinningRow(
            case="order",
            feature="o__row_number_ranked",
            inputs={"p": (_S, ("a", "a", "a")), "o": (_S, ("b", "a", "c"))},
            expected={"o__row_number_ranked": (_I, (2, 1, 3))},
            context=_CTX,
        ),
        PinningRow(
            case="order",
            feature="o__row_number_ranked",
            inputs={"p": (_S, ("a", "a", "a", "a")), "o": (_D, (2.0, 1.0, 2.0, 2.0))},
            expected={"o__row_number_ranked": (_I, (2, 1, 3, 4))},
            context=_CTX,
        ),
        PinningRow(
            case="order_tiers",
            feature="o__row_number_ranked",
            inputs={"p": (_S, ("a", "a", "a")), "o": (_D, (None, NAN, 1.0))},
            expected={"o__row_number_ranked": (_I, (3, 2, 1))},
            context=_CTX,
        ),
        PinningRow(
            case="order_tiers",
            feature="o__rank_ranked",
            inputs={"p": (_S, ("a",) * 5), "o": (_D, (NAN, 1.0, NAN, None, None))},
            expected={"o__rank_ranked": (_I, (2, 1, 2, 4, 4))},
            context=_CTX,
        ),
        PinningRow(
            case="order_tiers",
            feature="o__dense_rank_ranked",
            inputs={"p": (_S, ("a",) * 5), "o": (_D, (NAN, 1.0, NAN, None, None))},
            expected={"o__dense_rank_ranked": (_I, (2, 1, 2, 3, 3))},
            context=_CTX,
        ),
        PinningRow(
            case="ties",
            feature="o__rank_ranked",
            inputs={"p": (_S, ("a",) * 4), "o": (_D, (2.0, 1.0, 2.0, 3.0))},
            expected={"o__rank_ranked": (_I, (2, 1, 2, 4))},
            context=_CTX,
        ),
        PinningRow(
            case="ties",
            feature="o__dense_rank_ranked",
            inputs={"p": (_S, ("a",) * 4), "o": (_D, (2.0, 1.0, 2.0, 3.0))},
            expected={"o__dense_rank_ranked": (_I, (2, 1, 2, 3))},
            context=_CTX,
        ),
        PinningRow(
            case="ties",
            feature="o__percent_rank_ranked",
            inputs={"p": (_S, ("a",) * 5), "o": (_D, (3.0, 1.0, 2.0, 2.0, 4.0))},
            # ranks 4, 1, 2, 2, 5 over n = 5: (rank - 1) / 4
            expected={"o__percent_rank_ranked": (_D, (0.75, 0.0, 0.25, 0.25, 1.0))},
            context=_CTX,
        ),
        PinningRow(
            case="single_row",
            feature="o__percent_rank_ranked",
            inputs={"p": (_S, ("a", "b", "b")), "o": (_D, (5.0, 1.0, 2.0))},
            expected={"o__percent_rank_ranked": (_D, (0.0, 0.0, 1.0))},
            context=_CTX,
        ),
        PinningRow(
            case="all_none",
            feature="o__rank_ranked",
            inputs={"p": (_S, ("a", "a", "b")), "o": (_D, (None, None, 5.0))},
            expected={"o__rank_ranked": (_I, (1, 1, 1))},
            context=_CTX,
        ),
        PinningRow(
            case="all_none",
            feature="o__row_number_ranked",
            inputs={"p": (_S, ("a", "a", "b")), "o": (_I, (None, None, 5))},
            expected={"o__row_number_ranked": (_I, (1, 2, 1))},
            context=_CTX,
        ),
        PinningRow(
            case="dtype",
            feature="o__ntile_2_ranked",
            inputs={"p": (_S, ("a",) * 5), "o": (_D, (4.0, 1.0, 3.0, 2.0, 5.0))},
            # position p of n = 5 sits in bucket p * 2 // 5 + 1: 1, 1, 1, 2, 2 in order 1.0 .. 5.0
            expected={"o__ntile_2_ranked": (_I, (2, 1, 1, 1, 2))},
            context=_CTX,
        ),
        PinningRow(
            case="top_bottom",
            feature="o__top_2_ranked",
            inputs={"p": (_S, ("a",) * 5), "o": (_D, (1.0, NAN, 3.0, None, 2.0))},
            expected={"o__top_2_ranked": (_B, (False, False, True, False, True))},
            context=_CTX,
        ),
        PinningRow(
            case="top_bottom",
            feature="o__top_4_ranked",
            inputs={"p": (_S, ("a",) * 5), "o": (_D, (1.0, NAN, 3.0, None, 2.0))},
            # descending: 3.0, 2.0, 1.0, then NaN, then None (NaN sorts before None, both after real values)
            expected={"o__top_4_ranked": (_B, (True, True, True, False, True))},
            context=_CTX,
        ),
        PinningRow(
            case="top_bottom",
            feature="o__bottom_2_ranked",
            inputs={"p": (_S, ("a",) * 5), "o": (_D, (1.0, NAN, 3.0, None, 2.0))},
            expected={"o__bottom_2_ranked": (_B, (True, False, False, False, True))},
            context=_CTX,
        ),
        PinningRow(
            case="top_bottom",
            feature="o__bottom_4_ranked",
            inputs={"p": (_S, ("a",) * 5), "o": (_D, (1.0, NAN, 3.0, None, 2.0))},
            expected={"o__bottom_4_ranked": (_B, (True, True, True, False, True))},
            context=_CTX,
        ),
        PinningRow(
            case="top_bottom",
            feature="o__top_2_ranked",
            inputs={"p": (_S, ("a",) * 3), "o": (_D, (None, NAN, 1.0))},
            # order: 1.0, then NaN, then None: top 2 is the 1.0 row and the NaN row
            expected={"o__top_2_ranked": (_B, (False, True, True))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="o__row_number_ranked",
            inputs={"p": (_D, (NAN, 1.0, NAN)), "o": (_D, (5.0, 1.0, 3.0))},
            expected={"o__row_number_ranked": (_I, (2, 1, 1))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="o__row_number_ranked",
            inputs={"p": (_S, (None, "a", None, "a")), "o": (_I, (2, 1, 1, 2))},
            expected={"o__row_number_ranked": (_I, (2, 1, 1, 2))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="o__row_number_ranked",
            inputs={"p": (_D, (None, NAN, None, NAN)), "o": (_D, (2.0, 2.0, 1.0, 1.0))},
            # None and NaN keys are two groups: (idx 0, 2) and (idx 1, 3)
            expected={"o__row_number_ranked": (_I, (2, 2, 1, 1))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="o__row_number_ranked",
            inputs={"p": (_D, (0.0, -0.0, 0.0)), "o": (_D, (3.0, 2.0, 1.0))},
            expected={"o__row_number_ranked": (_I, (3, 2, 1))},
            context=_CTX,
        ),
        PinningRow(
            case="empty",
            feature="o__row_number_ranked",
            inputs={"p": (_S, ()), "o": (_D, ())},
            expected={"o__row_number_ranked": (_I, ())},
            context=_CTX,
        ),
        PinningRow(
            case="empty",
            feature="o__percent_rank_ranked",
            inputs={"p": (_S, ()), "o": (_D, ())},
            expected={"o__percent_rank_ranked": (_D, ())},
            context=_CTX,
        ),
        PinningRow(
            case="empty",
            feature="o__top_2_ranked",
            inputs={"p": (_S, ()), "o": (_D, ())},
            expected={"o__top_2_ranked": (_B, ())},
            context=_CTX,
        ),
    ),
)
