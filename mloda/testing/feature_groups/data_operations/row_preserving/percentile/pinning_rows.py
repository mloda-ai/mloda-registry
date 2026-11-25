"""Hand-computed pinning rows of the percentile reference."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

NAN = float("nan")
INF = float("inf")

PERCENTILE_RULES: Mapping[str, str] = MappingProxyType(
    {
        "linear": "The percentile interpolates linearly at position q * (n - 1) of the sorted values.",
        "endpoints": "An integral position gives that exact value, and a non-integral one next to one infinity gives that infinity (row 21, like pc.quantile), so infinities do not turn into NaN.",
        "none": "None values are skipped; an all-None partition gives None.",
        "nan": "NaN values are skipped, also on their own row; an all-NaN partition gives None.",
        "partition_keys": "NaN and signed-zero partition keys merge into one group; a None key is its own group.",
        "dtype": "The output is float64 whatever the numeric source type.",
        "empty": "An empty input gives an empty float64 column.",
    }
)

_I = DataType.INT64
_D = DataType.DOUBLE
_S = DataType.STRING
_CTX = {"partition_by": ["p"]}

PERCENTILE_PINNING = PinningTable(
    operation="percentile",
    rules=PERCENTILE_RULES,
    rows=(
        PinningRow(
            case="linear",
            feature="v__p50_percentile",
            inputs={"p": (_S, ("a",) * 4), "v": (_D, (4.0, 1.0, 3.0, 2.0))},
            # position 0.5 * 3 = 1.5: halfway between 2.0 and 3.0
            expected={"v__p50_percentile": (_D, (2.5,) * 4)},
            context=_CTX,
        ),
        PinningRow(
            case="linear",
            feature="v__p25_percentile",
            inputs={"p": (_S, ("a",) * 4), "v": (_D, (4.0, 1.0, 3.0, 2.0))},
            # position 0.75: 1.0 + 0.75 * (2.0 - 1.0)
            expected={"v__p25_percentile": (_D, (1.75,) * 4)},
            context=_CTX,
        ),
        PinningRow(
            case="linear",
            feature="v__p75_percentile",
            inputs={"p": (_S, ("a",) * 4), "v": (_D, (4.0, 1.0, 3.0, 2.0))},
            # position 2.25: 3.0 + 0.25 * (4.0 - 3.0)
            expected={"v__p75_percentile": (_D, (3.25,) * 4)},
            context=_CTX,
        ),
        PinningRow(
            case="linear",
            feature="v__p50_percentile",
            inputs={"p": (_S, ("a", "a", "b")), "v": (_D, (1.0, 3.0, 7.0))},
            expected={"v__p50_percentile": (_D, (2.0, 2.0, 7.0))},
            context=_CTX,
        ),
        PinningRow(
            case="endpoints",
            feature="v__p0_percentile",
            inputs={"p": (_S, ("a",) * 3), "v": (_D, (-INF, 1.0, INF))},
            expected={"v__p0_percentile": (_D, (-INF,) * 3)},
            context=_CTX,
        ),
        PinningRow(
            case="endpoints",
            feature="v__p100_percentile",
            inputs={"p": (_S, ("a",) * 3), "v": (_D, (-INF, 1.0, INF))},
            expected={"v__p100_percentile": (_D, (INF,) * 3)},
            context=_CTX,
        ),
        PinningRow(
            case="endpoints",
            feature="v__p50_percentile",
            inputs={"p": (_S, ("a",) * 3), "v": (_D, (-INF, 1.0, INF))},
            expected={"v__p50_percentile": (_D, (1.0,) * 3)},
            context=_CTX,
        ),
        PinningRow(
            case="endpoints",
            feature="v__p50_percentile",
            inputs={"p": (_S, ("a",) * 2), "v": (_D, (INF, INF))},
            # equal neighbours give that value, like pc.quantile, not inf - inf = NaN
            expected={"v__p50_percentile": (_D, (INF,) * 2)},
            context=_CTX,
        ),
        PinningRow(
            case="none",
            feature="v__p50_percentile",
            inputs={"p": (_S, ("a",) * 3), "v": (_D, (1.0, None, 3.0))},
            expected={"v__p50_percentile": (_D, (2.0, 2.0, 2.0))},
            context=_CTX,
        ),
        PinningRow(
            case="none",
            feature="v__p50_percentile",
            inputs={"p": (_S, ("a", "a", "b")), "v": (_I, (None, None, 4))},
            expected={"v__p50_percentile": (_D, (None, None, 4.0))},
            context=_CTX,
        ),
        PinningRow(
            case="nan",
            feature="v__p50_percentile",
            inputs={"p": (_S, ("a",) * 3), "v": (_D, (1.0, NAN, 3.0))},
            expected={"v__p50_percentile": (_D, (2.0, 2.0, 2.0))},
            context=_CTX,
        ),
        PinningRow(
            case="nan",
            feature="v__p50_percentile",
            inputs={"p": (_S, ("a", "a", "b")), "v": (_D, (NAN, NAN, 5.0))},
            expected={"v__p50_percentile": (_D, (None, None, 5.0))},
            context=_CTX,
        ),
        PinningRow(
            case="nan",
            feature="v__p50_percentile",
            inputs={"p": (_S, ("a", "a", "a")), "v": (_D, (NAN, None, NAN))},
            expected={"v__p50_percentile": (_D, (None, None, None))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="v__p50_percentile",
            inputs={"p": (_D, (NAN, 1.0, NAN)), "v": (_D, (1.0, 10.0, 3.0))},
            expected={"v__p50_percentile": (_D, (2.0, 10.0, 2.0))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="v__p50_percentile",
            inputs={"p": (_S, (None, "a", None, "a")), "v": (_D, (1.0, 10.0, 3.0, 20.0))},
            expected={"v__p50_percentile": (_D, (2.0, 15.0, 2.0, 15.0))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="v__p50_percentile",
            inputs={"p": (_D, (0.0, -0.0, 0.0)), "v": (_D, (1.0, 2.0, 3.0))},
            expected={"v__p50_percentile": (_D, (2.0, 2.0, 2.0))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="v__p50_percentile",
            inputs={"p": (_D, (None, NAN, None, NAN)), "v": (_D, (1.0, 10.0, 3.0, 20.0))},
            # None and NaN keys are two groups
            expected={"v__p50_percentile": (_D, (2.0, 15.0, 2.0, 15.0))},
            context=_CTX,
        ),
        PinningRow(
            case="dtype",
            feature="v__p50_percentile",
            inputs={"p": (_S, ("a",) * 4), "v": (_I, (1, 2, 3, 4))},
            expected={"v__p50_percentile": (_D, (2.5,) * 4)},
            context=_CTX,
        ),
        PinningRow(
            case="dtype",
            feature="v__p50_percentile",
            inputs={"p": (_S, ("a",)), "v": (_I, (7,))},
            expected={"v__p50_percentile": (_D, (7.0,))},
            context=_CTX,
        ),
        PinningRow(
            case="empty",
            feature="v__p50_percentile",
            inputs={"p": (_S, ()), "v": (_D, ())},
            expected={"v__p50_percentile": (_D, ())},
            context=_CTX,
        ),
        PinningRow(
            case="empty",
            feature="v__p50_percentile",
            inputs={"p": (_S, ()), "v": (_I, ())},
            expected={"v__p50_percentile": (_D, ())},
            context=_CTX,
        ),
        PinningRow(
            case="endpoints",
            feature="v__p25_percentile",
            inputs={"p": (_S, ("a",) * 3), "v": (_D, (-INF, 1.0, INF))},
            # a half-infinite interval gives the infinite neighbour, like pc.quantile
            expected={"v__p25_percentile": (_D, (-INF,) * 3)},
            context=_CTX,
        ),
    ),
)
