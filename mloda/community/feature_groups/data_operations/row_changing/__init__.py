"""Row-changing data operations (collapse / expand the row count).

Unlike ``row_preserving`` operations (which return one output row per input
row) and ``aggregation`` (which reduces to one row per partition key), the
operations in this package change the row count by collapsing events onto a
new index. The first member is ``resample`` (one row per non-empty time
bucket per partition).
"""
