"""Self-test: a toy ReadFileFG passes the source suite reached through the public mloda.testing door."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

from mloda.provider import FeatureSet, ReadFileFG

from mloda.testing.source.read_file_suite import ReadFileFGTestMixin


def _cell(text: str) -> Any:
    return int(text) if text.lstrip("-").isdigit() else text


class MlodaTestingToyFileFG(ReadFileFG):
    """Header line of comma-separated column names, then comma-separated rows; unique suffix keeps it inert."""

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (".mlodatestingtoy",)

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        with open(path, encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n")
        if not header:
            raise ValueError(f"{path} has an empty header")
        return header.split(",")

    @classmethod
    def load_neutral(cls, match: Any, features: FeatureSet) -> Any:
        import pyarrow as pa

        with open(match.access, encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split(",")
            rows = [line.rstrip("\n").split(",") for line in handle if line.strip()]
        return pa.table(
            {name: [_cell(row[header.index(name)]) for row in rows] for name in sorted(features.get_all_names())}
        )


class TestMlodaTestingToyFileFG(ReadFileFGTestMixin):
    feature_group_class = MlodaTestingToyFileFG
    present_column = "mlodatesting_toy_present"
    missing_column = "mlodatesting_toy_missing"

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        names = list(columns)
        rows = zip(*(columns[n] for n in names))
        path.write_text(",".join(names) + "\n" + "".join(",".join(str(v) for v in r) + "\n" for r in rows))
