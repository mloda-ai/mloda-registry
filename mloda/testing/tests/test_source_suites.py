"""Self-test: the guide's sensor CSV ReadFileFG passes the source suite reached through the public mloda.testing door."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

from mloda.provider import FeatureSet, ReadFileFG

from mloda.testing.source.read_file_suite import ReadFileFGTestMixin


def _cell(text: str) -> Any:
    return int(text) if text.lstrip("-").isdigit() else text


class MlodaTestingSensorCsvFG(ReadFileFG):
    """Sensor CSV from the input-data readers guide: a #sensor-schema header line, then comma-separated rows."""

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (".mlodatestingsensorcsv",)

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        with open(path, encoding="utf-8") as handle:
            header = handle.readline()
        if "#sensor-schema" not in header:
            raise ValueError("its header lacks the #sensor-schema marker")
        return header.replace("#sensor-schema", "").strip().split(",")

    @classmethod
    def load_neutral(cls, match: Any, features: FeatureSet) -> Any:
        import pyarrow as pa

        with open(match.access, encoding="utf-8") as handle:
            header = handle.readline().replace("#sensor-schema", "").strip().split(",")
            rows = [line.rstrip("\n").split(",") for line in handle if line.strip()]
        return pa.table(
            {name: [_cell(row[header.index(name)]) for row in rows] for name in sorted(features.get_all_names())}
        )


class TestMlodaTestingSensorCsvFG(ReadFileFGTestMixin):
    feature_group_class = MlodaTestingSensorCsvFG
    present_column = "mlodatesting_sensor_present"
    missing_column = "mlodatesting_sensor_missing"

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        names = list(columns)
        rows = zip(*(columns[name] for name in names))
        lines = ["#sensor-schema " + ",".join(names), *(",".join(str(v) for v in row) for row in rows)]
        path.write_text("\n".join(lines) + "\n")
