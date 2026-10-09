"""Small pipeline runners for exercising Extenders through mloda.run_all."""

from __future__ import annotations

import signal
import subprocess  # nosec
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.steward import Extender, ExtenderHook
from mloda.user import (
    Feature,
    FeatureName,
    Index,
    JoinSpec,
    Link,
    Options,
    ParallelizationMode,
    PluginCollector,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader

from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator


def expected_value_int() -> list[Any]:
    """Expected `value_int` column values from the canonical raw test fixture."""
    return PyArrowDataOpsTestDataCreator.get_raw_data()["value_int"]


def run_value_int(
    *extenders: Extender,
    parallelization_modes: set[ParallelizationMode] | None = None,
    flight_server: Any | None = None,
    child_bootstrap: Callable[[], None] | None = None,
) -> list[Any]:
    """Run `value_int` through the pipeline with the given extenders; return the column. Optional
    parallelization_modes, flight_server and child_bootstrap forward straight to mloda.run_all."""
    plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})
    results = mloda.run_all(
        ["value_int"],
        compute_frameworks=[PyArrowTable],
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
        parallelization_modes=parallelization_modes or {ParallelizationMode.SYNC},
        flight_server=flight_server,
        child_bootstrap=child_bootstrap,
    )
    for table in results:
        if isinstance(table, pa.Table) and "value_int" in table.column_names:
            column: list[Any] = table.to_pydict()["value_int"]
            return column
    raise AssertionError("No result table with value_int found")


def prepare_value_int(*extenders: Extender, parallelization_modes: set[ParallelizationMode] | None = None) -> mloda:
    """Prepare (but do not run) `value_int` through the pipeline with the given extenders; call
    session.run() to execute it. Optional parallelization_modes forwards straight to mloda.prepare."""
    plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})
    return mloda.prepare(
        ["value_int"],
        compute_frameworks=[PyArrowTable],
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
        parallelization_modes=parallelization_modes or {ParallelizationMode.SYNC},
    )


# Module-level so MULTIPROCESSING can pickle it by path.
# The prefix keeps it from colliding with a host's features once registered.
class MlodaTestingValueIntPlusOne(FeatureGroup):
    """Adds one to `value_int`, null-safe; used to exercise two chained calculate invocations."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("value_int")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        values = data["value_int"].to_pylist()
        return {cls.get_class_name(): [None if v is None else v + 1 for v in values]}


def run_two_features(
    *extenders: Extender,
    parallelization_modes: set[ParallelizationMode] | None = None,
    flight_server: Any | None = None,
) -> list[Any]:
    """Run a `value_int`-plus-one feature group through the pipeline, chaining two
    FEATURE_GROUP_CALCULATE_FEATURE invocations (the data creator, then this feature group); return the plus-one
    column. Optional parallelization_modes and flight_server forward straight to mloda.run_all."""
    feature_group = MlodaTestingValueIntPlusOne
    plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, feature_group})
    column_name = feature_group.get_class_name()
    results = mloda.run_all(
        [column_name],
        compute_frameworks=[PyArrowTable],
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
        parallelization_modes=parallelization_modes or {ParallelizationMode.SYNC},
        flight_server=flight_server,
    )
    for table in results:
        if isinstance(table, pa.Table) and column_name in table.column_names:
            column: list[Any] = table.to_pydict()[column_name]
            return column
    raise AssertionError(f"No result table with {column_name} found")


class _JoinSource:
    """Plain mixin (not a FeatureGroup) serving `_columns` as a PyArrow source indexed on `_index_column`."""

    _columns: ClassVar[dict[str, list[Any]]]
    _index_column: ClassVar[str]

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index((cls._index_column,))]

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(set(cls._columns))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(cls._columns)


# Module-level so MULTIPROCESSING can pickle the link by path.
# The MlodaTesting class prefix and the column prefix keep them from colliding with a host's features once registered
# (the default matcher also matches by class name).
class MlodaTestingJoinLeft(_JoinSource, FeatureGroup):
    """Left PyArrow source of the join."""

    _columns = {"mloda_testing_left_id": [1, 2, 3], "mloda_testing_left_value": [10, 20, 30]}
    _index_column = "mloda_testing_left_id"


class MlodaTestingJoinRight(_JoinSource, FeatureGroup):
    """Right PyArrow source of the join."""

    _columns = {"mloda_testing_right_id": [1, 2, 4], "mloda_testing_right_value": [100, 200, 400]}
    _index_column = "mloda_testing_right_id"


class MlodaTestingJoinedSum(FeatureGroup):
    """Adds `mloda_testing_left_value` and `mloda_testing_right_value`, null-safe; forces the join of both sources."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mloda_testing_left_value"), Feature("mloda_testing_right_value")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        pairs = zip(data["mloda_testing_left_value"].to_pylist(), data["mloda_testing_right_value"].to_pylist())
        return {cls.get_class_name(): [None if a is None or b is None else a + b for a, b in pairs]}


def run_joined_features(
    *extenders: Extender,
    parallelization_modes: set[ParallelizationMode] | None = None,
    flight_server: Any | None = None,
) -> list[Any]:
    """Run an inner join of two PyArrow sources (mloda_testing_left_id=mloda_testing_right_id) into a
    consumer that sums `mloda_testing_left_value` and `mloda_testing_right_value`; return the summed
    column. Optional parallelization_modes and flight_server forward straight to mloda.run_all."""
    link = Link.inner(
        JoinSpec(MlodaTestingJoinLeft, Index(("mloda_testing_left_id",))),
        JoinSpec(MlodaTestingJoinRight, Index(("mloda_testing_right_id",))),
    )
    plugin_collector = PluginCollector.enabled_feature_groups(
        {MlodaTestingJoinLeft, MlodaTestingJoinRight, MlodaTestingJoinedSum}
    )
    column_name = MlodaTestingJoinedSum.get_class_name()
    results = mloda.run_all(
        [column_name],
        compute_frameworks=[PyArrowTable],
        plugin_collector=plugin_collector,
        links={link},
        function_extender=set(extenders),
        parallelization_modes=parallelization_modes or {ParallelizationMode.SYNC},
        flight_server=flight_server,
    )
    for table in results:
        if isinstance(table, pa.Table) and column_name in table.column_names:
            column: list[Any] = table.to_pydict()[column_name]
            return column
    raise AssertionError(f"No result table with {column_name} found")


def run_csv_feature(
    directory: Path,
    *extenders: Extender,
    parallelization_modes: set[ParallelizationMode] | None = None,
    flight_server: Any | None = None,
    carrier: dict[str, str] | None = None,
) -> list[Any]:
    """Write a small CSV into `directory` and run its `alpha` column through the pipeline, firing a nested
    INPUT_DATA_LOAD hook with `data_access_identity` set to the CSV's path; return the column. Optional
    parallelization_modes, flight_server and carrier forward straight to mloda.run_all."""
    path = directory / "data.csv"
    path.write_text("alpha,beta\n1,2\n3,4\n", encoding="utf-8")
    plugin_collector = PluginCollector.enabled_feature_groups({ReadFileFeature})
    results = mloda.run_all(
        [Feature("alpha", options={CsvReader.__name__: str(path)})],
        compute_frameworks=[PyArrowTable],
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
        parallelization_modes=parallelization_modes or {ParallelizationMode.SYNC},
        flight_server=flight_server,
        carrier=carrier,
    )
    for table in results:
        if isinstance(table, pa.Table) and "alpha" in table.column_names:
            column: list[Any] = table.to_pydict()["alpha"]
            return column
    raise AssertionError("No result table with alpha found")


class CountingExtender(Extender):
    """Breaking pass-through probe that counts its own invocations.

    If marker_path is set, each call also appends one line to it, so calls made in a MULTIPROCESSING
    worker stay visible to the parent.
    """

    def __init__(self, marker_path: Path | None = None) -> None:
        self.raise_on_error = True
        self.calls = 0
        self.marker_path = marker_path
        # Above the default priority (100) so this probe always sorts downstream of a
        # default-priority host extender, regardless of set iteration order.
        self.priority = 200

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.marker_path is not None:
            with open(self.marker_path, "a", encoding="utf-8") as marker_file:
                marker_file.write("call\n")
        return func(*args, **kwargs)


# The MlodaTesting prefix keeps it and its minted subclasses from matching a host feature by class name.
class MlodaTestingFailingFeatureGroup(FeatureGroup):
    """Primary-source feature group that always raises; the sentinel feature_name never matches a real request."""

    feature_name: str = "mloda_testing_never_requested"
    calls: int = 0

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({cls.feature_name})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.calls += 1
        raise RuntimeError("inner boom")


def failing_feature_group(feature_name: str) -> type[MlodaTestingFailingFeatureGroup]:
    """Build a fresh MlodaTestingFailingFeatureGroup subclass per call so parallel tests never share state."""

    class MlodaTestingFailing(MlodaTestingFailingFeatureGroup):
        pass

    MlodaTestingFailing.feature_name = feature_name
    MlodaTestingFailing.calls = 0
    return MlodaTestingFailing


def run_failing_feature(feature_group: type[MlodaTestingFailingFeatureGroup], *extenders: Extender) -> Any:
    """Run feature_group.feature_name through the pipeline; calculate_feature always raises."""
    return run_feature(feature_group, *extenders)


def run_feature(feature_group: type[MlodaTestingFailingFeatureGroup], *extenders: Extender) -> Any:
    """Run feature_group.feature_name through the pipeline with the given extenders."""
    plugin_collector = PluginCollector.enabled_feature_groups({feature_group})
    return mloda.run_all(
        [feature_group.feature_name],
        compute_frameworks=[PyArrowTable],
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
    )


def run_until_ready_then_sigterm(
    args: list[str], *, env: dict[str, str] | None = None, timeout: float = 60.0
) -> tuple[int, str]:
    """Spawn args, wait for a stdout line "ready", send SIGTERM; return (returncode, merged stdout and stderr).
    timeout is one deadline over the whole run; on expiry the child is killed and the output so far is reported."""
    proc = subprocess.Popen(  # nosec
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )
    lines: list[str] = []
    ready = threading.Event()

    def read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if line.strip() == "ready":
                ready.set()
        ready.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        if not ready.wait(timeout) or not any(line.strip() == "ready" for line in lines):
            raise AssertionError(f"child never printed ready within {timeout}s: {''.join(lines)}")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=max(deadline - time.monotonic(), 0.0))
        except subprocess.TimeoutExpired:
            raise AssertionError(f"child did not exit within {timeout}s of start: {''.join(lines)}") from None
        reader.join(5.0)
        return proc.returncode, "".join(lines)
    finally:
        proc.kill()
        proc.wait()
