# Pattern 28: Binary-Backed Features

Run a compiled binary (a model shipped as a wheel, usually license-gated) as the compute step of a FeatureGroup.

**What**: A FeatureGroup that sends selected columns to an external executable over Arrow IPC and attaches the returned columns.
**When**: The model is a black box built outside Python (Rust, C++), and the vendor gates it with a license.
**Why**: One mixin owns the process envelope (binary resolution, minimal environment, transport, error mapping), so a plugin only declares options and maps the result.
**Where**: Mixin in `mloda.community.feature_groups.binary_model` (Apache-2.0); the template for a paid plugin in `mloda.enterprise.feature_groups.binary_example`.
**How**: Subclass `BinaryModelMixin` next to `FeatureGroup`, set `BINARY_PLUGIN_ID` and `BINARY_WHEEL_DISTRIBUTION`, call `run_binary_model()` from `calculate_feature()`.

## Key Characteristic

| Class attribute | Meaning |
|-----------------|---------|
| `BINARY_PLUGIN_ID` | Import package of the wheel that ships the binary (`from <id> import binary_path`); also the id the license entitles |
| `BINARY_WHEEL_DISTRIBUTION` | The wheel's PyPI distribution name, distinct from `BINARY_PLUGIN_ID` (the import name); used in `packages.toml` |
| `BINARY_COMMAND_OVERRIDE` | Explicit argv prefix or path used instead of the wheel; tests point it at the simulated binary. No environment variable can redirect the binary |
| `LICENSE_FILE_OVERRIDE`, `LICENSE_KEY_OVERRIDE` | Values for `MLODA_LICENSE_FILE` / `MLODA_LICENSE_KEY` in the binary's environment; unset, the caller's own values are forwarded, and an empty string suppresses that forwarding |
| `BINARY_TIMEOUT_SECONDS` | Wall-clock limit per call; the process is terminated and `BinaryTerminatedError` raised |
| `FILE_TRANSPORT_THRESHOLD_BYTES` | Inputs above it travel through `--input` / `--output` files instead of stdin / stdout |
| `MAX_BATCH_BYTES` | Upper bound per record batch sent to the binary; oversized batches are split until they fit, keeping `utf8` arrays clear of the 2 GiB offset limit |

The wheel is imported inside the call, never at module level of the FeatureGroup or its `manifest.py`: mloda's plugin loader aborts discovery on a `ModuleNotFoundError` it does not know as optional, and a missing wheel must only fail the call.

## Complete Example

```python
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pyarrow as pa
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet, property_spec
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.binary_model.mixin import BinaryModelMixin


def _is_column_list(value: object) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and all(isinstance(v, str) and v for v in value)


def _is_parameters_mapping(value: object) -> bool:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


class BinaryExampleFeatureGroup(BinaryModelMixin, FeatureGroup):
    """Keyed hash of the configured columns, computed by the ``example_binary`` wheel."""

    BINARY_PLUGIN_ID = "example_binary"
    BINARY_WHEEL_DISTRIBUTION = "mloda-example-binary"
    OUTPUT_KEY = "result"
    OPERATION = "binary_operation"
    INPUT_COLUMNS = "binary_input_columns"
    PARAMETERS = "binary_parameters"

    PROPERTY_MAPPING: ClassVar = {
        OPERATION: property_spec("Operation the binary runs", strict=True, allowed_values={"hash": "Keyed hash"}),
        INPUT_COLUMNS: property_spec(
            "Frame columns sent to the binary, in operation order", match_guard=_is_column_list
        ),
        PARAMETERS: property_spec(
            "Operation parameters passed through unchanged", default=None, match_guard=_is_parameters_mapping
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.not_typed(name) for name in options.get(self.INPUT_COLUMNS)}

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: FeatureName | str, options: Options, data_access_collection: Any = None
    ) -> bool:
        if options is None or options.get(cls.OPERATION) != "hash":
            return False
        if not _is_column_list(options.get(cls.INPUT_COLUMNS)):
            return False
        parameters = options.get(cls.PARAMETERS)
        return parameters is None or _is_parameters_mapping(parameters)

    @classmethod
    def calculate_feature(cls, data: pa.Table, features: FeatureSet) -> pa.Table:
        for feature in features.features:
            columns: Sequence[str] = feature.options.get(cls.INPUT_COLUMNS)
            parameters: Mapping[str, Any] = feature.options.get(cls.PARAMETERS) or {}
            result = cls.run_binary_model(data, columns, "hash", parameters, {cls.OUTPUT_KEY: feature.name})
            data = data.append_column(feature.name, result.column(feature.name))
        return data
```

The feature name becomes the written output column, so one class serves any number of hashed features:

```python
from mloda.user import Feature, Options

feature = Feature(
    "customer_hash", Options(context={"binary_operation": "hash", "binary_input_columns": ["first_name", "last_name"]})
)
```

## What `run_binary_model()` Does

Before the binary runs, in this order: resolves and probes the binary (`--version`, `--capabilities`; a wrong `contract`, a missing wheel or a non-executable path is `BinaryUnavailableError`), rejects an input column absent from the frame or a written name that collides with any frame column (`BinaryUsageError`), an operation the binary does not list or a column type outside `int64`, `float64`, `utf8`, `boolean` (`UnsupportedError`), and a single string cell of 2 GiB or more (`DataError`). Nothing is ever computed in Python instead.

Then it projects the frame to the input columns, strips Arrow metadata, casts `large_string` and `string_view` to `utf8`, writes the config to a file in a private per-invocation directory under `<temp>/mloda-binary/`, runs the binary with a minimal environment (the two license variables, `PATH`, a fixed `C.UTF-8` locale, and `SYSTEMROOT` on Windows), and verifies the answer: output column set, types and row count, else `OutputContractError`. `utf8` outputs are cast back to `large_string` when the frame uses it. The directory is removed on every exit path.

| Raised | Exit code | Meaning |
|--------|-----------|---------|
| `BinaryUsageError` | 1 | Bad flags, paths or config |
| `LicenseMissingError` | 2 | No license source set |
| `LicenseInvalidError` | 3 | License unreadable, expired or not entitled to `BINARY_PLUGIN_ID` |
| `UnsupportedError` | 4 | Operation or column type the binary does not support |
| `DataError` | 5 | Malformed input or schema mismatch |
| `BinaryInternalError` | 6 | The binary crashed or reported an internal error |
| `BinaryTerminatedError` | 6 | The mixin terminated it (timeout) |
| `OutputContractError` | 6 | Exit 0 but the output broke the contract |

All are `ValueError` subclasses carrying `code` and `message`. The binary's `--version` and exit code are logged at debug level; the config (which may carry secrets) never is.

## Test

Point the class at the simulated binary from `mloda-testing[binary-model]` and give it a signed test license (the vectors are signed with the published test key):

```python
import sys

from mloda.testing.binary_model.license_vectors import valid_license_token


class StubExample(BinaryExampleFeatureGroup):
    BINARY_COMMAND_OVERRIDE = [sys.executable, "-m", "mloda.testing.binary_model.simulated_binary"]
    LICENSE_KEY_OVERRIDE = valid_license_token(["example_binary"])
```

Expected values come from `mloda.testing.binary_model.hash_reference.compute_expected_hash_column`. Cover the three levels of the [testing guide](10-testing-guide.md); at level 3 pass `compute_frameworks={PyArrowTable}` and `PluginCollector.enabled_feature_groups({StubExample, ApiInputDataFeature})` (the `api_data` reader must stay enabled alongside your class). The production class without an override must raise `BinaryUnavailableError`, and a run without a license `LicenseMissingError`.

### Against the real wheel

The stub above stays the fast-tests path; a suite that needs the actual compiled binary points the production class at it directly, no `BINARY_COMMAND_OVERRIDE`. The shared test-signed license is accepted only by a test-key build:

```python
class RealWheelExample(BinaryExampleFeatureGroup):
    LICENSE_KEY_OVERRIDE = valid_license_token(["example_binary"])
```

See `tests/test_binary_model_real/` for the full suite.

In the checkout, `uv pip install mloda-example-binary` gives the release build from PyPI (the dev venv has no `pip`, and re-running the documented `uv sync --all-packages --extra dev` removes the wheel again; `--all-extras` would install it through the `wheel` extra). It trusts only production keys, so it rejects the shared test-signed license vectors as an unknown key id. Run the suite on its own, in the dev environment, with `pytest tests/test_binary_model_real/`. Against the release build it covers the version and capabilities probes and both license error paths; the end-to-end test skips. That test needs a test-key build, made with the [mloda-binary-wrapper repo's build steps](https://github.com/mloda-ai/mloda-binary-wrapper). Test-key wheels are never published, because they accept a publicly known signing key.

Keep the rest of the repo's suite out of a run that has the wheel: it assumes the wheel is absent, and two guards fail loudly if the wheel is installed by accident. Running only `tests/test_binary_model_real/` never hits them.

The full expired/in-grace/valid license state machine is covered against the real compiled binary in the mloda-binary-wrapper repo's own CI across platforms, so it is not duplicated here. The plain `tox` gate never installs the wheel, so the suite is skipped there; the `real-wheel` CI job installs the release wheel from `uv.lock` and runs the suite; `tox -e real-wheel` reproduces that locally in its own environment, leaving the dev `.venv` untouched. The job covers the platforms the wrapper's CI run-tests natively: Linux x86_64, macOS Intel and Apple silicon, and Windows x86_64. Linux aarch64 is left out because the wrapper only smoke-tests it under emulation. The probe's own logic is covered unconditionally against the simulated binary by `tests/test_binary_model_real/test_probe_classification.py`. The full production-license end-to-end path remains deferred follow-up work.

## Packaging Rules

- `mloda-testing[binary-model]` (the stub, Arrow helpers, license vectors) is a `dev` extra only; nothing under `mloda/community/` or `mloda/enterprise/` imports `mloda.testing` at runtime.
- The wheel is never a hard dependency of the plugin package; without it the call rejects, discovery still works.
- A binary that implements the contract is verified with `mloda.testing.binary_model.conformance.BinaryModelConformanceBase`, the same kit the simulated binary passes.
- The wheel's distribution (`BINARY_WHEEL_DISTRIBUTION`) is declared under `optional_dependencies` with a version range, never under `dependencies` or `dev`; install it with `pip install mloda-example-binary`.
- The `wheel` extra lives on `mloda-enterprise-binary-example`, which ships inside the `mloda-enterprise` bundle. The bundle does not re-export the extra, so install the wheel directly.

## Combines With

- **Pattern 9 (Framework-specific)**: the transport is Arrow, so restrict to `PyArrowTable` (Polars is Arrow-backed too)
- **Pattern 11 (Options)**: the operation and its columns are `context` options; the feature name is the output
