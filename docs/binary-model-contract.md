# Binary model interface contract

Black-box contract for any binary invoked by mloda as a model: how mloda calls it, how data
and configuration go in, how results and errors come out, how it finds its license, and what
it may and may not do with the data it is given. Every binary-backed FeatureGroup mixin and
every shipped binary implement this. Operations themselves (what the binary computes) are
opaque to mloda; this contract only specifies the envelope around them.

Rejection rule, used throughout: anything a binary or the mixin cannot honour (unsupported
type or operation, missing binary for the running platform, missing or invalid license) fails
up front with a clear error, before any data is processed; nothing is silently computed in
Python instead (the same rejection principle mloda-registry's own `CLAUDE.md` states for
compute-framework backends).

This is contract version 1. A binary states the version it implements as the integer
`contract` in `--capabilities`; the mixin refuses a binary whose `contract` is not the one it
was written for. The `--capabilities` invocation is frozen across all future versions: the
flag spelling, exit 0, no license required, one JSON object on stdout carrying an integer
`contract` key. Any deviation (non-zero exit, no JSON, no `contract` key) is an up-front
rejection with a clear error and no `run` is attempted, so a foreign executable behind the
path override never receives data. Not part of version 1: operation-level entitlement,
per-operation type and output lists, opting out of file transport, a license status query
without data, provenance metadata on the output, and operations that generate rows without
reading a column.

## Identifier

`plugin_id` is the binary's one identifier, matching `^[a-z][a-z0-9_]*$`. It is printed by
`--version`, reported by `--capabilities`, matched against the license token's `plugins`
entitlement list, and, for a shipped binary, equal to the wheel's import package name. The
mloda feature names a FeatureGroup answers to are unrelated to it.

## Invocation

One executable per platform:

- `<binary> --version`: prints exactly one line, `<plugin_id> <semver>`, to stdout, exits 0.
  The semver covers operation semantics: a change in an operation's output for the same input
  and parameters is a major bump, a new operation is a minor bump. The mixin logs it; pinning
  the wheel version is a packaging decision, not a runtime check. Contract compatibility is
  decided by `contract` in `--capabilities`, not by this version.
- `<binary> --capabilities`: prints exactly one JSON object to stdout, followed by at most one
  newline, nothing else, exits 0 (see Capabilities below).
- `<binary> run --config <path> [--input <path>] [--output <path>]`: executes the configured
  operation (see Errors for non-zero exits).

Any other argument combination, including no arguments and `--help`, is a usage error (exit
code 1); the binary is machine-invoked and has no interactive help.

`--config` is always a file path; there is no inline JSON form. `parameters` may carry
secrets (a hashing key), and argv is visible to every process on the host, while Windows
argv quoting of JSON is fragile; a file avoids both (at the price of the document touching
disk for the duration of the call, see Data handling). The mixin writes the document into
the private per-invocation directory and passes argv as a list, never through a shell.
`--input` and `--output` are independent, optional file paths for large data (see Data
below); a conforming binary supports all four combinations. Without `--input` the binary
reads stdin; with it, stdin is never read. Without `--output` the binary writes stdout; with
it, stdout stays empty. `--version` and `--capabilities` never require a license; license
verification applies only to `run`.

When stdin/stdout is used, the binary may write output before it finishes reading input;
callers must read stdout concurrently with writing stdin to avoid a pipe-buffer deadlock. The
binary exits before reading any data on license, configuration and operation errors, and
after reading only the schema on input schema errors, so a caller writing stdin must tolerate
a broken pipe (`EPIPE` on POSIX, the equivalent error on Windows) and then rely on exit code
and stderr. `subprocess.Popen.communicate()` does exactly this and is the caller primitive;
it buffers the whole output in memory, which is why the mixin switches to `--input`/`--output`
above a size threshold it defines. On Windows the binary reads stdin and writes stdout in
binary mode; no newline translation touches the IPC bytes.

The mixin resolves the binary from an explicit path given by the FeatureGroup (trusted
configuration) or else from the installed wheel (see Platform naming); there is no
environment variable that redirects it, since that would let anyone who controls the
pipeline's environment run an arbitrary executable on the caller's data. It runs `--version`
and `--capabilities` once per binary path per process, keyed on path, size and modification
time so an in-place upgrade is noticed, and caches the result. `run` is one process per
call: no batching across calls and no server mode in this version. The mixin may terminate a
binary at any point (a caller-configured timeout, or cancellation), with `SIGTERM` then
`SIGKILL` on POSIX and `TerminateProcess` on Windows. The binary keeps no state between
invocations, so termination is always safe. A termination the mixin initiated is reported as
code 6 from the mixin's own knowledge, whatever exit code the process shows (`TerminateProcess`
reports 1, which would otherwise read as a usage error).

Everything textual the binary reads or writes (config, `--version` and `--capabilities`
output, stderr) is UTF-8 with `\n` line endings, regardless of platform locale; the binary
configures its own streams for that and never depends on the locale it inherits.

## Capabilities

`--capabilities` output:

```json
{
  "contract": 1,
  "plugin_id": "example_binary",
  "operations": ["op-a", "op-b"],
  "column_types": ["int64", "float64", "utf8", "boolean"]
}
```

Callers ignore unknown keys. `operations` are identifiers the mixin passes through unchanged
in the config document. `column_types` is the union of the types the binary accepts for
`input_columns` and the types its outputs may have, drawn from the complete vocabulary of
this contract version: `int64`, `float64`, `utf8`, `boolean`, the Arrow types of the same
name (`utf8` is the 32-bit-offset string type, `pa.string()`, and only that type: a
`large_string` or `string_view` column is outside the vocabulary and rejected the same as any
other unsupported type). Other widths and parameterized or nested types (int32, float32,
timestamps, decimals, lists, structs, dictionaries) are not in the vocabulary and are
rejected up front by the mixin, or with code 4 by the binary. The mixin classifies pyarrow
types with pyarrow's predicates, never by string spelling: `pa.types.is_int64` -> `int64`,
`is_float64` -> `float64`, `is_boolean` -> `boolean`, `is_string` -> `utf8`. `large_string`
and `string_view` (what Polars emits) are cast to `utf8` by the mixin before sending; that
cast is lossless transport marshalling of the same logical type, which is why it is allowed
on the mixin's side while a width change such as int32 to int64 is not (it would change the
logical type and is left to the caller) -- but the binary itself only ever accepts the cast,
narrow `utf8` on the wire, never the wider types directly. Because `utf8` offsets are 32-bit,
the mixin splits the input into record batches small enough that no single array exceeds
2 GiB, and the binary splits its output batches the same way; output batch sizes are
otherwise unconstrained. Batching cannot split one cell: a single string value of 2 GiB or
more has no `utf8` representation and is rejected up front. On return the mixin casts `utf8`
output columns to the string type its frame uses (`large_string` for a Polars-backed frame)
before attaching them, so the frame stays homogeneous. The list is global, neither per
operation nor split by direction, so the mixin's up-front check is necessary but not
sufficient: a binary may still answer code 4 at `run` for a combination it cannot serve.

## Data

Arrow IPC stream format (record batches): on stdin and stdout by default, or the same stream
format written to/read from the `--input`/`--output` file paths for large data. The input
stream carries exactly the columns named in `input_columns` and nothing else: the mixin
projects its frame to those columns before sending, so the binary never sees, carries or
returns data it does not operate on. The mixin strips schema-level and field-level metadata
before writing (a pandas-backed frame otherwise carries its `pandas` schema blob); the binary
ignores any metadata it nonetheless receives, never fails on it, and emits none. A stream
must end with the IPC end-of-stream marker; "truncated" means end of file without that
marker, not "no more batches", and is a data error (code 5), as is any input that is not an
IPC stream at all: zero bytes, a zero-length `--input` file, the IPC file/Feather format
(`ARROW1` magic). A compressed record batch body is a data error (code 5); a
dictionary-encoded column is an unsupported column type (code 4), decided by the type check
like any other type outside the vocabulary.

The one logical Arrow IPC stream must also account for the whole input: trailing bytes after
its own end-of-stream marker (a second, concatenated stream, or arbitrary garbage) are a data
error too, not silently ignored or merged in. stdout (or `--output`) carries only the IPC
stream; all diagnostics go to stderr.

Input: the schema must contain exactly the `input_columns` names, in any order, without
duplicates; a missing name, an extra name or a duplicate is a data error (code 5). The order
of the `input_columns` list is part of the operation's input contract (a left/right pair);
the order of fields in the stream is not. Each column is then checked for a type from
`column_types` (code 4); presence errors precede type errors. Null values are permitted; how
an operation treats them belongs to the operation, not to this contract.

Output: exactly the operation's output columns under their written names, in the order the
operation defines them, each typed from `column_types`; no input column is echoed. The same
number of rows as the input, in the same order; batch boundaries may differ. The mixin
aligns the returned columns to its frame by row order and identifies them by the written
names in the output schema, never by ordinal, and verifies the result: the set of field
names equals the set of `output_columns` values, every type maps into `column_types`, the
row count matches, and the stream parses even on exit 0. Any mismatch is a binary bug
reported as code 6. A schema-only input (zero record batches, then the end-of-stream marker)
is valid and yields a schema-only output that already carries the output columns, so a
caller can learn output types without sending data; the license check still applies. The
end-of-stream marker on output is a binary obligation that the conformance kit checks on
the raw trailing bytes, since pyarrow's reader accepts a stream without it; at runtime the
mixin relies on the checks above.

Every operation is therefore row-preserving by construction: it cannot drop, add or reorder
rows. Record suppression (k-anonymity and similar) is expressed as null or replacement
values, never as removed rows. An operation that needs the whole input before it can write
(global statistics, k-anonymity) buffers every batch first; the contract permits that, and
`--input` file transport is the intended path when such an input is large. Row order cannot
be verified by the mixin at runtime without echoing a column, so it is a binary obligation:
a conforming implementation enforces it structurally by handing the underlying computation
each batch and emitting that batch's rows in input order, and the conformance kit tests it
with a distinct-valued input whose output must line up row for row.

## Configuration

The `--config` document requires all four keys below; unknown top-level keys are a usage
error (exit code 1):

```json
{
  "input_columns": ["col_a", "col_b"],
  "operation": "op-a",
  "parameters": {},
  "output_columns": {"result": "col_a_hash"}
}
```

- `input_columns`: the columns the operation reads, at least one, without duplicates (code 1
  otherwise); see Data for the checks against the input schema. The mixin rejects up front a
  name that is absent from the caller's frame, so the projection never fails downstream.
- `operation`: one of the identifiers from `--capabilities`; opaque to mloda.
- `parameters`: operation-specific; opaque to mloda; an empty object when the operation takes
  none. May contain secrets, which is why the document only ever travels as a file. An
  operation validates its own parameter shape exhaustively, not just the keys it recognizes:
  an unrecognized key is a usage error (code 1), the same as a recognized key of the wrong
  type.
- `output_columns`: maps each output name the operation produces (defined by the operation,
  known to the FeatureGroup that wraps it, opaque to the mixin; not required to correspond
  1:1 with `input_columns`) to the column name it is written under. Every output of the
  operation must be mapped. Written names must be unique among themselves and distinct from
  every `input_columns` entry (usage error, exit code 1, checked with the document). An
  unknown or unmapped output name is also exit code 1, but is checked only after the
  operation check, since it needs the operation's output list (see Errors). Whether a written
  name collides with any other column of the caller's frame is the mixin's up-front check,
  not the binary's: the binary never sees those columns. Key order is not significant;
  outputs are written in the order the operation defines them.

## License

The binary reads two variables from its own environment; the first one set to a non-empty
value wins:

1. `MLODA_LICENSE_FILE`: path to the license file.
2. `MLODA_LICENSE_KEY`: the license content inline.

The token is ASCII text; both sources are stripped of surrounding whitespace before parsing,
so a trailing newline in the file does not matter. The two variables name one license whose
`plugins` list covers every binary the deployment uses; several license files are not
supported. The mixin sets them from the FeatureGroup's overrides or from its own
environment, inside the minimal environment described under Data handling.

If neither variable is set to a non-empty value, or `MLODA_LICENSE_FILE` names a file that
does not exist, the binary exits with code 2 (license missing), before any configuration or
input data is read. If a source is set but the license cannot be read or verified, is expired,
or its `plugins` entitlement list does not contain this binary's `plugin_id`, the binary exits
with code 3 (license invalid); there is no fallback from a set but unusable
`MLODA_LICENSE_FILE` to `MLODA_LICENSE_KEY`. The error `message` names the source and the
reason (`MLODA_LICENSE_FILE /etc/mloda/license: not readable`), so a code 2 or 3 can be acted
on without guessing. Which other token states count as valid (grace window, not yet valid) is
defined by the token specification; every state it rejects is code 3. The token format,
signature scheme and payload fields are defined by a separate license token specification,
out of scope here.

Verification runs on every `run` and nothing is cached across invocations, so a token that
expires while a pipeline is running fails the next invocation with code 3, subject to the
token specification's grace rules. An offline token cannot be revoked before its expiry;
rotation and revocation are addressed by that same specification. The gate protects the
vendor's entitlement, not the caller's data: a binary patched to skip it is still bound by
the Data handling rules and changes nothing about where the data goes. Whether a deployment
allows the plugin at all is `PluginPolicy` governance in mloda core, not this contract.

## Data handling

The binary runs with the caller's full process privileges; the only technical controls are
the minimal environment and the private per-invocation directory below. Everything else in
this section is a promise the vendor makes and the conformance kit checks where it can.

The binary:

- makes no network connections; every check, the license one included, runs offline;
- reads no files other than `--config`, `--input` and `MLODA_LICENSE_FILE`, writes none other
  than `--output`, and keeps no state between invocations;
- never writes cell values to stderr or into an error `message`; diagnostics name columns,
  types, counts and positions only, and stay bounded (64 KiB of stderr in total is the soft
  cap the kit enforces; `message` is at most 1024 bytes, and reports only what it needs to,
  never the full text of an unexpected internal exception);
- reads no environment variables beyond the two license variables and what its runtime needs
  to start.

The mixin:

- invokes the binary by absolute path and passes a minimal environment: the two license
  variables, `PATH`, a fixed UTF-8 locale (`LC_ALL=C.UTF-8`, `LANG=C.UTF-8`) on POSIX, and
  `SYSTEMROOT` on Windows. Nothing else from the calling process reaches the binary;
- keeps `--config`, `--input` and `--output` in a fresh directory created per invocation
  under a fixed parent (`<temp>/mloda-binary/`), named with the mixin's process id, with
  owner-only permissions, so parallel invocations never collide and orphans are recognisable;
  removes it when the process ends, on failure and termination included, so a partial
  `--output` never outlives the call; and on each invocation removes sibling directories
  whose process id is no longer alive, which covers a mixin process killed outright. An
  existing `--output` file is overwritten by the binary;
- states the residual risk: for the duration of a call the config document (secrets
  included) and, with file transport, the raw input columns are at rest unencrypted in that
  directory. A deployment that cannot accept this points the temp directory (`TMPDIR`,
  `TEMP`) at an encrypted or memory-backed volume;
- logs `plugin_id`, version and exit code at debug level, never the config document.

## Errors

Checks run in this order: flag parsing, including that `--config` exists and is readable
(code 1); license; config parse and structural validation (JSON syntax, required and unknown
keys, `input_columns` non-empty and without duplicates, written-name uniqueness and
collision with `input_columns`; code 1); operation capability check (code 4);
`output_columns` completeness against the operation's output list (unknown or unmapped
output name; code 1); opening `--input` and creating `--output` (code 1 if either fails);
input schema check (exact column set, duplicates, types, see Data); data. A binary reports
the very first violation it finds at each stage rather than accumulating and reporting
several at once. Non-zero exit codes:

| Code | Class                                                              |
|------|--------------------------------------------------------------------|
| 1    | Usage error (bad flags or paths, malformed or unknown config keys) |
| 2    | License missing                                                    |
| 3    | License invalid, expired, or insufficient entitlement              |
| 4    | Unsupported operation or column type                               |
| 5    | Data error (malformed input, schema mismatch)                      |
| 6    | Internal error                                                     |

On any non-zero exit, the last non-empty line of stderr is exactly one JSON object; earlier
lines, if any, are free-form diagnostics:

```json
{"code": 3, "message": "license expired 2026-08-01"}
```

`code` matches the process exit code. `message` is a single human-readable line, no stack
traces or multi-line payloads; for an unexpected internal failure the message names only the
underlying exception's class, never its full text, since that text is not bounded and could
otherwise repeat caller data. Callers ignore any keys beyond `code` and `message`. Any exit
code not in this table (a signal-terminated process included), or a non-zero exit whose last
non-empty stderr line is not one parseable JSON object, is treated as code 6. A binary built
with `panic = "abort"` installs a panic hook that writes the code 6 object and exits 6, so an
internal failure never reaches the caller as a bare signal. On any non-zero exit, data already
written to stdout or `--output` is undefined and must be discarded by the caller. On exit 0,
stderr is free-form diagnostics that a caller may log but never parses. The mixin raises one
exception class per error class, all subclasses of `ValueError`, carrying `code` and
`message`, so a caller can tell a license problem from a data problem without parsing text;
a termination the mixin initiated and a contract violation in the output (wrong schema, row
count or unparseable stream on exit 0) are distinct classes that both carry code 6, so a
timeout is never mistaken for a crash.

## Platform naming and wheel binary path

| Platform         | Rust target                 | Wheel platform tag (minimum) |
|------------------|-----------------------------|------------------------------|
| `linux-x86_64`   | `x86_64-unknown-linux-gnu`  | `manylinux2014_x86_64`       |
| `linux-aarch64`  | `aarch64-unknown-linux-gnu` | `manylinux2014_aarch64`      |
| `macos-x86_64`   | `x86_64-apple-darwin`       | `macosx_10_12_x86_64`        |
| `macos-aarch64`  | `aarch64-apple-darwin`      | `macosx_11_0_arm64`          |
| `windows-x86_64` | `x86_64-pc-windows-msvc`    | `win_amd64`                  |

The platform identifier names the wrapper build a binary belongs to; the wheel tag selects
the wheel at install time; the mixin never computes or matches either. musl and 32-bit
targets are not supported.

The wrapper publishes wheels only, never an sdist, so on an unlisted platform `pip` finds no
distribution instead of attempting a Rust build. The registry package never makes the wheel
a hard dependency: it is an optional extra, so the base package installs everywhere;
requesting the extra on an unsupported platform still fails at install time, which is the
precise guarantee. Without the extra, the import of `<package>` raises `ModuleNotFoundError`,
which the mixin maps to the same clear error as a missing binary (rejection rule). That
import happens inside the call, never at module level of the FeatureGroup or its manifest:
mloda's plugin loader treats a `ModuleNotFoundError` outside its fixed optional-dependency
list as fatal for plugin discovery, so an eager import would take every plugin down with it.

Each wheel exposes its binary through one function, `binary_path() -> pathlib.Path`,
importable as `from <package> import binary_path`, where `<package>` equals `plugin_id`. The
binary is a data file inside the import package, not a console script in the environment's
scripts directory. maturin's `bindings = "bin"` places it in the wheel's scripts area and
`python-source` alone does not move it, so the wheel build stages the release binary into
the package explicitly before the wheel is assembled. On Windows the path has a `.exe`
suffix; on POSIX platforms the file is marked executable, and installers preserve that mode
from the wheel. A wheel whose binary is missing (a damaged or partial install, a packaging
tool that stripped the data file) makes `binary_path()` raise `FileNotFoundError`; importing
`<package>` never fails for that reason, so plugin discovery is unaffected, and no subprocess
is launched (rejection rule). A binary that exists but cannot be executed (`noexec` mount, an
executable blocked by endpoint policy) surfaces as `PermissionError` from the spawn, which
the mixin maps to the same clear error.

## Conformance

The simulated binary model and its conformance kit ship in the Apache-licensed
`mloda-testing` package, at `mloda.testing.binary_model` (installed via the `binary-model`
extra: `mloda-testing[binary-model]`, or `mloda-testing @ git+<repository-url>@<tag>#subdirectory=mloda/testing`
while a given integration branch is unreleased). `BinaryModelConformanceBase` carries every
contract-generic check and this list is authoritative; `HashOperationConformanceMixin` adds
the checks specific to the `hash` operation used in this contract's worked example. Both are
ordinary pytest classes whose binary command, `plugin_id`, supported operations, column-type
vocabulary and license fixture texts are overridable class attributes, so a binary
implementation subclasses them, points those attributes at itself, and inherits every
applicable check unmodified -- this is how the simulated binary, a wrapper build, and an
end-to-end run against a real binary all prove they behave identically without drifting
apart. A binary may honour the reserved operation `_conformance_internal_error` to let the
kit provoke code 6; the kit skips that case otherwise.

The kit checks: `--version` format and that `--capabilities`' `contract` value matches
`CONTRACT_VERSION`; all four transport combinations; every exit code and the check order; the
exact-column-set rule, including that a column typed outside the vocabulary
(`large_string`/`string_view` included -- only bare `utf8` counts) is rejected; the
schema-only round trip; the end-of-stream marker on the raw output bytes; rejection of
compression, dictionary columns, the file format, zero-byte input and trailing data after a
complete stream; the last-non-empty-line stderr rule and the size caps; that schema-level and
field-level Arrow metadata on input is accepted and never echoed back on output; data-free
diagnostics (no cell value of a marked input appears on stderr); no network (a run under a
network-denied sandbox such as `unshare -n` on Linux); no files created outside `--output`
(a run in a read-only working directory); the minimal environment (a run with the allowlist
only). Row count and row order preservation and the output column set and types are
operation-specific checks, not contract-generic ones -- `HashOperationConformanceMixin` is
the current example, for the `hash` operation.

## Out of scope

- License token payload, signature container, grace and validity rules, rotation and
  revocation stance: defined by a separate license token specification.
- The mloda-side mixin and example FeatureGroup: the wheel import package it names, the
  explicit binary path override, the size threshold for file transport, the exception
  classes it raises, and how it sets the license variables.
- The Rust trait between a wrapper build and a core crate (batches in, operation and
  parameters, outputs out, buffered or streaming), the wheel build matrix, panic hook, code
  signing and publishing.
- Provenance on the output (license id, binary version), a license status query,
  license-aware feature matching, and row-generating operations: reserved for later contract
  versions.
