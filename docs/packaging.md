# Packaging

All `pyproject.toml` files are auto-generated from `config/`. Never edit them directly.
For how those packages reach PyPI, see [Releasing](releasing.md).

```bash
python scripts/generate_pyproject.py          # Generate all
python scripts/generate_pyproject.py --check  # CI validation (tox -e check-generated)
```

## Architecture

```text
config/
├── shared.toml       # version, authors, urls, defaults
└── packages.toml     # per-package: description, deps, path
         │
         ▼
scripts/generate_pyproject.py
         │
         ├──► mloda/*/pyproject.toml
         └──► pyproject.toml (workspace members + mloda core dependency)
```

`--check` fails if any generated file has drifted, including the root
`pyproject.toml`'s `mloda` entry, which the generator rewrites from
`core_dependency`. `.github/dependabot.yml` excludes `mloda` from
uv-ecosystem updates for the same reason: its floor moves through
`config/shared.toml`, not a version-bump PR.

## Config files

### shared.toml

Single source for values every package shares. `core_dependency` is substituted
into any `{core_dependency}` placeholder in `packages.toml`, so the mloda floor
is declared once. The `[build-system]` setuptools floor tracks the PEP 639 SPDX
`license` string the generator emits (accepted only from setuptools 77.0.1 on),
so it cannot be lowered without changing that form. The root `pyproject.toml`
keeps its own, higher, independent setuptools floor: it is the dev-only
workspace package, is never published, declares no `license`, and that floor
is Dependabot-managed.

```toml
# Illustrative values; see config/shared.toml for the current ones.
[project]
version = "0.4.0"
requires-python = ">=3.10,<3.15"
authors = [{ name = "Tom Kaltofen", email = "info@mloda.ai" }]

[defaults]
license = "Apache-2.0"
core_dependency = "mloda>=0.11.0,<0.12.0"
optional_dependencies = { dev = ["mloda-testing", "pytest>=9.0.3"] }
```

### packages.toml

| Field | Required | Description |
|-------|----------|-------------|
| `description` | Yes | PyPI description |
| `path` | Yes | Package directory |
| `published` | No | `true` ships the distribution standalone on PyPI. Single source of the released set, read through `scripts/published_packages.py`. Must be a boolean. It governs the released set, and also wheel contents: a bundle must own (name in its own `dependencies` or a non-dev extra) every published package nested under its path, and ships only the unpublished rest |
| `dependencies` | By convention | Runtime deps; use `"{core_dependency}"` for the mloda floor, `"<sibling>>={version}"` for a sibling package, or, for a package nested under an `entry_point_bundle`'s own path, `"<sibling>=={version}"` to own it (see [Sibling dependency floors](#sibling-dependency-floors)). The generator defaults it to empty rather than failing, but every package declares it |
| `optional_dependencies` | No | Merged with defaults. The entry `"{published_children}"` expands to every published package nested under this package's path, in config order; an `entry_point_bundle` cannot use it and names each package it owns instead, through a non-dev extra the same as through `dependencies`. A test-only third-party dependency goes in `dev` here; see [Add a test-only dependency](#add-a-test-only-dependency) |
| `optional_dependency_indexes` | No | `{ "<dependency>" = "<index name>" }`, pinning an optional dependency to a named index from `[defaults.uv_indexes]` (see [UV workspace sources](#uv-workspace-sources)), for a dependency not published on the default index |
| `has_readme` | No | `true` points the package at its own `README.md` |
| `workspace_deps` | No | Marks a meta-package whose deps are workspace siblings. Mutually exclusive with `py_typed`; unused today |
| `entry_point_groups` | No | List of mloda entry-point groups the package's `manifest.py` populates (`mloda.feature_groups`, `mloda.compute_frameworks`, `mloda.extenders`) |
| `entry_point_bundle` | No | `true` on bundle packages (`mloda-community`, `mloda-enterprise`); aggregates the entry points of every nested plugin package under its path it does not own. An owned package declares its own entry points in its own generated pyproject instead. Mutually exclusive with `entry_point_groups` |
| `py_typed` | No | `true` adds the dotted path to `packages` (what ships the marker) and emits `[tool.setuptools.package-data]` for it. Requires a committed `<path>/py.typed`. Mutually exclusive with `workspace_deps` |

For a `data_operations` leaf package, `optional_dependencies` must declare exactly the backend
extras its `manifest.py` registers, no more and no less, and its `polars` extra must use the same
version floor as mloda core's own `polars` extra: `tests/test_end2end/test_backend_optional_dependencies.py`
derives the expected set from each manifest, reads core's floor from its installed metadata, and fails
the build on drift. A leaf may raise its `polars` floor above core's only through
`_POLARS_FLOOR_OVERRIDES` in that same test (today `mloda-community-frame-aggregate`, `>=1.38`,
because its time window's `rolling_*_by` rejects null values on older polars).

A marker declares its whole subtree typed, including third-party distributions installed into it: on a namespace portion (`mloda/community`, `mloda/enterprise`) that is the entire namespace, on a shared base package (`mloda/community/feature_groups/data_operations`, `mloda/community/feature_groups/example`) it is everything published from below that base. mypy returns at the first `py.typed` on the module path, so those leaf packages need no flag of their own. The sibling dependency floor below already keeps the leaf at or above the release that first shipped the marker.

### Sibling dependency floors

A dependency on another package of the same `packages` table (a sibling, in-repo
dependency) is written `"<sibling>>={version}"`. The generator expands `{version}` to
`[project].version` in `shared.toml` and rejects a sibling dependency written without
the placeholder. Every release regenerates `pyproject.toml`, so a leaf always requires
the base built from the same commit. The mloda core floor (`core_dependency`) is
unaffected: it stays a real, hand-set minimum in `shared.toml`.

The same rule covers `optional_dependencies`: a sibling entry there may also be listed
bare, as `"{published_children}"` expands to. The generator refuses any dependency string
left carrying an unexpanded placeholder after expansion. `{version}` is only accepted in
that exact spelling, `"<sibling>[extras]>={version}"`, for a name that normalizes to a
configured sibling package; any other use of `{version}` fails generation.

Naming a nested sibling in an `entry_point_bundle`'s own `dependencies` or a non-dev extra owns that
sibling: the bundle excludes its code from its own wheel, since the sibling's own distribution ships
it instead. Such a requirement must then be spelled exactly `"<sibling>[extras]=={version}"`, with no
environment marker, so the bundle pins the code it excludes to the exact version it releases in
lockstep with, on every platform alike; a bare name, a `">={version}"` floor, or a marker are all
rejected there. This exact-operator spelling is required only for a sibling nested under the bundle's
own path, and only in `dependencies` or a non-dev extra; everywhere else `"=={version}"` stays
rejected like any other hand-written floor.

The generator rejects a config that breaks ownership:

- a published package nested under a bundle that the bundle does not own
- a nested package named in a bundle's `dev` extra, which never ships and so cannot own anything
- an unowned package nested under a package the bundle owns only through an extra, since a bare
  install would ship it without its parent
- a published package that names an unpublished one in its `dependencies` or a non-dev extra: the
  name either fails to resolve or resolves to a stale release whose files a bundle also ships

**Generator infers:**

- `license` from path (`mloda/enterprise/*` → proprietary, else default)
- `packages` from filesystem (scans for `__init__.py`, excludes `tests/`, `build/`, etc.)
- wheel boundaries from the layout: a nested package stays out of its parent's wheel,
  published or not; an `entry_point_bundle` ships all nested code except the own wheel
  packages of each package it owns (named in its own `dependencies` or a non-dev extra),
  so an unowned package nested under one it owns through `dependencies` still ships in the
  bundle wheel

**Default dev deps skipped for:** `mloda-testing`, `mloda-community`, `mloda-enterprise`

## Package hierarchy

### Bundled packages

`mloda-community` and `mloda-enterprise` include all sub-package code directly, except a nested
package they own: named in the bundle's own `dependencies` or a non-dev extra, pinned exactly
`"<name>=={version}"` (all packages release in lockstep at one version). An owned package's own
wheel ships it instead of the bundle's, and its own generated pyproject declares its own entry
points, so each shipped path keeps exactly one published owner (`pip install mloda-community
mloda-community-aggregation` then `pip uninstall mloda-community-aggregation` must never delete
files the bundle still needs). An unpublished nested package ships inside the bundle wheel; a
published one the bundle does not name fails generation. A nested plugin that imports a sibling outside
the bundle's path (`mloda-enterprise-audit` uses `mloda-community-extenders-shared`) needs that
sibling in the bundle's own `dependencies` too, at the ordinary `">={version}"` floor: it is not
nested, so it cannot be owned.

Uninstalling one owned package (`pip uninstall mloda-community-aggregation`) removes only its own
files; the bundle still needs it, so `pip check` then reports the missing dependency. Uninstalling
the bundle itself (`pip uninstall mloda-community`) leaves every package it owns installed, since pip
does not remove a dependency's dependencies; their entry points still register.

```text
mloda-community (bundled)
  └── includes: mloda.community.*
        ├── feature_groups/*        (except example, example-a and data_operations, including its
        │                            plugin leaves: each owned by its own published distribution)
        ├── compute_frameworks/*
        └── extenders/*             (except shared, otel and openlineage: each owned by its own
                                      published distribution)
```

A bundled plugin whose runtime dependency is heavy sits behind a bundle extra instead of a
hard dependency (today `mloda-community[otel]` and `mloda-community[openlineage]`, or both
together via `mloda-community[all]`; also `mloda-enterprise[ed25519]`,
`mloda-enterprise[otel]` and `mloda-enterprise[openlineage]`, though the audit plugin still loads
without the first two). `mloda-community[otel]` and `mloda-community[openlineage]` are also how the
bundle owns `mloda-community-otel` and `mloda-community-openlineage`: each extra pins the leaf
exactly, and the leaf's own `dependencies` carry the third-party pin, so a bare `mloda-community`
install ships neither extender; each installs, and is probed, only through its own distribution or
the matching extra. Every extra member's manifest must import cleanly without that dependency
installed so entry-point loading of the rest of the bundle stays intact. The config steps are in
[Add an optional runtime dependency to a bundle-only plugin](#add-an-optional-runtime-dependency-to-a-bundle-only-plugin).

Moving a dependency behind an extra changes existing installs: when the dependency is missing,
PluginLoader skips the entry point with a WARNING, and discovery never registers the extender.
A direct import of the manifest module now raises instead of degrading.

When the dependency is missing, accessing the extender name on the package (`OtelExtender`,
`OpenLineageExtender`) raises `ModuleNotFoundError` through the package's lazy `__getattr__`, so
`hasattr(pkg, "OtelExtender")` raises rather than returning `False`. Probe with
`importlib.util.find_spec("mloda.community.extenders.otel")` (respectively `...openlineage`) for
whether the extender itself is installed at all, since a bare `mloda-community` install has neither
module regardless of what third-party packages happen to be present; `find_spec("opentelemetry.trace")`
or `find_spec("openlineage.client")` only tells you whether that one dependency is present, not
whether the extender module is.

### Individual packages

Aggregation uses optional dependencies to avoid a circular dependency: the base
does not require its children, the children require the base.

```toml
[packages.mloda-community-example]
description = "Example community FeatureGroup plugin for mloda"
dependencies = ["{core_dependency}"]
path = "mloda/community/feature_groups/example"
published = true
optional_dependencies = { all = ["{published_children}"] }
entry_point_groups = ["mloda.feature_groups"]
py_typed = true
```

| Command | Result |
|---------|--------|
| `pip install mloda-community` | All community plugins (bundled) |
| `pip install mloda-community[otel]` | The bundle plus `mloda-community-otel` (owned by the bundle, so a bare install skips it) |
| `pip install mloda-community[openlineage]` | The bundle plus `mloda-community-openlineage` (owned by the bundle, so a bare install skips it) |
| `pip install mloda-community[all]` | The bundle plus both extenders |
| `pip install mloda-enterprise[ed25519]` | The bundle plus the Ed25519 manifest signer's dependency |
| `pip install mloda-enterprise[otel]` | The bundle plus the OTel audit log sink's dependency |
| `pip install mloda-enterprise[openlineage]` | The bundle plus the OpenLineage emitter the lineage facets extender builds on |
| `pip install mloda-community-example` | Base example only |
| `pip install mloda-community-example[all]` | Base + its published variants |
| `pip install mloda-community-example-a` | Variant A + base |

The base's `all` extra uses `{published_children}`, so it can only name published variants.
`mloda-community-example-b`, being unpublished, ships in the bundle wheel only.

## Entry points

mloda discovers installed plugins through the entry-point groups
`mloda.feature_groups`, `mloda.compute_frameworks`, and `mloda.extenders`. Each
plugin package ships a `manifest.py` listing the package's concrete plugin classes
under a per-group attribute:

| Group | Attribute | Base type |
|-------|-----------|-----------|
| `mloda.feature_groups` | `FEATURE_GROUPS` | `FeatureGroup` |
| `mloda.compute_frameworks` | `COMPUTE_FRAMEWORKS` | `ComputeFramework` |
| `mloda.extenders` | `EXTENDERS` | `Extender` |
| `mloda.optional_dependencies` | `OPTIONAL_DEPENDENCIES` | n/a (tuple of import roots) |

Conventions:

- One `manifest.py` per plugin package. It lists concrete classes only, never the
  shared base class in `base.py` / `*_base.py` (those are non-abstract and would
  wrongly register).
- The generator emits `[project.entry-points."<group>"]` tables whose entry name is
  the distribution label and whose value is the canonical
  `<dotted.package.path>.manifest:<ATTR>` target, e.g.
  `mloda-community-ffill = "mloda.community.feature_groups.data_operations.row_preserving.ffill.manifest:FEATURE_GROUPS"`.
- Bundle packages set `entry_point_bundle = true` and aggregate the entry points of
  every nested plugin package under their path they do not own; an owned package
  declares its own entry points in its own generated pyproject instead (see
  [Bundled packages](#bundled-packages)).
- `mloda.optional_dependencies` is a companion marker group, not a plugin group: it
  declares a package's optional import roots for `PluginLoader` to consult when the
  manifest import fails. Its target is the sibling `_optional_dependencies.py`, not
  `manifest.py`, because the loader reads it only after that import has failed. The
  package `__init__.py` therefore imports nothing from the optional dependency (both
  extender packages re-export their extender lazily through a module `__getattr__`).

## UV workspace sources

The generator adds `mloda-testing = { workspace = true }` only for top-level packages
(depth <= 2) that receive default dev deps, plus one such entry for each sibling in a
top-level package's runtime `dependencies` or extras (uv will not lock without it). Nested
packages cannot use workspace sources due to uv resolution limits; they get dev deps
but rely on root workspace resolution.

A package's own `optional_dependency_indexes` (any depth) instead emits `[tool.uv.sources]`
with `{ index = "<name>" }`, plus a matching `[[tool.uv.index]]` block naming the URL from
`[defaults.uv_indexes]` in `config/shared.toml`. Both are generated into that package's own
`pyproject.toml`, not just the root's: a root-declared index is honored for in-workspace
resolution too, but a standalone (non-workspace) build of just that package needs the index
declared in its own `pyproject.toml` as well, so co-locating it there works in both cases.
Every referenced index must declare `explicit = true`; without it uv's first-index strategy
would let the index shadow PyPI for other packages too, not just the dependency naming it.

## Common workflows

### Bump version

```bash
vim config/shared.toml                  # Change version
python scripts/generate_pyproject.py    # Regenerate
```

### Add a new package

1. Add to `config/packages.toml` (description, dependencies, path; for a plugin
   package also `entry_point_groups = ["mloda.feature_groups" | ...]`), above the
   `# --- Bundles ---` marker and after every published package it depends on:
   `scripts/published_packages.py` rejects a config that is not dependency-first.
2. For a plugin package, create `<path>/manifest.py` listing the concrete classes.
3. If it should ship standalone on PyPI, set `published = true`. Two edits follow it,
   the way `py_typed` also needs its committed marker: the gate test
   `tests/test_end2end/test_published_set_single_source.py` pins the expected set in
   `_EXPECTED_PUBLISHED` (bundle-only packages go into `_BUNDLE_ONLY`), and every
   published distribution needs a smoke import line in the `verify-published` tox env.
   The flag takes effect at the next release. The weekly verification checks out the release
   tag, so only a local run of that env from main fails for it until then.
   If the package is nested under an `entry_point_bundle`'s own path (`mloda-community` or
   `mloda-enterprise`), the bundle must also own it: add `"<name>=={version}"` to the bundle's
   own `dependencies` or a non-dev extra (see [Bundled packages](#bundled-packages)), or
   generation fails.
4. Regenerate and sync:

```bash
python scripts/generate_pyproject.py
uv sync --all-extras --all-packages
```

### Add a test-only dependency

Declare it in the package's `optional_dependencies.dev` in `config/packages.toml`,
regenerate, then run `uv lock` and commit `uv.lock`. tox syncs every workspace
member's `dev` extra (`--all-packages`), so root `pyproject.toml` never repeats
the entry; but it installs the lock with `--frozen`, so a dependency missing
from `uv.lock` is not installed.

### Add an optional runtime dependency to a bundle-only plugin

For a plugin that ships only inside `mloda-community` or `mloda-enterprise` and loads without the
dependency (today `cryptography` behind `mloda-enterprise[ed25519]` and `opentelemetry-api` behind
`mloda-enterprise[otel]`, both used by `mloda-enterprise-audit`):

1. Add the extra to the bundle's `optional_dependencies` in `config/packages.toml`. For
   `mloda-community`, also add it to the `all` extra.
2. Add the same specifier to the leaf's `optional_dependencies.dev`, repeating the default `dev`
   entries (a package's `dev` replaces them). Not its `dependencies`: a bundle install never reads them.
3. Regenerate, run `uv lock` and commit `uv.lock`, as in
   [Add a test-only dependency](#add-a-test-only-dependency).
4. If a test imports the dependency, add its import name to `REQUIRED_TEST_DEPENDENCIES` in
   `tests/test_end2end/test_dev_dependencies.py`.
5. Add the install row to the README and to the install table under
   [Individual packages](#individual-packages).

Keep the floor in the bundle extra and in the leaf `dev` entry equal. `test_bundle_extra_floor_matches_leaf_dev_entry`
enforces that pair.

The dependency can also be a first-party sibling (`mloda-community-openlineage` behind
`mloda-enterprise[openlineage]`). Spell its floor `{version}` in both places
(`test_bundle_extra_sibling_floor_matches_leaf_dev_entry` enforces the pair); the generator adds the
bundle's workspace source. PluginLoader re-raises a missing module whose root equals the entry
point's own root (`mloda`), so such a leaf's manifest must catch the missing sibling itself.

A published community leaf goes behind a bundle extra the same way, spelled `"<leaf>=={version}"`
(as `mloda-community-otel` and `mloda-community-openlineage` do): the extra both owns the leaf (see
[Bundled packages](#bundled-packages)) and gates its third-party dependency.

### Add a variant to an existing plugin

Same as [Add a new package](#add-a-new-package), plus add the variant to the parent's `optional_dependencies.all`,
which only takes a variant with `published = true`.
If that extra is `["{published_children}"]`, do not edit it: set `published = true`
on the variant instead, and the placeholder picks it up.
