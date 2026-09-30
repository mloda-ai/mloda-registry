# Create a Plugin Package

Create a standalone, installable plugin package using the official template.

## From Template

The easiest way to start is using the [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template):

```bash
# Create a new repo from the template
gh repo create my-plugin --template mloda-ai/mloda-plugin-template --private
git clone git@github.com:yourname/my-plugin.git
cd my-plugin
```

Or use the "Use this template" button on GitHub.

## Structure

The template provides a ready-to-use structure:

```text
placeholder/
├── feature_groups/
│   ├── manifest.py              # FEATURE_GROUPS entry-point list
│   └── my_plugin/
│       ├── __init__.py
│       ├── my_feature_group.py
│       └── tests/
├── compute_frameworks/
│   ├── manifest.py              # COMPUTE_FRAMEWORKS
│   └── my_plugin/
└── extenders/
    ├── manifest.py              # EXTENDERS
    └── my_plugin/
```

## Package Layout and Import-Time Discovery

mloda resolves a feature against every FeatureGroup subclass loaded in the process, so a FeatureGroup becomes a candidate as soon as the module defining it is imported (`MLODA_PLUGIN_REGISTRY_STRICT=strict` additionally drops groups missing from the explicit plugin registry). Python runs every parent package's `__init__.py` before a submodule, and the template's `my_plugin/__init__.py` imports `MyFeatureGroup`. A helper subpackage under it, such as `my_plugin/core/parsers.py`, therefore loads mloda and `MyFeatureGroup` whenever anything imports the parser.

Keep plain logic (parsers, crosswalks, arithmetic) in a sibling package that imports neither mloda nor a FeatureGroup module:

```text
acme/
├── core/                        # plain logic, no mloda import
│   └── parsers.py
└── feature_groups/
    └── my_plugin/
        ├── __init__.py          # imports MyFeatureGroup
        └── my_feature_group.py  # imports acme.core.parsers
```

To pin what an import loads, see [Testing What an Import Loads](feature-group-patterns/10-testing-guide.md#testing-what-an-import-loads).

## Set Up Your Plugin

1. **Run the customization script**. It renames `placeholder/` to your namespace and rewrites imports, `pyproject.toml` fields and the entry-point paths:
   ```bash
   ./bin/customize.sh acme --author "Your Name" --email you@example.com
   ```

2. **Verify setup**:
   ```bash
   uv venv && source .venv/bin/activate && uv sync --all-extras && tox
   ```

See the [template README](https://github.com/mloda-ai/mloda-plugin-template#setup-your-plugin) for detailed setup instructions.

## Publish Entry Points

`PluginLoader.all()` discovers an installed package only through entry points. The template wires them up: each kind's `manifest.py` lists its concrete classes (`FEATURE_GROUPS`, `COMPUTE_FRAMEWORKS`, `EXTENDERS`), and `pyproject.toml` points one entry per group at it. Add a plugin by appending it to the list:

```python
# acme/feature_groups/manifest.py
from mloda.provider import FeatureGroup

from acme.feature_groups.my_plugin.my_feature_group import MyFeatureGroup

FEATURE_GROUPS: list[type[FeatureGroup]] = [MyFeatureGroup]
```

```toml
[project.entry-points."mloda.feature_groups"]
acme = "acme.feature_groups.manifest:FEATURE_GROUPS"
```

Without the template, add the same pair per group. The entry name is a label only; classes still register under their `module:qualname` key.

### Optional Backends

A manifest that raises `ImportError` is skipped only when the missing module is an optional root; otherwise `PluginLoader.all()` raises. Without a declaration the loader uses core's built-in roots (pandas, polars, duckdb, ...). For any other backend, add a companion `mloda.optional_dependencies` entry (mloda >= 0.13) named like the entry it protects. It replaces the built-in roots for that entry, so also list any built-in root that manifest imports:

```toml
[project.optional-dependencies]
rdf = ["rdflib"]

[project.entry-points."mloda.feature_groups"]
acme = "acme.feature_groups.manifest:FEATURE_GROUPS"
acme-rdf = "acme.feature_groups.rdf_manifest:FEATURE_GROUPS"

[project.entry-points."mloda.optional_dependencies"]
acme-rdf = "acme.feature_groups._optional_dependencies:RDF"
```

```python
# acme/feature_groups/_optional_dependencies.py
RDF: tuple[str, ...] = ("rdflib",)
```

Two rules:

- **One entry point per optional extra.** An `ImportError` skips the whole entry point, so a mixed manifest loses all its groups when `rdflib` is missing. Give each extra its own manifest and entry, and keep the base manifest free of optional imports.
- **The marker must import without the backend.** A marker that fails to import is ignored with a warning and the missing `rdflib` then makes `PluginLoader.all()` raise, so the marker cannot live in the manifest (or a package `__init__.py`) that imports `rdflib`.

See [Plugin Loader: Entry Points](https://mloda-ai.github.io/mloda/in_depth/plugin-loader/#entry-points) for validation, collision, and skip-logging behavior.

## Install Locally

```bash
pip install -e .
```

Entry points are recorded at install time, so reinstall after editing the entry-point tables. `PluginLoader().load_entry_points()` (from `mloda.user`) returns the registered keys, and `PluginLoader.skipped_plugins()` lists entries skipped for a missing backend.

## Next Steps

- [Share with your team](05-share-with-team.md)
- [Publish to community](06-publish-to-community.md)
