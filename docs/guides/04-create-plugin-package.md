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

mloda resolves a feature against every FeatureGroup subclass loaded in the process, so a FeatureGroup becomes a candidate as soon as the module defining it is imported (registry strict mode `strict`, set by `PluginCollector.set_strict_mode(...)`, else `MLODA_PLUGIN_REGISTRY_STRICT`, additionally drops groups missing from the explicit plugin registry). Python runs every parent package's `__init__.py` before a submodule, and the template's `my_plugin/__init__.py` imports `MyFeatureGroup`. A helper subpackage under it, such as `my_plugin/core/parsers.py`, therefore loads mloda and `MyFeatureGroup` whenever anything imports the parser.

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

```py
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

### Embedded Applications and Strict Mode

Installing entry points does not populate the runtime plugin registry: their classes are registered when `PluginLoader.all()` or `PluginLoader().load_entry_points()` loads them. Selecting feature groups with `PluginCollector.enabled_feature_groups(...)` or supplying extender instances also does not register their classes.

An embedded application that skips the loader should explicitly register its plugin classes needed for resolution before using `strict`. After importing its feature-group and extender classes:

```py
from mloda.user import PluginCollector, register_plugin

register_plugin(MyFeatureGroup)
register_plugin(MyExtender)

collector = PluginCollector.enabled_feature_groups({MyFeatureGroup})
collector.set_strict_mode("strict")
```

Pass this same collector as `plugin_collector` to the run call. `register_plugin` accepts a class and returns its `module:qualname` key; call it directly, not as a decorator. Alternatively, `off` and `warn` both allow unregistered plugins, with `warn` logging a warning. Choose either with `collector.set_strict_mode(...)` or `MLODA_PLUGIN_REGISTRY_STRICT`; the explicit setter takes precedence. See [Create an Extender](11-create-extender.md) for extender-specific strict-mode behavior.

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

### Artifacts Installed from a URL

Some backends need an artifact published only as a URL, such as a spaCy model wheel from Explosion's GitHub releases. It is never a dependency: PyPI rejects uploads whose metadata carries a direct URL requirement, and installing the same name from an index invites dependency confusion. Declare the backend library as an extra per [Optional Backends](#optional-backends), document the artifact's install command pinned to the backend's version, and take its name (or a local path) as configuration:

```bash
M=https://github.com/explosion/spacy-models/releases/download
pip install "acme[ner]" "$M/de_core_news_md-3.8.0/de_core_news_md-3.8.0-py3-none-any.whl"
```

Never list the artifact under `mloda.optional_dependencies` or import it in a manifest: a missing model would then skip the entry point instead of failing closed. Check it inside the call, as [Binary-Backed Features](feature-group-patterns/29-binary-backed-features.md) does for a wheel from an index, and before the backend loads it, since some backends download a missing artifact on load (Presidio's `SpacyNlpEngine` runs `spacy download`):

- **Check** the distribution's metadata (`importlib.metadata`, which `spacy.util.is_package` wraps) or an existing path. This proves the artifact is present, not that it is valid.
- **Fail closed** with a `ValueError` subclass that names the install URL. Wrap the backend's load in the same error, `from exc`, for an installed package that is no model or may not match the backend's version.
- **Keep paths out**: name what the artifact is for (the language), never the configured value, which may be a user path, and raise `from None` where the caught error would echo it. A `from exc` cause keeps its own detail for debugging, so a traceback can still show the path.

```python
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest

MODELS_URL = "https://github.com/explosion/spacy-models/releases"


class ModelNotInstalledError(ValueError):
    pass


def require_installed(lang: str, model: str) -> None:
    """Refuse before the backend loads (and so downloads) a missing model."""
    if model and Path(model).exists():  # Path("") is the working directory
        return
    try:
        distribution(model)
    except (PackageNotFoundError, ValueError):  # ValueError: an empty name
        raise ModelNotInstalledError(
            f"The model for {lang!r} is not installed; install its wheel from {MODELS_URL}."
        ) from None  # the caught error echoes the configured value


def test_missing_model_fails_closed_without_the_path() -> None:
    with pytest.raises(ModelNotInstalledError, match="spacy-models/releases") as info:
        require_installed("de", "/home/alice/models/de_core_news_md")
    assert "alice" not in str(info.value)
    assert info.value.__suppress_context__
```

## Install Locally

```bash
pip install -e .
```

Entry points are recorded at install time, so reinstall after editing the entry-point tables. `PluginLoader().load_entry_points()` (from `mloda.user`) returns the registered keys, and `PluginLoader.skipped_plugins()` lists entries skipped for a missing backend.

## Next Steps

- [Share with your team](05-share-with-team.md)
- [Publish to community](06-publish-to-community.md)
