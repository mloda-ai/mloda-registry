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
│   └── my_plugin/
│       ├── __init__.py
│       ├── my_feature_group.py
│       └── tests/
├── compute_frameworks/
│   └── my_framework/
└── extenders/
    └── my_extender/
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

1. **Rename the namespace** to your organization:
   ```bash
   mv placeholder acme
   ```

2. **Update `pyproject.toml`**:
   - `name`: Change `"placeholder-my-plugin"` to `"acme-my-plugin"`
   - `authors`: Your name and email
   - `tool.setuptools.packages.find.include`: Change to `["acme*"]`

3. **Update imports** in Python files from `placeholder.` to `acme.`

4. **Verify setup**:
   ```bash
   uv venv && source .venv/bin/activate && uv pip install -e ".[dev]" && tox
   ```

See the [template README](https://github.com/mloda-ai/mloda-plugin-template#setup-your-plugin) for detailed setup instructions.

## Install Locally

```bash
pip install -e .
```

## Next Steps

- [Share with your team](05-share-with-team.md)
- [Publish to community](06-publish-to-community.md)
