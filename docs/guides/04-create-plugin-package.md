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

> **Re-scanning after install.** `PluginLoader.all()` caches the plugin set on
> first use. If you install (or reinstall) this package into a Python process
> that has already called `all()`, pass `PluginLoader.all(force_reload=True)`
> (or call `PluginLoader.reset_cache()` first) so its entry points are picked up.
> A fresh process needs no flag.

## Next Steps

- [Share with your team](05-share-with-team.md)
- [Publish to community](06-publish-to-community.md)
