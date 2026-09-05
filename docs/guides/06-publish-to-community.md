# Publish to the Community Registry

Submit your plugin to mloda-registry for others to use.

## Prerequisites

- Working plugin with tests (start with [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template))
- README with usage examples

## Steps

1. **Fork** mloda-registry on GitHub

2. **Add your plugin** to the appropriate folder:
   ```
   mloda/community/feature_groups/your_plugin/
   ```

3. **Add to package config** in `config/packages.toml`. A plugin that should also ship as its own PyPI distribution, not only inside the `mloda-community` bundle wheel, needs `published = true`. For a typed plugin, set `py_typed = true` and commit an empty `<path>/py.typed`. A plugin nested under an already-typed path (anything under `data_operations/`) inherits the ancestor marker and needs neither the flag nor its own file. Its dependency on that base is declared as `"<base>>={version}"` in `config/packages.toml`, and the generator expands the placeholder (see [Sibling dependency floors](../packaging.md#sibling-dependency-floors)).

4. **Run tests** to ensure everything works:
   ```bash
   tox
   ```

5. **Create a Pull Request**

## After Merge

Your plugin will be available via:

```bash
pip install mloda-community
```

Users can then import your plugin:

```python
from mloda.community.feature_groups.your_plugin import YourFeatureGroup
```
