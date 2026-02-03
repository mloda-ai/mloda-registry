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

3. **Add to package config** in `config/packages.toml`

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
