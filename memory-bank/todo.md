# Memory Bank TODOs

## Pending Updates

### mloda py.typed fix
- **Issue:** mloda core package missing `py.typed` marker file
- **Impact:** mypy treats all mloda imports as `Any`, requiring `# type: ignore[misc]` on FeatureGroup subclasses
- **Fix:** When mloda releases with `py.typed`, remove `# type: ignore[misc]` comments from:
  - `mloda/community/feature_groups/example/community_example_feature_group.py`
  - `mloda/enterprise/feature_groups/example/enterprise_example_feature_group.py`
  - Any other plugin FeatureGroup implementations
- **Tracking:** Check mloda releases for py.typed addition
