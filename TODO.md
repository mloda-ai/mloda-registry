# mloda-registry Implementation TODO

Step-by-step implementation guide for building the mloda registry ecosystem.

**Full Guide:** [IMPLEMENTATION_GUIDE.md](/home/tom/project/marketing-assets/pivot/analysis/registry/IMPLEMENTATION_GUIDE.md)

---

## mloda Core Status

**mloda v0.4.2** released with full PEP 420 namespace package support:
- [x] `namespaces = true` in pyproject.toml
- [x] `namespace_packages = true` in mypy config
- [x] Deleted `mloda/__init__.py` (namespace root has no __init__)
- [x] Added `mloda` alias in `mloda/user/__init__.py`:
  ```python
  from mloda.core.api.request import mlodaAPI
  mloda = mlodaAPI
  ```

**New import pattern:**
```python
from mloda.user import mloda, Feature, Options  # or mlodaAPI
result = mloda.run_all(["feature_a", "feature_b"])
```

**Reference:** [09_namespace_migration.md](/home/tom/project/marketing-assets/pivot/analysis/registry/09_namespace_migration.md)

---

## Quick Start MVP

> Minimum viable registry to prove the architecture works.

- [x] Create namespace package structure (`mloda/`)
- [x] Implement `mloda-testing` package
- [x] Create example plugin (`mloda-community-example`)
- [x] Implement basic `mloda.registry.discover()`

---

## Phase 1: Repository Setup

**Reference:** [00_repositories.md](/home/tom/project/marketing-assets/pivot/analysis/registry/00_repositories.md)

### 1.1 mloda-registry Monorepo Structure

- [x] Set up PEP 420 namespace package structure
  ```
  mloda/                    # No __init__.py (namespace root)
  ├── community/
  ├── enterprise/
  ├── registry/
  └── testing/
  ```
- [x] Configure pytest for `mloda/**/tests`
- [ ] Update CI/CD workflows for namespace packages

### 1.2 mloda-template Repo

- [ ] Create GitHub template repo `mloda-ai/mloda-template`
- [ ] Add placeholder structure for private plugins
- [ ] Document rename process in README

---

## Phase 2: Package Specifications

### 2.1 pyproject.toml Standards

**Reference:** [02_pyproject_spec.md](/home/tom/project/marketing-assets/pivot/analysis/registry/02_pyproject_spec.md)

- [ ] Define pyproject.toml template for plugins
- [x] Set up meta-packages:
  - [x] `mloda-community` (depends on all community plugins)
  - [x] `mloda-enterprise` (depends on all enterprise plugins)
- [x] Configure individual plugin package structure

### 2.2 Access Model

**Reference:** [04_access_model.md](/home/tom/project/marketing-assets/pivot/analysis/registry/04_access_model.md)

- [ ] Implement tier-based access (community/enterprise)
- [ ] Design license key validation for enterprise plugins

---

## Phase 3: Core Packages

### 3.1 mloda-registry Package

**Reference:** [00_repositories.md](/home/tom/project/marketing-assets/pivot/analysis/registry/00_repositories.md)

```python
from mloda.registry import discover, search
```

- [x] Implement `discover()` - list available plugins (stub)
- [x] Implement `search()` - find plugins by criteria (stub)
- [ ] Add CLI commands for discovery

### 3.2 mloda-testing Package

```python
from mloda.testing import FeatureGroupTestBase
```

- [x] Create `FeatureGroupTestBase` class
- [ ] Add test utilities and fixtures
- [ ] Document usage patterns

---

## Phase 4: First Plugins

### 4.1 Plugin Template

**Reference:** [06_plugin_template.md](/home/tom/project/marketing-assets/pivot/analysis/registry/06_plugin_template.md)

- [ ] Create cookiecutter or copier template
- [ ] Include tests, CI, and documentation stubs

### 4.2 Example Community Plugin

- [ ] Create `mloda-community-timeseries` as reference implementation
- [ ] Include compute framework variants:
  - [ ] Pandas variant
  - [ ] PyArrow variant
- [ ] Full test coverage

### 4.3 Example Enterprise Plugin

**Reference:** [08_enterprise_plugin_example.md](/home/tom/project/marketing-assets/pivot/analysis/registry/08_enterprise_plugin_example.md)

- [ ] Create `mloda-enterprise-forecasting` as reference
- [ ] Implement license validation

---

## Phase 5: Publishing & Distribution

### 5.1 Publishing Guide

**Reference:** [05_publishing_guide.md](/home/tom/project/marketing-assets/pivot/analysis/registry/05_publishing_guide.md)

- [ ] Document PyPI publishing process
- [ ] Set up automated releases via CI

### 5.2 Enterprise Checkout

**Reference:** [07_fakecheckout.md](/home/tom/project/marketing-assets/pivot/analysis/registry/07_fakecheckout.md)

- [ ] Design checkout flow for enterprise plugins
- [ ] Implement license key delivery mechanism

---

## Phase 6: User Experiences

Validate each user journey works end-to-end.

| # | User Experience | Status | Reference |
|---|-----------------|--------|-----------|
| 1 | Use existing plugin | [ ] | [01_use_existing_plugin.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/01_use_existing_plugin.md) |
| 2a | Create plugin for myself | [ ] | [02_create_plugin_for_myself.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/02_create_plugin_for_myself.md) |
| 2b | Create plugin inline | [ ] | [02_b_create_plugin_inline.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/02_b_create_plugin_inline.md) |
| 3 | Share plugin with team | [ ] | [03_share_plugin_with_team.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/03_share_plugin_with_team.md) |
| 4 | Publish to community | [ ] | [04_publish_plugin_to_community.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/04_publish_plugin_to_community.md) |
| 5 | Contribute to official | [ ] | [05_contribute_to_official_plugin.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/05_contribute_to_official_plugin.md) |
| 6 | Become official plugin | [ ] | [06_become_official_plugin.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/06_become_official_plugin.md) |
| 7a | Discover via registry | [ ] | [07a_discover_via_registry.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/07a_discover_via_registry.md) |
| 7b | Discover loaded plugins | [ ] | [07b_discover_loaded_plugins.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/07b_discover_loaded_plugins.md) |
| 8 | Create compute framework | [ ] | [08_create_compute_framework_plugin.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/08_create_compute_framework_plugin.md) |
| 9 | Create extender | [ ] | [09_create_extender_plugin.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/09_create_extender_plugin.md) |
| 10 | Debug/troubleshoot | [ ] | [10_debug_troubleshoot_plugin.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/10_debug_troubleshoot_plugin.md) |
| 11 | Fork for enterprise | [ ] | [11_fork_registry_for_enterprise.md](/home/tom/project/marketing-assets/pivot/analysis/registry/user_experience/11_fork_registry_for_enterprise.md) |

---

## Additional References

| Document | Path |
|----------|------|
| Role Terminology | [01_role_terminology.md](/home/tom/project/marketing-assets/pivot/analysis/registry/01_role_terminology.md) |
| Pivot Learnings | [pivot_learnings.md](/home/tom/project/marketing-assets/pivot/analysis/learnings/pivot_learnings.md) |
| Handover Layer Positioning | [handover_layer_pivot.md](/home/tom/project/marketing-assets/pivot/handover_layer_pivot.md) |
