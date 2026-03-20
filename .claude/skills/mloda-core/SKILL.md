---
name: mloda-core
description: mloda core library source code and documentation. Use when understanding mloda internals, API, or implementation details.
---

# mloda Core Library

**Path detection:**
!`echo ${MLODA_PATH:-$(find .. -maxdepth 2 -name "mloda" -type d ! -name "mloda-registry" 2>/dev/null | head -1)}`

If path is empty, ask user to set `MLODA_PATH` environment variable or clone mloda repo.

**Key locations (relative to mloda path):**

| Path | Purpose |
|------|---------|
| `mloda/` | Main source package |
| `mloda/core/` | Core functionality (abstract plugins, API, filter, prepare, runtime) |
| `mloda/provider/` | Data provider interfaces |
| `mloda/steward/` | Data governance/stewardship |
| `mloda/user/` | User-facing functionality |
| `mloda_plugins/` | Built-in plugin implementations |
| `mloda_plugins/compute_framework/` | Compute framework plugins |
| `mloda_plugins/feature_group/` | Feature group plugins |
| `mloda_plugins/function_extender/` | Function extender plugins |
| `mloda_plugins/config/` | Plugin configuration |
| `docs/docs/` | Documentation source (MkDocs, same as mloda-ai.github.io/mloda/) |
| `docs/mkdocs.yml` | MkDocs configuration |
| `tests/` | Test suite (1500+ tests) |
| `memory-bank/` | Project context and architecture docs |

**Online fallback:**
- Docs: https://mloda-ai.github.io/mloda/
- Repo: https://github.com/mloda-ai/mloda
