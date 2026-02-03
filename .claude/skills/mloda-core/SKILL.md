---
name: mloda-core
description: mloda core library source code and documentation. Use when understanding mloda internals, API, or implementation details.
---

# mloda Core Library

**Path detection:**
!`echo ${MLODA_PATH:-$(find .. -maxdepth 2 -name "mloda" -type d ! -name "mloda-registry" 2>/dev/null | head -1)}`

If path is empty, ask user to set `MLODA_PATH` environment variable or clone mloda repo.

**Key locations (relative to mloda path):**
- `src/mloda_core/` - Core implementation
- `docs/` - Documentation source (same as mloda-ai.github.io/mloda/)
- `tests/` - Test examples

**Online fallback:**
- Docs: https://mloda-ai.github.io/mloda/
- Repo: https://github.com/mloda-ai/mloda
