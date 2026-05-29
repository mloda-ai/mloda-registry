# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Keep in sync:** `AGENTS.md` is a 1:1 copy of this file. When you change one, update the other so tool-agnostic agents and Claude see identical guidance.

**Core mloda project:** https://github.com/mloda-ai/mloda | **Docs:** https://mloda-ai.github.io/mloda/

**Skills available:**
- `/mloda-plugins` - Plugin development guides with decision trees for FeatureGroups, ComputeFrameworks, and Extenders
- `/mloda-core` - Core library source code and docs (uses `MLODA_PATH` env var or online fallback)

**IMPORTANT:** See [Plugin Development Guides](#plugin-development-guides) - these are practical how-to guides for development, whereas the docs describe mloda conceptually. Read `docs/guides/` frequently when working on plugin code.

## TDD Orchestrator Role

**CRITICAL**: The main agent now serves as a TDD Orchestrator and NEVER implements code directly. Instead:

- **Orchestration Only**: Coordinate Test-Driven Development cycles between specialized agents
- **No Code Implementation**: NEVER write implementation code or tests directly
- **Agent Delegation**: Use Red Agent for test writing, Green Agent for implementation

## TDD Workflow

1. **Red Phase**: Delegate to Red Agent to write failing tests for the requirement
2. **Validation**: Verify tests fail for the right reasons
3. **Green Phase**: Delegate to Green Agent for minimal implementation
4. **Validation**: Ensure tests pass and no regressions
5. **Repeat**: Continue cycle for next requirement

## CFW Backend Rejection over Python Fallback

When a compute framework backend cannot natively support an input or operation, reject it up-front with a clear `ValueError` rather than computing the result in Python inside the backend module. Both Red and Green agents must follow this rule (see `.claude/agents/green-agent.md` "CFW Backend Rules" for the full statement, rationale, and precedents; the Red agent writes a rejection test, not a fallback test). Reviewers should enforce it as well.

## Deadlock Protection

**CRITICAL**: If Red or Green agents get stuck or fail repeatedly:

1. **Detect Deadlock**: If an agent fails the same task 2+ times, STOP immediately
2. **Do NOT Loop**: Never retry the same failing operation more than twice
3. **Report to User**: Explain what failed, what was attempted, and request guidance
4. **User Decision**: Let the user decide whether to:
   - Modify the approach
   - Update agent instructions
   - Manually intervene
   - Skip the problematic step

**Never continue TDD cycles if agents are stuck** - this wastes resources and indicates a fundamental issue that requires human intervention.

## Avoid Duplication

- Before creating any new code, search the codebase for existing implementations that solve the same or a similar problem.
- If similar logic already exists, reuse it: extend, parameterise, or refactor the existing code rather than duplicating it.
- Never copy-paste blocks of code across files. If the same logic is needed in multiple places, extract it into a shared module, function, or utility.

## Follow Existing Conventions

- Before writing or modifying code (including tests), read surrounding files to understand the existing conventions (naming, formatting, structure, comments, indentation).
- Match the style of the repository by default. Only introduce new patterns, naming conventions, or structural choices when explicitly requested or when the feature genuinely requires it.
- If the repo uses specific file naming schemes (e.g. numbered prefixes, grouped by concern), follow the same scheme.
- For tests specifically, follow the existing test structure, assertion style, naming patterns, and strategies (e.g. mocking, stubbing, fixtures, test data setup) rather than introducing new ones.

## Git Commits

- No agent should add `Co-Authored-By` lines or any other commit authorship attribution.

## Project Practices

`tox` is the gate. It runs `pytest -n {env:PYTEST_WORKERS:2}` (default 2 workers, no timeout), then `ruff format --check --line-length 120 .`, `ruff check .`, `mypy --strict --ignore-missing-imports .`, and `bandit -c pyproject.toml -r -q .`. All of these must pass before a PR is mergeable.

- **Python**: supported range is `>=3.10`; tox envs cover `python310`, `python311`, `python312`, `python313`, `python314`.
- **Type hints**: use modern forms (`list[str]`, `dict[str, int]`, `X | None`). Ruff enforces this via `UP006` and `UP007` (extend-selected in `pyproject.toml`).
- **Formatting**: ruff format with line length 120.
- **Tests**: parallel-safe (pytest-xdist). Per-package envs are available for isolated runs: `tox -e testing`, `tox -e community-example`, `tox -e registry`, `tox -e enterprise-example`.
- **Supply chain**: `[tool.uv] exclude-newer = "7 days"` in `pyproject.toml` defers new dependency releases by 7 days. Do not edit this without a reason.
- **Auto-generated `pyproject.toml`**: edit `config/shared.toml` (version, authors, urls) or `config/packages.toml` (per-package), then run `python scripts/generate_pyproject.py`. Never edit `pyproject.toml` files directly. See `memory-bank/pyproject-generation.md`.
- **Commits**: use [Conventional Commits](https://www.conventionalcommits.org/) (`fix:`, `feat:`, `chore:`, `docs:`, `test:`, `refactor:`, `minor:`, `perf:`, `impr:`, `ci:`, `style:`, `build:`). semantic-release computes the next version. This project deviates from the standard: only `minor:` commits bump the minor version; `feat:` is treated as a patch bump along with everything else (see `.releaserc.yaml`).

## Issue Creation

When filing a GitHub issue (via `gh issue create` or otherwise), follow the structure in `.github/ISSUE_TEMPLATE/issue.yml`:

- Summary in one sentence
- Reproduction (for bugs) or motivation (for features)
- Code pointers if relevant (`file:line`)
- Definition of done if scoped (what counts as complete)

Issues that meet this bar are eligible for the `good first issue` label without further sharpening.

## Virtual Environment Setup

If not in a devcontainer, set up the environment:

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment
uv venv

# 3. Activate the environment
source .venv/bin/activate

# 4. Install dependencies
uv sync --all-extras
```

For subsequent sessions, just activate and sync:

```bash
source .venv/bin/activate
uv sync --all-extras
```

## Devcontainer Tox Usage

- Use `tox` when running tests without dependency changes (uses pre-built environment)
- Use `uv run tox` when dependencies have changed (rebuilds environment with new deps)

## Package Management

All `pyproject.toml` files are **auto-generated** from config files.

**Documentation (read on-demand):**
- `memory-bank/pyproject-generation.md` - Read when modifying config files or troubleshooting generation
- `memory-bank/package-hierarchy.md` - Read when understanding bundled vs individual package structure
- `memory-bank/release-pypi.md` - Read when preparing releases or debugging PyPI publishing

**Quick reference:**
- Edit `config/shared.toml` for version/authors/urls
- Edit `config/packages.toml` for per-package config
- Run `python scripts/generate_pyproject.py` to regenerate
- **Never edit pyproject.toml files directly**

## Plugin Development Guides

`docs/guides/` contains step-by-step guides for mloda plugin development. Use `/mloda-plugins` skill for decision trees and full guide index.
