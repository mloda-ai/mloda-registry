# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read the memory-bank prior.

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
