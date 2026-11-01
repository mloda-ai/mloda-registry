# Contribute to Official Plugins

Improve existing community plugins with bug fixes or new features.

> For general PR mechanics, code style, and the full review process, see [CONTRIBUTING.md](../../CONTRIBUTING.md). This guide focuses on the workflow specific to contributing to official plugins.

## Code Review Process

### What Happens After You Submit

Once you open a Pull Request:

1. A maintainer will review your PR within **7 days**
2. They may request changes or ask clarifying questions
3. If you haven't received a response after 7 days, feel free to ping the thread

You don't need to open an issue or discuss beforehand. PRs can be submitted directly.

### Acceptance Criteria

Your PR will be merged when:

- **CI passes**: `pytest`, `ruff format --check`, `ruff check`, `mypy --strict --ignore-missing-imports` and `bandit` must all pass (see [Code Style](../../CONTRIBUTING.md#code-style))
- **Tests included**: New functionality has corresponding tests
- **Docs updated**: If behavior changes, documentation reflects it
- **Style consistent**: Code follows existing patterns in the codebase
