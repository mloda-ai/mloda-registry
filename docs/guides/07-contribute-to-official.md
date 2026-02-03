# Contribute to Official Plugins

Improve existing community plugins with bug fixes or new features.

## Steps

1. **Fork and clone** mloda-registry:
   ```bash
   gh repo fork mloda-ai/mloda-registry --clone
   cd mloda-registry
   ```

2. **Create a branch**:
   ```bash
   git checkout -b fix/my-improvement
   ```

3. **Make changes** and add tests

4. **Run tests locally**:
   ```bash
   tox
   ```

5. **Create a Pull Request**

## Guidelines

- Follow existing code style
- Add tests for new functionality
- Update documentation if needed
- Keep changes focused and small

## Code Review Process

### What Happens After You Submit

Once you open a Pull Request:

1. A maintainer will review your PR within **7 days**
2. They may request changes or ask clarifying questions
3. If you haven't received a response after 7 days, feel free to ping the thread

You don't need to open an issue or discuss beforehand—PRs can be submitted directly.

### Acceptance Criteria

Your PR will be merged when:

- **CI passes**: All tests and ruff linting checks must pass
- **Tests included**: New functionality has corresponding tests
- **Docs updated**: If behavior changes, documentation reflects it
- **Style consistent**: Code follows existing patterns in the codebase

### Review Feedback

- Expect constructive feedback aimed at improving code quality
- Changes may be requested before merge—this is normal
- Discussion is encouraged; ask questions if something is unclear

### After Merge

- Your changes will be included in the next release
- You'll be credited in the git history as the author
