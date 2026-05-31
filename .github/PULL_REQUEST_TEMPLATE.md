<!--
Thanks for contributing to mloda-registry! A few notes before you submit:
- Use a Conventional Commit style PR title (e.g. `fix:`, `feat:`, `docs:`, `chore:`).
- Keep the description focused; delete sections that don't apply.
-->

## Summary

<!-- What does this PR change, and why? One or two sentences is plenty. -->

## Related issue

<!-- Link the issue this closes, e.g. "Closes #123". Leave blank if none. -->
Closes #

## Type of change

- [ ] Bug fix (`fix:`)
- [ ] New feature / new plugin (`feat:`)
- [ ] Documentation (`docs:`)
- [ ] Refactor / maintenance (`refactor:` / `chore:`)
- [ ] Other (explain below)

## Checklist

- [ ] `uv run tox` passes locally (tests, `ruff format --check`, `ruff check`, `mypy --strict`, `bandit`)
- [ ] Tests added or updated for the change
- [ ] Documentation updated where relevant
- [ ] `pyproject.toml` not edited by hand (regenerated from `config/` via `scripts/generate_pyproject.py` if dependencies changed)
- [ ] PR title follows [Conventional Commits](https://www.conventionalcommits.org/)
