"""mloda-community-extenders-shared: Pickle-safety, open-invocation and teardown helpers shared by mloda extenders, plus an opt-in SIGTERM handler."""

from mloda.steward import is_picklable

__all__ = ["is_picklable"]
