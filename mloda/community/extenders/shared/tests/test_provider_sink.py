"""Tests for provider_sink.py: configured_provider and drop_unpicklable_provider, the provider resolution and
pickle-drop steps shared by the OTel extenders. Imports no opentelemetry."""

from __future__ import annotations

import logging
import threading
from typing import Any
from unittest.mock import Mock

import pytest
from mloda.steward import WarnOncePerInstance


class _LockHolder:
    """A plain object holding a threading.Lock, which can never survive plain pickling."""

    def __init__(self) -> None:
        self.lock = threading.Lock()


class TestConfiguredProvider:
    def test_injected_wins_without_resolving_the_ambient_one(self) -> None:
        from mloda.community.extenders.shared.provider_sink import configured_provider

        injected, ambient = object(), Mock(return_value=object())

        assert configured_provider(injected, True, ambient) is injected
        ambient.assert_not_called()

    def test_ambient_is_used_under_sdk_defaults(self) -> None:
        from mloda.community.extenders.shared.provider_sink import configured_provider

        resolved = object()
        ambient = Mock(return_value=resolved)

        assert configured_provider(None, True, ambient) is resolved
        ambient.assert_called_once_with()

    def test_none_without_injected_or_sdk_defaults_and_never_resolves_ambient(self) -> None:
        from mloda.community.extenders.shared.provider_sink import configured_provider

        ambient = Mock(return_value=object())

        assert configured_provider(None, False, ambient) is None
        ambient.assert_not_called()

    def test_a_raising_ambient_lookup_propagates_to_the_caller(self) -> None:
        from mloda.community.extenders.shared.provider_sink import configured_provider

        with pytest.raises(RuntimeError, match="resolve boom"):
            configured_provider(None, True, Mock(side_effect=RuntimeError("resolve boom")))


class TestDropUnpicklableProvider:
    _LOG = logging.getLogger("mloda.test.provider_sink")

    def _drop(self, state: dict[str, Any], warning: WarnOncePerInstance | None = None, owner: str = "Owner") -> None:
        from mloda.community.extenders.shared.provider_sink import drop_unpicklable_provider

        drop_unpicklable_provider(
            state, "thing", owner_name=owner, warning=warning or WarnOncePerInstance(), log=self._LOG
        )

    def test_picklable_provider_is_kept_without_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        state: dict[str, Any] = {"_thing": [1, 2, 3]}

        with caplog.at_level(logging.WARNING):
            self._drop(state)

        assert state == {"_thing": [1, 2, 3]}
        assert caplog.records == []

    def test_absent_provider_is_left_alone_without_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        state: dict[str, Any] = {"_thing": None, "other": 1}

        with caplog.at_level(logging.WARNING):
            self._drop(state)

        assert state == {"_thing": None, "other": 1}
        assert caplog.records == []

    def test_unpicklable_provider_is_replaced_by_none_and_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        state: dict[str, Any] = {"_thing": _LockHolder(), "other": 1}

        with caplog.at_level(logging.WARNING):
            self._drop(state, owner="MyOwner")

        assert state["_thing"] is None
        assert state["other"] == 1
        assert [r.getMessage() for r in caplog.records] == [
            "MyOwner drops an injected thing when pickled or copied because it isn't picklable (TypeError); "
            "the copy is inert unless use_sdk_defaults=True, which lets it resolve a provider installed in its "
            "own process, e.g. via child_bootstrap under MULTIPROCESSING."
        ]

    def test_warns_once_per_warning_guard(self, caplog: pytest.LogCaptureFixture) -> None:
        guard = WarnOncePerInstance()

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                state: dict[str, Any] = {"_thing": _LockHolder()}
                self._drop(state, warning=guard)
                assert state["_thing"] is None

        assert len(caplog.records) == 1
