"""In-memory OpenLineage transport double: the OpenLineage equivalent of InMemorySpanExporter."""

from __future__ import annotations

from openlineage.client.client import Event
from openlineage.client.event_v2 import RunEvent
from openlineage.client.transport.transport import Config, Transport


class RecordingTransport(Transport):
    """Records every emitted RunEvent in memory instead of sending it anywhere."""

    kind = "recording"
    config_class = Config

    def __init__(self, config: Config | None = None) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: Event) -> None:
        if not isinstance(event, RunEvent):
            raise TypeError(f"RecordingTransport only records RunEvent, got {type(event).__name__}")
        self.events.append(event)
