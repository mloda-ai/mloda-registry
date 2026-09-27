"""Small process-liveness helper shared by ``test_transport.py`` and ``test_mixin.py``. Not a
``test_*`` module, so pytest never collects it on its own.
"""

from __future__ import annotations

from pathlib import Path

from mloda.community.feature_groups.binary_model.transport import pid_is_alive


def pid_running(pid: int) -> bool:
    """Alive and not a zombie (a killed process awaiting reaping counts as dead)."""
    if not pid_is_alive(pid):
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return True
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"
