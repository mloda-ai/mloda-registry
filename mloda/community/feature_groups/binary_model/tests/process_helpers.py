"""Small process-liveness helpers shared by ``test_binary.py``, ``test_transport.py``, and
``test_mixin.py``. Not a ``test_*`` module, so pytest never collects it on its own.
"""

from __future__ import annotations

import os
import signal
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


def kill_descendant_if_running(pid_file: Path, child_pid: int | None) -> None:
    """SIGKILL the descendant named in ``pid_file`` if it is still running, tolerating an exit race."""
    if child_pid is None and pid_file.exists():
        child_pid = int(pid_file.read_text(encoding="utf-8"))
    if child_pid is not None and pid_running(child_pid):
        try:
            os.kill(child_pid, signal.SIGKILL)
        except OSError:
            pass
