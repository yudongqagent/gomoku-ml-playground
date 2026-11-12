"""Graceful interrupt handling for long-running jobs.

Self-play and training runs last hours. When the operator (or the
training server, which sends SIGTERM then SIGKILL after 5s on cancel)
interrupts a job, the job should stop promptly, persist what it has,
and say exactly how to resume — instead of dying mid-write.

Usage (in a CLI script, main thread only)::

    flag = InterruptFlag()
    flag.install()
    try:
        train(..., stop_flag=flag)
    finally:
        flag.uninstall()
    if flag.requested:
        print(f"interrupted ({flag.signal_name}) — resumable from ...")

The library loops (``selfplay.generate_games`` / ``train.train``) check
``flag.requested`` at safe points and stop cleanly:

* self-play: the in-progress game is abandoned (a partial game has no
  outcome and cannot be a training sample); every completed game is
  already on disk as a shard, and the data manifest is up to date.
* training: the current batch finishes, then an ``interrupted.pt``
  checkpoint (weights + optimiser + step) is written and the loop
  returns normally.

Signal handling only works in the main thread; installing from a worker
thread raises ValueError (from :mod:`signal`). The flag itself is safe
to check from anywhere.
"""

from __future__ import annotations

import signal


class InterruptFlag:
    """A flag set by SIGINT / SIGTERM handlers installed on this object."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None
        self._previous: dict[int, object] = {}

    def _handler(self, signum: int, _frame) -> None:
        # The first signal wins; a second one still just sets the flag
        # (the process exits at the next safe point, which is seconds
        # away at most — a second signal does not need to force-kill).
        self.requested = True
        self.signal_name = signal.Signals(signum).name

    def install(self) -> "InterruptFlag":
        """Install SIGINT/SIGTERM handlers. Main thread only."""
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handler)
        return self

    def uninstall(self) -> None:
        """Restore the previous handlers (no-op if never installed)."""
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)  # type: ignore[arg-type]
        self._previous.clear()

    def exit_code(self) -> int:
        """Shell-conventional exit code for the received signal
        (128 + signum): 130 for SIGINT, 143 for SIGTERM. 0 if no
        signal was received."""
        if self.signal_name == "SIGINT":
            return 128 + signal.SIGINT
        if self.signal_name == "SIGTERM":
            return 128 + signal.SIGTERM
        return 0

    def __bool__(self) -> bool:
        return self.requested
