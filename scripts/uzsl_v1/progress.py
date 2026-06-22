from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(self, total: int, *, label: str = "progress", width: int = 28) -> None:
        self.total = max(0, total)
        self.label = label
        self.width = width
        self.start = time.monotonic()
        self.last_len = 0

    def update(self, done: int, *, suffix: str = "") -> None:
        done = min(max(done, 0), self.total) if self.total else done
        left = max(self.total - done, 0)
        ratio = done / self.total if self.total else 1.0
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.monotonic() - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = left / rate if rate > 0 else 0.0
        message = (
            f"\r{self.label}: [{bar}] {done}/{self.total} done, "
            f"{left} left, {rate:.2f}/s, eta {format_seconds(eta)}"
        )
        if suffix:
            message += f" | {suffix}"
        padding = " " * max(0, self.last_len - len(message))
        sys.stdout.write(message + padding)
        sys.stdout.flush()
        self.last_len = len(message)

    def finish(self, *, suffix: str = "") -> None:
        self.update(self.total, suffix=suffix)
        sys.stdout.write("\n")
        sys.stdout.flush()


def format_seconds(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"
