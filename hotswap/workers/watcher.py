"""File watcher for detecting new training data."""

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class DataWatcherHandler(FileSystemEventHandler):
    """Handler for file system events with debouncing."""

    def __init__(
        self,
        callback: Callable[[Path], None],
        patterns: list[str] | None = None,
        debounce_seconds: float = 2.0,
    ):
        self.callback = callback
        self.patterns = patterns or [".pt", ".pth", ".csv"]
        self.debounce_seconds = debounce_seconds
        self._last_event_time: dict[str, float] = {}
        self._lock = threading.Lock()
        self._pending_callbacks: dict[str, threading.Timer] = {}

    def _matches_pattern(self, path: Path) -> bool:
        """Check if path matches watched patterns."""
        return any(path.suffix == pattern for pattern in self.patterns)

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if not self._matches_pattern(path):
            return

        self._schedule_callback(path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if not self._matches_pattern(path):
            return

        self._schedule_callback(path)

    def _schedule_callback(self, path: Path) -> None:
        """Schedule a debounced callback for the file."""
        key = str(path)

        with self._lock:
            # Cancel existing timer if any
            if key in self._pending_callbacks:
                self._pending_callbacks[key].cancel()

            # Schedule new callback
            timer = threading.Timer(
                self.debounce_seconds,
                self._execute_callback,
                args=[path],
            )
            self._pending_callbacks[key] = timer
            timer.start()

            logger.debug(f"Scheduled callback for {path} in {self.debounce_seconds}s")

    def _execute_callback(self, path: Path) -> None:
        """Execute the callback for a file."""
        with self._lock:
            key = str(path)
            if key in self._pending_callbacks:
                del self._pending_callbacks[key]

        logger.info(f"New data detected: {path}")
        try:
            self.callback(path)
        except Exception as e:
            logger.error(f"Callback error for {path}: {e}")


class DataWatcher:
    """Watches a directory for new training data files.

    Uses watchdog to monitor the filesystem and triggers callbacks
    when new data files are detected. Includes debouncing to avoid
    duplicate triggers from partial writes.
    """

    def __init__(
        self,
        watch_path: str | Path,
        on_new_data: Callable[[Path], None],
        patterns: list[str] | None = None,
        debounce_seconds: float = 2.0,
    ):
        self.watch_path = Path(watch_path)
        self.watch_path.mkdir(parents=True, exist_ok=True)

        self._handler = DataWatcherHandler(
            callback=on_new_data,
            patterns=patterns,
            debounce_seconds=debounce_seconds,
        )
        self._observer = Observer()
        self._running = False

    def start(self) -> None:
        """Start watching the directory."""
        if self._running:
            return

        self._observer.schedule(
            self._handler,
            str(self.watch_path),
            recursive=False,
        )
        self._observer.start()
        self._running = True
        logger.info(f"Started watching: {self.watch_path}")

    def stop(self) -> None:
        """Stop watching the directory."""
        if not self._running:
            return

        self._observer.stop()
        self._observer.join(timeout=5.0)
        self._running = False
        logger.info("Stopped watching directory")

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def __enter__(self) -> "DataWatcher":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
