"""Central logging configuration for GhostFold.

Provides file-based logging (all messages) and console output (warnings/errors only)
so that progress bars remain clean on screen while verbose details go to the log file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_console: Optional[Console] = None
_log_file_path: Optional[Path] = None


def get_console() -> Console:
    """Return the shared Rich Console singleton.

    The same instance is used by both ``RichHandler`` and ``rich.progress.Progress``
    so they coordinate live-display rendering without conflicts.
    """
    global _console
    if _console is None:
        _console = Console(stderr=True)
    return _console


def setup_logging(
    project_dir: str,
    log_filename: str = "ghostfold.log",
    level: int = logging.DEBUG,
) -> Path:
    """Configure the ``ghostfold`` root logger.

    * A :class:`~logging.FileHandler` at *level* writes **all** messages to
      ``<project_dir>/<log_filename>``.
    * A :class:`~rich.logging.RichHandler` at ``WARNING`` renders only
      warnings/errors on the terminal.

    Returns the resolved path to the log file.
    """
    global _log_file_path

    os.makedirs(project_dir, exist_ok=True)
    _log_file_path = Path(project_dir) / log_filename

    file_handler = logging.FileHandler(_log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    console_handler = RichHandler(
        console=get_console(),
        show_time=False,
        show_path=False,
        markup=True,
        level=logging.WARNING,
    )

    root = logging.getLogger("ghostfold")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return _log_file_path


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``ghostfold`` namespace."""
    return logging.getLogger(f"ghostfold.{name}")


def setup_worker_logging(log_file_path: str, level: int = logging.DEBUG) -> None:
    """Configure logging inside a :class:`~concurrent.futures.ProcessPoolExecutor` worker.

    Workers write to the same log file as the parent.  Only a
    :class:`~logging.FileHandler` is attached — no console handler — so workers
    never interfere with the parent's Rich progress display.
    """
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    root = logging.getLogger("ghostfold")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
