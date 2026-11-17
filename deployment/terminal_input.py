"""Small cross-platform keyboard-input helpers for interactive terminal pickers."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator


def _line_input() -> str:
    """Read a portable line-mode picker command when raw input is unavailable."""

    value = input("[k/j move, space toggle, Enter save, q cancel]: ").strip()
    if not value:
        return "\n"
    return {"k": "UP", "j": "DOWN"}.get(value.lower(), value[0])


@contextmanager
def raw_terminal() -> Iterator[None]:
    """Use raw keyboard input where supported, otherwise leave stdin in line mode."""

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        yield
        return

    try:
        import msvcrt
    except ImportError:
        msvcrt = None
    if msvcrt is not None:
        # Enable ANSI escape processing in the Windows console when available.
        os.system("")
        yield
        return

    try:
        import termios
        import tty
    except ImportError:
        yield
        return

    old = termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setraw(sys.stdin.fileno())
        yield
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)


def read_key() -> str:
    """Read one picker key and normalize arrows across Windows and POSIX."""

    try:
        import msvcrt
    except ImportError:
        msvcrt = None
    if msvcrt is not None:
        first = msvcrt.getwch()
        if first == "\x03":
            raise KeyboardInterrupt
        if first in {"\x00", "\xe0"}:
            return {
                "H": "UP",
                "P": "DOWN",
            }.get(msvcrt.getwch(), "")
        return first

    if not sys.stdin.isatty():
        return _line_input()
    first = sys.stdin.read(1)
    if first != "\x1b":
        return first
    second, third = sys.stdin.read(1), sys.stdin.read(1)
    return {"\x1b[A": "UP", "\x1b[B": "DOWN"}.get(first + second + third, "ESC")
