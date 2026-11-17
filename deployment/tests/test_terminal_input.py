"""Regression tests for the Windows-safe terminal picker input helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from deployment import terminal_input


def test_windows_arrow_keys_are_normalized(monkeypatch) -> None:
    values = iter(("\xe0", "H"))
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(getwch=lambda: next(values)))

    assert terminal_input.read_key() == "UP"


def test_windows_ctrl_c_raises_keyboard_interrupt(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(getwch=lambda: "\x03"))

    try:
        terminal_input.read_key()
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Ctrl-C must interrupt an interactive picker.")


def test_line_input_maps_j_and_k(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "msvcrt", raising=False)
    monkeypatch.setattr(terminal_input.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _prompt: "j")

    assert terminal_input.read_key() == "DOWN"


def test_picker_modules_import_without_termios(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "termios", None)
    monkeypatch.setitem(sys.modules, "tty", None)

    import deployment.routing_configuration  # noqa: F401
    import deployment.specialist_selection  # noqa: F401
