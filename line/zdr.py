"""Request-scoped ZDR helpers for the Line SDK."""

from __future__ import annotations

from contextvars import ContextVar, Token

_zdr_enabled: ContextVar[bool] = ContextVar("line_zdr_enabled", default=False)


def set_zdr_enabled(enabled: bool) -> Token[bool]:
    return _zdr_enabled.set(bool(enabled))


def reset_zdr_enabled(token: Token[bool]) -> None:
    _zdr_enabled.reset(token)


def is_zdr_enabled() -> bool:
    return _zdr_enabled.get()


def safe_error_message(message: str) -> str:
    if is_zdr_enabled():
        return "Internal error"
    return message
