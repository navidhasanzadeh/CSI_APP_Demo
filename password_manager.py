"""Utility helpers for password-protected dialogs."""
from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_PASSWORD = "wirlabwirlab"
PASSWORD_FILE = Path("configs/password.txt")

_PASSWORD_BYPASS = False


def _hash_password(password: str) -> str:
    return hashlib.sha256((password or "").encode("utf-8")).hexdigest()


def set_password_bypass(enabled: bool = True) -> None:
    """Enable or disable bypassing password checks."""

    global _PASSWORD_BYPASS
    _PASSWORD_BYPASS = bool(enabled)


def is_password_required() -> bool:
    """Return False when password checks are bypassed."""

    return not _PASSWORD_BYPASS


def load_password_hash() -> str:
    """Return the stored password hash, creating a default if missing."""

    default_hash = _hash_password(DEFAULT_PASSWORD)
    try:
        PASSWORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return default_hash

    try:
        if PASSWORD_FILE.exists():
            stored = PASSWORD_FILE.read_text(encoding="utf-8").strip()
            if stored:
                return stored
    except OSError:
        pass

    try:
        PASSWORD_FILE.write_text(default_hash, encoding="utf-8")
    except OSError:
        pass
    return default_hash


def save_password(password: str) -> str:
    """Store a new password hash and return it."""

    hashed = _hash_password(password)
    try:
        PASSWORD_FILE.parent.mkdir(parents=True, exist_ok=True)
        PASSWORD_FILE.write_text(hashed, encoding="utf-8")
    except OSError:
        pass
    return hashed


def verify_password(candidate: str) -> bool:
    """Check whether a plaintext password matches the stored hash."""

    if not is_password_required():
        return True
    stored_hash = load_password_hash()
    return stored_hash == _hash_password(candidate)
