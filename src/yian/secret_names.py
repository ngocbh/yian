"""Names whose literal value is a credential a flow file must not hold.

The list is deliberately short and documented in ``docs/syntax.md``: exact names
plus a few suffixes, compared after normalizing case, camelCase and ``-``/``_``.
Broad words such as ``key``, ``token`` or ``auth`` are left out because they name
ordinary settings far more often than secrets. This module imports only the
standard library so any layer (compose today, memory later) can use it through
``from yian.secret_names import ...``.
"""

from __future__ import annotations

import re


SECRET_NAMES: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "client_secret",
        "private_key",
        "access_token",
        "refresh_token",
        "api_key",
    }
)
SECRET_SUFFIXES: tuple[str, ...] = ("_api_key", "_password", "_secret")

_ACRONYM_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")


def normalize_name(name: str) -> str:
    """Return ``name`` in lower snake case for comparison with the secret list.

    CamelCase and acronym boundaries become ``_`` and ``-`` becomes ``_``. No other
    character changes, so dots and spaces never act as separators.

    Args:
        name (`str`): A slot, field or key name as written.

    Returns:
        `str`: The normalized name, e.g. ``"HTTPProxyPassword"`` ->
        ``"http_proxy_password"``.
    """
    split = _CAMEL_BOUNDARY.sub(r"\1_\2", _ACRONYM_BOUNDARY.sub(r"\1_\2", name))
    return split.replace("-", "_").lower()


def is_secret_name(name: str) -> bool:
    """Return whether ``name`` clearly names a credential.

    Args:
        name (`str`): A slot, field or key name as written.

    Returns:
        `bool`: ``True`` when the normalized name, ignoring leading and trailing
        underscores, is in `SECRET_NAMES` or ends with one of `SECRET_SUFFIXES`.
    """
    normalized = normalize_name(name)
    return normalized.strip("_") in SECRET_NAMES or normalized.endswith(
        SECRET_SUFFIXES
    )
