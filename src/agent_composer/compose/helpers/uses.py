"""Pure parsing of a `uses:` ref string -> `UsesRef(scheme, path, version)`.

Grammar: `[<scheme>:]<path>[@<version>]`.

- **`<scheme>:`** — a leading `[A-Za-z][A-Za-z0-9+.-]*:` token BEFORE any `/`, stored
  lowercased. `None` = local; `"hub"` = marketplace; anything else is rejected by the
  *resolver* (this module only EXTRACTS the scheme — it does not judge which are known,
  so a Windows-drive-looking `C:/x` parses to scheme `c` and never touches the disk).
- **`<path>`** — the (relative) flow path, possibly nested. Required.
- **`@<version>`** — optional opaque tag, split on the LAST `@` (`a@b@c` -> path `a@b`,
  version `c`); a trailing `@` (empty version) is loud.

Imports only `compose.errors` (lowest layer — no cycle: both `loader` and the resolver
import this).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agent_composer.compose.helpers.errors import LoadError

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")


@dataclass(frozen=True)
class UsesRef:
    scheme: Optional[str]   # None = local; "hub" = marketplace; else rejected at resolve
    path: str               # relative flow path, possibly nested
    version: Optional[str]  # opaque tag, optional


def parse_uses_ref(value: str) -> UsesRef:
    """Parse a `uses:` value into a `UsesRef`, or raise `LoadError` on a malformed ref."""
    raw = (value or "").strip()
    if not raw:
        raise LoadError("uses: ref must be a non-empty string")

    scheme: Optional[str] = None
    rest = raw
    m = _SCHEME.match(raw)
    if m and "/" not in raw[: m.start(1)]:  # a scheme only if its colon precedes any '/'
        scheme = m.group(1).lower()
        rest = raw[m.end():]

    if "@" in rest:
        path, _, version = rest.rpartition("@")
    else:
        path, version = rest, None

    if not path:
        raise LoadError(f"uses: ref {value!r} has no flow path")
    if version == "":
        raise LoadError(f"uses: ref {value!r} has an empty version (trailing '@')")
    return UsesRef(scheme, path, version)
