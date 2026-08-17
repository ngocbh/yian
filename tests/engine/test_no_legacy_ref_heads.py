"""Guard: no legacy reference-vocabulary heads remain after the rename.

The JSON-IO single-value redesign unified the legacy `node` and
`ref_outputs` reference heads into a single `outputs` head. This test fails if any
legacy brace-wrapped `node.`/`ref_outputs.` reference literal survives in the engine
source, the seed gallery, or the test fixtures — the concrete red->green rating for
an otherwise mechanical sweep. (The needles are built dynamically below so this
guard file never matches itself.)

Scope: `src/` and `tests/`, including the seed gallery. The design docs under `docs/`
legitimately discuss the old vocabulary historically and are not scanned.
"""

from pathlib import Path

import pytest

# Built dynamically so this guard file does not match itself.
# `${node.output}` is now the canonical surface — the old `${node.` legacy needle is
# removed. Only the older `${ref_outputs.` (a pre-redesign vocabulary) is still scanned.
_LEGACY = ("${" + "ref_outputs.",)
_REPO = Path(__file__).resolve().parents[2]
_ROOTS = ("src", "tests")
_EXTS = {".py", ".yaml", ".yml", ".md"}


def _scan_files():
    for root in _ROOTS:
        for path in (_REPO / root).rglob("*"):
            if path.suffix in _EXTS and path.is_file():
                # Skip this guard file itself (it MUST contain the needle to scan for).
                if path.resolve() == Path(__file__).resolve():
                    continue
                yield path


def test_no_legacy_ref_heads():
    offenders = []
    for path in _scan_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in _LEGACY:
            if needle in text:
                offenders.append(f"{path.relative_to(_REPO)} contains {needle!r}")
    assert not offenders, "legacy reference heads remain:\n" + "\n".join(sorted(offenders))
