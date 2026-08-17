"""CODE-node callables referenced by the seed gallery.

Each function is a `module:function` target for a `kind: code` node: it receives
the node's **bound input record** (one dict) and returns the node's **one output
value** — a scalar, list, or object matching the seed's declared `output:`
shape. These are deterministic placeholders (no live data/IO) so the gallery
flows are runnable end-to-end; swap in real implementations as needed.
"""

from __future__ import annotations

from typing import Any, Dict, List


# --- seed 01 (structured-agent) -------------------------------------------- #
def one_line_summary(rec: Dict[str, Any]) -> str:
    """`{rating: float, rationale: str}` -> a one-line verdict string."""
    rating = rec.get("rating")
    rationale = rec.get("rationale", "")
    lean = "positive" if (rating or 0) >= 0 else "negative"
    return f"{lean} ({rating:+.2f}): {rationale}".strip()


# --- seed 03 (research-one) ------------------------------------------------ #
def fetch_facts(rec: Dict[str, Any]) -> Dict[str, Any]:
    """`{topic, as_of}` -> `{values: list[float], news: list[str]}`."""
    topic = str(rec.get("topic", "")).upper()
    return {
        "values": [100.0, 101.5, 99.75, 102.25],
        "news": [f"{topic}: quarterly update", f"{topic}: reviewer note"],
    }


# --- seed 13 (types-objects) ----------------------------------------------- #
def build_outline(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Compose a nested plan matching seed 13's declared output."""
    names: List[str] = list(rec.get("names") or [])
    quantity = rec.get("quantity") or 0
    items = [f"item {n}" for n in names]
    total = float(quantity)
    return {
        "items": items,
        "summary": {
            "count": len(items),
            "total": total,
            "meta": {"as_of": rec.get("as_of"), "dry_run": bool(rec.get("dry_run", False))},
        },
    }


# --- seed 12 (depends-on) -------------------------------------------------- #
def prime_cache(rec: Dict[str, Any]) -> bool:
    """Side-effect node: warm a cache, return a `bool` nobody binds."""
    return True


# --- seed 17 (effects-human-wait) ------------------------------------------ #
def confirm_action(rec: Dict[str, Any]) -> str:
    """Echo the confirmed action."""
    return str(rec.get("answer", "cancel"))


# --- errors/ negative-gallery CODE callables (deterministic, no IO) --------- #
def const_one(rec: Dict[str, Any]) -> int:
    """Return the integer 1 (a deterministic numeric producer for assert fixtures)."""
    return 1


def fail_always(rec: Dict[str, Any]):
    """Raise — a CODE node that always fails (runtime-failure fixture)."""
    raise RuntimeError("intentional CODE failure")


def wrong_type(rec: Dict[str, Any]) -> str:
    """Return a str even where the node declares `int` — trips the write-boundary check."""
    return "not-an-int"


def echo_value(rec: Dict[str, Any]) -> Any:
    """Return the single bound input value (a passthrough leaf)."""
    return next(iter(rec.values()), None)


# --- human_input questions ref source -------------------------------------- #
def questions_seed(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """`{seed}` -> a one-element `list[object]` of question records (a gate's `questions:` ref)."""
    return [{"question": inputs["seed"], "header": "H",
             "options": [{"label": "A"}, {"label": "B"}]}]


# --- seed 26 (human-questions) — route on the keyed answer record ----------- #
def chosen_framework(inputs: Dict[str, Any]) -> str:
    """`{ans: {Framework, Notes}}` -> the chosen framework label (routes on the answer)."""
    return str(inputs["ans"]["Framework"])


# --- seed 30 (reference-code) — the reference twin of seed 29's inline body -- #
def verdict_line(inputs: Dict[str, Any]) -> str:
    """`{rating: float, label: str}` -> a verdict string.

    The `module:function` twin of seed 29's inline body — same code, so the two
    seeds show one computation both ways (an inline body promotes to a reference by
    copy-paste).
    """
    lean = "positive" if inputs["rating"] >= 0 else "negative"
    return f"{inputs['label']}: {lean} ({inputs['rating']:+.2f})"
