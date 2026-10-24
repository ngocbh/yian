"""Every `composing-agents` skill template must compile via `load_flow` — the templates are
part of the authoring contract, so a template that no longer loads is a bug, not a chore."""

from pathlib import Path

import pytest

from yian.compose.loader import load_flow

_TEMPLATES = Path(__file__).resolve().parents[2] / ".claude/skills/composing-agents/templates"
# Templates written with TS yian's ask, sleep and take built-ins, or its list comprehensions,
# len() and slices, which Python yian doesn't have.
TS_ONLY = {
    "chat.yaml",
    "compact_history.yaml",
    "human-in-loop.yaml",
    "human-questions.yaml",
    "loop.yaml",
    "sleep-until.yaml",
}


@pytest.mark.parametrize("path", sorted(_TEMPLATES.glob("*.yaml")), ids=lambda p: p.name)
def test_template_loads(path):
    if path.name in TS_ONLY:
        pytest.skip("uses TS-only built-ins (#1025)")
    # search_paths = the templates dir so `uses:` siblings resolve.
    load_flow(path.read_text(), search_paths=[path.parent])


def test_llm_config_cascade_template_loads():
    p = _TEMPLATES / "llm-config-cascade.yaml"
    flow = load_flow(p.read_text(), search_paths=[p.parent])
    # The `reviewer` agent shows an `llm_config` field taken from a value.
    assert "__llm_config" in flow.compiled.wiring["reviewer"]
