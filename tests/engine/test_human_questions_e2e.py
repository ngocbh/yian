"""End-to-end pause/resume across the three `human_input` question surfaces.

A `human_input` gate suspends the run with a `HumanInputRequired.questions` list and
resumes when the host delivers a record keyed by each question's `header`. The three
ways the gate sources its questions:

  (A) static     — a literal `questions:` list in the YAML (seed 26).
  (C) manual ref — a `code` node produces the list, fed via `questions: ${qs}`.
  (B) adaptive   — an `adaptive_questions:` block that desugars at load into a synth
                   compose-agent + the gate; the agent authors the list.

Each test runs to the pause, asserts the carried questions, delivers an answer record,
and resumes to a succeeded terminal. The static and manual-ref forms need no LLM; the
adaptive form monkeypatches `model_from_config` with a fake structured-output chat.
"""

from pathlib import Path

import agent_composer.llm_clients as llm_clients_mod
from agent_composer.compose import load_flow
from agent_composer.run import resume_command, resume_flow, run_flow
from agent_composer.suspension.pause import HumanInputRequired


def _human_input_reason(result):
    """The single `HumanInputRequired` pause reason on a paused `RunResult`."""
    reasons = [r for r in result.pause_reasons if isinstance(r, HumanInputRequired)]
    assert len(reasons) == 1, f"expected one human-input pause, got {result.pause_reasons}"
    return reasons[0]


# --- (A) static: a literal questions list in the seed --------------------------------- #


def test_static_questions_pause_resume():
    """Seed 26: a static `questions:` list templates `${proj}` into a question, pauses,
    and the delivered header-keyed record routes the terminal output."""
    loaded = load_flow(Path("tests/seeds/26-human-questions.yaml").read_text())
    result = run_flow(loaded, {"proj": "Atlas"})

    assert result.status == "paused", result.error
    reason = _human_input_reason(result)
    assert reason.questions, "the gate must carry the static questions"
    headers = [q["header"] for q in reason.questions]
    assert headers == ["Framework", "Notes"]
    framework_q = next(q for q in reason.questions if q["header"] == "Framework")
    assert "Atlas" in framework_q["question"]  # ${proj} rendered

    # Deliver a record keyed by every question header (the gate's bare-object answer).
    answer = {"Framework": "React", "Notes": "ship it"}
    cmd = resume_command(loaded, reason, answer)
    final = resume_flow(loaded, engine=result.engine, commands=[cmd])

    assert final.status == "succeeded", final.error
    assert final.output == "React"  # routed on the delivered label


# --- (C) manual ref: a code node produces the questions list -------------------------- #

_MANUAL_REF = """
id: manual_ref
name: manual_ref
input:
  seed: str
nodes:
  src:
    kind: code
    input: {seed: "${input.seed}"}
    output: list[object]
    code: tests.seeds.fns:questions_seed
  ask:
    kind: human_input
    input: {qs: "${src.output}"}
    questions: "${qs}"
output: ${ask.output}
"""


def test_manual_ref_questions_pause_resume():
    """A `code` node builds the questions list; the gate reads it via `questions: ${qs}`,
    pauses with the code-produced question, and resumes on the delivered record."""
    loaded = load_flow(_MANUAL_REF)
    result = run_flow(loaded, {"seed": "Which option?"})

    assert result.status == "paused", result.error
    reason = _human_input_reason(result)
    assert len(reason.questions) == 1
    q = reason.questions[0]
    assert q["question"] == "Which option?"  # the seed input flowed through the code node
    assert q["header"] == "H"

    cmd = resume_command(loaded, reason, {"H": "A"})
    final = resume_flow(loaded, engine=result.engine, commands=[cmd])

    assert final.status == "succeeded", final.error
    assert final.output == {"H": "A"}  # the gate emits the bare answer record


# --- (B) adaptive: an LLM composes the questions, desugared into agent + gate ---------- #

_ADAPTIVE = """
id: adaptive
name: adaptive
input:
  ctx: str
nodes:
  ask:
    kind: human_input
    input: {ctx: "${input.ctx}"}
    prompt: "Help me decide on ${ctx}."
    adaptive_questions:
      prompt: "Design 1-3 questions with options for: ${ctx}"
output: ${ask.output}
"""

# The questions the fake compose-agent "authors" — one single-select choice.
_COMPOSED = [
    {
        "question": "Which database?",
        "header": "DB",
        "options": [
            {"label": "Postgres", "description": "Relational."},
            {"label": "SQLite", "description": "Embedded."},
        ],
        "multi_select": False,
    }
]


class _ComposeChat:
    """A fake chat for the synth compose-agent's `plain`/native structured path.

    The synth agent declares a top-level `list[Question]` output, which
    `type_to_schema` wraps in a single-field `ListWrapper` model (field `items`).
    `plain` mode calls `with_structured_output(schema).invoke(...)`, so the bound
    object must return `schema.model_validate({"items": [...]})` — a ListWrapper, NOT
    a bare list. `bind_tools`/`invoke` cover the plain-text path if it is ever reached.
    """

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):  # plain-text path (unused on the native structured route)
        raise AssertionError("structured path expected, not a plain-text invoke")

    def with_structured_output(self, schema):
        class _Bound:
            def invoke(self, messages):
                return schema.model_validate({"items": _COMPOSED})

        return _Bound()


def test_adaptive_questions_pause_resume(monkeypatch):
    """An `adaptive_questions:` block desugars into a compose-agent + gate. The agent
    composes the questions list (via the fake structured chat), the gate pauses carrying
    exactly those questions, and the delivered record resumes to a succeeded terminal."""
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: _ComposeChat())

    loaded = load_flow(_ADAPTIVE)
    result = run_flow(loaded, {"ctx": "storage"})

    assert result.status == "paused", result.error
    reason = _human_input_reason(result)
    # The gate pauses with EXACTLY the agent-composed questions.
    assert [q["header"] for q in reason.questions] == ["DB"]
    assert reason.questions[0]["question"] == "Which database?"
    assert [o["label"] for o in reason.questions[0]["options"]] == ["Postgres", "SQLite"]

    cmd = resume_command(loaded, reason, {"DB": "Postgres"})
    final = resume_flow(loaded, engine=result.engine, commands=[cmd])

    assert final.status == "succeeded", final.error
    assert final.output == {"DB": "Postgres"}
