"""End-to-end blocking answers across the three `human_input` question surfaces.

A `human_input` gate calls the installed host with a validated ``questions`` list
and returns the handler's record keyed by each question's ``header``. The three ways
the gate sources its questions are:

  (A) static     — a literal `questions:` list in the YAML (seed 26).
  (C) manual ref — a `code` node produces the list, fed via `questions: ${qs}`.
  (B) adaptive   — an `adaptive_questions:` block that desugars at load into a synth
                   compose-agent + the gate; the agent authors the list.

Each test asserts the trusted host payload and the succeeded terminal. The static and
manual-ref forms need no LLM; the
adaptive form monkeypatches `model_from_config` with a fake structured-output chat.
"""

from pathlib import Path

import agent_composer.llm_clients as llm_clients_mod
from agent_composer.compose import load_flow
from agent_composer.run import run_flow
from agent_composer.tools import install_ask_user


# --- (A) static: a literal questions list in the seed --------------------------------- #


def test_static_questions_block_and_return_answer():
    """Seed 26 renders its static questions and routes the host answer."""
    loaded = load_flow(Path("tests/seeds/26-human-questions.yaml").read_text())
    requests = []
    answer = {"Framework": "React", "Notes": "ship it"}
    with install_ask_user(lambda **request: requests.append(request) or answer):
        result = run_flow(loaded, {"proj": "Atlas"})

    assert result.status == "succeeded", result.error
    questions = requests[0]["questions"]
    assert questions, "the gate must pass the static questions"
    headers = [q["header"] for q in questions]
    assert headers == ["Framework", "Notes"]
    framework_q = next(q for q in questions if q["header"] == "Framework")
    assert "Atlas" in framework_q["question"]  # ${proj} rendered
    assert result.output == "React"  # routed on the returned label


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


def test_manual_ref_questions_block_and_return_answer():
    """A `code` node builds the questions list; the gate reads it via `questions: ${qs}`,
    passes the code-produced question to the host, and returns its record."""
    loaded = load_flow(_MANUAL_REF)
    requests = []
    with install_ask_user(
        lambda **request: requests.append(request) or {"H": "A"}
    ):
        result = run_flow(loaded, {"seed": "Which option?"})

    assert result.status == "succeeded", result.error
    assert len(requests[0]["questions"]) == 1
    q = requests[0]["questions"][0]
    assert q["question"] == "Which option?"  # the seed input flowed through the code node
    assert q["header"] == "H"
    assert result.output == {"H": "A"}


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


def test_adaptive_questions_block_and_return_answer(monkeypatch):
    """An `adaptive_questions:` block desugars into a compose-agent + gate. The agent
    composes the questions list and the gate passes exactly those questions to the host."""
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: _ComposeChat())

    loaded = load_flow(_ADAPTIVE)
    requests = []
    with install_ask_user(
        lambda **request: requests.append(request) or {"DB": "Postgres"}
    ):
        result = run_flow(loaded, {"ctx": "storage"})

    assert result.status == "succeeded", result.error
    questions = requests[0]["questions"]
    assert [q["header"] for q in questions] == ["DB"]
    assert questions[0]["question"] == "Which database?"
    assert [o["label"] for o in questions[0]["options"]] == ["Postgres", "SQLite"]
    assert result.output == {"DB": "Postgres"}
