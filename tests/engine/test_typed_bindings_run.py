"""Typed node input bindings checked when a run binds a node's inputs.

A binding written `name: !Type value` declares the value's type. The binder
checks the resolved value against it without converting, and a mismatch fails the
node with code `binding.type`, located on the typed binding.
"""

from yian.compose.loader import load_flow
from yian.run import run_flow


def _size_flow(qty: object) -> str:
    """Return a flow whose untyped producer emits `{"qty": qty}` to an `int` input."""
    return f"""
id: typed_size
name: typed_size
nodes:
  size:
    kind: code
    code: |
      return {{"qty": {qty!r}}}
  use:
    kind: code
    input:
      qty: !int ${{size.output.qty}}
    code: |
      return inputs["qty"]
output: ${{use.output}}
"""


def test_mismatched_typed_input_fails_the_consumer_on_its_binding():
    """A str where `int` is declared fails the consumer and locates the binding."""
    text = _size_flow("5")
    loaded = load_flow(text)

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "use"
    assert "binding 'qty'" in result.failure.message
    diagnostic = loaded.diagnose(result.failure)
    assert diagnostic.frames, diagnostic.notes
    frame = diagnostic.frames[-1]
    binding_line = next(
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if "qty: !int " in line
    )
    assert frame.text == text
    assert frame.range.covered_lines(text) == [binding_line]


def test_matching_typed_input_passes_the_value_through():
    """An int where `int` is declared reaches the node unchanged."""
    result = run_flow(load_flow(_size_flow(5)), {})

    assert result.status == "succeeded", result.failure
    assert result.output == 5
    assert type(result.output) is int


def _typed_consumer_flow(value: object, declared: str) -> str:
    """Return a flow whose producer emits `value` into an input declared `declared`."""
    return f"""
id: typed_consumer
name: typed_consumer
nodes:
  p:
    kind: code
    code: |
      return {value!r}
  c:
    kind: code
    input:
      x: !{declared} ${{p.output}}
    code: |
      return type(inputs["x"]).__name__
output: ${{c.output}}
"""


def test_typed_input_does_not_convert_an_int_declared_float():
    """An int bound to a `float` input stays an int; checking does not convert."""
    result = run_flow(load_flow(_typed_consumer_flow(30, "float")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == "int"


def test_typed_input_does_not_coerce_a_numeric_string():
    """A numeric string bound to an `int` input fails instead of being parsed."""
    result = run_flow(load_flow(_typed_consumer_flow("30", "int")), {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "c"


_HUMAN_FLOW = """
id: typed_gate
name: typed_gate
input:
  n: !str
nodes:
  src:
    kind: code
    input:
      n: ${input.n}
    code: |
      return int(inputs["n"]) if inputs["n"].isdigit() else inputs["n"]
  ask:
    kind: human_input
    input:
      count: !int ${src.output}
    prompt: "Proceed with ${count} items?"
output: ${ask.output}
"""


def test_typed_human_input_context_binding_fails_before_asking():
    """A mistyped context binding fails the gate before the host is asked."""
    asked = []
    result = run_flow(
        load_flow(_HUMAN_FLOW),
        {"n": "three"},
        tools={"ask_user": lambda **request: asked.append(request) or "yes"},
    )

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "ask"
    assert asked == []


def test_typed_human_input_context_binding_passes_a_matching_value():
    """A well-typed context binding renders into the prompt and the gate asks."""
    asked = []
    result = run_flow(
        load_flow(_HUMAN_FLOW),
        {"n": "3"},
        tools={"ask_user": lambda **request: asked.append(request) or "yes"},
    )

    assert result.status == "succeeded", result.failure
    assert len(asked) == 1
    assert asked[0]["question"] == "Proceed with 3 items?"
    assert result.output == "yes"


_ADAPTIVE_FLOW = """
id: typed_adaptive
name: typed_adaptive
input:
  n: !str
nodes:
  src:
    kind: code
    input:
      n: ${input.n}
    code: |
      return int(inputs["n"]) if inputs["n"].isdigit() else inputs["n"]
  ask:
    kind: human_input
    input:
      count: !int ${src.output}
    adaptive_questions:
      prompt: "Design one question about ${count} items."
output: ${ask.output}
"""


def test_adaptive_questions_composer_keeps_the_gate_input_types():
    """The synthetic composer inherits the gate's typed inputs, so both check them."""
    loaded = load_flow(_ADAPTIVE_FLOW)
    wiring = loaded.compiled.wiring

    gate_contract = wiring["ask"]["count"].contract
    composer_contract = wiring["ask__compose"]["count"].contract

    assert gate_contract is not None
    assert composer_contract == gate_contract
