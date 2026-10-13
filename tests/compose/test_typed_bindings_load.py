"""Loading typed node input bindings: `name: Type = value` splits at load."""

from __future__ import annotations

import pytest

from yian.compile.manifest import flow_fingerprint
from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.diagnostics import Failure
from yian.nodes.contracts import Binding
from yian.run import run_flow
from yian.typesys import TypeRegistry, type_for


def _binding(compiled, node_id: str, name: str) -> Binding:
    """Return one node's exact wiring binding by parameter name."""
    binding = compiled.wiring[node_id][name]
    assert type(binding) is Binding
    return binding


def _child(compiled, node_id: str):
    """Return the compiled child flow baked onto a CALL or MAP node."""
    child = compiled.nodes[node_id].child
    return getattr(child, "compiled", child)


def _typed_flow(binding: str) -> str:
    """Return a flow whose `use` node binds `qty` as `binding`."""
    return f"""
id: typed
name: typed
input:
  n: int
nodes:
  size:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${{input.n}}
    output:
      qty: int
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      qty: {binding}
    output: int
output: ${{use.output}}
"""


def test_typed_ref_binding_splits_and_carries_its_contract():
    loaded = load_flow(_typed_flow("int = ${size.output.qty}"))

    binding = _binding(loaded.compiled, "use", "qty")
    assert binding.source == "${size.output.qty}"
    assert binding.contract is not None
    assert binding.contract.type == type_for("int", TypeRegistry())


def test_typed_ref_binding_infers_the_same_edges_as_the_plain_binding():
    typed = load_flow(_typed_flow("int = ${size.output.qty}"))
    plain = load_flow(_typed_flow("${size.output.qty}"))

    def edges_from_size(loaded):
        return [edge for edge in loaded.compiled.edges if edge.from_ == "size"]

    typed_edges = edges_from_size(typed)
    assert typed_edges
    assert not any(edge.optional for edge in typed_edges)
    assert typed_edges == edges_from_size(plain)


def test_plain_binding_has_no_contract():
    loaded = load_flow(_typed_flow("${size.output.qty}"))

    assert _binding(loaded.compiled, "use", "qty").contract is None


def test_typed_binding_changes_the_flow_fingerprint():
    typed = load_flow(_typed_flow("int = ${size.output.qty}"))
    plain = load_flow(_typed_flow("${size.output.qty}"))

    assert flow_fingerprint(typed.compiled) != flow_fingerprint(plain.compiled)


def test_typed_literal_binding_reads_the_right_side_as_yaml():
    loaded = load_flow(_typed_flow("int = 0"))

    binding = _binding(loaded.compiled, "use", "qty")
    assert binding.source == 0
    assert type(binding.source) is int
    assert binding.contract.type == type_for("int", TypeRegistry())


def test_contract_origin_points_at_the_typed_binding_line():
    text = _typed_flow("int = ${size.output.qty}")
    loaded = load_flow(text)
    contract = _binding(loaded.compiled, "use", "qty").contract
    line = next(
        index
        for index, row in enumerate(text.splitlines(), start=1)
        if "qty: int = " in row
    )

    diagnostic = loaded.diagnose(
        Failure(
            code="binding.type",
            message="x",
            node_id="use",
            trace=(contract.origins.default,),
        )
    )

    assert len(diagnostic.frames) == 1
    assert diagnostic.frames[0].range.start_line == line


def test_typed_inline_call_binding_references_the_synthetic_call():
    loaded = load_flow(
        """
id: typed_call
name: typed_call
defs:
  f:
    input:
      a: int
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.a}
        output: int
    output: ${x.output}
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      x: int = call(f, a=1)
    output: int
output: ${use.output}
"""
    )

    binding = _binding(loaded.compiled, "use", "x")
    assert "__call_0" in loaded.compiled.nodes
    assert binding.source == "${__call_0.output}"
    assert binding.contract.type == type_for("int", TypeRegistry())


def test_typed_case_output_binding_expands_and_keeps_its_contract():
    loaded = load_flow(
        """
id: typed_case
name: typed_case
typedefs:
  Decision: Literal[go, stop]
input:
  flag: bool
nodes:
  route:
    kind: case
    cases:
      - when: ${input.flag}
        then: go
    else: stop
  go:
    kind: code
    code: tests.engine._compose_codefns:pick_pro
    input:
      flag: ${input.flag}
    output: Decision
  stop:
    kind: code
    code: tests.engine._compose_codefns:pick_pro
    input:
      flag: ${input.flag}
    output: Decision
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      x: Decision = ${route.output}
    output: str
output: ${use.output}
"""
    )

    binding = _binding(loaded.compiled, "use", "x")
    assert binding.source == "${go.output | stop.output}"
    decision = loaded.compiled.nodes["go"].output_contract.type
    assert binding.contract.type == decision
    assert decision.tags


@pytest.mark.parametrize(
    "note",
    ['"a = b"', "'a = b'", "|\n        a = b"],
    ids=["double", "single", "literal"],
)
def test_quoted_and_block_values_stay_plain_text(note):
    loaded = load_flow(
        f"""
id: plain
name: plain
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      note: {note}
    output: str
output: ${{use.output}}
"""
    )

    binding = _binding(loaded.compiled, "use", "note")
    assert binding.source.rstrip("\n") == "a = b"
    assert binding.contract is None


def _value_flow(binding: str) -> str:
    """Return a flow whose `use` node returns its `v` input, bound as `binding`."""
    return f"""
id: value
name: value
input:
  a: int
nodes:
  use:
    kind: code
    input:
      v: {binding}
    code: |
      return inputs["v"]
output: ${{use.output}}
"""


@pytest.mark.parametrize(
    ("binding", "expected"),
    [
        ('str = "${input.a} = 2"', "1 = 2"),
        ("str = 'n=${input.a}'", "n=1"),
        ('list[str] = ["${input.a}"]', ["${input.a}"]),
    ],
    ids=["quoted-template", "single-quoted-template", "literal-list"],
)
def test_quoted_or_bracketed_right_side_is_read_as_yaml_first(binding, expected):
    result = run_flow(load_flow(_value_flow(binding)), {"a": 1})

    assert result.status == "succeeded"
    assert result.output == expected


def test_plain_value_whose_left_side_is_not_a_type_fails_at_its_line():
    text = """
id: bad
name: bad
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      note: a = b
    output: str
output: ${use.output}
"""
    with pytest.raises(LoadError) as info:
        load_flow(text)

    assert str(info.value.diagnostic.message).startswith(
        "`a` in `note: a = b` is not a type"
    )
    frames = info.value.diagnostic.frames
    assert frames
    line = text.splitlines().index("      note: a = b") + 1
    assert frames[0].range.start_line == line


def test_typed_inputs_inside_a_multi_node_def_carry_contracts():
    loaded = load_flow(
        """
id: typed_def
name: typed_def
defs:
  f:
    input:
      a: int
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: int = ${input.a}
        output: int
    output: ${x.output}
nodes:
  use:
    kind: call
    call: f
    input:
      a: 1
output: ${use.output}
"""
    )

    child = _child(loaded.compiled, "use")
    binding = _binding(child, "x", "topic")
    assert binding.source == "${input.a}"
    assert binding.contract.type == type_for("int", TypeRegistry())


def test_typed_inputs_inside_an_inline_call_body_carry_contracts():
    loaded = load_flow(
        """
id: typed_inline_call
name: typed_inline_call
input:
  n: int
nodes:
  wrapper:
    kind: call
    input:
      n: ${input.n}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          value: int = ${input.n}
        output: int
    output: ${echo.output}
output: ${wrapper.output}
"""
    )

    child = _child(loaded.compiled, "wrapper")
    binding = _binding(child, "echo", "value")
    assert binding.source == "${input.n}"
    assert binding.contract.type == type_for("int", TypeRegistry())


def test_typed_inputs_inside_an_inline_map_body_carry_contracts():
    loaded = load_flow(
        """
id: typed_inline_map
name: typed_inline_map
nodes:
  each:
    kind: map
    over: [1, 2]
    input:
      x: ${item}
    nodes:
      echo:
        kind: code
        code: |
          return inputs["value"]
        input:
          value: int = ${input.x}
        output: int
    output: ${echo.output}
output: ${each.output}
"""
    )

    child = _child(loaded.compiled, "each")
    binding = _binding(child, "echo", "value")
    assert binding.source == "${input.x}"
    assert binding.contract.type == type_for("int", TypeRegistry())


def test_single_node_def_parameter_defaults_are_not_split():
    loaded = load_flow(
        """
id: single
name: single
defs:
  f:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      p: int = 3
    output: int
nodes:
  use:
    kind: call
    call: f
output: ${use.output}
"""
    )

    child = _child(loaded.compiled, "use")
    parameters = {parameter.name: parameter for parameter in child.nodes[child.start_id].params}
    assert parameters["p"].has_default
    assert parameters["p"].default == 3
    node_id = next(nid for nid, wiring in child.wiring.items() if "p" in wiring)
    binding = _binding(child, node_id, "p")
    assert binding.source == "${input.p}"
    assert binding.contract is None


_QUOTED_INLINE_CALL = """
id: quoted_inline_call
name: quoted_inline_call
nodes:
  host:
    kind: call
    input:
      n: 1
    nodes:
      use:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          note: "a = b"
        output: str
    output: ${use.output}
output: ${host.output}
"""

_QUOTED_INLINE_LOOP = """
id: quoted_inline_loop
name: quoted_inline_loop
nodes:
  host:
    kind: loop
    times: 1
    input:
      n: 1
    nodes:
      use:
        kind: code
        code: |
          return inputs["n"]
        input:
          n: ${input.n}
          note: "a = b"
        output: int
    output:
      n: ${use.output}
output: ${host.output}
"""

_QUOTED_DEF = """
id: quoted_def
name: quoted_def
defs:
  f:
    nodes:
      use:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          note: "a = b"
        output: str
    output: ${use.output}
nodes:
  host:
    kind: call
    call: f
output: ${host.output}
"""


@pytest.mark.parametrize(
    "text",
    [_QUOTED_INLINE_CALL, _QUOTED_INLINE_LOOP, _QUOTED_DEF],
    ids=["inline-call", "inline-loop", "multi-node-def"],
)
def test_quoted_input_stays_plain_wherever_it_is_written(text):
    loaded = load_flow(text)

    compiled = _child(loaded.compiled, "host")
    binding = _binding(compiled, "use", "note")
    assert binding.source == "a = b"
    assert binding.contract is None
