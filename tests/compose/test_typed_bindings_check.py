"""Checking typed bindings at load: literals exactly, references by inferred type.

A typed binding whose value can never fit its declared type is a load error located
at the binding. A literal right side is fully known, so it is checked exactly; a
reference or expression is rejected only when its inferred type is disjoint from the
declared one, while a narrower inferred type loads and is checked at runtime.
"""

from __future__ import annotations

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.run import run_flow


def _flow(binding: str, *, producers: str = "", typedefs: str = "") -> str:
    """Return a flow whose `use` node binds `x` as `binding` after `producers`."""
    return f"""
id: checked
name: checked
{typedefs}nodes:
{producers}  use:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: {binding}
output: ${{use.output}}
"""


def _producer(node_id: str, value: str, output: str | None) -> str:
    """Return a CODE node `node_id` returning `value`, declaring `output` when given."""
    declared = f"    output: {output}\n" if output is not None else ""
    return (
        f"  {node_id}:\n"
        "    kind: code\n"
        "    code: |\n"
        f"      return {value}\n"
        f"{declared}"
    )


def _load_error(text: str) -> LoadError:
    """Load `text`, expecting a load error, and return it."""
    with pytest.raises(LoadError) as info:
        load_flow(text)
    return info.value


def _error_line(error: LoadError) -> int:
    """Return the 1-based line of the error's first diagnostic frame."""
    frames = error.diagnostic.frames
    assert frames, error.diagnostic.message
    return frames[0].range.start_line


def _line_of(text: str, line: str) -> int:
    """Return the 1-based line number of `line` in `text`."""
    return text.splitlines().index(line) + 1


# --- disjoint references ------------------------------------------------- #


def test_disjoint_node_input_fails_at_its_line():
    text = _flow("int = ${note.output}", producers=_producer("note", '"hi"', "str"))

    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "node 'use' input 'x': declared int, but the value is str"
    )
    assert _error_line(error) == _line_of(text, "      x: int = ${note.output}")


def test_disjoint_flow_output_field_fails_at_its_line():
    text = """
id: checked
name: checked
nodes:
  note:
    kind: code
    code: |
      return "hi"
    output: str
output:
  x: int = ${note.output}
"""
    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "output 'x': declared int, but the value is str"
    )
    assert _error_line(error) == _line_of(text, "  x: int = ${note.output}")


def test_disjoint_single_flow_output_fails():
    text = """
id: checked
name: checked
nodes:
  note:
    kind: code
    code: |
      return "hi"
    output: str
output: int = ${note.output}
"""
    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "output: declared int, but the value is str"
    )
    assert _error_line(error) == _line_of(text, "output: int = ${note.output}")


def test_record_flow_output_field_named_result_keeps_its_name():
    text = """
id: checked
name: checked
nodes:
  note:
    kind: code
    code: |
      return "hi"
    output: str
output:
  result: int = ${note.output}
"""
    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "output 'result': declared int, but the value is str"
    )


def test_embedded_template_is_text_and_disjoint_from_int():
    text = _flow("int = n=${a.output}", producers=_producer("a", "1", "int"))

    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "node 'use' input 'x': declared int, but the value is str"
    )


def test_record_with_an_extra_required_field_is_disjoint_from_a_smaller_record():
    typedefs = """typedefs:
  Big:
    a: int
    b: str
  Small:
    a: int
"""
    text = _flow(
        "Small = ${big.output}",
        producers=_producer("big", '{"a": 1, "b": "x"}', "Big"),
        typedefs=typedefs,
    )

    error = _load_error(text)

    assert str(error.diagnostic.message) == (
        "node 'use' input 'x': declared {\"a\":int}, "
        'but the value is {"a":int,"b":str}'
    )


# --- literals ------------------------------------------------------------ #


def test_text_literal_for_int_fails_at_its_line():
    text = _flow("int = abc")

    error = _load_error(text)

    assert str(error.diagnostic.message) == "binding 'x': `abc` is not a valid int"
    assert _error_line(error) == _line_of(text, "      x: int = abc")


def test_number_literal_for_str_suggests_quoting():
    error = _load_error(_flow("str = 30"))

    assert str(error.diagnostic.message) == (
        "binding 'x': `30` is not a valid str; quote it to keep it as text: "
        '`x: str = "30"`'
    )


def test_true_literal_for_str_explains_yaml_booleans():
    error = _load_error(_flow("str = true"))

    message = str(error.diagnostic.message)
    assert message.startswith("binding 'x': `true` is not a valid str")
    assert '`x: str = "true"`' in message
    assert "yes/no/on/off/true/false as booleans" in message


@pytest.mark.parametrize("word", ["yes", "no"])
def test_unquoted_yes_or_no_literal_for_str_is_text(word):
    result = run_flow(load_flow(_flow(f"str = {word}")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == word


def test_null_literal_for_a_record_output_field_fails_at_its_line():
    text = """
id: checked
name: checked
nodes: {}
output:
  x: int = null
"""
    error = _load_error(text)

    assert str(error.diagnostic.message) == "binding 'x': `null` is not a valid int"
    assert _error_line(error) == _line_of(text, "  x: int = null")


def test_escaped_template_is_text_and_does_not_fit_int():
    text = _flow("int = $${x}")

    error = _load_error(text)

    assert str(error.diagnostic.message) == (
        "node 'use' input 'x': declared int, but the value \"${x}\" does not fit"
    )
    assert _error_line(error) == _line_of(text, "      x: int = $${x}")


@pytest.mark.parametrize(
    "binding",
    [
        "Optional[int] = null",
        "int = 3",
        "list[str] = []",
        "float = 3",
        "str = abc",
        "str = $${x}",
    ],
)
def test_fitting_literals_load(binding):
    load_flow(_flow(binding))


def test_escaped_template_binds_its_text():
    result = run_flow(load_flow(_flow("str = $${x}")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == "${x}"


# --- narrower, opaque, and fitting references ------------------------------ #


def test_optional_into_non_optional_loads_and_fails_at_runtime():
    text = _flow(
        "int = ${maybe.output}", producers=_producer("maybe", "None", "Optional[int]")
    )
    loaded = load_flow(text)

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "use"


def test_str_into_literal_loads():
    load_flow(_flow("Literal[a, b] = ${s.output}", producers=_producer("s", '"a"', "str")))


def test_int_into_float_loads():
    load_flow(_flow("float = ${i.output}", producers=_producer("i", "1", "int")))


@pytest.mark.parametrize(
    ("binding", "producers"),
    [
        ("int = ${a.output + 1}", _producer("a", "1", None)),
        ("int = ${p.output}", _producer("p", "1", None)),
        ("str = ${system.run_id}", ""),
    ],
)
def test_unproven_sources_load(binding, producers):
    load_flow(_flow(binding, producers=producers))


# --- sources from calls, bodies, and loops --------------------------------- #


_STR_CHILD = """
defs:
  shout:
    input:
      y: str
    nodes:
      up:
        kind: code
        input:
          y: ${input.y}
        output: str
        code: |
          return inputs["y"].upper()
    output: ${up.output}
"""


def test_inline_call_source_that_returns_str_is_disjoint_from_int():
    text = f"""
id: checked
name: checked
{_STR_CHILD}nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: int = call(shout, y=hi)
output: ${{use.output}}
"""
    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "node 'use' input 'x': declared int, but the value is str"
    )
    assert _error_line(error) == _line_of(text, "      x: int = call(shout, y=hi)")


def test_disjoint_binding_inside_a_defs_body_names_the_body():
    text = """
id: checked
name: checked
defs:
  child:
    nodes:
      note:
        kind: code
        output: str
        code: |
          return "hi"
      use:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: int = ${note.output}
    output: ${use.output}
nodes:
  run:
    kind: call
    call: child
output: ${run.output}
"""
    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "defs entry 'child': node 'use' input 'x': declared int, but the value is str"
    )
    # The first frame is the enclosing definition; the last is the binding.
    frames = error.diagnostic.frames
    assert frames[-1].range.start_line == _line_of(text, "          x: int = ${note.output}")


def test_disjoint_binding_into_an_inline_body_fails():
    text = """
id: checked
name: checked
nodes:
  note:
    kind: code
    output: str
    code: |
      return "hi"
  wrap:
    kind: call
    input:
      n: int = ${note.output}
    nodes:
      twice:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] * 2
    output: ${twice.output}
output: ${wrap.output}
"""
    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "node 'wrap' input 'n': declared int, but the value is str"
    )
    assert _error_line(error) == _line_of(text, "      n: int = ${note.output}")


def _loop_flow(seed: str) -> str:
    """Return a flow whose LOOP seeds `n` as `seed` into a body returning `n: int`.

    Args:
        seed (`str`): The right side of the LOOP node's `n:` seed.
    """
    return f"""
id: checked
name: checked
defs:
  bump:
    input:
      n: int
    nodes:
      step:
        kind: code
        input:
          n: ${{input.n}}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      n: ${{step.output}}
nodes:
  note:
    kind: code
    output: str
    code: |
      return "hi"
  loop:
    kind: loop
    times: 2
    input:
      n: {seed}
    call: bump
output: ${{loop.output}}
"""


def test_disjoint_loop_seed_fails():
    error = _load_error(_loop_flow("int = ${note.output}"))

    assert str(error.diagnostic.message).startswith(
        "node 'loop' input 'n': declared int, but the value is str"
    )


# --- the typed binding is reported before downstream checks ---------------- #


def _call_str_child(binding: str) -> str:
    """Return a flow that calls `shout` (taking `y: str`) with `y` bound as `binding`.

    Args:
        binding (`str`): The right side of the CALL node's `y:` binding.
    """
    return f"""
id: checked
name: checked
{_STR_CHILD}nodes:
  a:
    kind: code
    output: str
    code: |
      return "hi"
  run:
    kind: call
    call: shout
    input:
      y: {binding}
output: ${{run.output}}
"""


def test_disjoint_reference_into_a_call_reports_the_binding_not_the_child():
    error = _load_error(_call_str_child("int = ${a.output}"))

    assert str(error.diagnostic.message).startswith(
        "node 'run' input 'y': declared int, but the value is str"
    )


def test_bad_literal_into_a_call_reports_the_binding_not_the_child():
    error = _load_error(_call_str_child("int = abc"))

    assert str(error.diagnostic.message) == "binding 'y': `abc` is not a valid int"


def test_bad_literal_loop_seed_reports_the_binding_not_the_carried_type():
    error = _load_error(_loop_flow("str = 5"))

    assert str(error.diagnostic.message).startswith("binding 'n': `5` is not a valid str")


# --- MAP item bindings ----------------------------------------------------- #


def _named_map_flow(binding: str, *, over: str, param_type: str, typedefs: str = "") -> str:
    """Return a flow that maps the named `child` over `over`, binding `t` as `binding`.

    Args:
        binding (`str`): The right side of the MAP node's `t:` binding.
        over (`str`): The MAP node's `over:` source.
        param_type (`str`): The type the `child` body declares for `t`.
        typedefs (`str`, *optional*, defaults to `""`): A `typedefs:` section.
    """
    return f"""
id: checked
name: checked
{typedefs}input:
  words: list[int]
defs:
  child:
    input:
      t: {param_type}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${{input.t}}
    output: ${{echo.output}}
nodes:
  opaque:
    kind: code
    code: |
      return [1, 2]
  m:
    kind: map
    over: {over}
    call: child
    input:
      t: {binding}
output: ${{m.output}}
"""


def test_disjoint_item_binding_on_a_named_map_fails_at_its_line():
    text = _named_map_flow("str = ${item}", over="${input.words}", param_type="str")

    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "node 'm' input 't': declared str, but the value is int"
    )
    assert _error_line(error) == _line_of(text, "      t: str = ${item}")


def test_disjoint_item_binding_on_an_inline_map_fails():
    text = """
id: checked
name: checked
input:
  words: list[int]
nodes:
  m:
    kind: map
    over: ${input.words}
    input:
      t: str = ${item}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.t}
    output: ${echo.output}
output: ${m.output}
"""
    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "node 'm' input 't': declared str, but the value is int"
    )


def test_fitting_item_binding_loads():
    load_flow(_named_map_flow("int = ${item}", over="${input.words}", param_type="int"))


def test_disjoint_item_field_binding_fails():
    typedefs = """typedefs:
  Rec:
    name: str
"""
    text = _named_map_flow(
        "int = ${item.name}",
        over="${input.recs}",
        param_type="int",
        typedefs=typedefs,
    ).replace("input:\n  words: list[int]\n", "input:\n  recs: list[Rec]\n", 1)

    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "node 'm' input 't': declared int, but the value is str"
    )


def test_item_binding_over_an_untyped_list_loads():
    load_flow(_named_map_flow("str = ${item}", over="${opaque.output}", param_type="str"))
