"""Compose provenance contracts for authored flow-input declarations."""

from __future__ import annotations

from textwrap import indent

import pytest

from agent_composer.compile.manifest import flow_fingerprint
from agent_composer.compile.model import START_ID
from agent_composer.compose import LoadError, load_flow
from agent_composer.compose.source import SourceCursor, SourceDocument
from agent_composer.diagnostics import OriginRef, OriginResolver
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.contracts import ContractOrigins, Parameter, ValueContract
from agent_composer.run import run_flow


_ROOT_FLOW = """\
id: input-contract-origins
name: input-contract-origins
input:
  topic: str
  window: int = 30
  payload:
    profile:
      age: int
      nickname: Optional[str]
  flags:
    enabled: bool
    limits:
      count: int
nodes:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${input.topic}
    output: str
output: ${echo.output}
"""


def _parameter(parameters: list[Parameter], name: str) -> Parameter:
    """Return one exact parameter selected by name."""
    matches = [parameter for parameter in parameters if parameter.name == name]
    assert len(matches) == 1
    parameter = matches[0]
    assert type(parameter) is Parameter
    return parameter


def _start_parameters(compiled) -> list[Parameter]:
    """Return one compiled flow's canonical START signature."""
    parameters = compiled.nodes[START_ID].params
    assert type(parameters) is list
    assert all(type(parameter) is Parameter for parameter in parameters)
    return parameters


def _contract(parameter: Parameter) -> ValueContract:
    """Return one parameter's exact value contract."""
    contract = parameter.contract
    assert type(contract) is ValueContract
    return contract


def _assert_site(
    resolver: OriginResolver | None,
    origin: OriginRef | None,
    cursor: SourceCursor,
    *,
    anchor: str = "value",
) -> OriginRef:
    """Assert that one semantic origin resolves to the expected source occurrence."""
    assert resolver is not None
    assert isinstance(origin, OriginRef)
    frame = resolver.resolve(origin)
    assert frame is not None
    assert cursor.document is not None
    assert frame.text == cursor.document.text
    expected = cursor.key_range if anchor == "key" else cursor.range
    assert frame.range == expected
    return origin


def test_agent_variables_keep_declaration_contract_and_default_source_sites():
    text = """\
id: variable-origins
name: variable-origins
typedefs:
  Settings:
    limit: int[1,12]
    tags: list[str]
nodes:
  writer:
    kind: agent
    variables:
      settings: 'Settings = {limit: 4, tags: [a, b]}'
      note: Optional[str] = null
    prompt: ${note}
    output: str
output: ${writer.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    variables = loaded.compiled.nodes["writer"].variables
    for name in ("settings", "note"):
        variable = _parameter(variables, name)
        cursor = document.root.child("nodes").child("writer").child("variables").child(name)
        declaration = _assert_site(loaded.origin_resolver, variable.origin, cursor, anchor="key")
        contract = _assert_site(loaded.origin_resolver, _contract(variable).origins.default, cursor)
        default = _assert_site(loaded.origin_resolver, variable.default_origin, cursor)
        assert len({declaration, contract, default}) == 3
    settings = _parameter(variables, "settings")
    assert _contract(settings).origins.at(("limit",)) == _contract(settings).origins.default


@pytest.mark.parametrize("placement", ["compact", "definition", "flat-definition", "inline"])
def test_agent_variable_locations_follow_existing_node_projections(placement):
    body = "kind: agent\nvariables:\n  count: int[1,12] = 4\nprompt: ${count}\noutput: str\n"
    header = "id: projected_variables\nname: projected_variables\n"
    if placement == "compact":
        text = header + body
        cursor_path = ()
    elif placement == "flat-definition":
        text = (header + "defs:\n  child:\n" + indent(body, "    ")
                + "nodes:\n  invoke:\n    kind: call\n    call: child\noutput: ${invoke.output}\n")
        cursor_path = ("defs", "child")
    elif placement == "definition":
        text = (header + "defs:\n  child:\n    nodes:\n      writer:\n" + indent(body, "        ")
                + "    output: ${writer.output}\nnodes:\n  invoke:\n    kind: call\n"
                + "    call: child\noutput: ${invoke.output}\n")
        cursor_path = ("defs", "child", "nodes", "writer")
    else:
        text = (header + "nodes:\n  invoke:\n    kind: call\n    nodes:\n      writer:\n"
                + indent(body, "        ") + "    output: ${writer.output}\noutput: ${invoke.output}\n")
        cursor_path = ("nodes", "invoke", "nodes", "writer")

    document = SourceDocument.parse(text)
    cursor = document.root
    for segment in cursor_path:
        cursor = cursor.child(segment)
    cursor = cursor.child("variables").child("count")
    loaded = load_flow(text)
    compiled = loaded.compiled if placement == "compact" else loaded.compiled.nodes["invoke"].child
    agents = [node for node in compiled.nodes.values() if isinstance(node, AgentNode)]
    assert len(agents) == 1
    variable = agents[0].variables[0]
    _assert_site(loaded.origin_resolver, variable.origin, cursor, anchor="key")
    _assert_site(loaded.origin_resolver, variable.contract.origins.default, cursor)
    _assert_site(loaded.origin_resolver, variable.default_origin, cursor)

    invalid = text.replace("int[1,12] = 4", "int[1,12] = 13")
    with pytest.raises(LoadError, match="default.*does not satisfy") as caught:
        load_flow(invalid)
    invalid_cursor = SourceDocument.parse(invalid).root
    for segment in cursor_path:
        invalid_cursor = invalid_cursor.child(segment)
    assert caught.value.diagnostic.frames[-1].range == invalid_cursor.child("variables").child("count").range


def test_external_aliases_rebase_agent_variables_without_changing_identity():
    child_text = """\
id: variable_child
name: variable_child
kind: agent
variables:
  count: int[1,12] = 4
prompt: ${count}
output: str
"""
    child = load_flow(child_text, source_label="variable-child.yaml")
    parent = load_flow("""\
id: variable-parent
name: variable-parent
uses:
  left: child
  right: child
nodes:
  first:
    kind: call
    call: left
  second:
    kind: call
    call: right
output:
  first: ${first.output}
  second: ${second.output}
""", child_resolver=lambda _path, _version=None: child)
    fragments = [parent.compiled.nodes[name].child for name in ("first", "second")]
    variables = [next(node for node in fragment.nodes.values() if isinstance(node, AgentNode)).variables[0]
                 for fragment in [child.compiled, *fragments]]
    origins = [tuple(variable.iter_origins()) for variable in variables]
    assert all(len(sites) == 3 for sites in origins)
    for index in range(3):
        assert len({sites[index] for sites in origins}) == 3
        assert parent.origin_resolver.resolve(origins[1][index]) == child.origin_resolver.resolve(origins[0][index])
        assert parent.origin_resolver.resolve(origins[2][index]) == child.origin_resolver.resolve(origins[0][index])
    assert flow_fingerprint(fragments[0]) == flow_fingerprint(fragments[1])


def test_root_inputs_compile_parameter_contract_default_and_nested_origins():
    """One traversal retains every semantic site on mapping and list-form records."""
    document = SourceDocument.parse(_ROOT_FLOW)
    loaded = load_flow(_ROOT_FLOW)
    inputs = document.root.child("input")

    topic = _parameter(loaded.input, "topic")
    topic_cursor = inputs.child("topic")
    _assert_site(loaded.origin_resolver, topic.origin, topic_cursor, anchor="key")
    _assert_site(
        loaded.origin_resolver,
        _contract(topic).origins.default,
        topic_cursor,
    )
    assert topic.origin != _contract(topic).origins.default

    window = _parameter(loaded.input, "window")
    window_cursor = inputs.child("window")
    parameter_origin = _assert_site(
        loaded.origin_resolver,
        window.origin,
        window_cursor,
        anchor="key",
    )
    contract_origin = _assert_site(
        loaded.origin_resolver,
        _contract(window).origins.default,
        window_cursor,
    )
    default_origin = _assert_site(
        loaded.origin_resolver,
        window.default_origin,
        window_cursor,
    )
    assert len({parameter_origin, contract_origin, default_origin}) == 3

    payload = _parameter(loaded.input, "payload")
    payload_cursor = inputs.child("payload")
    payload_entries = dict(_contract(payload).origins.entries)
    _assert_site(
        loaded.origin_resolver,
        _contract(payload).origins.default,
        payload_cursor,
    )
    _assert_site(
        loaded.origin_resolver,
        payload_entries[("profile",)],
        payload_cursor.child("profile"),
    )
    _assert_site(
        loaded.origin_resolver,
        payload_entries[("profile", "age")],
        payload_cursor.child("profile").child("age"),
    )
    _assert_site(
        loaded.origin_resolver,
        payload_entries[("profile", "nickname")],
        payload_cursor.child("profile").child("nickname"),
    )

    flags = _parameter(loaded.input, "flags")
    flags_cursor = inputs.child("flags")
    flags_entries = dict(_contract(flags).origins.entries)
    _assert_site(
        loaded.origin_resolver,
        flags_entries[("enabled",)],
        flags_cursor.child("enabled"),
    )
    _assert_site(
        loaded.origin_resolver,
        flags_entries[("limits",)],
        flags_cursor.child("limits"),
    )
    _assert_site(
        loaded.origin_resolver,
        flags_entries[("limits", "count")],
        flags_cursor.child("limits").child("count"),
    )


def test_required_flow_input_failure_selects_the_authored_parameter_key():
    """An omitted required value reports the declaration that required it."""
    document = SourceDocument.parse(_ROOT_FLOW, label="required.yaml")
    loaded = load_flow(_ROOT_FLOW, source_label="required.yaml")
    topic = _parameter(loaded.input, "topic")

    result = run_flow(
        loaded,
        {
            "payload": {"profile": {"age": 7}},
            "flags": {"enabled": True, "limits": {"count": 1}},
        },
    )

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "binding.evaluate"
    assert result.failure.trace[-1] == topic.origin
    diagnostic = loaded.diagnose(result.failure)
    assert (
        diagnostic.frames[-1].range
        == document.root.child("input").child("topic").key_range
    )


def test_nested_flow_input_failure_selects_the_exact_contract_field():
    """A structural type path resolves to its nested authored type occurrence."""
    document = SourceDocument.parse(_ROOT_FLOW, label="nested.yaml")
    loaded = load_flow(_ROOT_FLOW, source_label="nested.yaml")
    payload = _parameter(loaded.input, "payload")
    age_origin = dict(_contract(payload).origins.entries)[("profile", "age")]

    result = run_flow(
        loaded,
        {
            "topic": "hello",
            "payload": {"profile": {"age": "not-an-int"}},
            "flags": {"enabled": True, "limits": {"count": 1}},
        },
    )

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "contract.input"
    assert result.failure.trace[-1] == age_origin
    diagnostic = loaded.diagnose(result.failure)
    expected = (
        document.root.child("input").child("payload").child("profile").child("age")
    )
    assert diagnostic.frames[-1].range == expected.range


def test_scalar_flow_input_failure_selects_the_contract_root():
    """A root mismatch resolves to the declaration value, not the parameter key."""
    document = SourceDocument.parse(_ROOT_FLOW, label="scalar.yaml")
    loaded = load_flow(_ROOT_FLOW, source_label="scalar.yaml")
    window = _parameter(loaded.input, "window")
    contract_origin = _contract(window).origins.default

    result = run_flow(
        loaded,
        {
            "topic": "hello",
            "window": "not-an-int",
            "payload": {"profile": {"age": 7}},
            "flags": {"enabled": True, "limits": {"count": 1}},
        },
    )

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "contract.input"
    assert result.failure.trace[-1] == contract_origin
    diagnostic = loaded.diagnose(result.failure)
    expected = document.root.child("input").child("window")
    assert diagnostic.frames[-1].range == expected.range


def test_non_string_record_field_is_a_located_load_error():
    """Invalid YAML record keys never escape the Compose error boundary."""
    text = """\
id: invalid-record-field
name: invalid-record-field
input:
  payload:
    true: int
nodes: {}
"""
    document = SourceDocument.parse(text)

    with pytest.raises(LoadError, match="record field names") as caught:
        load_flow(text)

    expected = document.root.child("input").child("payload").child(True)
    assert caught.value.diagnostic.frames[-1].range == expected.key_range


@pytest.mark.parametrize("key", ["true", "null", "1.5", "''"])
@pytest.mark.parametrize("compact", [False, True], ids=["multi-node", "compact"])
def test_invalid_definition_input_name_is_a_located_load_error(
    key: str,
    compact: bool,
):
    """Invalid nested signature keys never escape the Compose error boundary."""
    body = (
        f"""    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      {key}: int
    output: int
"""
        if compact
        else f"""    input:
      {key}: int
    nodes: {{}}
"""
    )
    text = f"""\
id: invalid-definition-input
name: invalid-definition-input
defs:
  child:
{body}nodes: {{}}
"""
    document = SourceDocument.parse(text)
    input_cursor = document.root.child("defs").child("child").child("input")
    invalid_name = next(iter(input_cursor.value))

    with pytest.raises(LoadError, match="flow input names") as caught:
        load_flow(text)

    expected = input_cursor.child(invalid_name)
    assert caught.value.diagnostic.frames[-1].range == expected.key_range


@pytest.mark.parametrize("empty", ["null", "[]", "false", "0", "''"])
def test_falsy_definition_input_keeps_existing_empty_normalization(empty: str):
    """Located compilation preserves the loader's established empty-def shorthand."""
    load_flow(
        f"""id: empty-definition-input
name: empty-definition-input
defs:
  child:
    input: {empty}
    nodes: {{}}
nodes: {{}}
"""
    )


def test_compact_root_and_both_definition_forms_retain_input_contract_sites():
    """Root and definition lowering share the physical declaration reader."""
    compact_root = """\
id: compact_input_contract
name: compact_input_contract
input:
  x: int = 3
output: int
kind: code
code: tests.engine._compose_codefns:echo_x
"""
    compact_document = SourceDocument.parse(compact_root)
    compact = load_flow(compact_root)
    root_parameter = _parameter(compact.input, "x")
    root_cursor = compact_document.root.child("input").child("x")
    _assert_site(
        compact.origin_resolver, root_parameter.origin, root_cursor, anchor="key"
    )
    _assert_site(
        compact.origin_resolver,
        _contract(root_parameter).origins.default,
        root_cursor,
    )
    _assert_site(compact.origin_resolver, root_parameter.default_origin, root_cursor)

    text = """\
id: definition-input-contracts
name: definition-input-contracts
input:
  x: int
defs:
  named:
    input:
      x: int
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.x}
        output: int
    output: ${echo.output}
  compact:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: int
    output: int
nodes:
  named_call:
    kind: call
    call: named
    input:
      x: ${input.x}
  compact_call:
    kind: call
    call: compact
    input:
      x: ${input.x}
output:
  named: ${named_call.output}
  compact: ${compact_call.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    sites = (
        (
            _parameter(
                _start_parameters(loaded.compiled.nodes["named_call"].child),
                "x",
            ),
            document.root.child("defs").child("named").child("input").child("x"),
        ),
        (
            _parameter(
                _start_parameters(loaded.compiled.nodes["compact_call"].child),
                "x",
            ),
            document.root.child("defs").child("compact").child("input").child("x"),
        ),
    )
    for parameter, cursor in sites:
        _assert_site(loaded.origin_resolver, parameter.origin, cursor, anchor="key")
        _assert_site(
            loaded.origin_resolver,
            _contract(parameter).origins.default,
            cursor,
        )


def test_generated_inline_child_input_contract_stays_source_free():
    """Inferred anonymous-child signatures do not claim an authored declaration."""
    text = """\
id: generated-inline-input
name: generated-inline-input
input:
  x: int
nodes:
  invoke:
    kind: call
    input:
      x: ${input.x}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.x}
        output: int
    output: ${echo.output}
output: ${invoke.output}
"""
    loaded = load_flow(text)
    generated = _parameter(
        _start_parameters(loaded.compiled.nodes["invoke"].child),
        "x",
    )

    assert generated.origin is None
    assert generated.default_origin is None
    assert _contract(generated).origins == ContractOrigins()


def test_external_alias_rebases_every_input_contract_origin():
    """An alias view owns distinct keys that resolve to the canonical child sites."""
    child_text = """\
id: external-input-child
name: external-input-child
input:
  payload:
    profile:
      age: int
  window: int = 30
nodes:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${input.payload}
    output:
      profile:
        age: int
output: ${echo.output}
"""
    child = load_flow(child_text, source_label="child.yaml")
    parent_text = """\
id: external-input-parent
name: external-input-parent
input:
  payload:
    profile:
      age: int
uses:
  imported: child
nodes:
  invoke:
    kind: call
    call: imported
    input:
      payload: ${input.payload}
output: ${invoke.output}
"""
    parent = load_flow(
        parent_text,
        source_label="parent.yaml",
        child_resolver=lambda _path, _version=None: child,
    )
    embedded_inputs = _start_parameters(parent.compiled.nodes["invoke"].child)

    pairs = (
        (
            _parameter(child.input, "payload").origin,
            _parameter(embedded_inputs, "payload").origin,
        ),
        (
            _contract(_parameter(child.input, "payload")).origins.default,
            _contract(_parameter(embedded_inputs, "payload")).origins.default,
        ),
        (
            dict(_contract(_parameter(child.input, "payload")).origins.entries)[
                ("profile", "age")
            ],
            dict(_contract(_parameter(embedded_inputs, "payload")).origins.entries)[
                ("profile", "age")
            ],
        ),
        (
            _parameter(child.input, "window").default_origin,
            _parameter(embedded_inputs, "window").default_origin,
        ),
    )
    assert child.origin_resolver is not None
    assert parent.origin_resolver is not None
    for canonical, rebased in pairs:
        assert isinstance(canonical, OriginRef)
        assert isinstance(rebased, OriginRef)
        assert rebased != canonical
        assert parent.origin_resolver.resolve(rebased) == child.origin_resolver.resolve(
            canonical
        )


def test_formatting_only_relocation_keeps_keys_and_moves_resolved_ranges():
    """Semantic identities stay stable while the retained document owns coordinates."""
    moved_text = "# formatting-only relocation\n\n" + _ROOT_FLOW
    baseline = load_flow(_ROOT_FLOW, source_label="flow.yaml")
    moved = load_flow(moved_text, source_label="flow.yaml")
    baseline_payload = _parameter(baseline.input, "payload")
    moved_payload = _parameter(moved.input, "payload")
    baseline_age = dict(_contract(baseline_payload).origins.entries)[("profile", "age")]
    moved_age = dict(_contract(moved_payload).origins.entries)[("profile", "age")]

    assert moved_payload.origin == baseline_payload.origin
    assert _contract(moved_payload).origins == _contract(baseline_payload).origins
    assert moved_age == baseline_age
    assert baseline.origin_resolver is not None
    assert moved.origin_resolver is not None
    baseline_frame = baseline.origin_resolver.resolve(baseline_age)
    moved_frame = moved.origin_resolver.resolve(moved_age)
    assert baseline_frame is not None
    assert moved_frame is not None
    assert moved_frame.range.start_line == baseline_frame.range.start_line + 2
    assert moved_frame.range.end_line == baseline_frame.range.end_line + 2
