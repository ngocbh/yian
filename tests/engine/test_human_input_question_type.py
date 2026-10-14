from yian.nodes.human_input.questions import question_list_type
from yian.typesys.values import ValueKind
from yian.nodes.agent.structured import type_to_schema


def test_question_list_type_structure():
    sh = question_list_type()
    assert sh.kind == ValueKind.LIST_OBJECT
    q = sh.element
    assert q.kind == ValueKind.OBJECT
    assert {"question", "header", "options", "multi_select"} <= set(q.fields)
    assert {"question", "header"} <= q.required
    opt = q.fields["options"]
    assert opt.kind == ValueKind.LIST_OBJECT
    assert {"label", "description"} <= set(opt.element.fields)


def test_question_type_drives_structured_schema():
    # the synth compose-agent generates against this; it must build a pydantic model
    model = type_to_schema(question_list_type())
    assert model is not None  # list -> ListWrapper(items=List[Record])


_ANY_GATE_FLOW = """
id: gate
name: gate
nodes:
  ask:
    kind: human_input
    prompt: anything to add?
    output: Any
output: ${ask.output}
"""


def test_human_input_with_any_output_stores_a_free_text_answer_as_text():
    from yian.compose.loader import load_flow
    from yian.run import run_flow

    loaded = load_flow(_ANY_GATE_FLOW)
    assert loaded.compiled.nodes["ask"].answer_schema == [
        {"name": "answer", "type": "Any", "required": True},
    ]
    result = run_flow(loaded, {}, tools={"ask_user": lambda **request: "yes"})
    assert result.status == "succeeded", result.failure
    assert result.output == "yes"
