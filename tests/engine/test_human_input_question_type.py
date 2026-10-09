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
