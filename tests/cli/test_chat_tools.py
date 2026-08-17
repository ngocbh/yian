"""Flow-op tools for `ac chat` — workspace-confined discover/read/validate/run/write."""
from concurrent.futures import CancelledError

import pytest

from agent_composer.cli.chat import tools as T
from agent_composer.tools import install_ask_user


def test_list_and_read_flow(tmp_path):
    (tmp_path / "a.yaml").write_text("id: a\nname: a\nnodes: {}\noutput: {}\n")
    T.set_workspace(tmp_path)
    listed = T.list_flows()
    assert "a.yaml" in listed
    assert "id: a" in T.read_flow("a.yaml")


def test_read_flow_rejects_escape(tmp_path):
    T.set_workspace(tmp_path)
    with pytest.raises(ValueError):
        T.read_flow("../secret.txt")


GOOD = "id: g\nname: g\ninput:\n  x: str\nnodes:\n  e:\n    kind: code\n    input:\n      x: ${input.x}\n    output: str\n    code: tests.seeds.fns:echo_value\noutput:\n  x: ${e.output}\n"


def test_validate_good_and_bad(tmp_path):
    T.set_workspace(tmp_path)
    (tmp_path / "g.yaml").write_text(GOOD)
    (tmp_path / "b.yaml").write_text("id: b\nname: b\nnodes:\n  x: {kind: nope}\n")
    assert "OK" in T.validate_flow("g.yaml")
    assert "OK" not in T.validate_flow("b.yaml")


def test_run_flow_tool(tmp_path):
    T.set_workspace(tmp_path)
    (tmp_path / "g.yaml").write_text(GOOD)
    # inputs is a JSON OBJECT (dict), matching how a tool-calling model passes structured
    # args — not a hand-serialized JSON string.
    out = T.run_flow("g.yaml", {"x": "hi"})
    assert "hi" in out


def test_run_flow_tool_defaults_inputs(tmp_path):
    # Omitting inputs runs the flow with an empty input set (default None -> {}).
    T.set_workspace(tmp_path)
    (tmp_path / "n.yaml").write_text(
        "id: n\nname: n\ninput: {}\noutput: int\nkind: code\ncode: tests.seeds.fns:const_one\n"
    )
    assert "OUTPUT: 1" in T.run_flow("n.yaml")


def test_write_flow_approved_create_shows_target_and_content(tmp_path):
    requests = []

    def approve(**request):
        requests.append(request)
        return {"Decision": "Approve"}

    T.set_workspace(tmp_path)
    with install_ask_user(approve):
        result = T.write_flow("new.yaml", GOOD)

    assert result == "WROTE: new.yaml"
    assert (tmp_path / "new.yaml").read_text() == GOOD
    assert len(requests) == 1
    assert "Create new.yaml" in requests[0]["question"]
    assert GOOD in requests[0]["question"]
    assert requests[0]["questions"][0]["header"] == "Decision"
    assert [
        option["label"] for option in requests[0]["questions"][0]["options"]
    ] == ["Approve", "Reject"]


def test_write_flow_approved_overwrite_shows_unified_diff(tmp_path):
    original = GOOD.replace("id: g", "id: old").replace("name: g", "name: old")
    dest = tmp_path / "existing.yaml"
    dest.write_text(original)
    requests = []

    def approve(**request):
        requests.append(request)
        return {"Decision": "Approve"}

    T.set_workspace(tmp_path)
    with install_ask_user(approve):
        result = T.write_flow("existing.yaml", GOOD)

    assert result == "WROTE: existing.yaml"
    assert dest.read_text() == GOOD
    preview = requests[0]["question"]
    assert "Overwrite existing.yaml" in preview
    assert "--- a/existing.yaml" in preview
    assert "+++ b/existing.yaml" in preview
    assert "-id: old" in preview
    assert "+id: g" in preview


@pytest.mark.parametrize(
    "answer",
    [
        {"Decision": "Reject"},
        {"Decision": None},
        None,
        {"Decision": "approve"},
    ],
)
def test_write_flow_requires_exact_approval_without_mutating_existing_bytes(
    tmp_path, answer
):
    dest = tmp_path / "existing.yaml"
    original = b"original bytes\r\n"
    dest.write_bytes(original)
    T.set_workspace(tmp_path)

    with install_ask_user(lambda **_request: answer):
        result = T.write_flow("existing.yaml", GOOD)

    assert result == "NOT WRITTEN: existing.yaml (approval was not granted)"
    assert dest.read_bytes() == original


def test_write_flow_rejected_create_has_no_filesystem_effect(tmp_path):
    T.set_workspace(tmp_path)

    with install_ask_user(lambda **_request: {"Decision": "Reject"}):
        result = T.write_flow("new/subdir/flow.yaml", GOOD)

    assert result == "NOT WRITTEN: new/subdir/flow.yaml (approval was not granted)"
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("failure", [RuntimeError("prompt failed"), CancelledError()])
def test_write_flow_prompt_failure_leaves_existing_bytes_unchanged(tmp_path, failure):
    dest = tmp_path / "existing.yaml"
    original = b"original bytes\n"
    dest.write_bytes(original)
    T.set_workspace(tmp_path)

    def fail(**_request):
        raise failure

    with install_ask_user(fail), pytest.raises(type(failure)):
        T.write_flow("existing.yaml", GOOD)

    assert dest.read_bytes() == original


def test_write_flow_without_host_approval_fails_closed(tmp_path):
    T.set_workspace(tmp_path)

    result = T.write_flow("new.yaml", GOOD)

    assert result == "NOT WRITTEN: new.yaml (host approval is unavailable)"
    assert not (tmp_path / "new.yaml").exists()


def test_write_flow_validates_before_prompt_and_confines_path(tmp_path):
    T.set_workspace(tmp_path)
    # invalid content is refused before writing
    assert "INVALID" in T.write_flow(
        "bad.yaml",
        "id: b\nname: b\nnodes:\n  x: {kind: nope}\n",
    )
    assert not (tmp_path / "bad.yaml").exists()
    # escape refused
    with pytest.raises(ValueError):
        T.write_flow("../evil.yaml", GOOD)
