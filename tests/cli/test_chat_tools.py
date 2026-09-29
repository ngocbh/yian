"""Flow-op tools for `composer chat` — workspace-confined discover/read/validate/run/write."""
from concurrent.futures import CancelledError
import os
from pathlib import Path
import stat

import pytest

from agent_composer.cli.chat import tools as T
from agent_composer.tools import TOOL_REGISTRY


@pytest.fixture
def approval_tty(monkeypatch):
    """Make host approval available without depending on the test runner's stdin."""
    monkeypatch.setattr(T, "_approval_prompt_available", lambda: True, raising=False)


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


def test_write_flow_approved_create_shows_target_and_content(tmp_path, approval_tty):
    requests = []

    def approve(**request):
        requests.append(request)
        return {"Decision": "Approve"}

    T.set_workspace(tmp_path)
    with T._bind_ask_user(approve):
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


def test_write_flow_approved_overwrite_shows_unified_diff(tmp_path, approval_tty):
    original = (
        GOOD.replace("id: g", "id: old")
        .replace("name: g", "name: old")
        .replace("\n", "\r\n")
    )
    dest = tmp_path / "existing.yaml"
    dest.write_bytes(original.encode())
    requests = []

    def approve(**request):
        requests.append(request)
        return {"Decision": "Approve"}

    T.set_workspace(tmp_path)
    with T._bind_ask_user(approve):
        result = T.write_flow("existing.yaml", GOOD)

    assert result == "WROTE: existing.yaml"
    assert dest.read_text() == GOOD
    preview = requests[0]["question"]
    assert "Overwrite existing.yaml" in preview
    assert "--- a/existing.yaml" in preview
    assert "+++ b/existing.yaml" in preview
    assert "-id: old" in preview
    assert "+id: g" in preview
    assert "[CRLF]" in preview
    assert "[LF]" in preview


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
    tmp_path, answer, approval_tty
):
    dest = tmp_path / "existing.yaml"
    original = b"original bytes\r\n"
    dest.write_bytes(original)
    T.set_workspace(tmp_path)

    with T._bind_ask_user(lambda **_request: answer):
        result = T.write_flow("existing.yaml", GOOD)

    assert result == "NOT WRITTEN: existing.yaml (approval was not granted)"
    assert dest.read_bytes() == original


def test_write_flow_rejected_create_has_no_filesystem_effect(tmp_path, approval_tty):
    T.set_workspace(tmp_path)

    with T._bind_ask_user(lambda **_request: {"Decision": "Reject"}):
        result = T.write_flow("new/subdir/flow.yaml", GOOD)

    assert result == "NOT WRITTEN: new/subdir/flow.yaml (approval was not granted)"
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("failure", [RuntimeError("prompt failed"), CancelledError()])
def test_write_flow_prompt_failure_leaves_existing_bytes_unchanged(
    tmp_path, failure, approval_tty
):
    dest = tmp_path / "existing.yaml"
    original = b"original bytes\n"
    dest.write_bytes(original)
    T.set_workspace(tmp_path)

    def fail(**_request):
        raise failure

    with T._bind_ask_user(fail), pytest.raises(type(failure)):
        T.write_flow("existing.yaml", GOOD)

    assert dest.read_bytes() == original


def test_write_flow_without_host_approval_fails_closed(tmp_path, approval_tty):
    T.set_workspace(tmp_path)

    result = T.write_flow("new.yaml", GOOD)

    assert result == "NOT WRITTEN: new.yaml (host approval is unavailable)"
    assert not (tmp_path / "new.yaml").exists()


def test_write_flow_noninteractive_input_fails_closed_without_prompting(
    tmp_path, monkeypatch
):
    T.set_workspace(tmp_path)

    class NonInteractiveInput:
        def isatty(self):
            return False

    monkeypatch.setattr(T.sys, "stdin", NonInteractiveInput())

    def unexpected_prompt(**_request):
        raise AssertionError("non-interactive write must not prompt")

    with T._bind_ask_user(unexpected_prompt):
        result = T.write_flow("new/subdir/flow.yaml", GOOD)

    assert result == "NOT WRITTEN: new/subdir/flow.yaml (interactive approval requires a TTY)"
    assert not (tmp_path / "new").exists()


def test_write_flow_without_secure_dirfd_support_fails_closed_before_prompt(
    tmp_path, approval_tty, monkeypatch
):
    T.set_workspace(tmp_path)
    monkeypatch.setattr(
        T,
        "_secure_workspace_write_available",
        lambda: False,
        raising=False,
    )

    def unexpected_prompt(**_request):
        raise AssertionError("unsupported host must not prompt")

    with T._bind_ask_user(unexpected_prompt):
        result = T.write_flow("new/subdir/flow.yaml", GOOD)

    assert result == (
        "NOT WRITTEN: new/subdir/flow.yaml "
        "(secure workspace writes are unsupported on this platform)"
    )
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("original", [None, b"original bytes\n"])
def test_write_flow_refuses_target_changed_while_approval_is_open(
    tmp_path, original, approval_tty
):
    dest = tmp_path / "flow.yaml"
    if original is not None:
        dest.write_bytes(original)
    changed = b"changed while approval was open\n"
    T.set_workspace(tmp_path)

    def approve_after_change(**_request):
        dest.write_bytes(changed)
        return {"Decision": "Approve"}

    with T._bind_ask_user(approve_after_change):
        result = T.write_flow("flow.yaml", GOOD)

    assert result == "NOT WRITTEN: flow.yaml (target changed during approval)"
    assert dest.read_bytes() == changed


def test_write_flow_does_not_follow_dangling_symlink_created_during_approval(
    tmp_path, approval_tty
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.yaml"
    link = workspace / "new.yaml"
    T.set_workspace(workspace)

    def approve_after_link(**_request):
        link.symlink_to(outside)
        return {"Decision": "Approve"}

    with T._bind_ask_user(approve_after_link):
        result = T.write_flow("new.yaml", GOOD)

    assert result == "NOT WRITTEN: new.yaml (target changed during approval)"
    assert link.is_symlink()
    assert not outside.exists()


def test_write_flow_rejects_same_byte_hardlink_swap_during_approval(
    tmp_path, approval_tty
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "flow.yaml"
    outside = tmp_path / "outside.yaml"
    original = b"original bytes\n"
    target.write_bytes(original)
    outside.write_bytes(original)
    T.set_workspace(workspace)

    def approve_after_hardlink_swap(**_request):
        target.unlink()
        os.link(outside, target)
        return {"Decision": "Approve"}

    with T._bind_ask_user(approve_after_hardlink_swap):
        result = T.write_flow("flow.yaml", GOOD)

    assert result == "NOT WRITTEN: flow.yaml (target changed during approval)"
    assert target.read_bytes() == original
    assert outside.read_bytes() == original


def test_write_flow_rejects_preexisting_multi_link_target_before_prompt(
    tmp_path, approval_tty
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_bytes(b"original bytes\n")
    os.link(outside, workspace / "flow.yaml")
    T.set_workspace(workspace)

    def unexpected_prompt(**_request):
        raise AssertionError("multi-link target must not prompt")

    with T._bind_ask_user(unexpected_prompt):
        result = T.write_flow("flow.yaml", GOOD)

    assert result == "NOT WRITTEN: flow.yaml (target has multiple hard links)"
    assert outside.read_bytes() == b"original bytes\n"


@pytest.mark.parametrize(
    ("original", "substitute_after_check"),
    [(None, 1), (b"original bytes\n", 1)],
)
def test_write_flow_commit_does_not_follow_symlink_substituted_after_recheck(
    tmp_path,
    approval_tty,
    monkeypatch,
    original,
    substitute_after_check,
):
    outside = tmp_path / "outside.yaml"
    link = tmp_path / "new.yaml"
    if original is not None:
        link.write_bytes(original)
    T.set_workspace(tmp_path)
    original_check = T._target_is_unchanged
    checks = 0

    def substitute_after_recheck(*args):
        nonlocal checks
        unchanged = original_check(*args)
        checks += 1
        if checks == substitute_after_check:
            if original is not None:
                link.unlink()
            link.symlink_to(outside)
        return unchanged

    monkeypatch.setattr(T, "_target_is_unchanged", substitute_after_recheck)
    with T._bind_ask_user(lambda **_request: {"Decision": "Approve"}):
        result = T.write_flow("new.yaml", GOOD)

    assert result == "NOT WRITTEN: new.yaml (target changed during approval)"
    assert link.is_symlink()
    assert not outside.exists()


@pytest.mark.parametrize("original", [None, b"original bytes\n"])
def test_write_flow_commit_is_anchored_across_parent_symlink_swap(
    tmp_path, approval_tty, monkeypatch, original
):
    workspace = tmp_path / "workspace"
    safe = workspace / "safe"
    held = workspace / "held"
    outside = tmp_path / "outside"
    safe.mkdir(parents=True)
    outside.mkdir()
    target = safe / "flow.yaml"
    outside_target = outside / "flow.yaml"
    if original is not None:
        target.write_bytes(original)
        outside_target.write_bytes(original)
    T.set_workspace(workspace)
    original_open = T._open_no_follow
    swapped = False
    leaf_opens = 0

    def swap_parent_before_leaf_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal leaf_opens, swapped
        if Path(path).name == "flow.yaml":
            leaf_opens += 1
        if not swapped and leaf_opens == 3:
            swapped = True
            safe.rename(held)
            safe.symlink_to(outside, target_is_directory=True)
        try:
            return original_open(path, flags, mode=mode, dir_fd=dir_fd)
        except TypeError:
            return original_open(path, flags)

    monkeypatch.setattr(T, "_open_no_follow", swap_parent_before_leaf_open)
    with T._bind_ask_user(lambda **_request: {"Decision": "Approve"}):
        result = T.write_flow("safe/flow.yaml", GOOD)

    assert result == "NOT WRITTEN: safe/flow.yaml (target changed during approval)"
    if original is None:
        assert not outside_target.exists()
        assert not (held / "flow.yaml").exists()
    else:
        assert outside_target.read_bytes() == original
        assert (held / "flow.yaml").read_bytes() == original


@pytest.mark.parametrize("original", [None, b"original bytes\n"])
def test_write_flow_rolls_back_hardlink_added_during_commit(
    tmp_path, approval_tty, monkeypatch, original
):
    target = tmp_path / "flow.yaml"
    outside = tmp_path / "outside.yaml"
    if original is not None:
        target.write_bytes(original)
    T.set_workspace(tmp_path)
    replace_contents = T._replace_fd_contents
    linked = False

    def link_before_write(descriptor, content):
        nonlocal linked
        if not linked:
            linked = True
            os.link(target, outside)
        replace_contents(descriptor, content)

    monkeypatch.setattr(T, "_replace_fd_contents", link_before_write)
    with T._bind_ask_user(lambda **_request: {"Decision": "Approve"}):
        result = T.write_flow("flow.yaml", GOOD)

    assert result == "NOT WRITTEN: flow.yaml (target changed during approval)"
    if original is None:
        assert not target.exists()
        assert outside.read_bytes() == b""
    else:
        assert target.read_bytes() == original
        assert outside.read_bytes() == original


def test_write_flow_create_uses_normal_file_permissions(tmp_path, approval_tty):
    T.set_workspace(tmp_path)
    previous_umask = os.umask(0o022)
    try:
        with T._bind_ask_user(lambda **_request: {"Decision": "Approve"}):
            result = T.write_flow("new.yaml", GOOD)
    finally:
        os.umask(previous_umask)

    assert result == "WROTE: new.yaml"
    assert stat.S_IMODE((tmp_path / "new.yaml").stat().st_mode) == 0o644


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


_ASKING_FLOW = """
id: asking
name: asking
nodes:
  ask:
    kind: human_input
    prompt: "Your name?"
output: ${ask.output}
"""

_LISTING_FLOW = """
id: listing
name: listing
nodes:
  ls:
    kind: tool
    tool_id: list_flows
  ask:
    kind: human_input
    input: {found: "${ls.output}"}
    prompt: "Saw ${found}?"
output: ${ask.output}
"""


def test_write_flow_approval_uses_the_bound_handler_not_the_registry(
    tmp_path, approval_tty
):
    requests = []

    def approve(**request):
        requests.append(request)
        assert "ask_user" not in TOOL_REGISTRY
        return {"Decision": "Approve"}

    T.set_workspace(tmp_path)
    with T._bind_ask_user(approve):
        result = T.write_flow("new.yaml", GOOD)

    assert result == "WROTE: new.yaml"
    assert len(requests) == 1
    assert "ask_user" not in TOOL_REGISTRY


def test_run_flow_tool_serves_nested_human_input_from_the_bound_handler(tmp_path):
    (tmp_path / "asking.yaml").write_text(_ASKING_FLOW)
    T.set_workspace(tmp_path)

    with T._bind_ask_user(lambda **request: f"answer to {request['question']}"):
        result = T.run_flow("asking.yaml")

    assert result == "OUTPUT: answer to Your name?"
    assert "ask_user" not in TOOL_REGISTRY


def test_run_flow_tool_keeps_registered_tools_next_to_the_bound_handler(tmp_path):
    (tmp_path / "listing.yaml").write_text(_LISTING_FLOW)
    T.set_workspace(tmp_path)

    with T._bind_ask_user(lambda **request: request["question"]):
        result = T.run_flow("listing.yaml")

    assert result == "OUTPUT: Saw listing.yaml?"
