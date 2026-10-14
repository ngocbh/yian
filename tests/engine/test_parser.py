"""Unit tests for the section parser — strict sections + x-* + anchors."""

from pathlib import Path

import pytest

from yian.compose import LoadError
from yian.compose.parse.file import (
    ComposeFile,
    parse_file,
    parse_source_document,
)
from yian.diagnostics import SourceRange

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


_MINIMAL = """
id: min
name: minimal
input:
  topic: str
nodes:
  note:
    kind: agent
    input:
      topic: ${input.topic}
    output: str
    prompt: "Write a note on ${topic}."
output: ${note.output}
"""


def test_minimal_parses():
    f = parse_file(_MINIMAL)
    assert isinstance(f, ComposeFile)
    assert f.id == "min"
    assert f.name == "minimal"
    assert f.description is None
    assert f.inputs == {"topic": "str"}
    assert "note" in f.nodes  # nodes stays a raw dict at this step
    assert f.nodes["note"]["kind"] == "agent"
    assert f.outputs == "${note.output}"
    assert f.asserts == []  # optional, defaults empty
    assert f.typedefs == {}


def test_unknown_top_level_key_is_loud():
    text = "id: x\nname: x\nnodes: {}\nbogus: 1\n"
    with pytest.raises(LoadError) as exc:
        parse_file(text)
    assert "bogus" in str(exc.value)
    assert exc.value.diagnostic.frames[-1].range == SourceRange(4, 1, 4, 6)
    assert exc.value.diagnostic.frames[-1].range.start_line == 4


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "id: x\nname: x\nnodes: {}\nasserts:\n  - true\n",
            SourceRange(5, 5, 5, 9),
        ),
        (
            "id: x\nname: x\nnodes: {}\ninput: []\n",
            SourceRange(4, 8, 4, 10),
        ),
        (
            "id: x\nname: x\nnodes: {}\nuses:\n  7: child\n",
            SourceRange(5, 3, 5, 4),
        ),
    ],
)
def test_schema_error_replays_pydantic_location(text, expected):
    with pytest.raises(LoadError) as exc:
        parse_file(text)

    assert exc.value.diagnostic.frames[-1].range == expected
    assert exc.value.diagnostic.frames[-1].range.start_line == expected.start_line


def test_schema_error_uses_first_location_that_replays():
    document = parse_source_document("name: [bad]\nnodes: {}\n", "flow.yaml")

    with pytest.raises(LoadError) as exc:
        parse_file(document)

    assert exc.value.diagnostic.frames[-1].range == SourceRange(1, 7, 1, 12)
    assert exc.value.diagnostic.frames[-1].range.start_line == 1


def test_missing_schema_field_falls_back_to_root_range():
    document = parse_source_document("id: x\nname: x\n", "flow.yaml")

    with pytest.raises(LoadError) as exc:
        parse_file(document)

    assert exc.value.diagnostic.frames[-1].range == document.root.range
    assert (
        exc.value.diagnostic.frames[-1].range.start_line
        == document.root.range.start_line
    )


def test_x_keys_ignored():
    text = (
        "x-owner: ngocbh\n"
        "x-agent-defaults:\n"
        "  kind: agent\n" + _MINIMAL
    )
    f = parse_file(text)
    assert isinstance(f, ComposeFile)
    assert f.id == "min"
    # x-* keys are not surfaced on the strict model
    assert not hasattr(f, "x-owner")


def test_asserts_and_typedefs_optional():
    text = _MINIMAL + (
        "asserts:\n"
        '  - "input.topic != \\"\\""\n'
        "typedefs:\n"
        "  Rating:\n"
        "    category: str\n"
    )
    f = parse_file(text)
    assert len(f.asserts) == 1
    assert "Rating" in f.typedefs


def test_non_mapping_top_level_is_loud():
    with pytest.raises(LoadError):
        parse_file("- just\n- a\n- list\n")


def test_seed_00_section_parse():
    f = parse_file((_SEEDS / "00-hello-agent.yaml").read_text())
    assert f.id == "hello-agent"
    assert f.name == "hello_agent"
    assert "note" in f.nodes
    assert f.outputs == "${note.output}"


def test_seed_11_anchors_load():
    # PyYAML expands &anchor / *alias / <<: merge before strict validation; the
    # x-agent-defaults anchor holder is stripped as an x-* key.
    f = parse_file((_SEEDS / "11-reuse-anchors.yaml").read_text())
    assert f.id == "reuse-anchors"
    for nid in ("pro", "con", "judge"):
        assert f.nodes[nid]["kind"] == "agent"  # merged in from the anchor
        assert f.nodes[nid]["outputs"] == "str"
        assert f.nodes[nid]["llm_config"]["provider"] == "anthropic"


def test_composefile_parses_optional_version():
    from yian.compose import parse_file
    f = parse_file("id: x\nname: x\nversion: v1\nnodes:\n  a: {kind: code, code: m:f}\n")
    assert f.version == "v1"


def test_composefile_version_defaults_none():
    from yian.compose import parse_file
    f = parse_file("id: x\nname: x\nnodes:\n  a: {kind: code, code: m:f}\n")
    assert f.version is None
