"""Unit test for parsing `kind: loop` into a LoopDescriptor.

The parser layer only: `parse_file` -> `ComposeFile` (raw `nodes` dict), then
`parse_nodes(document.root.child("nodes"))` reads each keyed-map body into its typed descriptor.
`kind: loop` must produce a `LoopDescriptor` with `call`/`while_`/`max`/`inputs`
populated. Runtime construction is covered by the loop build tests.
"""

from yian.compose.parse.file import parse_file
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceDocument

LOOP_YAML = """
id: chat
name: chat
input:
  history: list
output:
  messages: list
nodes:
  chat_loop:
    kind: loop
    call: chat_turn
    input:
      messages: ${input.history}
      exited: false
    while: not ${exited}
    max: 1000
"""


def _parse_nodes(text: str):
    """Parse one physical flow document and return its located descriptors."""

    document = SourceDocument.parse(text)
    parse_file(document)
    return parse_nodes(document.root.child("nodes"))


def test_parser_reads_loop_node():
    desc = _parse_nodes(LOOP_YAML)["chat_loop"]
    assert desc.call == "chat_turn"
    assert desc.while_ == "not ${exited}"
    assert desc.max == 1000
    assert desc.inputs["exited"] is False


UNTIL_TIMES_YAML = """
id: p
name: p
input:
  n: int
output:
  n: int
nodes:
  L:
    kind: loop
    call: body
    input:
      n: 3
    until: ${n} <= 0
    max: 5
"""


def test_parser_accepts_until_and_times_keys():
    desc = _parse_nodes(UNTIL_TIMES_YAML)["L"]
    assert desc.until_ == "${n} <= 0"
    # `times:` is also an accepted loop key. `LoopDescriptor.build`, not parsing, enforces
    # exactly one termination form.
    times_desc = _parse_nodes(
        UNTIL_TIMES_YAML.replace("until: ${n} <= 0", "times: 4")
    )["L"]
    assert times_desc.times == 4
