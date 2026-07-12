"""Unit test for parsing `kind: loop` into a LoopDescriptor.

The parser layer only: `parse_file` -> `ComposeFile` (raw `nodes` dict), then
`parse_nodes(NodeReader.spanless(f.nodes))` reads each keyed-map body into its typed descriptor.
`kind: loop` must produce a `LoopDescriptor` with `call`/`while_`/`max`/`inputs`
populated. Baking the descriptor into a runtime LoopNode is a later step.
"""

from agent_composer.compose.parser import parse_file
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.stages.parse import parse_nodes

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
    inputs:
      messages: ${input.history}
      exited: false
    while: not ${exited}
    max: 1000
"""


def test_parser_reads_loop_node():
    cf = parse_file(LOOP_YAML)
    desc = parse_nodes(NodeReader.spanless(cf.nodes))["chat_loop"]
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
    inputs:
      n: 3
    until: ${n} <= 0
    max: 5
"""


def test_parser_accepts_until_and_times_keys():
    cf = parse_file(UNTIL_TIMES_YAML)
    desc = parse_nodes(NodeReader.spanless(cf.nodes))["L"]
    assert desc.until_ == "${n} <= 0"
    # `times:` is also an accepted loop key (exactly-one-of is a later build check, not the
    # parser's job) — a node declaring it parses into `LoopDescriptor.times`.
    times_desc = parse_nodes(
        NodeReader.spanless(parse_file(UNTIL_TIMES_YAML.replace("until: ${n} <= 0", "times: 4")).nodes)
    )["L"]
    assert times_desc.times == 4
