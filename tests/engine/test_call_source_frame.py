"""The render-only `SourceFrame` carried onto `call`/`map` nodes at load.

A `call`/`map` node bakes its child flow's `SourceFrame` as `child_source` so the CLI can
box a `.yaml` frame for the child a call descends into (the nested-error traceback). A def
child indexes its inner nodes at their absolute lines in the PARENT file (label
`defs:<name>`); an external `uses:` child indexes its own file (label = the filename). The
frame is frozen + render-only, so a per-callsite deepcopy shares the one instance.
"""

import copy
from pathlib import Path

from agent_composer.compose.loader import SourceFrame, load_flow

_SEEDS = Path(__file__).resolve().parents[1] / "seeds"


def test_def_call_node_carries_source_frame():
    text = (_SEEDS / "25-nested-suspension.yaml").read_text()
    loaded = load_flow(text, search_paths=[_SEEDS])
    frame = loaded.compiled.nodes["gate"].child_source
    assert isinstance(frame, SourceFrame)
    assert frame.label == "defs:review"
    assert frame.text == text                                # a def indexes the PARENT file
    assert frame.node_ranges["approve"].start_line == 26     # inner nodes at their parent HEADER lines
    assert frame.node_ranges["record"].start_line == 32
    assert frame.field_ranges["approve"]["outputs"].start_line == 31


def test_external_call_node_carries_source_frame_labelled_by_filename():
    text = (_SEEDS / "24-uses-external.yaml").read_text()
    loaded = load_flow(text, search_paths=[_SEEDS])
    frame = loaded.compiled.nodes["say"].child_source
    assert isinstance(frame, SourceFrame)
    assert frame.label == "lib_verdict.yaml"           # the resolved filename, not name:
    assert frame.text == (_SEEDS / "lib_verdict.yaml").read_text()  # the external file's OWN text
    assert frame.node_ranges["v"].start_line == 16                 # the inner node HEADER line in that file


def test_source_frame_shared_across_deepcopy():
    # `clone_child` deep-copies each child node per callsite; the frozen render-only frame
    # returns `self` from __deepcopy__ so the full YAML text is not copied per clone.
    text = (_SEEDS / "25-nested-suspension.yaml").read_text()
    loaded = load_flow(text, search_paths=[_SEEDS])
    node = loaded.compiled.nodes["gate"]
    assert copy.deepcopy(node).child_source is node.child_source
