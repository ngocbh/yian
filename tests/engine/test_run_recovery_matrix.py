"""Recovery equivalence across nested growth and unfinished live effects.

The append-only log is the sole run authority.  These tests capture the live
``RunState`` after every committed record, replay every resulting prefix through
the public-boundary recovery seam, and compare the complete reducer state.  The
effect cases stop the dispatcher immediately after durable ``NodeStarted`` so no
model, human interaction, or wall-clock wait is invoked.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

import pytest

import agent_composer.nodes.wait.node as wait_module
import agent_composer.run as run_module
import agent_composer.runtime.engine as engine_module
from agent_composer.compile.expand import build_grow
from agent_composer.compile.llm_cascade import resolve_llm_cascade
from agent_composer.compile.manifest import flow_fingerprint, flow_manifest
from agent_composer.compile.model import START_ID
from agent_composer.compose import load_flow
from agent_composer.events import NodeStarted, NodeSucceeded, RunSucceeded
from agent_composer.nodes.base import NodeKind, Output
from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.eval_node import eval_node
from agent_composer.runtime.log import (
    MemoryRunLog,
    NodeExpandedRecord,
    NodeStartedRecord,
    RunResumedRecord,
    encode_pool,
)
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_run_resumed,
    prepare_run_started,
)
from agent_composer.typesys import VariablePool


# Fixed writer output; keep these bytes independent of the current codec.
_RECORDED_GROWTH_PREFIX = (
    (
        b'{"kind":"run_started","payload":{"effective_llm_by_static_path":{},'
        b'"flow_fingerprint":"f1d89a1792c63e9f6ab2bdb87eb8cabe5a4ff3ef58f732cb5d65c52c6c0e0eb0",'
        b'"initial_pool":{"start_id":"__start__","store":{"__start__":{"kind":"object","value":{}}},'
        b'"system":{"now":{"kind":"datetime","value":"2026-09-20T01:23:31.303163+00:00"},'
        b'"run_id":{"kind":"str","value":"historical-growth"},"today":{"kind":"date",'
        b'"value":"2026-09-20"}}},"learnable":true,"memory_layout":[{"config":{},'
        b'"policy_name":"episodic","ref":{"owner":"flow","root_flow_id":"recovery-matrix-growth",'
        b'"static_path":["recovery-matrix-growth"]}},{"config":{},"policy_name":"reflective",'
        b'"ref":{"owner":"flow","root_flow_id":"recovery-matrix-growth",'
        b'"static_path":["recovery-matrix-growth","outer"]}}],'
        b'"memory_snapshots":[{"policy_name":"episodic","snapshot":{"epoch":0,'
        b'"format":"agent_composer.episodic_memory","frames":[{"config":{"max_episodes":100,'
        b'"recall_limit":10},"episodes":[],"policy_name":"episodic","ref":{"owner":"flow",'
        b'"root_flow_id":"recovery-matrix-growth","static_path":["recovery-matrix-growth"]}}],'
        b'"namespace":"581e810e9b54407e8463be3d1cdb7d76","revision":0,"version":1}},'
        b'{"policy_name":"reflective","snapshot":{"epoch":0,"format":"agent_composer.reflective_memory",'
        b'"frames":[{"config":{"max_items":100,"recall_limit":20},"lessons":[],'
        b'"policy_name":"reflective","ref":{"owner":"flow","root_flow_id":"recovery-matrix-growth",'
        b'"static_path":["recovery-matrix-growth","outer"]}}],"llm_config":{"anthropic_effort":null,'
        b'"google_thinking_level":null,"model":"claude-sonnet-4-5","openai_reasoning_effort":null,'
        b'"provider":"anthropic","temperature":null},"namespace":"33f37e507bcb4469bc43a6a974337f04",'
        b'"revision":0,"version":1}}],"num_workers":0,"start_wiring_override":{}},'
        b'"run_id":"historical-growth","schema":9,"seq":0}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"__start__"},"run_id":"historical-growth",'
        b'"schema":9,"seq":1}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"__start__","evidence":null,'
        b'"forward_capture":{"bound_inputs":{},"evaluated_keys":[],"input_names":[],"items":[],'
        b'"node_data":{"input_output_paths":{}},"route":{"children":[],"content_index":0,"count":0,'
        b'"expression":"{}","kind":"record","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":null},"scope":{"boundary":"start",'
        b'"flow_path":[],"occurrence_path":[],"static_path":["__start__"]},"variable_names":[]},'
        b'"node_id":"__start__","value":{"kind":"object","value":{}}},"run_id":"historical-growth",'
        b'"schema":9,"seq":2}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"outer"},"run_id":"historical-growth","schema":9,'
        b'"seq":3}'
    ),
    (
        b'{"kind":"node_expanded","payload":{"evidence":null,"forward_capture":{"bound_inputs":{},'
        b'"evaluated_keys":[],"input_names":[],"items":[],"node_data":{},"route":{"children":[],'
        b'"content_index":0,"count":0,"expression":"{}","kind":"record","names":[],"path":[],'
        b'"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},'
        b'"scope":{"boundary":null,"flow_path":[],"occurrence_path":[],"static_path":["outer"]},'
        b'"variable_names":[]},"prune_ids":[],"seed":{},"spawner_id":"outer",'
        b'"topology_digest":"a99de1538df593752120f06f64a03e2338a426bae5b3192dcd024e10bdb56407"},'
        b'"run_id":"historical-growth","schema":9,"seq":4}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"outer/__start__"},"run_id":"historical-growth",'
        b'"schema":9,"seq":5}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"outer/__start__","evidence":null,'
        b'"forward_capture":{"bound_inputs":{},"evaluated_keys":[],"input_names":[],"items":[],'
        b'"node_data":{"input_output_paths":{}},"route":{"children":[],"content_index":0,"count":0,'
        b'"expression":"{}","kind":"record","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":null},"scope":{"boundary":"start",'
        b'"flow_path":["outer"],"occurrence_path":[{"index":null,"kind":"call","static_path":["outer"]}],'
        b'"static_path":["outer","__start__"]},"variable_names":[]},"node_id":"outer/__start__",'
        b'"value":{"kind":"object","value":{}}},"run_id":"historical-growth","schema":9,"seq":6}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"outer/mapped"},"run_id":"historical-growth",'
        b'"schema":9,"seq":7}'
    ),
    (
        b'{"kind":"node_expanded","payload":{"evidence":null,'
        b'"forward_capture":{"bound_inputs":{"over":[1,2]},"evaluated_keys":["over"],'
        b'"input_names":["value"],"items":[{"capture":{"bound_inputs":{"value":1},'
        b'"evaluated_keys":["value"],"input_names":[],"items":[],"node_data":{},'
        b'"route":{"children":[{"children":[{"children":[{"children":[{"children":[],"content_index":0,'
        b'"count":0,"expression":"1","kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":"1"},{"children":[],"content_index":0,'
        b'"count":0,"expression":"2","kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":"2"}],"content_index":0,"count":0,'
        b'"expression":"[1, 2]","kind":"list","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":null}],"content_index":0,"count":0,'
        b'"expression":"[1, 2][0]","kind":"project","names":[],"path":[0],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":null}],"content_index":0,"count":0,'
        b'"expression":"item","kind":"selected","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":null}],"content_index":0,"count":0,'
        b'"expression":"{value: item}","kind":"record","names":["value"],"path":[],"result_type":null,'
        b'"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},"scope":null,'
        b'"variable_names":[]},"index":0},{"capture":{"bound_inputs":{"value":2},'
        b'"evaluated_keys":["value"],"input_names":[],"items":[],"node_data":{},'
        b'"route":{"children":[{"children":[{"children":[{"children":[{"children":[],"content_index":0,'
        b'"count":0,"expression":"1","kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":"1"},{"children":[],"content_index":0,'
        b'"count":0,"expression":"2","kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":"2"}],"content_index":0,"count":0,'
        b'"expression":"[1, 2]","kind":"list","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":null}],"content_index":0,"count":0,'
        b'"expression":"[1, 2][1]","kind":"project","names":[],"path":[1],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":null}],"content_index":0,"count":0,'
        b'"expression":"item","kind":"selected","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":null}],"content_index":0,"count":0,'
        b'"expression":"{value: item}","kind":"record","names":["value"],"path":[],"result_type":null,'
        b'"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},"scope":null,'
        b'"variable_names":[]},"index":1}],"node_data":{},'
        b'"route":{"children":[{"children":[{"children":[],"content_index":0,"count":0,"expression":"1",'
        b'"kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],"snapshot":null,'
        b'"source":null,"spans":[],"value_json":"1"},{"children":[],"content_index":0,"count":0,'
        b'"expression":"2","kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":"2"}],"content_index":0,"count":0,'
        b'"expression":"[1, 2]","kind":"list","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":null}],"content_index":0,"count":0,'
        b'"expression":"{over: [1, 2]}","kind":"record","names":["over"],"path":[],"result_type":null,'
        b'"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},'
        b'"scope":{"boundary":null,"flow_path":["outer"],"occurrence_path":[{"index":null,"kind":"call",'
        b'"static_path":["outer"]}],"static_path":["outer","mapped"]},"variable_names":[]},'
        b'"prune_ids":[],"seed":[{"value":1},{"value":2}],"spawner_id":"outer/mapped",'
        b'"topology_digest":"8b0f3e24016bdd6cb652a3514aa79a527c17e8466940a4adae0aa1f7dc9fa0a8"},'
        b'"run_id":"historical-growth","schema":9,"seq":8}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"outer/repeated"},"run_id":"historical-growth",'
        b'"schema":9,"seq":9}'
    ),
    (
        b'{"kind":"node_expanded","payload":{"evidence":null,'
        b'"forward_capture":{"bound_inputs":{"exited":false,"n":0},"evaluated_keys":[],'
        b'"input_names":["n","exited"],"items":[],"node_data":{"input_output_paths":{"exited":["exited"],'
        b'"n":["n"]}},"route":{"children":[{"children":[],"content_index":0,"count":0,"expression":"0",'
        b'"kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],"snapshot":null,'
        b'"source":null,"spans":[],"value_json":"0"},{"children":[],"content_index":0,"count":0,'
        b'"expression":"false","kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":"false"}],"content_index":0,"count":0,'
        b'"expression":"{n: 0, exited: false}","kind":"record","names":["n","exited"],"path":[],'
        b'"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},'
        b'"scope":{"boundary":null,"flow_path":["outer"],"occurrence_path":[{"index":null,"kind":"call",'
        b'"static_path":["outer"]},{"index":0,"kind":"loop_driver","static_path":["outer","repeated"]}],'
        b'"static_path":["outer","repeated"]},"variable_names":[]},"prune_ids":[],'
        b'"seed":[{"exited":false,"n":0},0],"spawner_id":"outer/repeated",'
        b'"topology_digest":"0e73507a96feb5997c6eaed03da58fb9fef3bb88e22b12c52e2b41560ae2a056"},'
        b'"run_id":"historical-growth","schema":9,"seq":10}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"outer/mapped/__start__"},'
        b'"run_id":"historical-growth","schema":9,"seq":11}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"outer/mapped/__start__","evidence":null,'
        b'"forward_capture":{"bound_inputs":{},"evaluated_keys":[],"input_names":[],"items":[],'
        b'"node_data":{"input_output_paths":{}},"route":{"children":[],"content_index":0,"count":0,'
        b'"expression":"{}","kind":"record","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":null},"scope":{"boundary":null,'
        b'"flow_path":["outer"],"occurrence_path":[{"index":null,"kind":"call","static_path":["outer"]}],'
        b'"static_path":["outer","mapped"]},"variable_names":[]},"node_id":"outer/mapped/__start__",'
        b'"value":{"kind":"object","value":{}}},"run_id":"historical-growth","schema":9,"seq":12}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"outer/repeated#0/__start__"},'
        b'"run_id":"historical-growth","schema":9,"seq":13}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"outer/repeated#0/__start__",'
        b'"evidence":null,"forward_capture":{"bound_inputs":{"exited":false,"n":0},"evaluated_keys":[],'
        b'"input_names":["n","exited"],"items":[],"node_data":{"input_output_paths":{"exited":["exited"],'
        b'"n":["n"]}},"route":{"children":[{"children":[],"content_index":0,"count":0,"expression":"0",'
        b'"kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],"snapshot":null,'
        b'"source":null,"spans":[],"value_json":"0"},{"children":[],"content_index":0,"count":0,'
        b'"expression":"false","kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],'
        b'"snapshot":null,"source":null,"spans":[],"value_json":"false"}],"content_index":0,"count":0,'
        b'"expression":"{n: 0, exited: false}","kind":"record","names":["n","exited"],"path":[],'
        b'"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},'
        b'"scope":{"boundary":"start","flow_path":["outer","repeated"],"occurrence_path":[{"index":null,'
        b'"kind":"call","static_path":["outer"]},{"index":0,"kind":"loop_body","static_path":["outer",'
        b'"repeated"]}],"static_path":["outer","repeated","__start__"]},"variable_names":[]},'
        b'"node_id":"outer/repeated#0/__start__","value":{"kind":"object","value":{"exited":false,'
        b'"n":0}}},"run_id":"historical-growth","schema":9,"seq":14}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"outer/mapped#0/__start__"},'
        b'"run_id":"historical-growth","schema":9,"seq":15}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"outer/mapped#0/__start__","evidence":null,'
        b'"forward_capture":{"bound_inputs":{"value":1},"evaluated_keys":[],"input_names":["value"],'
        b'"items":[],"node_data":{"input_output_paths":{"value":["value"]}},'
        b'"route":{"children":[{"children":[],"content_index":0,"count":0,"expression":"1",'
        b'"kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],"snapshot":null,'
        b'"source":null,"spans":[],"value_json":"1"}],"content_index":0,"count":0,'
        b'"expression":"{value: 1}","kind":"record","names":["value"],"path":[],"result_type":null,'
        b'"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},'
        b'"scope":{"boundary":"start","flow_path":["outer","mapped"],"occurrence_path":[{"index":null,'
        b'"kind":"call","static_path":["outer"]},{"index":0,"kind":"map_item","static_path":["outer",'
        b'"mapped"]}],"static_path":["outer","mapped","__start__"]},"variable_names":[]},'
        b'"node_id":"outer/mapped#0/__start__","value":{"kind":"object","value":{"value":1}}},'
        b'"run_id":"historical-growth","schema":9,"seq":16}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"outer/mapped#1/__start__"},'
        b'"run_id":"historical-growth","schema":9,"seq":17}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"outer/mapped#1/__start__","evidence":null,'
        b'"forward_capture":{"bound_inputs":{"value":2},"evaluated_keys":[],"input_names":["value"],'
        b'"items":[],"node_data":{"input_output_paths":{"value":["value"]}},'
        b'"route":{"children":[{"children":[],"content_index":0,"count":0,"expression":"2",'
        b'"kind":"literal","names":[],"path":[],"result_type":null,"sizes":[],"snapshot":null,'
        b'"source":null,"spans":[],"value_json":"2"}],"content_index":0,"count":0,'
        b'"expression":"{value: 2}","kind":"record","names":["value"],"path":[],"result_type":null,'
        b'"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},'
        b'"scope":{"boundary":"start","flow_path":["outer","mapped"],"occurrence_path":[{"index":null,'
        b'"kind":"call","static_path":["outer"]},{"index":1,"kind":"map_item","static_path":["outer",'
        b'"mapped"]}],"static_path":["outer","mapped","__start__"]},"variable_names":[]},'
        b'"node_id":"outer/mapped#1/__start__","value":{"kind":"object","value":{"value":2}}},'
        b'"run_id":"historical-growth","schema":9,"seq":18}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"outer/repeated#0/bump"},'
        b'"run_id":"historical-growth","schema":9,"seq":19}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"outer/repeated#0/bump","evidence":null,'
        b'"forward_capture":{"bound_inputs":{"exited":false,"n":0},"evaluated_keys":["exited","n"],'
        b'"input_names":["n","exited"],"items":[],"node_data":{},"route":{"children":[{"children":[],'
        b'"content_index":0,"count":0,"expression":"outer/repeated#0/__start__.output.n","kind":"source",'
        b'"names":[],"path":[],"result_type":null,"sizes":[],"snapshot":null,'
        b'"source":{"execution_seq":13,"path":["n"],"run_id":"historical-growth"},"spans":[],'
        b'"value_json":null},{"children":[],"content_index":0,"count":0,'
        b'"expression":"outer/repeated#0/__start__.output.exited","kind":"source","names":[],"path":[],'
        b'"result_type":null,"sizes":[],"snapshot":null,"source":{"execution_seq":13,"path":["exited"],'
        b'"run_id":"historical-growth"},"spans":[],"value_json":null}],"content_index":0,"count":0,'
        b'"expression":"{n: outer/repeated#0/__start__.output.n,'
        b' exited: outer/repeated#0/__start__.output.exited}","kind":"record","names":["n","exited"],'
        b'"path":[],"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],'
        b'"value_json":null},"scope":{"boundary":null,"flow_path":["outer","repeated"],'
        b'"occurrence_path":[{"index":null,"kind":"call","static_path":["outer"]},{"index":0,'
        b'"kind":"loop_body","static_path":["outer","repeated"]}],"static_path":["outer","repeated",'
        b'"bump"]},"variable_names":[]},"node_id":"outer/repeated#0/bump","value":{"kind":"object",'
        b'"value":{"exited":false,"n":1}}},"run_id":"historical-growth","schema":9,"seq":20}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"outer/mapped#0/echo"},'
        b'"run_id":"historical-growth","schema":9,"seq":21}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"outer/mapped#0/echo","evidence":null,'
        b'"forward_capture":{"bound_inputs":{"x":1},"evaluated_keys":["x"],"input_names":["x"],'
        b'"items":[],"node_data":{},"route":{"children":[{"children":[],"content_index":0,"count":0,'
        b'"expression":"outer/mapped#0/__start__.output.value","kind":"source","names":[],"path":[],'
        b'"result_type":null,"sizes":[],"snapshot":null,"source":{"execution_seq":15,"path":["value"],'
        b'"run_id":"historical-growth"},"spans":[],"value_json":null}],"content_index":0,"count":0,'
        b'"expression":"{x: outer/mapped#0/__start__.output.value}","kind":"record","names":["x"],'
        b'"path":[],"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],'
        b'"value_json":null},"scope":{"boundary":null,"flow_path":["outer","mapped"],'
        b'"occurrence_path":[{"index":null,"kind":"call","static_path":["outer"]},{"index":0,'
        b'"kind":"map_item","static_path":["outer","mapped"]}],"static_path":["outer","mapped","echo"]},'
        b'"variable_names":[]},"node_id":"outer/mapped#0/echo","value":{"kind":"int","value":1}},'
        b'"run_id":"historical-growth","schema":9,"seq":22}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"outer/mapped#1/echo"},'
        b'"run_id":"historical-growth","schema":9,"seq":23}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"outer/mapped#1/echo","evidence":null,'
        b'"forward_capture":{"bound_inputs":{"x":2},"evaluated_keys":["x"],"input_names":["x"],'
        b'"items":[],"node_data":{},"route":{"children":[{"children":[],"content_index":0,"count":0,'
        b'"expression":"outer/mapped#1/__start__.output.value","kind":"source","names":[],"path":[],'
        b'"result_type":null,"sizes":[],"snapshot":null,"source":{"execution_seq":17,"path":["value"],'
        b'"run_id":"historical-growth"},"spans":[],"value_json":null}],"content_index":0,"count":0,'
        b'"expression":"{x: outer/mapped#1/__start__.output.value}","kind":"record","names":["x"],'
        b'"path":[],"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],'
        b'"value_json":null},"scope":{"boundary":null,"flow_path":["outer","mapped"],'
        b'"occurrence_path":[{"index":null,"kind":"call","static_path":["outer"]},{"index":1,'
        b'"kind":"map_item","static_path":["outer","mapped"]}],"static_path":["outer","mapped","echo"]},'
        b'"variable_names":[]},"node_id":"outer/mapped#1/echo","value":{"kind":"int","value":2}},'
        b'"run_id":"historical-growth","schema":9,"seq":24}'
    ),
)


_RUN_ID = "recovery-matrix"
_SYSTEM_TODAY = "2026-08-14"
_SYSTEM_NOW = "2026-08-14T12:00:00+00:00"
_WAIT_DEADLINE = "2026-08-14T12:00:10+00:00"

_NESTED_GROWTH = """
id: recovery-matrix-growth
name: recovery_matrix_growth
defs:
  echo_one:
    input:
      value: int
    nodes:
      echo:
        kind: code
        input:
          x: ${input.value}
        output: int
        code: tests.engine._compose_codefns:echo_x
    output: ${echo.output}
  bump_once:
    input:
      n: int
      exited: bool
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
        code: tests.engine._compose_codefns:loop_bump
    output: ${bump.output}
  composite:
    nodes:
      mapped:
        kind: map
        call: echo_one
        over: ${[1, 2]}
        input:
          value: ${item}
      repeated:
        kind: loop
        call: bump_once
        input:
          n: 0
          exited: false
        times: 2
    output:
      mapped: ${mapped.output}
      repeated: ${repeated.output}
nodes:
  outer:
    kind: call
    call: composite
output: ${outer.output}
"""

_NESTED_INLINE_GROWTH = """
id: recovery-matrix-inline-growth
name: recovery_matrix_inline_growth
nodes:
  outer:
    kind: call
    input:
      items: [1, 2]
    nodes:
      mapped:
        kind: map
        over: ${input.items}
        input:
          value: ${item}
        nodes:
          echo:
            kind: code
            input:
              x: ${input.value}
            output: int
            code: tests.engine._compose_codefns:echo_x
        output: ${echo.output}
      repeated:
        kind: loop
        input:
          n: 0
          exited: false
        times: 2
        nodes:
          bump:
            kind: code
            input:
              n: ${input.n}
              exited: ${input.exited}
            output:
              n: int
              exited: bool
            code: tests.engine._compose_codefns:loop_bump
        output: ${bump.output}
    output:
      mapped: ${mapped.output}
      repeated: ${repeated.output}
output: ${outer.output}
"""


def _recorded_growth_flow():
    """Build the authored program retained by the fixed historical wire sample."""
    source = _NESTED_GROWTH.replace(
        "name: recovery_matrix_growth\n",
        "name: recovery_matrix_growth\nlong_term_memory: episodic\n",
    ).replace(
        "  composite:\n    nodes:",
        "  composite:\n    long_term_memory: reflective\n    nodes:",
    )
    return load_flow(source)


def test_recorded_nested_growth_resumes_with_exact_memory_and_trace(tmp_path, monkeypatch):
    from agent_composer.memory.episodic import EpisodicMemory
    from agent_composer.memory.reflective import ReflectiveMemory
    from agent_composer.nodes.code import CodeNode
    from agent_composer.runtime.log import JsonlRunLog, SCHEMA_VERSION, decode_record, encode_record

    flow = _recorded_growth_flow()
    path = tmp_path / "historical-growth.jsonl"
    prefix = b"\n".join(_RECORDED_GROWTH_PREFIX) + b"\n"
    path.write_bytes(prefix)
    initial = decode_record(_RECORDED_GROWTH_PREFIX[0]).record
    snapshots = {item.policy_name: item.snapshot for item in initial.memory_snapshots}
    assert set(snapshots) == {"episodic", "reflective"}
    for name, snapshot in snapshots.items():
        assert snapshot["format"] == "agent_composer." + name + "_memory"
        assert snapshot["version"] == 1
    memory = {"episodic": EpisodicMemory(), "reflective": ReflectiveMemory()}
    restored = {}
    for name, backend in memory.items():
        original_open = backend.open_run

        def open_run(*, snapshot=None, name=name, original_open=original_open):
            session = original_open(snapshot=snapshot)
            restored[name] = session.snapshot
            return session

        monkeypatch.setattr(backend, "open_run", open_run)
    original_forward = CodeNode.forward
    calls = []

    def forward(node, *args, **kwargs):
        calls.append(node.id)
        assert node.id == "outer/repeated#1/bump", "accepted historical CODE reran"
        return original_forward(node, *args, **kwargs)

    monkeypatch.setattr(CodeNode, "forward", forward)
    result = run_module.resume_flow(flow, log=path, memory=memory)
    assert result.status == "succeeded"
    assert result.output == {"mapped": [1, 2], "repeated": {"n": 2, "exited": False}}
    assert calls == ["outer/repeated#1/bump"]
    assert restored == snapshots
    assert path.read_bytes().startswith(prefix)
    with JsonlRunLog.resume(path) as log:
        envelopes = tuple(log)
        replayed = run_module._replay_run_log(flow, log)
    length = len(_RECORDED_GROWTH_PREFIX)
    assert tuple(encode_record(item) for item in envelopes[:length]) == _RECORDED_GROWTH_PREFIX
    assert all(item.schema == SCHEMA_VERSION for item in envelopes[length:])
    assert replayed.flow_fingerprint == initial.flow_fingerprint == flow_fingerprint(flow.compiled)
    assert replayed.effective_llm_by_static_path == initial.effective_llm_by_static_path
    assert replayed.learnable is True
    historical = [item for item in replayed.trace.occurrences if item.outcome_seq < length]
    assert {item.node_id for item in historical if isinstance(item.record, NodeExpandedRecord)} == {
        "outer", "outer/mapped", "outer/repeated",
    }
    for occurrence in historical:
        assert occurrence._record_bytes == _RECORDED_GROWTH_PREFIX[occurrence.outcome_seq]
    complete = path.read_bytes()
    inspected = run_module.resume_flow(flow, log=path, memory={"episodic": object(), "reflective": object()})
    assert inspected.output == result.output and inspected.events == []
    assert path.read_bytes() == complete and calls == ["outer/repeated#1/bump"]


_NESTED_HUMAN_EFFECTS = {
    "call": """
id: recovery-nested-human-call
name: recovery_nested_human_call
defs:
  gate:
    input:
      token: str
    nodes:
      ask:
        kind: human_input
        prompt: nested call input
        output: str
    output: ${ask.output}
nodes:
  invoke:
    kind: call
    call: gate
    input:
      token: ready
output: ${invoke.output}
""",
    "map": """
id: recovery-nested-human-map
name: recovery_nested_human_map
defs:
  gate:
    input:
      token: str
    nodes:
      ask:
        kind: human_input
        prompt: nested map input
        output: str
    output: ${ask.output}
nodes:
  invoke:
    kind: map
    call: gate
    over: ${["a", "b"]}
    input:
      token: ${item}
output: ${invoke.output}
""",
}

_LOOP_BODY_SPAWNERS = """
id: recovery-loop-body-spawners
name: recovery_loop_body_spawners
defs:
  echo_one:
    input:
      value: int
    nodes:
      echo:
        kind: code
        input:
          x: ${input.value}
        output: int
        code: tests.engine._compose_codefns:echo_x
    output: ${echo.output}
  turn:
    input:
      n: int
      exited: bool
    nodes:
      nested_call:
        kind: call
        call: echo_one
        input:
          value: ${input.n}
      nested_map:
        kind: map
        call: echo_one
        over: ${[1, 2]}
        input:
          value: ${item}
      bump:
        kind: code
        depends_on: [nested_call, nested_map]
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
        code: tests.engine._compose_codefns:loop_bump
    output: ${bump.output}
nodes:
  repeated:
    kind: loop
    call: turn
    input:
      n: 0
      exited: false
    times: 2
output: ${repeated.output}
"""

_EFFECT_FLOWS = {
    "agent": """
id: recovery-agent
name: recovery_agent
nodes:
  effect:
    kind: agent
    mode: plain
    prompt: this model must never be invoked
    llm_config:
      provider: openai
      model: offline-test
output: ${effect.output}
""",
    "human_input": """
id: recovery-human
name: recovery_human
nodes:
  effect:
    kind: human_input
    prompt: this host interaction must never be invoked
    output: str
output: ${effect.output}
""",
    "wait": f"""
id: recovery-wait
name: recovery_wait
nodes:
  effect:
    kind: wait
    until: {_WAIT_DEADLINE!r}
output: ${{effect.output}}
""",
}

_ORDERED_EFFECTS = f"""
id: recovery-ordered-effects
name: recovery_ordered_effects
nodes:
  release:
    kind: code
    input: {{x: ready}}
    output: str
    code: tests.engine._compose_codefns:echo_x
  wait:
    kind: wait
    depends_on: [release]
    until: {_WAIT_DEADLINE!r}
  agent:
    kind: agent
    depends_on: [release]
    mode: plain
    prompt: never invoked
    llm_config: {{provider: openai, model: offline-test}}
  human:
    kind: human_input
    depends_on: [release]
    prompt: never invoked
    output: str
  later:
    kind: code
    depends_on: [release]
    input: {{x: still-ready}}
    output: str
    code: tests.engine._compose_codefns:echo_x
output:
  agent: ${{agent.output}}
  human: ${{human.output}}
  wait: ${{wait.output}}
  later: ${{later.output}}
"""


def _state_snapshot(state: RunState) -> dict[str, Any]:
    """Return every topology and reducer coordinate as detached values."""
    edges = tuple(
        (
            edge.id,
            edge.from_,
            edge.to,
            edge.source_handle,
            edge.input_group,
            edge.optional,
            edge.ordering,
        )
        for edge in state.flow.edges
    )
    return {
        "topology": flow_manifest(state.flow),
        "origin_context": {
            node_id: node.origin_context
            for node_id, node in state.flow.nodes.items()
        },
        "node_order": tuple(state.flow.nodes),
        "edges": edges,
        "wiring": copy.deepcopy(state.flow.wiring),
        "incoming": {
            node_id: tuple(edge.id for edge in state.flow.incoming(node_id))
            for node_id in state.flow.nodes
        },
        "outgoing": {
            node_id: tuple(edge.id for edge in state.flow.outgoing(node_id))
            for node_id in state.flow.nodes
        },
        "pool": encode_pool(state.pool),
        "node_state": dict(state.node_state),
        "edge_state": dict(state.edge_state),
        "ready": tuple(state.ready),
        "in_flight": tuple(state.in_flight),
        "depth": dict(state.depth),
        "wiring_overrides": copy.deepcopy(state.wiring_overrides),
        "effective_llm_by_static_path": copy.deepcopy(
            state.effective_llm_by_static_path
        ),
        "result": copy.deepcopy(state.result),
        "terminal_unreachable_failure": state.terminal_unreachable_failure,
        "next_sequence": state.next_sequence,
        "run_id": state.run_id,
        "flow_fingerprint": state.flow_fingerprint,
        "num_workers": state.num_workers,
        "input": copy.deepcopy(state.input),
        "initial_static_ids": state.initial_static_ids,
    }


def _new_started_run(loaded, *, num_workers: int) -> tuple[RunState, MemoryRunLog]:
    """Create one private explicitly configured run through its first record."""
    private_flow = copy.deepcopy(loaded.compiled)
    effective = resolve_llm_cascade(
        private_flow,
        {},
        defaults={"provider": "openai", "model": "offline-test"},
    )
    pool = VariablePool(start_id=private_flow.start_id)
    pool.set(private_flow.start_id, {})
    pool.add_system("run_id", _RUN_ID)
    pool.add_system("today", _SYSTEM_TODAY)
    pool.add_system("now", _SYSTEM_NOW)

    state = RunState(private_flow)
    log = MemoryRunLog(_RUN_ID)
    commit_prepared(
        log,
        state,
        prepare_run_started(
            state,
            RunStartData(
                run_id=_RUN_ID,
                flow_fingerprint=flow_fingerprint(loaded.compiled),
                initial_pool=pool,
                start_wiring_override={},
                effective_llm_by_static_path=effective,
                num_workers=num_workers,
            ),
        ),
    )
    return state, log


def _prefix_log(log: MemoryRunLog, length: int) -> MemoryRunLog:
    """Copy the first ``length`` authoritative records into an open memory log."""
    prefix = MemoryRunLog(log.run_id)
    for envelope in tuple(log)[:length]:
        prefix.append(envelope.record)
    return prefix


def _pause_after_effect_started(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> tuple[Any, RunState, MemoryRunLog]:
    """Stop a serial dispatcher at the target's post-durable start yield."""
    loaded = load_flow(source)
    state, log = _new_started_run(loaded, num_workers=0)
    dispatcher = FlowEngine(
        state,
        log,
        grow_builder=build_grow,
        num_workers=0,
    )
    drive = dispatcher.run()

    assert next(drive) == NodeStarted(START_ID)
    assert next(drive) == NodeSucceeded(START_ID, output={})

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("effect work ran after only its durable NodeStarted was requested")

    monkeypatch.setattr(dispatcher, "_evaluate", unexpected_evaluation)
    assert next(drive) == NodeStarted("effect")
    assert tuple(log)[-1].record == NodeStartedRecord(node_id="effect")
    drive.close()
    return loaded, state, log


@pytest.mark.parametrize(
    "source",
    [_NESTED_GROWTH, _NESTED_INLINE_GROWTH],
    ids=["named", "inline"],
)
@pytest.mark.parametrize("num_workers", [0, 2], ids=["serial", "pooled"])
def test_every_nested_growth_prefix_replays_to_the_live_state(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    num_workers: int,
) -> None:
    """Named and inline CALL/MAP/LOOP replay identically after every commit."""
    loaded = load_flow(source)
    state, log = _new_started_run(loaded, num_workers=num_workers)
    snapshots = [_state_snapshot(state)]
    original_commit = engine_module.commit_prepared

    def capture_after_commit(cursor, live_state, prepared):
        observers = original_commit(cursor, live_state, prepared)
        assert live_state is state
        snapshots.append(_state_snapshot(live_state))
        return observers

    monkeypatch.setattr(engine_module, "commit_prepared", capture_after_commit)
    observers = list(
        FlowEngine(
            state,
            log,
            grow_builder=build_grow,
            num_workers=num_workers,
        ).run()
    )

    assert isinstance(state.result, RunSucceeded)
    assert state.result.output == {
        "mapped": [1, 2],
        "repeated": {"n": 2, "exited": False},
    }
    assert observers[-1] == state.result
    records = tuple(log)
    assert len(snapshots) == len(records)

    for length, expected in enumerate(snapshots, start=1):
        recovered = run_module._replay_run_log(loaded, _prefix_log(log, length))
        assert _state_snapshot(recovered) == expected, (
            f"replayed state diverged after authoritative prefix {length}/{len(records)}"
        )


def test_every_loop_body_call_and_map_prefix_replays_to_the_live_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CALL and MAP nested inside each grown LOOP turn replay after every commit."""
    loaded = load_flow(_LOOP_BODY_SPAWNERS)
    state, log = _new_started_run(loaded, num_workers=0)
    snapshots = [_state_snapshot(state)]
    original_commit = engine_module.commit_prepared

    def capture_after_commit(cursor, live_state, prepared):
        observers = original_commit(cursor, live_state, prepared)
        assert live_state is state
        snapshots.append(_state_snapshot(live_state))
        return observers

    monkeypatch.setattr(engine_module, "commit_prepared", capture_after_commit)
    observers = list(
        FlowEngine(
            state,
            log,
            grow_builder=build_grow,
            num_workers=0,
        ).run()
    )

    assert isinstance(observers[-1], RunSucceeded)
    assert observers[-1].output == {"n": 2, "exited": False}
    expanded_ids = tuple(
        envelope.record.spawner_id
        for envelope in log
        if isinstance(envelope.record, NodeExpandedRecord)
    )
    assert any(node_id.endswith("/nested_call") for node_id in expanded_ids)
    assert any(node_id.endswith("/nested_map") for node_id in expanded_ids)
    records = tuple(log)
    assert len(snapshots) == len(records)

    for length, expected in enumerate(snapshots, start=1):
        recovered = run_module._replay_run_log(loaded, _prefix_log(log, length))
        assert _state_snapshot(recovered) == expected, (
            f"replayed state diverged after loop-body prefix {length}/{len(records)}"
        )


@pytest.mark.parametrize(
    "source",
    tuple(_EFFECT_FLOWS.values()),
    ids=tuple(_EFFECT_FLOWS),
)
def test_effect_start_prefix_replays_without_running_effect_work(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """AGENT, HUMAN_INPUT, and WAIT starts remain recoverable before execution."""
    loaded, live, log = _pause_after_effect_started(monkeypatch, source)
    recovered = run_module._replay_run_log(loaded, log)

    assert _state_snapshot(recovered) == _state_snapshot(live)
    assert tuple(recovered.in_flight) == ("effect",)
    assert tuple(recovered.ready) == ()

    prepared = prepare_run_resumed(recovered, num_workers=0)
    assert prepared.record == RunResumedRecord(
        retry_node_ids=("effect",),
        num_workers=0,
    )
    commit_prepared(log, recovered, prepared)
    assert tuple(recovered.in_flight) == ()
    assert tuple(recovered.ready) == ("effect",)


@pytest.mark.parametrize(
    "source",
    tuple(_NESTED_HUMAN_EFFECTS.values()),
    ids=tuple(_NESTED_HUMAN_EFFECTS),
)
def test_namespaced_human_input_start_replays_without_host_interaction(
    source: str,
) -> None:
    """Unfinished HUMAN_INPUT work under grown CALL/MAP keeps its live namespace."""
    loaded = load_flow(source)
    state, log = _new_started_run(loaded, num_workers=0)
    drive = FlowEngine(
        state,
        log,
        grow_builder=build_grow,
        num_workers=0,
    ).run()
    effect_id = None
    try:
        for observer in drive:
            if not isinstance(observer, NodeStarted):
                continue
            node = state.flow.nodes[observer.node_id]
            if node.kind is NodeKind.HUMAN_INPUT:
                effect_id = observer.node_id
                break
    finally:
        drive.close()

    assert effect_id is not None
    assert "/ask" in effect_id
    assert tuple(state.in_flight) == (effect_id,)
    assert tuple(log)[-1].record == NodeStartedRecord(node_id=effect_id)

    recovered = run_module._replay_run_log(loaded, log)
    assert _state_snapshot(recovered) == _state_snapshot(state)
    assert tuple(recovered.in_flight) == (effect_id,)

    prepared = prepare_run_resumed(recovered, num_workers=0)
    assert prepared.record == RunResumedRecord(
        retry_node_ids=(effect_id,),
        num_workers=0,
    )


def test_resume_prepends_ordered_effect_retries_before_existing_ready_work() -> None:
    """Recovery preserves nonlexicographic starts ahead of untouched readiness."""
    loaded = load_flow(_ORDERED_EFFECTS)
    state, log = _new_started_run(loaded, num_workers=2)
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(log, state, prepare_node_succeeded(state, START_ID, {}))
    commit_prepared(log, state, prepare_node_started(state, "release"))
    commit_prepared(log, state, prepare_node_succeeded(state, "release", "ready"))
    assert tuple(state.ready) == ("wait", "agent", "human", "later")

    retry_order = ("wait", "agent", "human")
    assert retry_order != tuple(sorted(retry_order))
    for node_id in retry_order:
        commit_prepared(log, state, prepare_node_started(state, node_id))
    started_order = tuple(
        envelope.record.node_id
        for envelope in log
        if isinstance(envelope.record, NodeStartedRecord)
    )
    assert started_order == (START_ID, "release", *retry_order)
    assert tuple(state.in_flight) == retry_order
    assert tuple(state.ready) == ("later",)

    recovered = run_module._replay_run_log(loaded, log)
    assert tuple(recovered.in_flight) == retry_order
    assert tuple(recovered.ready) == ("later",)

    prepared = prepare_run_resumed(recovered, num_workers=2)
    assert prepared.record == RunResumedRecord(
        retry_node_ids=retry_order,
        num_workers=2,
    )
    commit_prepared(log, recovered, prepared)
    assert tuple(log)[-1].record == prepared.record
    assert tuple(recovered.in_flight) == ()
    assert tuple(recovered.ready) == (*retry_order, "later")


class _ImmediateTimeoutEvent:
    """Record requested timeout durations while returning without sleeping."""

    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    def is_set(self) -> bool:
        """Report an active run before evaluation begins."""
        return False

    def wait(self, timeout: float | None = None) -> bool:
        """Capture one absolute-deadline remainder and simulate its timeout."""
        self.timeouts.append(timeout)
        return False


def test_recovered_wait_recomputes_from_the_authored_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WAIT recovery retains the deadline and derives a new remaining duration."""
    loaded, _, log = _pause_after_effect_started(
        monkeypatch,
        _EFFECT_FLOWS["wait"],
    )
    recovered = run_module._replay_run_log(loaded, log)
    authored = loaded.compiled.wiring["effect"]["until"]
    replayed = recovered.flow.wiring["effect"]["until"]
    assert replayed == authored
    assert replayed.source == _WAIT_DEADLINE
    assert replayed.origin is not None

    prepared = prepare_run_resumed(recovered, num_workers=0)
    commit_prepared(log, recovered, prepared)
    node = recovered.flow.nodes["effect"]

    observed_timeouts = []
    for second, expected in ((4, 6.0), (7, 3.0)):
        event = _ImmediateTimeoutEvent()
        monkeypatch.setattr(
            wait_module,
            "_utc_now",
            lambda second=second: datetime(
                2026,
                8,
                14,
                12,
                0,
                second,
                tzinfo=timezone.utc,
            ),
        )
        outcome = eval_node(
            node,
            recovered.flow,
            recovered.pool,
            stop_event=event,
        ).outcome
        assert outcome == Output(None)
        assert event.timeouts == [expected]
        observed_timeouts.append(event.timeouts[0])

    assert observed_timeouts == [6.0, 3.0]
