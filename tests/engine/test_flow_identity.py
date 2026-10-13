"""Stable authored-flow manifests and deterministic expansion-topology digests."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, replace

import pytest
from pydantic import BaseModel, ConfigDict

import yian.compile.manifest as manifest_module
import yian.typesys.values as values_module
from yian.compile.manifest import expansion_digest, flow_fingerprint, flow_manifest
from yian.compile.memory_layout import apply_memory_layout, resolve_memory_layout
from yian.compile.model import START_ID, CompiledFlow, Edge, Flow, FlowOutput
from yian.compose import load_flow
from yian.diagnostics import OriginRef
from yian.memory import (
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryScope,
)
from yian.nodes.agent import AgentNode
from yian.nodes.agent.structured import type_to_schema
from yian.nodes.call import CallNode
from yian.nodes.case import Case, CaseNode
from yian.nodes.code import CodeNode
from yian.nodes.contracts import (
    Assertion,
    Binding,
    ContractOrigins,
    Parameter,
    ValueContract,
)
from yian.nodes.end import EndNode
from yian.nodes.human_input import HumanInputNode
from yian.nodes.loop import LoopNode
from yian.nodes.map import MapNode
from yian.nodes.start import StartNode
from yian.nodes.tool import ToolNode
from yian.nodes.wait import WaitNode
from yian.typesys.values import Type, ValueKind


_INLINE_FLOW = """
id: stable
name: Stable
input:
  x: int = 1
env:
  beta: 2
  alpha: 1
nodes:
  step:
    kind: code
    input:
      x: ${input.x}
    output: int
    code: |
      return inputs["x"] + 1
    asserts:
      - x >= 0
      - x < 100
      - output > x
output: ${step.output}
"""


_INLINE_FLOW_WITH_COMMENTS = """
# This comment and the blank lines move every source range and inline-code frame.

id: stable
name: Stable
input:
  x: int = 1
env:
  alpha: 1
  beta: 2  # mapping presentation is not program identity
nodes:
  step:
    kind: code
    input:
      x: ${input.x}
    output: int
    code: |
      return inputs["x"] + 1
    asserts:
      - x >= 0
      - x < 100
      - output > x
output: ${step.output}
"""


def test_loaded_flow_stores_only_exact_parameter_carriers():
    """Compose builders and synthesized boundaries share one compiled parameter shape."""
    loaded = load_flow(_INLINE_FLOW)

    for node in loaded.compiled.nodes.values():
        if node.params is not None:
            assert type(node.params) is list
            assert all(type(parameter) is Parameter for parameter in node.params)


_INLINE_COMPOSITE_IDENTITY = """
id: inline-identity
name: inline_identity
input:
  value: int
nodes:
  wrapper:
    kind: call
    input:
      value: ${input.value}
    nodes:
      step:
        kind: code
        input:
          value: ${input.value}
        output: int
        code: |
          return inputs["value"] + 1
    output: ${step.output}
output: ${wrapper.output}
"""


def _compiled(source: str = _INLINE_FLOW):
    """Load one source string and return its representation-neutral compiled flow."""
    return load_flow(source).compiled


def _typed_mapping_value(mapping: dict, key: str):
    """Return one string-keyed value from the manifest's typed mapping form."""
    canonical_key = {"type": "str", "value": key}
    matches = [
        entry["value"]
        for entry in mapping["entries"]
        if entry["key"] == canonical_key
    ]
    assert len(matches) == 1
    return matches[0]


def test_flow_fingerprint_is_stable_across_reloads():
    first = _compiled()
    second = _compiled()

    assert first is not second
    assert first.nodes["step"] is not second.nodes["step"]
    assert first.nodes["step"]._code is not second.nodes["step"]._code
    assert flow_manifest(first) == flow_manifest(second)
    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert re.fullmatch(r"[0-9a-f]{64}", flow_fingerprint(first))


def test_flow_fingerprint_ignores_comments_and_python_object_identity():
    plain = _compiled()
    moved = _compiled(_INLINE_FLOW_WITH_COMMENTS)

    assert plain.nodes["step"]._code is not moved.nodes["step"]._code
    assert flow_manifest(plain) == flow_manifest(moved)
    assert flow_fingerprint(plain) == flow_fingerprint(moved)


def test_inline_identity_ignores_formatting_but_tracks_owner_and_child_semantics():
    baseline = load_flow(_INLINE_COMPOSITE_IDENTITY).compiled
    formatted = load_flow(
        "# moves every authored range without changing the program\n\n"
        + _INLINE_COMPOSITE_IDENTITY
    ).compiled
    changed_child = load_flow(
        _INLINE_COMPOSITE_IDENTITY.replace(
            'return inputs["value"] + 1',
            'return inputs["value"] + 2',
        )
    ).compiled
    renamed_owner = load_flow(
        _INLINE_COMPOSITE_IDENTITY.replace("wrapper:", "renamed:").replace(
            "${wrapper.output}",
            "${renamed.output}",
        )
    ).compiled

    baseline_id = baseline.nodes["wrapper"].flow_id
    assert formatted.nodes["wrapper"].flow_id == baseline_id
    assert changed_child.nodes["wrapper"].flow_id == baseline_id
    assert renamed_owner.nodes["renamed"].flow_id != baseline_id

    assert flow_manifest(formatted) == flow_manifest(baseline)
    assert flow_fingerprint(formatted) == flow_fingerprint(baseline)
    assert flow_fingerprint(changed_child) != flow_fingerprint(baseline)
    assert flow_fingerprint(renamed_owner) != flow_fingerprint(baseline)


def test_inline_manifest_canonicalizes_generated_structural_input_types():
    compiled = load_flow(
        """id: inline-manifest-structural-input
name: inline_manifest_structural_input
typedefs:
  Row:
    z: int
    a: Optional[str]
input:
  row: Row
nodes:
  wrapper:
    kind: call
    input:
      row: ${input.row}
    nodes:
      echo:
        kind: code
        code: |
          return inputs["row"]
        input:
          row: ${input.row}
        output: Row
    output: ${echo.output}
output: ${wrapper.output}
"""
    ).compiled

    wrapper_manifest = flow_manifest(compiled)["nodes"]["wrapper"]
    child_start = wrapper_manifest["child"]["nodes"][START_ID]
    generated = child_start["params"][0]

    assert [
        field["name"] for field in generated["contract"]["type"]["fields"]
    ] == ["a", "z"]


def test_flow_fingerprint_changes_with_node_kind_and_execution_source():
    code = _compiled(
        """
id: kind
name: Kind
nodes:
  leaf:
    kind: code
    output: str
    code: |
      return "one"
output: ${leaf.output}
"""
    )
    tool = _compiled(
        """
id: kind
name: Kind
nodes:
  leaf:
    kind: tool
    tool_id: one
output: ${leaf.output}
"""
    )
    changed_code = _compiled(
        """
id: kind
name: Kind
nodes:
  leaf:
    kind: code
    output: str
    code: |
      return "two"
output: ${leaf.output}
"""
    )

    assert flow_fingerprint(code) != flow_fingerprint(tool)
    assert flow_fingerprint(code) != flow_fingerprint(changed_code)


def test_flow_fingerprint_preserves_semantic_edge_order():
    original = _compiled()
    reordered = CompiledFlow.from_parts(
        nodes=copy.deepcopy(original.nodes),
        edges=list(reversed(copy.deepcopy(original.edges))),
        outputs=copy.deepcopy(original.outputs),
        wiring=copy.deepcopy(original.wiring),
        flow_llm_config=copy.deepcopy(original.flow_llm_config),
    )

    assert flow_fingerprint(original) != flow_fingerprint(reordered)


def test_flow_fingerprint_changes_with_wiring_types_asserts_and_env():
    original = _compiled()

    literal_one = _compiled(_INLINE_FLOW.replace("x: ${input.x}", "x: 7"))
    literal_two = _compiled(_INLINE_FLOW.replace("x: ${input.x}", "x: 8"))

    output_type = _compiled(_INLINE_FLOW.replace("output: int", "output: float"))

    input_type = _compiled(_INLINE_FLOW.replace("x: int = 1", "x: float = 1"))

    asserts = copy.deepcopy(original)
    asserts.nodes["step"].pre_asserts = list(reversed(asserts.nodes["step"].pre_asserts))

    env = copy.deepcopy(original)
    env.nodes["step"].env["alpha"] = 99

    title = copy.deepcopy(original)
    title.nodes["step"].title = "A different title"

    baseline = flow_fingerprint(original)
    assert flow_fingerprint(literal_one) != flow_fingerprint(literal_two)
    assert flow_fingerprint(output_type) != baseline
    assert flow_fingerprint(input_type) != baseline
    assert flow_fingerprint(asserts) != baseline
    assert flow_fingerprint(env) != baseline
    assert flow_fingerprint(title) == baseline


def test_flow_fingerprint_includes_authored_llm_layers_but_not_effective_values():
    original = _compiled(
        """
id: llm
name: LLM
llm_config:
  provider: openai
  temperature: 0.2
nodes:
  ask:
    kind: agent
    prompt: Answer briefly.
    output: str
    llm_config:
      model: gpt-5.5
      inherit: false
output: ${ask.output}
"""
    )
    baseline = flow_fingerprint(original)

    flow_layer = copy.deepcopy(original)
    flow_layer.flow_llm_config["temperature"] = 0.7

    own_layer = copy.deepcopy(original)
    own_layer.nodes["ask"].own_llm_config["model"] = "gpt-5.5-pro"

    inherit = copy.deepcopy(original)
    inherit.nodes["ask"].llm_inherit = True

    effective = copy.deepcopy(original)
    effective.nodes["ask"].llm_config = {
        "provider": "anthropic",
        "model": "claude-opus-4-8",
    }

    assert flow_fingerprint(flow_layer) != baseline
    assert flow_fingerprint(own_layer) != baseline
    assert flow_fingerprint(inherit) != baseline
    assert flow_fingerprint(effective) == baseline


def test_flow_fingerprint_excludes_config_credentials():
    first = _compiled()
    second = copy.deepcopy(first)
    first.flow_llm_config["api_key"] = "first-secret"
    second.flow_llm_config["api_key"] = "second-secret"
    first.nodes["step"].env["access_token"] = "first-token"
    second.nodes["step"].env["access_token"] = "second-token"
    first.nodes["step"].env["auth"] = {"client_secret": "first-client-secret"}
    second.nodes["step"].env["auth"] = {"client_secret": "second-client-secret"}
    first.nodes["step"].env["nested_auth"] = [
        {"token": "first-generic-token"},
        {"hf_token": "first-hf-token"},
        {"aws_session_token": "first-session-token"},
        {"private_key": "first-private-key"},
        {"aws_secret_access_key": "first-secret-access-key"},
    ]
    second.nodes["step"].env["nested_auth"] = [
        {"token": "second-generic-token"},
        {"hf_token": "second-hf-token"},
        {"aws_session_token": "second-session-token"},
        {"private_key": "second-private-key"},
        {"aws_secret_access_key": "second-secret-access-key"},
    ]
    first.nodes["step"].env["credentials"] = {"secret_key": "first-secret-key"}
    second.nodes["step"].env["credentials"] = {"secret_key": "second-secret-key"}
    first.nodes["step"].env["AZURE_SERVICE_BUS_SHARED_ACCESS_KEY"] = (
        "first-shared-access-key"
    )
    second.nodes["step"].env["AZURE_SERVICE_BUS_SHARED_ACCESS_KEY"] = (
        "second-shared-access-key"
    )
    first.nodes["step"].env["AZURE_SERVICE_BUS_SHARED_ACCESS_SIGNATURE"] = (
        "first-shared-access-signature"
    )
    second.nodes["step"].env["AZURE_SERVICE_BUS_SHARED_ACCESS_SIGNATURE"] = (
        "second-shared-access-signature"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    assert "first-secret" not in manifest_text
    assert "first-token" not in manifest_text
    assert "first-client-secret" not in manifest_text
    assert "first-generic-token" not in manifest_text
    assert "first-hf-token" not in manifest_text
    assert "first-session-token" not in manifest_text
    assert "first-private-key" not in manifest_text
    assert "first-secret-access-key" not in manifest_text
    assert "first-secret-key" not in manifest_text
    assert "first-shared-access-key" not in manifest_text
    assert "first-shared-access-signature" not in manifest_text


def test_flow_fingerprint_excludes_webhook_bearer_urls():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["SLACK_WEBHOOK_URL"] = (
        "https://hooks.slack.com/services/T000/B000/FIRST_SECRET"
    )
    second.nodes["step"].env["SLACK_WEBHOOK_URL"] = (
        "https://hooks.slack.com/services/T000/B000/SECOND_SECRET"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert "FIRST_SECRET" not in json.dumps(flow_manifest(first), sort_keys=True)


@pytest.mark.parametrize(
    "credential_key",
    ["AZURE_SUBSCRIPTION_KEY", "MINIO_ACCESS_KEY", "SESSION_COOKIE"],
)
def test_flow_fingerprint_excludes_standard_flat_credentials(credential_key):
    first = _compiled()
    rotated = copy.deepcopy(first)
    first.nodes["step"].env[credential_key] = "first-credential"
    rotated.nodes["step"].env[credential_key] = "second-credential"

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    assert "first-credential" not in json.dumps(flow_manifest(first), sort_keys=True)


def test_flow_fingerprint_keeps_structured_auth_semantics_without_credentials():
    first = _compiled()
    rotated = copy.deepcopy(first)
    first.nodes["step"].env["auth"] = {
        "type": "basic",
        "endpoint": "https://login.example/token",
        "username": "alice",
        "password": "first-password",
    }
    rotated.nodes["step"].env["auth"] = {
        "type": "basic",
        "endpoint": "https://login.example/token",
        "username": "bob",
        "password": "second-password",
    }

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    assert "basic" in manifest_text
    assert "login.example" in manifest_text
    for credential in ("alice", "first-password"):
        assert credential not in manifest_text

    for field, value in (
        ("type", "digest"),
        ("endpoint", "https://other.example/token"),
    ):
        changed = copy.deepcopy(rotated)
        changed.nodes["step"].env["auth"][field] = value
        assert flow_fingerprint(first) != flow_fingerprint(changed)


def test_flow_fingerprint_excludes_scalar_auth():
    first = _compiled()
    rotated = copy.deepcopy(first)
    first.nodes["step"].env["auth"] = "Bearer first-secret"
    rotated.nodes["step"].env["auth"] = "Bearer second-secret"

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    assert "first-secret" not in json.dumps(flow_manifest(first), sort_keys=True)


def test_flow_fingerprint_excludes_credential_declaration_defaults():
    source = """
id: input-credentials
name: Input credentials
input:
  api_key: str = __SECRET__
nodes:
  step:
    kind: code
    output: str
    code: |
      return "ok"
output: ${step.output}
"""
    first = _compiled(source.replace("__SECRET__", "secret-a"))
    rotated = _compiled(source.replace("__SECRET__", "secret-b"))

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    assert "secret-a" not in json.dumps(flow_manifest(first), sort_keys=True)


def test_flow_fingerprint_redacts_credentials_in_programmatic_parameter_defaults():
    first = _compiled()
    rotated = copy.deepcopy(first)
    changed = copy.deepcopy(first)
    first.nodes["step"].params = [
        Parameter(
            name="config",
            has_default=True,
            default={"region": "us", "auth": {"token": "first-secret"}},
        )
    ]
    rotated.nodes["step"].params = [
        Parameter(
            name="config",
            has_default=True,
            default={"region": "us", "auth": {"token": "second-secret"}},
        )
    ]
    changed.nodes["step"].params = [
        Parameter(
            name="config",
            has_default=True,
            default={"region": "eu", "auth": {"token": "second-secret"}},
        )
    ]

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    assert flow_fingerprint(first) != flow_fingerprint(changed)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    assert "first-secret" not in manifest_text
    assert "us" in manifest_text


def test_flow_fingerprint_redacts_uri_credentials_in_declaration_defaults():
    source = """
id: input-endpoint
name: Input endpoint
input:
  endpoint: str = __ENDPOINT__
nodes:
  step:
    kind: code
    output: str
    code: |
      return "ok"
output: ${step.output}
"""
    first_uri = "https://alice:first-secret@api.example/v1?region=us&key=first-key"
    second_uri = "https://bob:second-secret@api.example/v1?region=us&key=second-key"
    first = _compiled(source.replace("__ENDPOINT__", first_uri))
    rotated = _compiled(source.replace("__ENDPOINT__", second_uri))

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice", "first-secret", "first-key"):
        assert credential not in manifest_text

    changed_endpoint = _compiled(
        source.replace(
            "__ENDPOINT__", second_uri.replace("api.example", "other.example")
        )
    )
    assert flow_fingerprint(first) != flow_fingerprint(changed_endpoint)


@pytest.mark.parametrize(
    "declaration_name",
    ["connection_string", "warehouse_connection_string"],
)
def test_connection_string_declaration_default_keeps_sanitized_identity(
    declaration_name,
):
    source = f"""
id: input-connection
name: Input connection
input:
  {declaration_name}: str = __CONNECTION__
nodes:
  step:
    kind: code
    output: str
    code: |
      return "ok"
output: ${{step.output}}
"""
    first_value = "postgres://alice:first@db.example:5432/app?sslmode=require"
    second_value = "postgres://bob:second@db.example:5432/app?sslmode=require"
    first = _compiled(source.replace("__CONNECTION__", first_value))
    rotated = _compiled(source.replace("__CONNECTION__", second_value))

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice", "first"):
        assert credential not in manifest_text

    for old, new in (
        ("db.example", "other.example"),
        (":5432", ":5433"),
        ("/app?", "/other?"),
        ("sslmode=require", "sslmode=verify-full"),
    ):
        changed = _compiled(
            source.replace("__CONNECTION__", second_value.replace(old, new))
        )
        assert flow_fingerprint(first) != flow_fingerprint(changed)


def test_connection_string_declaration_default_excludes_unrecognized_scalar():
    source = """
id: input-connection
name: Input connection
input:
  connection_string: str = __CONNECTION__
nodes:
  step:
    kind: code
    output: str
    code: |
      return "ok"
output: ${step.output}
"""
    first = _compiled(source.replace("__CONNECTION__", "first-raw-secret"))
    rotated = _compiled(source.replace("__CONNECTION__", "second-raw-secret"))

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    assert "first-raw-secret" not in json.dumps(flow_manifest(first), sort_keys=True)


def test_flow_fingerprint_redacts_uri_credentials_in_node_env():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["DATABASE_URL"] = (
        "postgres://alice:first-password@db.example:5432/app"
        "?sslmode=require&api_key=first-api-key&token=first-token"
        "&clientSecret=first-client-secret"
    )
    second.nodes["step"].env["DATABASE_URL"] = (
        "postgres://bob:second-password@db.example:5432/app"
        "?sslmode=require&api_key=second-api-key&token=second-token"
        "&clientSecret=second-client-secret"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in (
        "alice",
        "first-password",
        "first-api-key",
        "first-token",
        "first-client-secret",
    ):
        assert credential not in manifest_text


def test_flow_fingerprint_redacts_credentials_in_jdbc_wrapper_uri():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["DATABASE_URL"] = (
        "jdbc:postgresql://alice:first-password@db.example:5432/app"
        "?sslmode=require&password=first-query-password&application_name=worker"
    )
    second.nodes["step"].env["DATABASE_URL"] = (
        "jdbc:postgresql://bob:second-password@db.example:5432/app"
        "?sslmode=require&password=second-query-password&application_name=worker"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice", "first-password", "first-query-password"):
        assert credential not in manifest_text

    for changed_uri in (
        second.nodes["step"].env["DATABASE_URL"].replace(
            "db.example", "other.example"
        ),
        second.nodes["step"].env["DATABASE_URL"].replace("/app?", "/other?"),
        second.nodes["step"].env["DATABASE_URL"].replace(
            "application_name=worker", "application_name=scheduler"
        ),
    ):
        changed = copy.deepcopy(second)
        changed.nodes["step"].env["DATABASE_URL"] = changed_uri
        assert flow_fingerprint(first) != flow_fingerprint(changed)


def test_flow_fingerprint_redacts_jdbc_sqlserver_semicolon_credentials():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["DATABASE_URL"] = (
        "jdbc:sqlserver://db.example:1433;databaseName=app;user=alice;"
        "password={first;password};encrypt=true"
    )
    second.nodes["step"].env["DATABASE_URL"] = (
        "jdbc:sqlserver://db.example:1433;databaseName=app;user=bob;"
        "password={second;password};encrypt=true"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice", "first;password"):
        assert credential not in manifest_text

    for old, new in (
        ("db.example", "other.example"),
        (":1433", ":1434"),
        ("databaseName=app", "databaseName=other"),
        ("encrypt=true", "encrypt=false"),
    ):
        changed = copy.deepcopy(second)
        changed.nodes["step"].env["DATABASE_URL"] = changed.nodes["step"].env[
            "DATABASE_URL"
        ].replace(old, new)
        assert flow_fingerprint(first) != flow_fingerprint(changed)


@pytest.mark.parametrize(
    ("first_url", "second_url", "semantic_change"),
    [
        (
            "jdbc:oracle:thin:alice/first-password@//db.example:1521/appservice",
            "jdbc:oracle:thin:bob/second-password@//db.example:1521/appservice",
            "jdbc:oracle:thin:bob/second-password@//db.example:1521/otherservice",
        ),
        (
            "jdbc:oracle:thin:alice/first-password@db.example:1521:APPSID",
            "jdbc:oracle:thin:bob/second-password@db.example:1521:APPSID",
            "jdbc:oracle:thin:bob/second-password@db.example:1521:OTHERSID",
        ),
    ],
    ids=["service", "sid"],
)
def test_flow_fingerprint_redacts_oracle_thin_inline_credentials(
    first_url, second_url, semantic_change
):
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["DATABASE_URL"] = first_url
    second.nodes["step"].env["DATABASE_URL"] = second_url

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice", "first-password"):
        assert credential not in manifest_text

    changed = copy.deepcopy(second)
    changed.nodes["step"].env["DATABASE_URL"] = semantic_change
    assert flow_fingerprint(first) != flow_fingerprint(changed)


@pytest.mark.parametrize(
    ("first_url", "second_url"),
    [
        (
            "https://api.example/v1?region=us&user=alice",
            "https://api.example/v1?region=us&user=bob",
        ),
        (
            "jdbc:postgresql://db.example:5432/app?sslmode=require&username=alice",
            "jdbc:postgresql://db.example:5432/app?sslmode=require&username=bob",
        ),
    ],
    ids=["uri-user", "jdbc-username"],
)
def test_flow_fingerprint_redacts_uri_user_query_parameters(first_url, second_url):
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["ENDPOINT"] = first_url
    second.nodes["step"].env["ENDPOINT"] = second_url

    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert "alice" not in json.dumps(flow_manifest(first), sort_keys=True)


@pytest.mark.parametrize(
    ("key", "first_value", "second_value", "semantic_change"),
    [
        (
            "connection_string",
            "postgres://alice:first@db.example:5432/app?sslmode=require",
            "postgres://bob:second@db.example:5432/app?sslmode=require",
            "postgres://bob:second@other.example:5433/other?sslmode=require",
        ),
        (
            "AZURE_STORAGE_CONNECTION_STRING",
            "Server=db.example;Port=1433;Database=app;User Id=alice;"
            "Password={first;password};Encrypt=true",
            "Server=db.example;Port=1433;Database=app;User Id=bob;"
            "Password={second;password};Encrypt=true",
            "Server=other.example;Port=1434;Database=other;User Id=bob;"
            "Password={second;password};Encrypt=false",
        ),
        (
            "DATABASE_CONNECTION_STRING",
            "jdbc:sqlserver://db.example:1433;databaseName=app;user=alice;"
            "password=first;encrypt=true",
            "jdbc:sqlserver://db.example:1433;databaseName=app;user=bob;"
            "password=second;encrypt=true",
            "jdbc:sqlserver://other.example:1434;databaseName=other;user=bob;"
            "password=second;encrypt=false",
        ),
    ],
    ids=["uri", "semicolon-dsn", "jdbc-sqlserver"],
)
def test_connection_string_keys_keep_sanitized_endpoint_identity(
    key, first_value, second_value, semantic_change
):
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env[key] = first_value
    second.nodes["step"].env[key] = second_value

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice", "first", "first;password"):
        assert credential not in manifest_text

    changed = copy.deepcopy(second)
    changed.nodes["step"].env[key] = semantic_change
    assert flow_fingerprint(first) != flow_fingerprint(changed)


@pytest.mark.parametrize("quote", ['"', "'"], ids=["double", "single"])
def test_connection_string_key_redacts_quoted_semicolon_credentials(quote):
    first = _compiled()
    rotated = copy.deepcopy(first)
    first.nodes["step"].env["DATABASE_CONNECTION_STRING"] = (
        "Server=db.example;Database=app;UserID=alice;"
        f"Password={quote}first;secret{quote};Encrypt=true"
    )
    rotated.nodes["step"].env["DATABASE_CONNECTION_STRING"] = (
        "Server=db.example;Database=app;UserID=bob;"
        f"Password={quote}second;secret{quote};Encrypt=true"
    )

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice", "first;secret"):
        assert credential not in manifest_text

    for old, new in (
        ("Server=db.example", "Server=other.example"),
        ("Database=app", "Database=other"),
        ("Encrypt=true", "Encrypt=false"),
    ):
        changed = copy.deepcopy(rotated)
        changed.nodes["step"].env["DATABASE_CONNECTION_STRING"] = changed.nodes[
            "step"
        ].env["DATABASE_CONNECTION_STRING"].replace(old, new)
        assert flow_fingerprint(first) != flow_fingerprint(changed)


def test_unparsed_semicolon_connection_string_is_excluded_safely():
    first = _compiled()
    rotated = copy.deepcopy(first)
    first.nodes["step"].env["DATABASE_CONNECTION_STRING"] = (
        'Server=db.example;Password="first;secret'
    )
    rotated.nodes["step"].env["DATABASE_CONNECTION_STRING"] = (
        'Server=db.example;Password="second;secret'
    )

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    assert "first;secret" not in json.dumps(flow_manifest(first), sort_keys=True)


def test_connection_string_key_keeps_structured_endpoint_identity():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["connection_string"] = {
        "server": "db.example",
        "port": 5432,
        "database": "app",
        "user": "alice",
        "password": "first-password",
        "sslmode": "require",
    }
    second.nodes["step"].env["connection_string"] = {
        "server": "db.example",
        "port": 5432,
        "database": "app",
        "user": "bob",
        "password": "second-password",
        "sslmode": "require",
    }

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice", "first-password"):
        assert credential not in manifest_text

    changed = copy.deepcopy(second)
    changed.nodes["step"].env["connection_string"]["database"] = "other"
    assert flow_fingerprint(first) != flow_fingerprint(changed)


def test_connection_string_key_keeps_sanitized_libpq_conninfo_identity():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["POSTGRES_CONNECTION_STRING"] = (
        "host='db example' port=5432 dbname=app user='alice role' "
        "password='first\\' secret' sslpassword='first ssl secret' "
        "sslmode=require application_name=worker\\ one"
    )
    second.nodes["step"].env["POSTGRES_CONNECTION_STRING"] = (
        "host='db example' port=5432 dbname=app user='bob role' "
        "password='second\\' secret' sslpassword='second ssl secret' "
        "sslmode=require application_name=worker\\ one"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice role", "first\\' secret", "first ssl secret"):
        assert credential not in manifest_text

    for old, new in (
        ("host='db example'", "host='other example'"),
        ("port=5432", "port=5433"),
        ("dbname=app", "dbname=other"),
        ("sslmode=require", "sslmode=verify-full"),
        ("application_name=worker\\ one", "application_name=scheduler\\ one"),
    ):
        changed = copy.deepcopy(second)
        changed.nodes["step"].env["POSTGRES_CONNECTION_STRING"] = changed.nodes[
            "step"
        ].env["POSTGRES_CONNECTION_STRING"].replace(old, new)
        assert flow_fingerprint(first) != flow_fingerprint(changed)


@pytest.mark.parametrize(
    ("first_password", "second_password"),
    [
        ("first;secret", "second;secret"),
        ("'first;secret'", "'second;secret'"),
        (r"first\;secret", r"second\;secret"),
    ],
    ids=["ordinary", "quoted", "escaped"],
)
def test_connection_string_key_redacts_libpq_semicolon_credentials(
    first_password, second_password
):
    first = _compiled()
    rotated = copy.deepcopy(first)
    first.nodes["step"].env["POSTGRES_CONNECTION_STRING"] = (
        "host=db.example dbname=app user=alice "
        f"password={first_password} sslmode=require"
    )
    rotated.nodes["step"].env["POSTGRES_CONNECTION_STRING"] = (
        "host=db.example dbname=app user=bob "
        f"password={second_password} sslmode=require"
    )

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice", "first"):
        assert credential not in manifest_text

    changed = copy.deepcopy(rotated)
    changed.nodes["step"].env["POSTGRES_CONNECTION_STRING"] = changed.nodes[
        "step"
    ].env["POSTGRES_CONNECTION_STRING"].replace("db.example", "other.example")
    assert flow_fingerprint(first) != flow_fingerprint(changed)


def test_connection_string_key_redacts_azure_shared_access_credentials():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["AZURE_SERVICE_BUS_CONNECTION_STRING"] = (
        "Endpoint=sb://bus.example/;AccountName=account;"
        "SharedAccessKeyName=policy;SharedAccessKey={first;key};"
        "SharedAccessSignature=first-sas;EntityPath=orders;TransportType=amqp"
    )
    second.nodes["step"].env["AZURE_SERVICE_BUS_CONNECTION_STRING"] = (
        "Endpoint=sb://bus.example/;AccountName=account;"
        "SharedAccessKeyName=policy;SharedAccessKey={second;key};"
        "SharedAccessSignature=second-sas;EntityPath=orders;TransportType=amqp"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("first;key", "first-sas"):
        assert credential not in manifest_text

    for old, new in (
        ("bus.example", "other.example"),
        ("AccountName=account", "AccountName=other"),
        ("SharedAccessKeyName=policy", "SharedAccessKeyName=other-policy"),
        ("EntityPath=orders", "EntityPath=payments"),
        ("TransportType=amqp", "TransportType=amqp-websockets"),
    ):
        changed = copy.deepcopy(second)
        changed.nodes["step"].env[
            "AZURE_SERVICE_BUS_CONNECTION_STRING"
        ] = changed.nodes["step"].env["AZURE_SERVICE_BUS_CONNECTION_STRING"].replace(
            old, new
        )
        assert flow_fingerprint(first) != flow_fingerprint(changed)


def test_connection_string_key_keeps_azure_cosmos_account_endpoint_identity():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["AZURE_COSMOS_CONNECTION_STRING"] = (
        "AccountEndpoint=https://account.documents.azure.com:443/;"
        "AccountKey=first-account-key"
    )
    second.nodes["step"].env["AZURE_COSMOS_CONNECTION_STRING"] = (
        "AccountEndpoint=https://account.documents.azure.com:443/;"
        "AccountKey=second-account-key"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    assert "first-account-key" not in manifest_text

    changed = copy.deepcopy(second)
    changed.nodes["step"].env["AZURE_COSMOS_CONNECTION_STRING"] = (
        "AccountEndpoint=https://other.documents.azure.com:443/;"
        "AccountKey=second-account-key"
    )
    assert flow_fingerprint(first) != flow_fingerprint(changed)


def test_connection_string_key_excludes_unrecognized_scalar():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["connection_string"] = "first raw secret"
    second.nodes["step"].env["connection_string"] = "second raw secret"

    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert "first raw secret" not in json.dumps(flow_manifest(first), sort_keys=True)


def test_flow_fingerprint_redacts_uri_credentials_after_leading_whitespace():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["ENDPOINT"] = (
        "  https://alice:first-password@api.example/v1?token=first-token"
    )
    second.nodes["step"].env["ENDPOINT"] = (
        "  https://bob:second-password@api.example/v1?token=second-token"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice", "first-password", "first-token"):
        assert credential not in manifest_text


def test_flow_fingerprint_redacts_uri_credentials_in_nested_mapping_keys():
    """Credential-only key changes collapse without losing duplicate entry semantics."""
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["routes"] = {
        (
            "https://alice:first-password@api.example/route"
            "?region=us&token=first-alpha-token"
        ): {"target": "alpha"},
        (
            "https://bob:first-password@api.example/route"
            "?region=us&token=first-beta-token"
        ): {"target": "beta"},
    }
    # Reverse insertion order and replace only credentials. Both original keys redact to
    # one canonical key, so the entry list must retain and deterministically sort both
    # values.
    second.nodes["step"].env["routes"] = {
        (
            "https://dave:second-password@api.example/route"
            "?region=us&token=second-beta-token"
        ): {"target": "beta"},
        (
            "https://carol:second-password@api.example/route"
            "?region=us&token=second-alpha-token"
        ): {"target": "alpha"},
    }

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest = flow_manifest(first)
    route_entries = _typed_mapping_value(
        manifest["nodes"]["step"]["env"], "routes"
    )["entries"]
    assert len(route_entries) == 2
    canonical_keys = {
        json.dumps(entry["key"], sort_keys=True) for entry in route_entries
    }
    assert len(canonical_keys) == 1
    manifest_text = json.dumps(manifest, sort_keys=True)
    for credential in (
        "alice",
        "bob",
        "first-password",
        "first-alpha-token",
        "first-beta-token",
    ):
        assert credential not in manifest_text

    changed_endpoint = copy.deepcopy(second)
    changed_endpoint.nodes["step"].env["routes"] = {
        key.replace("api.example", "other.example"): value
        for key, value in second.nodes["step"].env["routes"].items()
    }
    assert flow_fingerprint(first) != flow_fingerprint(changed_endpoint)


def test_top_level_node_env_uri_keys_are_redacted_without_collapsing_duplicates():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env = {
        "https://alice:first-password@api.example/route?region=us": "alpha",
        "https://bob:first-password@api.example/route?region=us": "beta",
    }
    second.nodes["step"].env = {
        "https://dave:second-password@api.example/route?region=us": "beta",
        "https://carol:second-password@api.example/route?region=us": "alpha",
    }

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest = flow_manifest(first)
    env_entries = manifest["nodes"]["step"]["env"]["entries"]
    assert len(env_entries) == 2
    assert len({json.dumps(entry["key"], sort_keys=True) for entry in env_entries}) == 1
    manifest_text = json.dumps(manifest, sort_keys=True)
    for credential in ("alice", "bob", "first-password"):
        assert credential not in manifest_text

    changed_endpoint = copy.deepcopy(second)
    changed_endpoint.nodes["step"].env = {
        key.replace("api.example", "other.example"): value
        for key, value in second.nodes["step"].env.items()
    }
    assert flow_fingerprint(first) != flow_fingerprint(changed_endpoint)


@pytest.mark.parametrize("scope", ["agent", "flow"])
def test_authored_llm_uri_keys_use_credential_redacted_typed_mappings(scope):
    first_uri = "https://alice:first-password@llm.example/v1?region=us"
    second_uri = "https://bob:second-password@llm.example/v1?region=us"

    if scope == "agent":
        first = _single_node_flow(
            AgentNode("node", prompt="p", llm_config={first_uri: "route"})
        )
        second = _single_node_flow(
            AgentNode("node", prompt="p", llm_config={second_uri: "route"})
        )
        config_manifest = flow_manifest(first)["nodes"]["node"]["own_llm_config"]
    else:
        first = _compiled()
        second = copy.deepcopy(first)
        first.flow_llm_config = {first_uri: "route"}
        second.flow_llm_config = {second_uri: "route"}
        config_manifest = flow_manifest(first)["flow_llm_config"]

    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert config_manifest["type"] == "mapping"
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("alice", "first-password"):
        assert credential not in manifest_text

    changed_endpoint = copy.deepcopy(second)
    if scope == "agent":
        config = changed_endpoint.nodes["node"].own_llm_config
        changed_endpoint.nodes["node"].own_llm_config = {
            key.replace("llm.example", "other.example"): value
            for key, value in config.items()
        }
    else:
        changed_endpoint.flow_llm_config = {
            key.replace("llm.example", "other.example"): value
            for key, value in changed_endpoint.flow_llm_config.items()
        }
    assert flow_fingerprint(first) != flow_fingerprint(changed_endpoint)


def test_flow_fingerprint_redacts_uri_credentials_in_authored_llm_config():
    first = AgentNode(
        "node",
        prompt="p",
        llm_config={
            "transport": {
                "endpoint": (
                    "https://first-user:first-password@llm.example/v1/models"
                    "?region=us&token=first-token"
                )
            }
        },
    )
    second = AgentNode(
        "node",
        prompt="p",
        llm_config={
            "transport": {
                "endpoint": (
                    "https://second-user:second-password@llm.example/v1/models"
                    "?region=us&token=second-token"
                )
            }
        },
    )

    assert flow_fingerprint(_single_node_flow(first)) == flow_fingerprint(
        _single_node_flow(second)
    )
    manifest_text = json.dumps(flow_manifest(_single_node_flow(first)), sort_keys=True)
    for credential in ("first-user", "first-password", "first-token"):
        assert credential not in manifest_text


def test_flow_fingerprint_redacts_signed_uri_query_credentials():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["SIGNED_URL"] = (
        "https://objects.example/bucket/model.bin?X-Amz-Expires=900"
        "&response-content-type=application/octet-stream&sig=first-sig"
        "&signature=first-signature&X-Amz-Signature=first-amz-signature"
        "&X-Goog-Signature=first-goog-signature"
    )
    second.nodes["step"].env["SIGNED_URL"] = (
        "https://objects.example/bucket/model.bin?X-Amz-Expires=900"
        "&response-content-type=application/octet-stream&sig=second-sig"
        "&signature=second-signature&X-Amz-Signature=second-amz-signature"
        "&X-Goog-Signature=second-goog-signature"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in (
        "first-sig",
        "first-signature",
        "first-amz-signature",
        "first-goog-signature",
    ):
        assert credential not in manifest_text


def test_flow_fingerprint_redacts_uri_key_parameter_only():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["MODEL_URL"] = (
        "https://models.example/v1/model?region=us&key=first-google-api-key"
    )
    second.nodes["step"].env["MODEL_URL"] = (
        "https://models.example/v1/model?region=us&key=second-google-api-key"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    assert "first-google-api-key" not in manifest_text

    changed_host = copy.deepcopy(second)
    changed_host.nodes["step"].env["MODEL_URL"] = changed_host.nodes["step"].env[
        "MODEL_URL"
    ].replace("models.example", "other.example")
    assert flow_fingerprint(first) != flow_fingerprint(changed_host)

    changed_region = copy.deepcopy(second)
    changed_region.nodes["step"].env["MODEL_URL"] = changed_region.nodes["step"].env[
        "MODEL_URL"
    ].replace("region=us", "region=eu")
    assert flow_fingerprint(first) != flow_fingerprint(changed_region)


@pytest.mark.parametrize("delimiter", ["?", "#"], ids=["query", "fragment"])
@pytest.mark.parametrize(
    "credential_param",
    [
        "output_signature",
        "function_signature",
        "eos_token",
        "continuation_token",
        "page_token",
        "fim_prefix_token",
        "image_token",
    ],
)
def test_uri_params_apply_strict_token_and_signature_policy(
    delimiter, credential_param
):
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["SIGNED_URL"] = (
        f"https://api.example/v1{delimiter}region=us"
        f"&{credential_param}=first-credential"
    )
    second.nodes["step"].env["SIGNED_URL"] = (
        f"https://api.example/v1{delimiter}region=us"
        f"&{credential_param}=second-credential"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert "first-credential" not in json.dumps(flow_manifest(first), sort_keys=True)

    changed_region = copy.deepcopy(second)
    changed_region.nodes["step"].env["SIGNED_URL"] = changed_region.nodes["step"].env[
        "SIGNED_URL"
    ].replace("region=us", "region=eu")
    assert flow_fingerprint(first) != flow_fingerprint(changed_region)


@pytest.mark.parametrize(
    "changed_uri",
    [
        "https://other.example/bucket/model.bin?X-Amz-Expires=900&sig=value",
        "https://objects.example/other/model.bin?X-Amz-Expires=900&sig=value",
        "https://objects.example/bucket/model.bin?X-Amz-Expires=901&sig=value",
    ],
    ids=["host", "path", "signed-url-expiry"],
)
def test_flow_fingerprint_keeps_signed_uri_endpoint_parameters(changed_uri):
    original = _compiled()
    changed = copy.deepcopy(original)
    original.nodes["step"].env["SIGNED_URL"] = (
        "https://objects.example/bucket/model.bin?X-Amz-Expires=900&sig=value"
    )
    changed.nodes["step"].env["SIGNED_URL"] = changed_uri

    assert flow_fingerprint(original) != flow_fingerprint(changed)


def test_flow_fingerprint_redacts_uri_fragment_credentials_only():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["CALLBACK_URI"] = (
        "https://auth.example/callback"
        "#access_token=first-access-token&id_token=first-id-token&state=stable"
    )
    second.nodes["step"].env["CALLBACK_URI"] = (
        "https://auth.example/callback"
        "#access_token=second-access-token&id_token=second-id-token&state=stable"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    assert "first-access-token" not in manifest_text
    assert "first-id-token" not in manifest_text

    changed_state = copy.deepcopy(first)
    changed_state.nodes["step"].env["CALLBACK_URI"] = (
        "https://auth.example/callback"
        "#access_token=first-access-token&id_token=first-id-token&state=changed"
    )
    assert flow_fingerprint(changed_state) != flow_fingerprint(first)

    first.nodes["step"].env["CALLBACK_URI"] = "https://auth.example/callback#first-section"
    second.nodes["step"].env["CALLBACK_URI"] = "https://auth.example/callback#second-section"
    assert flow_fingerprint(first) != flow_fingerprint(second)


@pytest.mark.parametrize(
    "changed_uri",
    [
        "postgres://user:secret@other.example:5432/app?sslmode=require&token=value",
        "postgres://user:secret@db.example:5432/other?sslmode=require&token=value",
        "postgres://user:secret@db.example:5432/app?sslmode=disable&token=value",
    ],
    ids=["host", "path", "noncredential-query"],
)
def test_flow_fingerprint_keeps_noncredential_uri_semantics(changed_uri):
    original = _compiled()
    changed = copy.deepcopy(original)
    original.nodes["step"].env["DATABASE_URL"] = (
        "postgres://user:secret@db.example:5432/app?sslmode=require&token=value"
    )
    changed.nodes["step"].env["DATABASE_URL"] = changed_uri

    assert flow_fingerprint(original) != flow_fingerprint(changed)


def test_uri_redaction_is_scoped_away_from_prompts_code_and_tool_ids():
    first_uri = "https://first-user:first-secret@example.test/path?token=first-token"
    second_uri = "https://second-user:second-secret@example.test/path?token=second-token"

    first_prompt = _single_node_flow(AgentNode("node", prompt=first_uri))
    second_prompt = _single_node_flow(AgentNode("node", prompt=second_uri))
    first_code = _single_node_flow(CodeNode("node", code=f"return {first_uri!r}"))
    second_code = _single_node_flow(CodeNode("node", code=f"return {second_uri!r}"))
    first_tool_id = _single_node_flow(ToolNode("node", tool_id=first_uri))
    second_tool_id = _single_node_flow(ToolNode("node", tool_id=second_uri))

    for first, second in (
        (first_prompt, second_prompt),
        (first_code, second_code),
        (first_tool_id, second_tool_id),
    ):
        assert flow_fingerprint(first) != flow_fingerprint(second)
        assert "first-secret" in json.dumps(flow_manifest(first), sort_keys=True)


def test_flow_fingerprint_redacts_uri_credentials_in_wiring():
    first = _fragment()
    second = copy.deepcopy(first)
    first.wiring["body"]["x"] = Binding(
        "https://first-user:first-secret@example.test/path?token=first-token"
    )
    second.wiring["body"]["x"] = Binding(
        "https://second-user:second-secret@example.test/path?token=second-token"
    )

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for credential in ("first-user", "first-secret", "first-token"):
        assert credential not in manifest_text


def test_flow_fingerprint_excludes_wiring_credentials_but_keeps_semantics():
    source = """
id: wiring-credentials
name: Wiring credentials
nodes:
  step:
    kind: code
    input:
      api_key: __SECRET__
      region: __REGION__
    output: str
    code: |
      return inputs["region"]
output: ${step.output}
"""
    first = _compiled(
        source.replace("__SECRET__", "secret-a").replace("__REGION__", "us")
    )
    rotated = _compiled(
        source.replace("__SECRET__", "secret-b").replace("__REGION__", "us")
    )
    changed_region = _compiled(
        source.replace("__SECRET__", "secret-b").replace("__REGION__", "eu")
    )

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    assert "secret-a" not in json.dumps(flow_manifest(first), sort_keys=True)
    assert flow_fingerprint(first) != flow_fingerprint(changed_region)


@pytest.mark.parametrize(
    ("slot", "primary_source", "backup_source"),
    [
        (
            "password",
            "${input.primary_password}",
            "${input.backup_password}",
        ),
        (
            "authorization",
            '"Bearer ${input.primary_token}"',
            '"Bearer ${input.backup_token}"',
        ),
        (
            "connection_string",
            "${input.primary_dsn}",
            "${input.backup_dsn}",
        ),
        (
            "connection_string",
            '"postgres://${input.primary_user}:${input.primary_password}@db.example/app"',
            '"postgres://${input.backup_user}:${input.backup_password}@db.example/app"',
        ),
        (
            "connection_string",
            '"Server=db.example;Password=${input.primary_password}"',
            '"Server=db.example;Password=${input.backup_password}"',
        ),
        (
            "endpoint",
            '"https://api.example/v1?token=${input.primary_token}&region=us"',
            '"https://api.example/v1?token=${input.backup_token}&region=us"',
        ),
        (
            "endpoint",
            '"https://${input.primary_user}:literal@api.example/v1"',
            '"https://${input.backup_user}:literal@api.example/v1"',
        ),
    ],
    ids=[
        "whole",
        "embedded",
        "connection",
        "connection-uri",
        "connection-dsn",
        "uri-query",
        "uri-userinfo",
    ],
)
def test_flow_fingerprint_keeps_credential_wiring_reference_identity(
    slot, primary_source, backup_source
):
    source = """
id: credential-routing
name: Credential routing
input:
  primary_password: str
  backup_password: str
  primary_token: str
  backup_token: str
  primary_dsn: str
  backup_dsn: str
  primary_user: str
  backup_user: str
nodes:
  step:
    kind: code
    input:
      __SLOT__: __SOURCE__
    output: str
    code: |
      return inputs["__SLOT__"]
output: ${step.output}
"""
    source = source.replace("__SLOT__", slot)
    primary = _compiled(source.replace("__SOURCE__", primary_source))
    backup = _compiled(source.replace("__SOURCE__", backup_source))

    assert flow_fingerprint(primary) != flow_fingerprint(backup)
    primary_manifest = json.dumps(flow_manifest(primary), sort_keys=True)
    backup_manifest = json.dumps(flow_manifest(backup), sort_keys=True)
    assert "__redacted_binding__:" in primary_manifest
    assert "__redacted_binding__:" in backup_manifest


def test_embedded_connection_binding_keeps_endpoint_semantics():
    first = _fragment()
    changed = copy.deepcopy(first)
    first.wiring["body"]["connection_string"] = Binding(
        "postgres://${input.primary_user}:${input.primary_password}@db.example/app"
    )
    changed.wiring["body"]["connection_string"] = Binding(
        first.wiring["body"]["connection_string"].source.replace(
            "db.example", "other.example"
        )
    )

    assert flow_fingerprint(first) != flow_fingerprint(changed)


def test_encoded_binding_like_uri_credentials_remain_literal():
    first = _fragment()
    rotated = copy.deepcopy(first)
    first.wiring["body"]["endpoint"] = Binding(
        "https://api.example/v1?token=%24%7Bfirst.secret%7D&region=us"
    )
    rotated.wiring["body"]["endpoint"] = Binding(
        "https://api.example/v1?token=%24%7Bsecond.secret%7D&region=us"
    )

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    assert "first.secret" not in json.dumps(flow_manifest(first), sort_keys=True)


def test_credential_binding_marker_redacts_literal_operands():
    first = _fragment()
    rotated = copy.deepcopy(first)
    first.wiring["body"]["authorization"] = Binding(
        'Bearer ${input.primary_token:-"first-secret"}'
    )
    rotated.wiring["body"]["authorization"] = Binding(
        'Bearer ${input.primary_token:-"second-secret"}'
    )

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    assert "first-secret" not in json.dumps(flow_manifest(first), sort_keys=True)

    changed = copy.deepcopy(rotated)
    changed.wiring["body"]["authorization"] = Binding(
        changed.wiring["body"]["authorization"].source.replace(
            "primary_token", "backup_token"
        )
    )
    assert flow_fingerprint(first) != flow_fingerprint(changed)


def test_credential_binding_marker_keeps_http_auth_scheme_semantics():
    first = _fragment()
    changed = copy.deepcopy(first)
    first.wiring["body"]["authorization"] = Binding("Bearer ${input.token}")
    changed.wiring["body"]["authorization"] = Binding("Basic ${input.token}")

    assert flow_fingerprint(first) != flow_fingerprint(changed)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    assert "Bearer" not in manifest_text


@pytest.mark.parametrize(
    ("slot", "first_source", "rotated_source"),
    [
        (
            "authorization",
            "prefix-${input.token}",
            "suffix-${input.token}",
        ),
        (
            "password",
            "Bearer ${input.tail}",
            "Basic ${input.tail}",
        ),
    ],
    ids=["unknown-affix", "scheme-outside-authorization"],
)
def test_credential_binding_marker_redacts_unsafe_literal_runs(
    slot, first_source, rotated_source
):
    first = _fragment()
    rotated = copy.deepcopy(first)
    first.wiring["body"][slot] = Binding(first_source)
    rotated.wiring["body"][slot] = Binding(rotated_source)

    assert flow_fingerprint(first) == flow_fingerprint(rotated)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    for literal in ("prefix-", "Bearer"):
        assert literal not in manifest_text


def test_binding_like_static_credentials_remain_literal():
    for scope in (
        "env",
        "env-uri",
        "env-connection",
        "structured-env",
        "nested-wiring",
    ):
        first = _compiled()
        rotated = copy.deepcopy(first)
        if scope == "env":
            first.nodes["step"].env["password"] = "${first.secret}"
            rotated.nodes["step"].env["password"] = "${second.secret}"
        elif scope == "env-uri":
            first.nodes["step"].env["endpoint"] = (
                "https://api.example/v1?token=${first.secret}"
            )
            rotated.nodes["step"].env["endpoint"] = (
                "https://api.example/v1?token=${second.secret}"
            )
        elif scope == "env-connection":
            first.nodes["step"].env["connection_string"] = (
                "Server=db.example;Password=${first.secret}"
            )
            rotated.nodes["step"].env["connection_string"] = (
                "Server=db.example;Password=${second.secret}"
            )
        elif scope == "structured-env":
            first.nodes["step"].env["auth"] = {"password": "${first.secret}"}
            rotated.nodes["step"].env["auth"] = {"password": "${second.secret}"}
        else:
            first.wiring["step"]["auth"] = Binding(
                {"password": "${first.secret}"}
            )
            rotated.wiring["step"]["auth"] = Binding(
                {"password": "${second.secret}"}
            )

        assert flow_fingerprint(first) == flow_fingerprint(rotated)
        manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
        assert "first.secret" not in manifest_text
        assert "__redacted_binding__:" not in manifest_text


def test_wait_fingerprint_and_expansion_digest_cover_absolute_until_wiring():
    first = _single_node_flow(WaitNode("node"))
    second = _single_node_flow(WaitNode("node"))
    first.wiring["node"] = {"until": Binding("2026-01-01T00:00:00+00:00")}
    second.wiring["node"] = {"until": Binding("2026-01-02T00:00:00+00:00")}

    assert flow_fingerprint(first) != flow_fingerprint(second)
    assert expansion_digest(first) != expansion_digest(second)


@pytest.mark.parametrize(
    "credential_key",
    [
        "DATABRICKS_TOKEN",
        "CUSTOM_SERVICE_SIGNATURE",
        "CUSTOM_SERVICE_TOKEN",
        " CUSTOM_SERVICE_TOKEN ",
        "GITHUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "NPM_TOKEN",
        "OPENAI_API_TOKEN",
        "PGPASSWORD",
        "MYSQL_PWD",
        "SERVICE_ACCESS_TOKEN",
        "SHARED_ACCESS_KEY",
        "SHARED_ACCESS_SIGNATURE",
        "SLACK_BOT_TOKEN",
        "AWS_SESSION_TOKEN",
        "id_token",
        "X-Amz-Security-Token",
        "AWS_ACCESS_KEY_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_APPLICATION_CREDENTIALS",
        "AZURE_STORAGE_ACCOUNT_KEY",
        "AZURE_STORAGE_CONNECTION_STRING",
        "AUTH",
        "PASSPHRASE",
        "clientSecret",
        "ClientSecret",
        "apiKey",
        "APIKey",
        "secretAccessKey",
        "AWSSecretAccessKey",
        "client.secret",
        "client secret",
        "client/secret",
        "apikey",
        "clientsecret",
        "accesstoken",
        "secretaccesskey",
        "connectionstring",
        "proxyAuth",
        "proxyAuthorization",
        "sshPassphrase",
        "basicAuth",
        "http.authorization",
        "basicauth",
        "basic_auth",
        "proxy_auth",
        "CUSTOM_AUTH",
        "SIG",
        "SIGNATURE",
        "X-Amz-Signature",
        "X-Goog-Signature",
        "xAmzSignature",
        "xgoogsignature",
        "WEBHOOK_SIGNATURE",
        "databricks_token",
        "github_token",
        "hf_token",
        "npm_token",
        "slack_bot_token",
        "webhook_signature",
        "x_amz_security_token",
        "x_amz_signature",
        "x_goog_signature",
    ],
)
def test_flow_fingerprint_redacts_common_provider_credentials(credential_key):
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["provider_settings"] = [
        {credential_key: "first-provider-secret"}
    ]
    second.nodes["step"].env["provider_settings"] = [
        {credential_key: "second-provider-secret"}
    ]

    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert "first-provider-secret" not in json.dumps(flow_manifest(first), sort_keys=True)


def test_flow_fingerprint_redacts_nested_docker_auth_config():
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env["container"] = {
        "settings": [
            {
                "DOCKER_AUTH_CONFIG": {
                    "auths": {
                        "registry.example": {
                            "auth": "first-docker-auth",
                            "identitytoken": "first-identity-token",
                        }
                    }
                }
            }
        ]
    }
    second.nodes["step"].env["container"] = {
        "settings": [
            {
                "DOCKER_AUTH_CONFIG": {
                    "auths": {
                        "registry.example": {
                            "auth": "second-docker-auth",
                            "identitytoken": "second-identity-token",
                        }
                    }
                }
            }
        ]
    }

    assert flow_fingerprint(first) == flow_fingerprint(second)
    manifest_text = json.dumps(flow_manifest(first), sort_keys=True)
    assert "DOCKER_AUTH_CONFIG" not in manifest_text
    assert "first-docker-auth" not in manifest_text
    assert "first-identity-token" not in manifest_text


@pytest.mark.parametrize(
    "config_key",
    [
        "key",
        "tokenizer",
        "secretariat",
        "keynote",
        "apiVersion",
        "connectionPool",
    ],
)
def test_compact_credential_matching_keeps_unrelated_config(config_key):
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env[config_key] = "first-semantic-value"
    second.nodes["step"].env[config_key] = "second-semantic-value"

    assert flow_fingerprint(first) != flow_fingerprint(second)
    assert "first-semantic-value" in json.dumps(flow_manifest(first), sort_keys=True)


@pytest.mark.parametrize(
    "config_key",
    [
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "mask_token",
        "sep_token",
        "cls_token",
        "continuation_token",
        "fim_prefix_token",
        "image_token",
        "page_token",
        "start_token",
        "end_token",
        "stop_token",
        "decoder_start_token",
        "input_signature",
        "output_signature",
        "function_signature",
        "method_signature",
        "call_signature",
        "type_signature",
        "model_signature",
        "serving_signature",
        "bosToken",
        "decoderstarttoken",
        "inputSignature",
        "modelsignature",
        "pageToken",
        "custom_service_signature",
        "custom_service_token",
        "oauth",
        "reauth",
        "custom_auth",
    ],
)
def test_semantic_credential_like_config_remains_identity_bearing(config_key):
    first = _compiled()
    second = copy.deepcopy(first)
    first.nodes["step"].env[config_key] = "first-semantic-value"
    second.nodes["step"].env[config_key] = "second-semantic-value"

    assert flow_fingerprint(first) != flow_fingerprint(second)
    assert "first-semantic-value" in json.dumps(flow_manifest(first), sort_keys=True)


def _compiled_with_env_literal(literal: str):
    """Load a flow whose node environment contains one YAML literal."""
    return _compiled(
        f"""
id: literal
name: Literal
nodes:
  step:
    kind: code
    env:
      value: {literal}
    output: str
    code: |
      return "ok"
output: ${{step.output}}
"""
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("2026-01-01", '{"$date": "2026-01-01"}'),
        ("!!set {a: null, b: null}", "[a, b]"),
    ],
    ids=["date-vs-map", "set-vs-list"],
)
def test_flow_fingerprint_distinguishes_yaml_literal_types(first, second):
    assert flow_fingerprint(_compiled_with_env_literal(first)) != flow_fingerprint(
        _compiled_with_env_literal(second)
    )


def test_flow_fingerprint_supports_binary_nonfinite_and_non_string_mapping_keys():
    binary = _compiled_with_env_literal("!!binary aGVsbG8=")
    nonfinite = _compiled_with_env_literal(".nan")
    ordered = _compiled_with_env_literal("{1: one, 2: two}")
    reordered = _compiled_with_env_literal("{2: two, 1: one}")

    assert flow_fingerprint(binary) == flow_fingerprint(
        _compiled_with_env_literal("!!binary aGVsbG8=")
    )
    assert flow_fingerprint(nonfinite) == flow_fingerprint(_compiled_with_env_literal(".nan"))
    assert flow_fingerprint(ordered) == flow_fingerprint(reordered)
    json.dumps(flow_manifest(binary), sort_keys=True, allow_nan=False)
    json.dumps(flow_manifest(nonfinite), sort_keys=True, allow_nan=False)
    json.dumps(flow_manifest(ordered), sort_keys=True, allow_nan=False)


def test_flow_fingerprint_tags_programmatic_container_types():
    list_value = _compiled()
    tuple_value = copy.deepcopy(list_value)
    set_value = copy.deepcopy(list_value)
    frozen_set_value = copy.deepcopy(list_value)
    list_value.nodes["step"].env["value"] = ["a", "b"]
    tuple_value.nodes["step"].env["value"] = ("a", "b")
    set_value.nodes["step"].env["value"] = {"a", "b"}
    frozen_set_value.nodes["step"].env["value"] = frozenset({"a", "b"})

    fingerprints = {
        flow_fingerprint(list_value),
        flow_fingerprint(tuple_value),
        flow_fingerprint(set_value),
        flow_fingerprint(frozen_set_value),
    }
    assert len(fingerprints) == 4


def _parent_with_child(child_body: str):
    """Load a parent whose baked CALL child differs only by its inline body."""
    return _compiled(
        f"""
id: recursive
name: Recursive
input:
  x: int
defs:
  child:
    input:
      x: int
    nodes:
      inner:
        kind: code
        input:
          x: ${{input.x}}
        output: int
        code: |
          {child_body}
    output: ${{inner.output}}
nodes:
  invoke:
    kind: call
    call: child
    input:
      x: ${{input.x}}
output: ${{invoke.output}}
"""
    )


def test_flow_fingerprint_recursively_includes_static_child_flows():
    add_one = _parent_with_child('return inputs["x"] + 1')
    add_two = _parent_with_child('return inputs["x"] + 2')

    assert flow_fingerprint(add_one) != flow_fingerprint(add_two)


def test_flow_manifest_sorts_mapping_keys_without_reordering_semantic_lists():
    original = _compiled()
    shuffled = copy.deepcopy(original)
    shuffled.nodes = dict(reversed(list(shuffled.nodes.items())))
    shuffled.wiring = {
        node_id: dict(reversed(list(wiring.items())))
        for node_id, wiring in reversed(list(shuffled.wiring.items()))
    }
    shuffled.nodes["step"].env = dict(reversed(list(shuffled.nodes["step"].env.items())))

    manifest = flow_manifest(shuffled)
    assert manifest == flow_manifest(original)
    assert list(manifest["nodes"]) == sorted(manifest["nodes"])
    assert list(manifest["wiring"]) == sorted(manifest["wiring"])
    assert manifest["edges"] == flow_manifest(original)["edges"]


def test_manifest_rejects_unknown_execution_fields_instead_of_omitting_them():
    flow = _compiled()
    flow.nodes["step"].future_execution_flag = True

    with pytest.raises(ValueError, match="future_execution_flag"):
        flow_manifest(flow)


def test_manifest_rejects_non_json_values_without_repr_fallback():
    flow = _compiled()
    flow.nodes["step"].env["opaque"] = object()

    with pytest.raises(TypeError, match="opaque|object"):
        flow_fingerprint(flow)


@dataclass(slots=True)
class _DataclassFixture:
    """Slotted dataclass fixture for the strict field census."""

    known: int


class _AttrsAttributeFixture:
    """Minimal attrs metadata entry without adding attrs as a test dependency."""

    def __init__(self, name: str) -> None:
        self.name = name


class _AttrsFixture:
    """Attrs-shaped fixture exposing the metadata consumed by the census."""

    __attrs_attrs__ = (_AttrsAttributeFixture("known"),)
    __slots__ = ("known",)

    def __init__(self, known: int) -> None:
        self.known = known


class _PydanticFixture(BaseModel):
    """Pydantic fixture whose framework slots must not count as authored fields."""

    model_config = ConfigDict(extra="allow")

    known: int


class _SlotsFixture:
    """Plain slotted fixture for the strict field census."""

    __slots__ = ("known",)

    def __init__(self, known: int) -> None:
        self.known = known


@pytest.mark.parametrize(
    "value",
    [_DataclassFixture(1), _AttrsFixture(1), _PydanticFixture(known=1), _SlotsFixture(1)],
    ids=["dataclass", "attrs", "pydantic", "slots"],
)
def test_field_census_supports_declared_object_models(value):
    assert manifest_module._field_names(value) == frozenset({"known"})
    manifest_module._check_fields(value, frozenset({"known"}), "fixture")


def test_field_census_detects_pydantic_extra_fields_without_internal_slots():
    value = _PydanticFixture(known=1, future=2)

    assert manifest_module._field_names(value) == frozenset({"known", "future"})
    with pytest.raises(ValueError, match="future"):
        manifest_module._check_fields(value, frozenset({"known"}), "fixture")


def _fragment() -> Flow:
    """Build a small valid fragment with literal wiring and two ordered edges."""
    start = StartNode("entry", params=[])
    body = CodeNode("body", code='return inputs["x"]')
    body.params = [
        Parameter(
            name="x",
            contract=ValueContract(Type.scalar(ValueKind.INTEGER)),
        )
    ]
    body.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))
    end = EndNode.value("terminal", output_name="result")
    return Flow(
        nodes={"entry": start, "body": body, "terminal": end},
        edges=[
            Edge("entry->body", "entry", "body", ordering=True),
            Edge("body->terminal", "body", "terminal", input_group="result"),
        ],
        wiring={
            "entry": {},
            "body": {"x": 1},
            "terminal": {"result": "${body.output}"},
        },
        start_id="entry",
        end_id="terminal",
    )


class _ParameterSubclass(Parameter):
    """Near-canonical fixture for exact manifest carrier checks."""


class _ValueContractSubclass(ValueContract):
    """Near-canonical fixture for exact output-contract manifest checks."""


def _replace_body(fragment: Flow) -> None:
    body = CodeNode("body", code='return inputs["x"] + 1')
    body.params = [
        Parameter(
            name="x",
            contract=ValueContract(Type.scalar(ValueKind.INTEGER)),
        )
    ]
    body.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))
    fragment.nodes["body"] = body


@pytest.mark.parametrize(
    "mutate",
    [
        lambda fragment: setattr(fragment, "start_id", "body"),
        lambda fragment: setattr(fragment, "end_id", "body"),
        _replace_body,
        lambda fragment: setattr(fragment, "edges", list(reversed(fragment.edges))),
        lambda fragment: fragment.wiring["body"].__setitem__("x", Binding(2)),
        lambda fragment: fragment.edges.append(
            Edge("entry->terminal", "entry", "terminal", ordering=True)
        ),
    ],
    ids=["entry", "terminal", "node-fields", "edge-order", "wiring", "topology"],
)
def test_expansion_digest_changes_for_every_builder_topology_component(mutate):
    original = _fragment()
    changed = copy.deepcopy(original)
    mutate(changed)

    assert expansion_digest(changed) != expansion_digest(original)
    assert re.fullmatch(r"[0-9a-f]{64}", expansion_digest(original))


def test_expansion_digest_is_stable_across_canonical_mapping_order():
    original = _fragment()
    shuffled = copy.deepcopy(original)
    shuffled.nodes = dict(reversed(list(shuffled.nodes.items())))
    shuffled.wiring = {
        node_id: dict(reversed(list(wiring.items())))
        for node_id, wiring in reversed(list(shuffled.wiring.items()))
    }

    assert expansion_digest(shuffled) == expansion_digest(original)
    json.dumps(flow_manifest(shuffled), sort_keys=True, allow_nan=False)


def test_expansion_digest_applies_topology_only_policy_to_recursive_children():
    original = _parent_with_child('return inputs["x"] + 1')
    changed = copy.deepcopy(original)
    changed.nodes["invoke"].child.flow_llm_config["temperature"] = 0.7

    assert expansion_digest(changed) == expansion_digest(original)


def _single_node_flow(node) -> Flow:
    """Wrap one canonical node so kind-specific manifest fields can be compared."""
    return Flow(
        nodes={node.id: node},
        edges=[],
        wiring={node.id: {}},
        start_id=node.id,
        end_id=node.id,
    )


def test_agent_variable_manifest_preserves_the_variable_free_fingerprint():
    flow = _single_node_flow(AgentNode("node", prompt="hi"))
    assert "variables" not in flow_manifest(flow)["nodes"]["node"]
    # Pinned after `mode` left the AGENT manifest.
    assert flow_fingerprint(flow) == (
        "4bf7361e46300e1e8c9250fbca0505b9ad985ce4159558c1528ac1bc29bf8bd4"
    )


def test_agent_variable_identity_tracks_names_domains_defaults_and_order_only():
    count = Parameter(
        "count", ValueContract(Type(ValueKind.INTEGER, bounds=(1, 12))),
        has_default=True, default=4,
    )
    tone = Parameter(
        "tone", ValueContract(Type(ValueKind.STRING, tags=frozenset({"brief", "detailed"}))),
        has_default=True, default="brief",
    )
    declarations = [
        [count, tone],
        [replace(count, name="sentences"), tone],
        [replace(count, default=5), tone],
        [replace(count, contract=ValueContract(Type(ValueKind.INTEGER, bounds=(1, 8)))), tone],
        [tone, count],
    ]
    flows = [_single_node_flow(AgentNode("node", variables=variables)) for variables in declarations]
    assert len({flow_fingerprint(flow) for flow in flows}) == len(flows)
    assert len({expansion_digest(flow) for flow in flows}) == len(flows)
    assert [variable["name"] for variable in flow_manifest(flows[0])["nodes"]["node"]["variables"]] == ["count", "tone"]

    located = _single_node_flow(AgentNode("node", variables=[
        replace(count, origin=OriginRef("declaration"), default_origin=OriginRef("default"),
                contract=ValueContract(count.contract.type, ContractOrigins(OriginRef("type")))),
        tone,
    ]))
    assert flow_fingerprint(located) == flow_fingerprint(flows[0])
    assert expansion_digest(located) == expansion_digest(flows[0])


@pytest.mark.parametrize("variables", [None, (), (Parameter("x"),)])
def test_agent_variable_manifest_rejects_noncanonical_containers(variables):
    node = AgentNode("node")
    node.variables = variables
    with pytest.raises(TypeError, match="expected list"):
        flow_manifest(_single_node_flow(node))


def _changed_fragment() -> Flow:
    """Return the canonical fragment with only its CODE body changed."""
    fragment = _fragment()
    _replace_body(fragment)
    return fragment


_TOTAL_LOOP_FLOW = """
id: total-loop
name: total_loop
defs:
  bump:
    input:
      n: int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: ${step.output}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: 0
    times: 2
output: ${loop.output}
"""


def test_total_loop_manifest_omits_returned_fields():
    manifest = flow_manifest(_compiled(_TOTAL_LOOP_FLOW))

    loop_manifest = manifest["nodes"]["loop"]
    assert loop_manifest["kind"] == "loop"
    assert "returned_fields" not in loop_manifest


def test_flow_manifest_emits_the_literal_node_kind():
    code_manifest = flow_manifest(_single_node_flow(CodeNode("node", code='return "ok"')))
    tool_manifest = flow_manifest(_single_node_flow(ToolNode("node", tool_id="t")))

    assert code_manifest["nodes"]["node"]["kind"] == "code"
    assert tool_manifest["nodes"]["node"]["kind"] == "tool"


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("id", "renamed-body"),
        ("output_contract", ValueContract(Type.scalar(ValueKind.NUMBER))),
        ("pre_asserts", [Assertion("x > 0", "x > 0")]),
        ("post_asserts", [Assertion("output > 0", "output > 0")]),
        ("commit_as", "owner"),
        ("origin_id", "origin"),
        ("env", {"threshold": 2}),
    ],
    ids=["id", "output-contract", "pre-asserts", "post-asserts", "commit-as", "origin-id", "env"],
)
def test_flow_fingerprint_covers_each_included_common_node_field(field, changed_value):
    original = _fragment()
    changed = copy.deepcopy(original)
    setattr(changed.nodes["body"], field, changed_value)

    assert flow_fingerprint(changed) != flow_fingerprint(original)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "renamed"},
        {"contract": ValueContract(Type.scalar(ValueKind.NUMBER))},
        {"required": True},
        {"has_default": True},
        {"coerce_strings": True},
    ],
    ids=["name", "contract", "required", "default", "coerce-strings"],
)
def test_flow_fingerprint_covers_each_parameter_field(changes):
    original = _fragment()
    changed = copy.deepcopy(original)
    param = changed.nodes["body"].params[0]
    changed.nodes["body"].params[0] = replace(param, **changes)

    assert flow_fingerprint(changed) != flow_fingerprint(original)


def test_flow_fingerprint_covers_parameter_default_independently():
    original = _fragment()
    original_param = original.nodes["body"].params[0]
    original.nodes["body"].params[0] = replace(
        original_param,
        has_default=True,
        default=1,
    )
    changed = copy.deepcopy(original)
    changed_param = changed.nodes["body"].params[0]
    changed.nodes["body"].params[0] = replace(changed_param, default=2)

    assert flow_fingerprint(changed) != flow_fingerprint(original)


def test_flow_fingerprint_preserves_parameter_order():
    original = _fragment()
    original.nodes["body"].params.append(Parameter(name="y"))
    changed = copy.deepcopy(original)
    changed.nodes["body"].params.reverse()

    assert flow_fingerprint(changed) != flow_fingerprint(original)


def test_parameter_origins_do_not_affect_program_or_expansion_identity():
    """Parameter, default, and nested contract origins are diagnostic-only."""
    first = _fragment()
    second = copy.deepcopy(first)
    int_type = Type.scalar(ValueKind.INTEGER)
    first.nodes["body"].params = [
        Parameter(
            "x",
            contract=ValueContract(int_type),
            has_default=True,
            default=1,
        )
    ]
    second.nodes["body"].params = [
        Parameter(
            "x",
            contract=ValueContract(
                int_type,
                ContractOrigins(
                    default=OriginRef("second/contract"),
                    entries=((('nested',), OriginRef("second/contract/nested")),),
                ),
            ),
            has_default=True,
            default=1,
            origin=OriginRef("second/parameter"),
            default_origin=OriginRef("second/default"),
        )
    ]

    assert flow_manifest(first) == flow_manifest(second)
    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert expansion_digest(first) == expansion_digest(second)


def test_output_contract_origins_do_not_affect_program_or_expansion_identity():
    """Root and nested result origins are diagnostic-only manifest data."""
    first = _fragment()
    second = copy.deepcopy(first)
    int_type = Type.scalar(ValueKind.INTEGER)
    first.nodes["body"].output_contract = ValueContract(int_type)
    second.nodes["body"].output_contract = ValueContract(
        int_type,
        ContractOrigins(
            default=OriginRef("second/output"),
            entries=((('nested',), OriginRef("second/output/nested")),),
        ),
    )

    assert flow_manifest(first) == flow_manifest(second)
    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert expansion_digest(first) == expansion_digest(second)


def test_node_origin_does_not_affect_program_or_expansion_identity():
    """A node's authored source identity is diagnostic-only manifest data."""
    first = _fragment()
    second = copy.deepcopy(first)
    second.nodes["body"].origin = OriginRef("root/node/body")

    assert flow_manifest(first) == flow_manifest(second)
    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert expansion_digest(first) == expansion_digest(second)


def test_node_origin_context_does_not_affect_program_or_expansion_identity():
    """Generated failure ancestry is diagnostic-only manifest data."""
    first = _fragment()
    second = copy.deepcopy(first)
    second.nodes["body"].origin_context = (
        OriginRef("root/nodes/invoke"),
        OriginRef("child/nodes/map"),
    )

    assert flow_manifest(first) == flow_manifest(second)
    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert expansion_digest(first) == expansion_digest(second)


def test_memory_scope_is_not_authored_identity_but_is_expansion_topology():
    plain = _fragment()
    stamped = copy.deepcopy(plain)
    stamped.nodes["body"].memory_scope = MemoryScope(("body",))

    assert flow_manifest(stamped) == flow_manifest(plain)
    assert flow_fingerprint(stamped) == flow_fingerprint(plain)
    assert expansion_digest(stamped) != expansion_digest(plain)


def test_memory_expansion_projection_omits_root_id_and_config_but_hashes_keys():
    def compiled(root_id: str, policy: str, limit: int) -> CompiledFlow:
        source = f"""id: {root_id}
name: memory
long_term_memory:
  {policy}:
    type: {policy}
    config: {{limit: {limit}}}
nodes:
  body:
    kind: code
    code: tests.engine._compose_codefns:echo
    input: {{topic: hello}}
    output: str
output: ${{body.output}}
"""
        flow = load_flow(source).compiled
        apply_memory_layout(flow, resolve_memory_layout(flow))
        return flow

    first = compiled("first", "outer", 3)
    second = compiled("second", "outer", 99)
    other_key = compiled("first", "inner", 3)

    assert flow_fingerprint(first) != flow_fingerprint(second)
    assert expansion_digest(first) == expansion_digest(second)
    assert expansion_digest(first) != expansion_digest(other_key)


def test_memory_expansion_projection_covers_keys_and_occurrence_topology():
    scope_path = ("nested", "body")
    segment = MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ("nested",), 0)
    baseline = _fragment()
    baseline.nodes["body"].memory_scope = MemoryScope(scope_path)
    variants = [baseline]
    for memories, occurrence_path in (
        (("notes",), ()),
        (("chat",), ()),
        (("notes", "chat"), ()),
        (("chat", "notes"), ()),
        (("notes",), (segment,)),
    ):
        flow = copy.deepcopy(baseline)
        flow.nodes["body"].memory_scope = MemoryScope(scope_path, memories, occurrence_path)
        variants.append(flow)

    assert len({expansion_digest(flow) for flow in variants}) == 6


def test_memory_expansion_projection_fields():
    from yian.compile.manifest import _memory_scope_topology

    segment = MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ("nested",), 1)
    scope = MemoryScope(("nested", "body"), ("notes",), (segment,))
    assert _memory_scope_topology(scope, "scope") == {
        "memories": ["notes"],
        "occurrence_path": [
            {"index": 1, "kind": "map_item", "static_path": ["nested"]}
        ],
        "static_path": ["nested", "body"],
    }


def test_case_predicate_origins_do_not_affect_program_or_expansion_identity():
    """Ordered predicate provenance is diagnostic-only manifest data."""
    first = _single_node_flow(CaseNode("case", [Case("branch", "true")]))
    second = copy.deepcopy(first)
    second.nodes["case"].cases[0].origin = OriginRef("case/0/when")

    assert flow_manifest(first) == flow_manifest(second)
    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert expansion_digest(first) == expansion_digest(second)


def test_case_manifest_rejects_a_mutated_non_origin_predicate_carrier():
    """Identity projection must not silently ignore malformed diagnostic state."""
    flow = _single_node_flow(CaseNode("case", [Case("branch", "true")]))
    flow.nodes["case"].cases[0].origin = "flow.yaml:12"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="expected OriginRef or None"):
        flow_manifest(flow)


def test_output_contract_cutover_preserves_manifest_shape_and_fingerprint():
    """The carrier cutover must not invalidate equivalent durable flow identity."""
    fragment = _fragment()
    node_manifest = flow_manifest(fragment)["nodes"]["body"]

    assert "output_contract" not in node_manifest
    assert node_manifest["output_type"] == {
        "element": None,
        "fields": None,
        "kind": "int",
        "nullable": False,
        "required": None,
        "tags": None,
    }
    assert flow_fingerprint(fragment) == (
        "16d1ca51cfd1c4730aea134054211d87708d8aa08bc8f592b41df96c74982d00"
    )
    assert expansion_digest(fragment) == flow_fingerprint(fragment)


def test_numeric_bounds_are_part_of_type_identity_only_when_present():
    plain = _fragment()
    assert "bounds" not in flow_manifest(plain)["nodes"]["body"]["output_type"]
    fingerprints = {flow_fingerprint(plain)}
    digests = {expansion_digest(plain)}
    for bounds in ((1, 12), (2, 12), (1, 13)):
        bounded = copy.deepcopy(plain)
        bounded.nodes["body"].output_contract = ValueContract(Type(ValueKind.INTEGER, bounds=bounds))
        assert flow_manifest(bounded)["nodes"]["body"]["output_type"]["bounds"] == list(bounds)
        fingerprints.add(flow_fingerprint(bounded))
        digests.add(expansion_digest(bounded))
    assert len(fingerprints) == len(digests) == 4


def test_numeric_bounds_remain_in_nested_parameter_contract_identity():
    from yian.typesys.types import type_for

    first = _fragment()
    first.nodes["body"].params = [Parameter(
        "scores", ValueContract(type_for("list[Optional[float[0, 1]]]", {})),
        has_default=True, default=[0.5],
    )]
    first.wiring["body"] = {}
    second = copy.deepcopy(first)
    second.nodes["body"].params[0] = replace(
        second.nodes["body"].params[0],
        contract=ValueContract(type_for("list[Optional[float[0, 2]]]", {})),
    )
    nested = flow_manifest(first)["nodes"]["body"]["params"][0]["contract"]["type"]["element"]
    assert nested["bounds"] == [0, 1] and nested["nullable"] is True
    assert flow_fingerprint(first) != flow_fingerprint(second)

def test_output_contract_manifest_rejects_unknown_raw_and_subclass_bypasses():
    """Only an exact ValueContract may occupy canonical node output storage."""
    unknown = _fragment()
    object.__setattr__(
        unknown.nodes["body"].output_contract,
        "future_field",
        True,
    )
    with pytest.raises(ValueError, match="future_field"):
        flow_manifest(unknown)

    raw = _fragment()
    raw.nodes["body"].output_contract = Type.scalar(ValueKind.INTEGER)
    with pytest.raises(TypeError, match="expected ValueContract"):
        flow_manifest(raw)

    subclass = _fragment()
    subclass.nodes["body"].output_contract = _ValueContractSubclass(
        Type.scalar(ValueKind.INTEGER)
    )
    with pytest.raises(TypeError, match="expected ValueContract"):
        flow_manifest(subclass)


def test_parameter_manifest_rejects_unknown_fields_and_raw_bypass():
    """The manifest enforces the exact carrier and an explicit field policy."""
    with_unknown = _fragment()
    object.__setattr__(with_unknown.nodes["body"].params[0], "future_field", True)

    with pytest.raises(ValueError, match="future_field"):
        flow_manifest(with_unknown)

    raw = _fragment()
    raw.nodes["body"].params = ["x"]  # type: ignore[list-item]
    with pytest.raises(TypeError, match="expected Parameter"):
        flow_manifest(raw)

    subclass = _fragment()
    subclass.nodes["body"].params = [_ParameterSubclass("x")]
    with pytest.raises(TypeError, match="expected Parameter"):
        flow_manifest(subclass)


@pytest.mark.parametrize("params", [(), (Parameter("x"),)])
def test_parameter_manifest_rejects_noncanonical_containers(params):
    flow = _fragment()
    flow.nodes["body"].params = params  # type: ignore[assignment]

    with pytest.raises(TypeError, match="expected list"):
        flow_manifest(flow)


def _member(tag: str, **fields: ValueKind) -> Type:
    """Build a closed record tagged by a single-value `action` Literal."""
    record_fields = {"action": Type(kind=ValueKind.STRING, tags=frozenset({tag}))}
    record_fields.update({name: Type.scalar(kind) for name, kind in fields.items()})
    return Type(
        kind=ValueKind.OBJECT,
        fields=record_fields,
        required=frozenset(record_fields),
    )


def _union(*members: Type) -> Type:
    return Type(kind=ValueKind.OBJECT, variants=members)


def test_union_free_manifest_has_no_union_key_and_keeps_its_fingerprint():
    fragment = _fragment()
    fragment.nodes["body"].output_contract = ValueContract(_member("buy", qty=ValueKind.INTEGER))
    output_type = flow_manifest(fragment)["nodes"]["body"]["output_type"]

    assert "union" not in output_type
    assert all("union" not in field["type"] for field in output_type["fields"])
    assert flow_fingerprint(_fragment()) == (
        "16d1ca51cfd1c4730aea134054211d87708d8aa08bc8f592b41df96c74982d00"
    )


def test_union_manifest_orders_members_canonically():
    buy = _member("buy", qty=ValueKind.INTEGER)
    sell = _member("sell", price=ValueKind.NUMBER)
    first = _fragment()
    second = copy.deepcopy(first)
    first.nodes["body"].output_contract = ValueContract(_union(buy, sell))
    second.nodes["body"].output_contract = ValueContract(_union(sell, buy))

    union_manifest = flow_manifest(second)["nodes"]["body"]["output_type"]["union"]
    assert [member["fields"][0]["type"]["tags"] for member in union_manifest] == [
        ["buy"],
        ["sell"],
    ]
    assert flow_fingerprint(first) == flow_fingerprint(second)
    assert expansion_digest(first) == expansion_digest(second)


def test_deep_untagged_union_manifests_quickly(monkeypatch):
    def record(**fields: Type) -> Type:
        return Type(kind=ValueKind.OBJECT, fields=fields, required=frozenset(fields))

    depth = 30
    typ = _union(
        record(a=Type.scalar(ValueKind.STRING)), record(b=Type.scalar(ValueKind.INTEGER))
    )
    for level in range(depth):
        typ = _union(record(inner=typ), record(**{f"leaf{level}": Type.scalar(ValueKind.INTEGER)}))
    fragment = _fragment()
    fragment.nodes["body"].output_contract = ValueContract(typ)

    calls = [0]
    render = values_module._canonical_nonnullable_label

    def counted(typ: Type) -> str:
        calls[0] += 1
        return render(typ)

    monkeypatch.setattr(values_module, "_canonical_nonnullable_label", counted)
    flow_fingerprint(fragment)
    # Ordering each level's members labels that level's subtree, so the work is
    # quadratic in depth; the bound rules out the exponential relabelling of nested
    # members.
    assert depth < calls[0] <= 3 * (depth + 2) ** 2


@pytest.mark.parametrize(
    ("first_type", "second_type"),
    [
        (Type.scalar(ValueKind.INTEGER), Type.scalar(ValueKind.NUMBER)),
        (
            Type(kind=ValueKind.OBJECT, fields={"a": Type.scalar(ValueKind.INTEGER)}),
            Type(kind=ValueKind.OBJECT, fields={"b": Type.scalar(ValueKind.INTEGER)}),
        ),
        (
            Type(
                kind=ValueKind.OBJECT,
                fields={"a": Type.scalar(ValueKind.INTEGER)},
                required=frozenset(),
            ),
            Type(
                kind=ValueKind.OBJECT,
                fields={"a": Type.scalar(ValueKind.INTEGER)},
                required=frozenset({"a"}),
            ),
        ),
        (
            Type(kind=ValueKind.STRING, tags=frozenset({"first"})),
            Type(kind=ValueKind.STRING, tags=frozenset({"second"})),
        ),
        (
            Type(kind=ValueKind.LIST_ANY, element=Type.scalar(ValueKind.INTEGER)),
            Type(kind=ValueKind.LIST_ANY, element=Type.scalar(ValueKind.STRING)),
        ),
        (
            Type(kind=ValueKind.STRING, nullable=False),
            Type(kind=ValueKind.STRING, nullable=True),
        ),
        (
            _union(_member("buy", qty=ValueKind.INTEGER), _member("sell")),
            _union(_member("buy", qty=ValueKind.NUMBER), _member("sell")),
        ),
        (
            _union(_member("buy"), _member("sell")),
            _union(_member("buy"), _member("short")),
        ),
        (
            _union(_member("buy"), _member("sell")),
            _union(_member("buy"), _member("sell"), _member("hold")),
        ),
    ],
    ids=[
        "kind",
        "fields",
        "required",
        "tags",
        "element",
        "nullable",
        "variants-member-field",
        "variants-tag",
        "variants-member-added",
    ],
)
def test_flow_fingerprint_covers_each_structural_type_field(first_type, second_type):
    first = _fragment()
    second = copy.deepcopy(first)
    first.nodes["body"].output_contract = ValueContract(first_type)
    second.nodes["body"].output_contract = ValueContract(second_type)

    assert flow_fingerprint(first) != flow_fingerprint(second)


def test_flow_fingerprint_preserves_structural_field_order_used_by_agent_schema():
    first_type = Type(
        kind=ValueKind.OBJECT,
        fields={
            "first": Type.scalar(ValueKind.INTEGER),
            "second": Type.scalar(ValueKind.STRING),
        },
    )
    second_type = Type(
        kind=ValueKind.OBJECT,
        fields={
            "second": Type.scalar(ValueKind.STRING),
            "first": Type.scalar(ValueKind.INTEGER),
        },
    )
    first_schema = type_to_schema(first_type)
    second_schema = type_to_schema(second_type)
    assert first_schema is not None
    assert second_schema is not None
    assert list(first_schema.model_json_schema()["properties"]) == ["first", "second"]
    assert list(second_schema.model_json_schema()["properties"]) == ["second", "first"]

    first = _fragment()
    second = copy.deepcopy(first)
    first.nodes["body"].output_contract = ValueContract(first_type)
    second.nodes["body"].output_contract = ValueContract(second_type)

    assert flow_fingerprint(first) != flow_fingerprint(second)


def test_start_manifest_uses_canonical_parameters():
    parameter = Parameter(
        "x",
        contract=ValueContract(Type.scalar(ValueKind.INTEGER)),
        required=True,
        coerce_strings=True,
    )
    start = StartNode("node", params=[parameter])

    manifest = flow_manifest(_single_node_flow(start))["nodes"]["node"]

    assert manifest["params"][0]["name"] == "x"
    assert manifest["params"][0]["contract"]["type"]["kind"] == "int"


def test_flow_fingerprint_preserves_start_parameter_order():
    first_parameter = Parameter("first")
    second_parameter = Parameter("second")
    first = StartNode("node", params=[first_parameter, second_parameter])
    second = StartNode("node", params=[second_parameter, first_parameter])

    assert flow_fingerprint(_single_node_flow(first)) != flow_fingerprint(
        _single_node_flow(second)
    )


@pytest.mark.parametrize(
    ("factory", "first_kwargs", "second_kwargs"),
    [
        (CodeNode, {"code": 'return "first"'}, {"code": 'return "second"'}),
        (AgentNode, {"prompt": "first"}, {"prompt": "second"}),
        (AgentNode, {"prompt": "p", "tools": ["first"]}, {"prompt": "p", "tools": ["second"]}),
        (AgentNode, {"prompt": "p", "retries": 1}, {"prompt": "p", "retries": 2}),
        (
            AgentNode,
            {"prompt": "p", "llm_config": {"provider": "openai"}},
            {"prompt": "p", "llm_config": {"provider": "anthropic"}},
        ),
        (
            AgentNode,
            {"prompt": "p", "llm_inherit": True},
            {"prompt": "p", "llm_inherit": False},
        ),
        (ToolNode, {"tool_id": "first"}, {"tool_id": "second"}),
        (
            CaseNode,
            {"cases": [Case("first", "x")]},
            {"cases": [Case("second", "x")]},
        ),
        (
            CaseNode,
            {"cases": [Case("branch", "first")]},
            {"cases": [Case("branch", "second")]},
        ),
        (
            CaseNode,
            {"cases": [Case("first", "x"), Case("second", "y")]},
            {"cases": [Case("second", "y"), Case("first", "x")]},
        ),
        (HumanInputNode, {"prompt": "first"}, {"prompt": "second"}),
        (
            HumanInputNode,
            {"prompt": "p", "answer_schema": [{"name": "first"}]},
            {"prompt": "p", "answer_schema": [{"name": "second"}]},
        ),
        (
            HumanInputNode,
            {"questions": [{"question": "first"}]},
            {"questions": [{"question": "second"}]},
        ),
        (
            HumanInputNode,
            {"questions_input": "first"},
            {"questions_input": "second"},
        ),
        (
            HumanInputNode,
            {"prompt": "p", "title": "first"},
            {"prompt": "p", "title": "second"},
        ),
        (
            CallNode,
            {"flow_id": "first", "child": _fragment()},
            {"flow_id": "second", "child": _fragment()},
        ),
        (
            CallNode,
            {"flow_id": "child", "flow_version": 1, "child": _fragment()},
            {"flow_id": "child", "flow_version": 2, "child": _fragment()},
        ),
        (
            CallNode,
            {"flow_id": "child", "child": _fragment()},
            {"flow_id": "child", "child": _changed_fragment()},
        ),
        (
            MapNode,
            {"flow_id": "first", "child": _fragment()},
            {"flow_id": "second", "child": _fragment()},
        ),
        (
            MapNode,
            {"flow_id": "child", "flow_version": 1, "child": _fragment()},
            {"flow_id": "child", "flow_version": 2, "child": _fragment()},
        ),
        (
            MapNode,
            {"flow_id": "child", "child": _fragment()},
            {"flow_id": "child", "child": _changed_fragment()},
        ),
        (
            LoopNode,
            {"flow_id": "first", "child": _fragment()},
            {"flow_id": "second", "child": _fragment()},
        ),
        (
            LoopNode,
            {"flow_id": "child", "flow_version": 1, "child": _fragment()},
            {"flow_id": "child", "flow_version": 2, "child": _fragment()},
        ),
        (
            LoopNode,
            {"flow_id": "child", "child": _fragment()},
            {"flow_id": "child", "child": _changed_fragment()},
        ),
        (
            LoopNode,
            {"flow_id": "child", "child": _fragment(), "predicate_kind": "while"},
            {"flow_id": "child", "child": _fragment(), "predicate_kind": "until"},
        ),
        (
            LoopNode,
            {"flow_id": "child", "child": _fragment(), "predicate": "first"},
            {"flow_id": "child", "child": _fragment(), "predicate": "second"},
        ),
        (
            LoopNode,
            {"flow_id": "child", "child": _fragment(), "times": 1},
            {"flow_id": "child", "child": _fragment(), "times": 2},
        ),
        (
            LoopNode,
            {"flow_id": "child", "child": _fragment(), "max_iters": 1},
            {"flow_id": "child", "child": _fragment(), "max_iters": 2},
        ),
        (
            LoopNode,
            {"flow_id": "child", "child": _fragment(), "max_iters": None},
            {"flow_id": "child", "child": _fragment(), "max_iters": 5},
        ),
        (
            LoopNode,
            {"flow_id": "child", "child": _fragment(), "iteration": 1},
            {"flow_id": "child", "child": _fragment(), "iteration": 2},
        ),
        (
            LoopNode,
            {"flow_id": "child", "child": _fragment(), "returned_fields": None},
            {"flow_id": "child", "child": _fragment(), "returned_fields": ("x",)},
        ),
    ],
    ids=[
        "code-source",
        "agent-prompt",
        "agent-tools",
        "agent-retries",
        "agent-own-llm-config",
        "agent-llm-inherit",
        "tool-id",
        "case-handle",
        "case-when",
        "case-order",
        "human-prompt",
        "human-answer-schema",
        "human-questions",
        "human-questions-input",
        "human-title",
        "call-flow-id",
        "call-flow-version",
        "call-child",
        "map-flow-id",
        "map-flow-version",
        "map-child",
        "loop-flow-id",
        "loop-flow-version",
        "loop-child",
        "loop-predicate-kind",
        "loop-predicate",
        "loop-times",
        "loop-max-iters",
        "loop-max-iters-none",
        "loop-iteration",
        "loop-returned-fields",
    ],
)
def test_flow_fingerprint_covers_each_included_kind_field(
    factory, first_kwargs, second_kwargs
):
    first = factory("node", **first_kwargs)
    second = factory("node", **second_kwargs)

    assert flow_fingerprint(_single_node_flow(first)) != flow_fingerprint(
        _single_node_flow(second)
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            EndNode.record("node", output_names=["first"]),
            EndNode.record("node", output_names=["second"]),
        ),
        (EndNode.list_("node", n=1), EndNode.list_("node", n=2)),
        (EndNode.record("node", output_names=[]), EndNode.list_("node", n=0)),
    ],
    ids=["end-record-names", "end-list-n", "end-mode"],
)
def test_flow_fingerprint_covers_each_end_field(first, second):
    first.params = second.params = None

    assert flow_fingerprint(_single_node_flow(first)) != flow_fingerprint(
        _single_node_flow(second)
    )


def test_value_end_manifest_keeps_the_record_encoding():
    manifest = flow_manifest(_single_node_flow(EndNode.value("t", output_name="result")))

    assert manifest["nodes"]["t"]["mode"] == "record"
    assert manifest["nodes"]["t"]["n"] is None
    assert manifest["nodes"]["t"]["names"] == [{"type": "str", "value": "result"}]
    assert "keyed" not in manifest["nodes"]["t"]


@pytest.mark.parametrize("names", [[], ["result"]], ids=["zero", "one"])
def test_zero_and_one_name_record_end_manifests_are_keyed(names):
    value_end = EndNode.value("t", output_name=names[0] if names else None)
    record_end = EndNode.record("t", output_names=names)

    value_manifest = flow_manifest(_single_node_flow(value_end))["nodes"]["t"]
    record_manifest = flow_manifest(_single_node_flow(record_end))["nodes"]["t"]
    assert record_manifest["keyed"] is True
    assert record_manifest == {**value_manifest, "keyed": True}


def test_two_name_record_end_manifest_is_not_keyed():
    end = EndNode.record("t", output_names=["a", "b"])

    assert "keyed" not in flow_manifest(_single_node_flow(end))["nodes"]["t"]


_PIN_FLOW = """
id: pin
name: pin
nodes:
  step:
    kind: code
    output: str
    code: |
      return "filled"
%s"""

_PIN_DEF_FLOW = """
id: pdef
name: pdef
defs:
  greet:
    kind: code
    output: str
    code: |
      return "hi"
nodes:
  c:
    kind: call
    call: greet
output: ${c.output}
"""


# These pins guard that the value form, an absent output, and multi-field records keep their
# encoding when END record semantics change; the END-manifest tests above are the primary
# guard.
@pytest.mark.parametrize(
    ("output_section", "expected"),
    [
        (
            "output: ${step.output}\n",
            "86083b953d7368601eef3283de683f3fc74a56e71d88a8f5cedc85a7df8f0b7c",
        ),
        ("", "a8c3106bd5e10a83d0d7e1e740ad3c47327e3631a41660a83e23be82f9041278"),
        (
            "output:\n  order: ${step.output}\n  note: ${step.output}\n",
            "6aaae9f1cab8afbf121830b92d53d4e7272f4e10f67445c38b91208ee3b307ad",
        ),
    ],
    ids=["value", "none", "two-fields"],
)
def test_loaded_flow_output_forms_keep_their_fingerprints(output_section, expected):
    flow = load_flow(_PIN_FLOW % output_section).compiled

    assert flow_fingerprint(flow) == expected


def test_loaded_one_field_record_flow_has_its_pinned_fingerprint():
    flow = load_flow(_PIN_FLOW % "output:\n  order: ${step.output}\n").compiled

    assert flow_fingerprint(flow) == (
        "dacabe0326df2af4e935cec6b57bdb3abcdc401c93af076e60c900a553c84203"
    )


def test_loaded_zero_and_one_field_records_differ_from_their_value_forms():
    named = flow_fingerprint(load_flow(_PIN_FLOW % "output:\n  result: ${step.output}\n").compiled)
    value = flow_fingerprint(load_flow(_PIN_FLOW % "output: ${step.output}\n").compiled)
    empty = flow_fingerprint(load_flow(_PIN_FLOW % "output: {}\n").compiled)
    absent = flow_fingerprint(load_flow(_PIN_FLOW % "").compiled)

    assert named != value
    assert empty != absent


def test_call_over_a_single_node_def_keeps_its_fingerprint():
    flow = load_flow(_PIN_DEF_FLOW).compiled

    assert flow_fingerprint(flow) == (
        "350227763e763bc56cdfce0b6392ebcd6b408012e816d9e464820705b9b363f5"
    )


def test_loaded_value_output_and_single_node_def_use_a_value_end():
    flow = load_flow(_PIN_FLOW % "output: ${step.output}\n").compiled
    assert flow.nodes[flow.end_id].returns_record is False

    call = load_flow(_PIN_DEF_FLOW).compiled.nodes["c"]
    assert call.child.nodes[call.child.end_id].returns_record is False


@pytest.mark.parametrize(
    ("attribute", "changed_value"),
    [
        ("id", "changed-edge"),
        ("from_", "changed-source"),
        ("to", "changed-target"),
        ("source_handle", "branch"),
        ("input_group", "value"),
        ("optional", True),
        ("ordering", False),
    ],
    ids=["id", "from", "to", "source-handle", "input-group", "optional", "ordering"],
)
def test_flow_fingerprint_covers_each_edge_field(attribute, changed_value):
    original = _fragment()
    changed = copy.deepcopy(original)
    changed.edges[0] = replace(changed.edges[0], **{attribute: changed_value})

    assert flow_fingerprint(changed) != flow_fingerprint(original)


@pytest.mark.parametrize(
    ("first_outputs", "second_outputs"),
    [
        ([FlowOutput("first", 1)], [FlowOutput("second", 1)]),
        ([FlowOutput("value", 1)], [FlowOutput("value", 2)]),
        (
            [FlowOutput("first", 1), FlowOutput("second", 2)],
            [FlowOutput("second", 2), FlowOutput("first", 1)],
        ),
    ],
    ids=["name", "from", "order"],
)
def test_flow_fingerprint_covers_flow_output_fields_and_order(first_outputs, second_outputs):
    first = _compiled()
    second = copy.deepcopy(first)
    first.outputs = first_outputs
    second.outputs = second_outputs

    assert flow_fingerprint(first) != flow_fingerprint(second)


@pytest.mark.parametrize(
    ("attribute", "changed_value"),
    [("start_id", "body"), ("end_id", "body")],
    ids=["start-id", "end-id"],
)
def test_flow_fingerprint_covers_flow_boundary_ids(attribute, changed_value):
    original = _fragment()
    changed = copy.deepcopy(original)
    setattr(changed, attribute, changed_value)

    assert flow_fingerprint(changed) != flow_fingerprint(original)


def test_expansion_digest_excludes_effective_but_includes_authored_llm_config():
    first = AgentNode("node", prompt="answer", llm_config={"provider": "openai"})
    changed_effective = copy.deepcopy(first)
    changed_effective.llm_config = {"provider": "anthropic"}
    changed_authored = AgentNode(
        "node", prompt="answer", llm_config={"provider": "anthropic"}
    )

    assert expansion_digest(_single_node_flow(first)) == expansion_digest(
        _single_node_flow(changed_effective)
    )
    assert expansion_digest(_single_node_flow(first)) != expansion_digest(
        _single_node_flow(changed_authored)
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (ToolNode("node", tool_id="first"), ToolNode("node", tool_id="second")),
        (
            CaseNode("node", [Case("first", "x"), Case("second", "y")]),
            CaseNode("node", [Case("second", "y"), Case("first", "x")]),
        ),
        (
            HumanInputNode("node", prompt="first", answer_schema=[{"name": "answer"}]),
            HumanInputNode("node", prompt="second", answer_schema=[{"name": "answer"}]),
        ),
        (
            CallNode("node", flow_id="first", child=_fragment()),
            CallNode("node", flow_id="second", child=_fragment()),
        ),
        (
            LoopNode(
                "node",
                flow_id="child",
                child=_fragment(),
                predicate_kind="while",
                predicate="keep_going",
                max_iters=2,
            ),
            LoopNode(
                "node",
                flow_id="child",
                child=_fragment(),
                predicate_kind="while",
                predicate="still_going",
                max_iters=2,
            ),
        ),
        (
            EndNode.record("node", output_names=["result"]),
            EndNode.list_("node", n=1),
        ),
    ],
    ids=["tool", "case-order", "human-input", "call", "loop", "end-mode"],
)
def test_flow_manifest_covers_remaining_kind_execution_fields(first, second):
    assert flow_fingerprint(_single_node_flow(first)) != flow_fingerprint(_single_node_flow(second))


def test_flow_manifest_includes_human_input_title_but_ignores_inert_map_parallel():
    first_human = HumanInputNode("node", prompt="answer", title="First title")
    second_human = HumanInputNode("node", prompt="answer", title="Second title")
    serial_map = MapNode("node", flow_id="child", child=_fragment(), parallel=False)
    parallel_map = MapNode("node", flow_id="child", child=_fragment(), parallel=True)

    assert flow_fingerprint(_single_node_flow(first_human)) != flow_fingerprint(
        _single_node_flow(second_human)
    )
    assert flow_fingerprint(_single_node_flow(serial_map)) == flow_fingerprint(
        _single_node_flow(parallel_map)
    )


def test_flow_manifest_allows_shared_child_aliases_but_rejects_child_cycles():
    shared = _fragment()
    aliased = Flow(
        nodes={
            "left": CallNode("left", flow_id="child", child=shared),
            "right": CallNode("right", flow_id="child", child=shared),
        },
        edges=[],
        wiring={"left": {}, "right": {}},
        start_id="left",
        end_id="right",
    )
    flow_manifest(aliased)

    cyclic = Flow(nodes={}, edges=[], wiring={}, start_id="call", end_id="call")
    cyclic.nodes["call"] = CallNode("call", flow_id="self", child=cyclic)
    cyclic.wiring["call"] = {}
    with pytest.raises(ValueError, match="child-flow cycle"):
        flow_manifest(cyclic)


_LOOP_IDENTITY = """
id: count
name: count
nodes:
  loop:
    kind: loop
    input:
      n: 0
    %s
    max: 5
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: ${bump.output}
output: ${loop.output.n}
"""


@pytest.mark.parametrize(
    ("predicate", "digest"),
    [
        (
            "while: n < 3",
            "6a791c968ef6c0b1b7d8ea90563cc251c5316a41dbb96f1c1b77fa29bd564b53",
        ),
        (
            "until: n >= 3",
            "12570531551fa07862337f35d38e2d864095b9393d3fa786a756d2915d0561dd",
        ),
    ],
)
def test_loop_with_max_keeps_its_fingerprint(predicate, digest):
    # Pinned before `max:` became optional: a loop that sets `max:` keeps its identity.
    compiled = load_flow(_LOOP_IDENTITY % predicate).compiled
    assert flow_fingerprint(compiled) == digest


def test_loop_without_max_has_null_max_in_manifest_and_its_own_fingerprint():
    src_with_max = _LOOP_IDENTITY % "while: n < 3"
    src_without = src_with_max.replace("    max: 5\n", "")
    with_max = load_flow(src_with_max).compiled
    without = load_flow(src_without).compiled

    assert flow_manifest(without)["nodes"]["loop"]["max_iters"] is None
    assert flow_manifest(with_max)["nodes"]["loop"]["max_iters"] == 5
    assert flow_fingerprint(without) != flow_fingerprint(with_max)
    # Loading the same source again gives the same fingerprint.
    reloaded = load_flow(src_without).compiled
    assert flow_fingerprint(without) == flow_fingerprint(reloaded)
