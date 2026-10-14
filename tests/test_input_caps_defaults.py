"""Input-size caps come from `yian.defaults`, and the CLI checks the variables first."""

from __future__ import annotations

import re
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from typer.testing import CliRunner

from yian import defaults
from yian.cli import app
from yian.cli import utils as cli_utils
from yian.cli.run import _ResumeKeyError, _resume_env_key
from yian.compose.skills import load_skill_dir
from yian.llm_clients.auth.common import ApiKeyCredentials, AuthError
from yian.llm_clients.auth.http import _request
from yian.llm_clients.auth.openai import OpenAIAuth
from yian.llm_clients.auth.store import AuthStore
from yian.llm_clients.config import validate_base_url
from yian.llm_clients.model_list import ModelListError, list_models
from yian.skills import Skill, SkillError
from yian.skills.catalog import check_skill_name
from yian.skills.tool import system_prompt_with_skills
from yian.tools.registry import _normalize_options


def _serve_once(body: bytes):
    """Serve one HTTP response with `body` on a loopback port; return (port, thread, errors)."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(2)
    errors = []

    def serve():
        try:
            connection, _ = server.accept()
            with connection:
                connection.settimeout(2)
                request = bytearray()
                while b"\r\n\r\n" not in request:
                    chunk = connection.recv(4096)
                    if not chunk:
                        raise ConnectionError("client closed before sending headers")
                    request.extend(chunk)
                connection.sendall(
                    f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n".encode() + body
                )
        except Exception as error:
            errors.append(error)
        finally:
            server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return server.getsockname()[1], thread, errors


@pytest.fixture
def no_proxy(monkeypatch):
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")


def test_base_url_cap(set_defaults):
    url = "https://a.example/v1"
    assert len(url) == 20
    set_defaults(MAX_BASE_URL_CHARS=20)
    assert validate_base_url(url) == url
    with pytest.raises(ValueError, match="at most 20 characters"):
        validate_base_url(url + "x")


def test_skill_file_cap(tmp_path: Path, set_defaults):
    directory = tmp_path / "s"
    directory.mkdir()
    (directory / "SKILL.md").write_bytes(b"x" * 11)
    set_defaults(MAX_SKILL_FILE_BYTES=10)
    with pytest.raises(SkillError, match="larger than 10 bytes"):
        load_skill_dir(directory)


def test_skill_name_cap(set_defaults):
    set_defaults(MAX_SKILL_NAME_CHARS=3)
    assert check_skill_name("abc") == "abc"
    with pytest.raises(SkillError, match="use 1-3 lowercase"):
        check_skill_name("abcd")


def test_skill_description_cap(set_defaults):
    set_defaults(MAX_SKILL_DESCRIPTION_CHARS=5)
    prompt = system_prompt_with_skills("base", [Skill("a", "abcdefgh", "")])
    assert prompt.endswith("- a: abcd…")


def test_question_option_caps(set_defaults):
    set_defaults(MAX_QUESTION_OPTIONS=2, MAX_QUESTION_OPTION_CHARS=5)
    assert _normalize_options(["abcdefg", "b", "c"]) == ["abcd…", "b"]


def test_key_prompt_cap(monkeypatch, set_defaults):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    prompts = []

    def password(label, **kwargs):
        prompts.append(label)
        return SimpleNamespace(ask=lambda: "")

    monkeypatch.setitem(sys.modules, "questionary", SimpleNamespace(password=password))
    set_defaults(MAX_KEY_PROMPTS=1)
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        cli_utils.ensure_api_key("anthropic")
    assert len(prompts) == 1


def test_resume_key_prompt_cap(monkeypatch, set_defaults):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    asked = []
    set_defaults(MAX_KEY_PROMPTS=1)
    with pytest.raises(_ResumeKeyError, match="no API key provided"):
        _resume_env_key(
            "openai",
            "OPENAI_API_KEY",
            None,
            ask_secret=lambda label: asked.append(label) or "",
            prompt_available=True,
        )
    assert len(asked) == 1


def test_auth_store_cap(set_defaults):
    AuthStore().save("anthropic", ApiKeyCredentials("sk-saved"), "claude-x")
    assert AuthStore().read().selections
    set_defaults(MAX_AUTH_STORE_BYTES=10)
    with pytest.raises(AuthError, match="size limit"):
        AuthStore().read()


def test_auth_store_cap_applies_to_writes(set_defaults):
    set_defaults(MAX_AUTH_STORE_BYTES=10)
    with pytest.raises(AuthError, match="size limit"):
        AuthStore().save("anthropic", ApiKeyCredentials("sk-saved"), "claude-x")


@pytest.mark.parametrize("limit", [10, None])
def test_auth_response_cap(no_proxy, set_defaults, limit):
    body = b'{"a": "xx"}'
    assert len(body) == 11
    if limit is not None:
        set_defaults(MAX_AUTH_RESPONSE_BYTES=limit)
    port, thread, errors = _serve_once(body)
    try:
        if limit is None:
            response = _request("GET", f"http://127.0.0.1:{port}", headers={}, timeout=2)
            assert response.data == {"a": "xx"}
        else:
            with pytest.raises(ValueError, match="too large"):
                _request("GET", f"http://127.0.0.1:{port}", headers={}, timeout=2)
    finally:
        thread.join(3)
    assert not thread.is_alive()
    assert not errors


@pytest.mark.parametrize("limit", [10, None])
def test_model_list_cap(no_proxy, set_defaults, limit):
    body = b'{"models": []}'
    if limit is not None:
        set_defaults(MAX_MODEL_LIST_BYTES=limit)
    port, thread, errors = _serve_once(body)
    try:
        base_url = f"http://127.0.0.1:{port}"
        if limit is None:
            assert list_models("ollama", None, base_url=base_url, timeout=2) == ()
        else:
            with pytest.raises(ModelListError, match="invalid response"):
                list_models("ollama", None, base_url=base_url, timeout=2)
    finally:
        thread.join(3)
    assert not thread.is_alive()
    assert not errors


def test_oauth_callback_cap(set_defaults):
    set_defaults(MAX_OAUTH_CALLBACK_BYTES=20)
    replies = []
    errors = []
    threads = []

    def open_url(url):
        state = parse_qs(urlparse(url).query)["state"][0]

        def send():
            try:
                with socket.create_connection(("127.0.0.1", 1455), timeout=2) as connection:
                    connection.sendall(
                        f"GET /auth/callback?code=code&state={state} HTTP/1.1\r\n"
                        "Host: localhost:1455\r\n\r\n".encode()
                    )
                    replies.append(connection.recv(4096))
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=send)
        threads.append(thread)
        thread.start()

    def no_http(*args, **kwargs):
        raise AssertionError("no token exchange expected")

    try:
        with pytest.raises(AuthError, match="timed out"):
            OpenAIAuth(request=no_http).browser_login(open_url, timeout=0.5)
    finally:
        for thread in threads:
            thread.join(3)
    assert not errors
    assert replies and replies[0].startswith(b"HTTP/1.1 400")


def test_cli_reports_a_bad_variable_without_a_traceback(monkeypatch, tmp_path: Path):
    flow = tmp_path / "flow.yaml"
    flow.write_text("id: f\nname: f\nnodes: {}\n", encoding="utf-8")
    monkeypatch.setenv("YIAN_MAX_KEY_PROMPTS", "abc")
    # The autouse fixture already read a clean environment.
    defaults._forget()

    result = CliRunner().invoke(app, ["run", str(flow)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    text = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))
    assert "YIAN_MAX_KEY_PROMPTS='abc' is not an integer" in text
    assert "Traceback" not in text
