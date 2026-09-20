"""Offline stdio app-server fixture; arguments are scenario and optional audit file."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "normal"
AUDIT = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def fake_worker():
    """Exercise the supervisor without importing the optional SDK."""
    from agent_composer.llm_clients.codex.protocol import Decoder, encode

    decoder = Decoder()
    replies = 0
    turns = 0
    callback_sequence = 0

    def emit(kind, identity, payload):
        os.write(1, encode(kind, identity, payload))

    def tool_request(identity):
        nonlocal callback_sequence
        callback_sequence += 1
        params = {
            "threadId": "actual-thread",
            "turnId": f"turn-{turns}",
            "callId": "native-call",
            "tool": "lookup",
            "arguments": {},
        }
        suffix = SCENARIO.removeprefix("fake_worker_tool_")
        if suffix == "wrong_thread":
            params["threadId"] = "wrong"
        elif suffix == "wrong_turn":
            params["turnId"] = "wrong"
        elif suffix == "missing_call":
            del params["callId"]
        elif suffix == "empty_call":
            params["callId"] = ""
        elif suffix in {"missing_thread", "missing_turn"}:
            del params["threadId" if suffix == "missing_thread" else "turnId"]
        emit(
            "callback",
            identity,
            {
                "callback_id": callback_sequence,
                "method": "item/tool/call",
                "params": params,
            },
        )

    while True:
        chunk = os.read(0, 65536)
        if not chunk:
            return
        for frame in decoder.feed(chunk):
            if frame["kind"] == "shutdown":
                return
            if frame["kind"] == "callback_result":
                replies += 1
                if SCENARIO == "fake_worker_tool_duplicate" and replies == 1:
                    tool_request(frame["id"])
                else:
                    terminal = {
                        "threadId": "actual-thread",
                        "turn": {
                            "id": f"turn-{turns}",
                            "status": "completed",
                            "items": [],
                        },
                    }
                    emit(
                        "event",
                        frame["id"],
                        {"method": "turn/completed", "params": terminal},
                    )
                    emit("result", frame["id"], {"value": terminal})
                continue
            if frame["kind"] != "command":
                continue
            identity, op = frame["id"], frame["payload"]["op"]
            if SCENARIO.startswith("fake_worker_result_"):
                batch = (
                    encode(
                        "event",
                        identity,
                        {"method": "offline/before_result_fault", "params": {}},
                    )
                    + encode("result", identity, {"value": {}})
                    + encode(
                        "event",
                        identity,
                        {"method": "offline/after_result_fault", "params": {}},
                    )
                )
                if SCENARIO == "fake_worker_result_broken":
                    batch += b"\x00\x00\x00\x01{"
                os.write(1, batch)
                if SCENARIO == "fake_worker_result_exit":
                    return
                continue
            if SCENARIO.startswith("fake_worker_tool_") and op == "run_turn":
                turns += 1
                if SCENARIO == "fake_worker_tool_broken_ipc":
                    emit(
                        "turn",
                        identity,
                        {"threadId": "actual-thread", "turnId": f"turn-{turns}"},
                    )
                    emit(
                        "event",
                        identity,
                        {"method": "offline/before_fault", "params": {}},
                    )
                    tool_request(identity)
                    emit(
                        "event",
                        identity,
                        {
                            "method": "offline/observed_after_callback",
                            "params": {"actual": 42},
                        },
                    )
                    time.sleep(0.1)
                    os.write(1, b"\x00\x00\x00\x03bad")
                    continue
                if SCENARIO == "fake_worker_tool_after_complete":
                    emit(
                        "turn",
                        identity,
                        {"threadId": "actual-thread", "turnId": f"turn-{turns}"},
                    )
                    terminal = {
                        "threadId": "actual-thread",
                        "turn": {
                            "id": f"turn-{turns}",
                            "status": "completed",
                            "items": [],
                        },
                    }
                    emit(
                        "event",
                        identity,
                        {"method": "turn/completed", "params": terminal},
                    )
                    tool_request(identity)
                    continue
                # Deliberately precedes the authoritative turn-start response.
                tool_request(identity)
                if SCENARIO == "fake_worker_tool_two_pending":
                    tool_request(identity)
                if SCENARIO == "fake_worker_tool_start_failed":
                    emit("fault", identity, {"code": "sdk"})
                elif SCENARIO == "fake_worker_tool_start_closed":
                    emit("closed", 0, {"ok": True})
                else:
                    emit(
                        "event",
                        identity,
                        {"method": "offline/before_turn", "params": {}},
                    )
                    emit(
                        "turn",
                        identity,
                        {"threadId": "actual-thread", "turnId": f"turn-{turns}"},
                    )
                continue
            if SCENARIO == "fake_worker_malformed":
                os.write(1, b"\x00\x00\x00\x03bad")
                continue
            if SCENARIO == "fake_worker_wrong_id":
                identity += 1
            if SCENARIO == "fake_worker_partial":
                os.write(1, b"\x00\x00\x00\x05x")
                return
            if SCENARIO == "fake_worker_exit":
                return
            if SCENARIO == "fake_worker_invalid_thread" and op == "start_thread":
                os.write(
                    1, encode("result", identity, {"value": {"thread": {"id": 1}}})
                )
                continue
            result = {} if op == "initialize" else {"thread": {"id": "actual-thread"}}
            if op == "start_thread":
                os.write(
                    1,
                    encode(
                        "event",
                        identity,
                        {"method": "offline/before_result", "params": {}},
                    ),
                )
            if SCENARIO == "fake_worker_late_result" and op == "start_thread":
                time.sleep(0.12)
            os.write(1, encode("result", identity, {"value": result}))
            if SCENARIO == "fake_worker_duplicate" and op == "start_thread":
                os.write(1, encode("result", identity, {"value": result}))


if SCENARIO.startswith("fake_worker"):
    fake_worker()
    raise SystemExit
LOCK = threading.Lock()
CHILD = None
if SCENARIO in {"descendants", "stuck_shutdown"}:
    CHILD = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
if SCENARIO == "stuck_shutdown":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def send(value):
    with LOCK:
        print(json.dumps(value), flush=True)


def notify(method, params):
    send({"method": method, "params": params})


def thread_response(identity):
    return {
        "approvalPolicy": "never",
        "approvalsReviewer": "user",
        "cwd": "/tmp",
        "model": "offline",
        "modelProvider": "openai",
        "sandbox": {"type": "readOnly"},
        "thread": {
            "id": identity,
            "cliVersion": "0.155.1",
            "createdAt": 1,
            "updatedAt": 1,
            "cwd": "/tmp",
            "ephemeral": False,
            "modelProvider": "openai",
            "preview": "",
            "sessionId": "offline-session",
            "source": "appServer",
            "status": {"type": "idle"},
            "turns": [],
        },
    }


def finish(thread_id, turn_id, status="completed"):
    item = {
        "id": "message-1",
        "type": "agentMessage",
        "text": "offline answer",
        "phase": "final_answer",
    }
    notify(
        "item/agentMessage/delta",
        {
            "threadId": thread_id,
            "turnId": turn_id,
            "itemId": "message-1",
            "delta": "offline ",
        },
    )
    notify("item/completed", {"threadId": thread_id, "turnId": turn_id, "item": item})
    turn = {"id": turn_id, "status": status, "items": [item]}
    if status == "failed":
        turn["error"] = {
            "message": "offline native failure",
            "codexErrorInfo": "other",
            "additionalDetails": None,
        }
    notify("turn/completed", {"threadId": thread_id, "turn": turn})


thread_id, turn_id = "thread-1", "turn-1"
for line in sys.stdin:
    message = json.loads(line)
    if AUDIT is not None:
        with AUDIT.open("a") as output:
            output.write(json.dumps(message) + "\n")
    method, identity = message.get("method"), message.get("id")
    params = message.get("params", {})
    if method == "initialize":
        if SCENARIO == "init_failure":
            send(
                {
                    "id": identity,
                    "error": {"code": -32000, "message": "api_key=secret-marker"},
                }
            )
            continue
        send({"id": identity, "result": {"serverInfo": {"version": "0.155.1"}}})
        notify(
            "offline/started",
            {
                "server_pid": os.getpid(),
                "child_pid": None if CHILD is None else CHILD.pid,
            },
        )
    elif method in {"thread/start", "thread/resume"}:
        thread_id = params.get("threadId", "thread-1")
        send({"id": identity, "result": thread_response(thread_id)})
    elif method == "turn/start":
        if SCENARIO == "reader_death":
            print("invalid-json-secret-marker", flush=True)
            continue
        send(
            {
                "id": identity,
                "result": {
                    "turn": {"id": turn_id, "status": "inProgress", "items": []}
                },
            }
        )
        if SCENARIO == "callback":
            send(
                {
                    "id": "callback-1",
                    "method": "item/tool/call",
                    "params": {
                        "threadId": thread_id,
                        "turnId": turn_id,
                        "callId": "call-1",
                        "tool": "lookup",
                        "arguments": {"value": 1},
                    },
                }
            )
        elif SCENARIO in {"quiet", "stuck_shutdown"}:
            pass
        elif SCENARIO == "slow":
            threading.Timer(0.6, finish, args=(thread_id, turn_id)).start()
        else:
            finish(
                thread_id, turn_id, "failed" if SCENARIO == "failed" else "completed"
            )
    elif method == "turn/interrupt":
        send({"id": identity, "result": {}})
        if SCENARIO != "stuck_shutdown":
            finish(thread_id, turn_id, "interrupted")
    elif identity == "callback-1" and "result" in message:
        notify("offline/callback_result", {"result": message["result"]})
        finish(thread_id, turn_id)
if SCENARIO == "stuck_shutdown":
    time.sleep(60)
