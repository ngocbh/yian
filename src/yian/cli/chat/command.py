"""``yian chat`` — a broker-driven interactive flow.

Loads the bundled yian chat flow (or a user-supplied one) and runs it behind the
main-thread terminal broker. HUMAN_INPUT blocks inside the live run; assistant replies
are enqueued before the next prompt. Textual exit is flow-owned. Only terminal EOF is
normalized by the broker to ``/exit``.

A single bad turn does not end the session. The reply agent can fail mid-turn (a raise,
or hitting its tool-iteration cap), which terminates the underlying flow *run* as
`failed`. Because the transcript is mirrored host-side off the turn body's terminal
(`__end__`) output, the loop surfaces the error and RESTARTS the flow seeded with the
accumulated transcript, so the conversation continues from where it left off. Every
initial run and failed-turn restart gets its own reported durable log.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from functools import partial
import os
from pathlib import Path
import select
import sys
from threading import Lock
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from yian.cli.chat import tools as chat_tools
from yian.cli.run import (  # reused helpers
    _default_run_log_path,
    _ensure_cwd_importable,
    _ensure_provider_keys,
    _last_segment,
    _prompt_request,
    _report_run_log,
    _render_load_error,
    _run_with_terminal_controller,
)
from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.run import _run_flow_controlled
from yian.events import NodeSucceeded
from yian.tools import registered_tools
from yian.typesys.seeding import default_run_id

console = Console()
err_console = Console(stderr=True)  # errors go to stderr, mirroring `yian run` (run.py:45-46)

# The yian chat flow shipped with the package — the default when no flow is given.
_BUNDLED = Path(__file__).parent / "chat.yaml"


def _prompt_chat_request(request) -> Any:
    """Read one chat line on the broker-owned main thread.

    Structured question sets and questions with options use the general CLI
    renderer. The ordinary chat surface is deliberately line-oriented: it preserves
    typed text exactly (apart from the line ending), raises EOF for the broker to
    normalize, and applies no textual command policy.
    """
    if request.questions or request.options:
        return _prompt_request(request)
    label = request.question or f"input for {request.node_id}"
    console.print(Text(f"{label} ", style="bold cyan"), end="")
    try:
        input_fd = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        # Click's in-memory CliRunner stdin has no descriptor, but its scripted
        # line is already available and therefore cannot hold controller cleanup.
        input_fd = None

    if input_fd is None:
        line = sys.stdin.readline()
        is_eof = line == ""
    else:
        # Read exactly one line from the descriptor. TextIOWrapper.readline() may
        # prefetch later pasted lines into its private buffer, making the next
        # select() wait despite data already buffered. It can also block after a
        # partial line arrives and stop observing cancellation. Descriptor-specific
        # reads avoid both failures while retaining stdin's codec: byte-exact for
        # pipes and one canonical boundary at a time for a terminal.
        cancel_read, cancel_write = os.pipe()
        raw = bytearray()
        reached_eof = False
        canonical_tty = os.isatty(input_fd)
        cancel_lock = Lock()
        cancel_active = True

        def cancel() -> None:
            # PromptRequest extracts callbacks under its own lock and invokes them
            # afterward. Serialize that delayed invocation with descriptor teardown
            # so a closed numeric fd can never be reused before this write.
            with cancel_lock:
                if not cancel_active:
                    return
                try:
                    os.write(cancel_write, b"x")
                except OSError:
                    return

        request.set_cancel_callback(cancel)
        try:
            while True:
                readable, _, _ = select.select([input_fd, cancel_read], [], [])
                if cancel_read in readable or request.cancelled:
                    raise CancelledError
                # A canonical TTY becomes readable only for one completed line or
                # VEOF. Read that whole boundary at once: byte-at-a-time reads can
                # consume the partial text preceding Ctrl-D and then wait forever
                # for a boundary the terminal driver has already discharged. Pipes
                # stay byte-exact so later pasted lines cannot be prefetched.
                chunk = os.read(input_fd, 65536 if canonical_tty else 1)
                if chunk == b"":
                    reached_eof = True
                    break
                if canonical_tty:
                    raw.extend(chunk)
                    break
                if chunk == b"\n":
                    break
                raw.extend(chunk)
        finally:
            request.set_cancel_callback(None)
            with cancel_lock:
                cancel_active = False
                os.close(cancel_read)
                os.close(cancel_write)
        is_eof = reached_eof and not raw
        if is_eof:
            line = ""
        else:
            encoding = getattr(sys.stdin, "encoding", None) or "utf-8"
            errors = getattr(sys.stdin, "errors", None) or "strict"
            line = bytes(raw).decode(encoding, errors)
    if request.cancelled:
        raise CancelledError
    if is_eof:
        console.print()
        raise EOFError
    return line.rstrip("\r\n")


def chat(
    flow: Optional[Path] = typer.Argument(
        None, exists=True, dir_okay=False, readable=True,
        help="A chat flow .yaml (default: the bundled yian chat).",
    ),
    workspace: Path = typer.Option(
        Path.cwd(), "--workspace", "-C",
        help="Directory the flow-op tools are confined to.",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Override the LLM provider for agents that set none (cascade)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override the LLM model for agents that set none (cascade)."
    ),
) -> None:
    """Start an interactive chat session over a chat flow."""
    _ensure_cwd_importable()  # so a chat flow's `code: pkg.mod:fn` ref resolves (mirrors `yian run`)
    chat_tools.set_workspace(workspace)
    path = flow or _BUNDLED
    text = path.read_text()
    search = [path.parent, workspace]
    # The CLI flags supply the OUTERMOST cascade layer (fill-the-gap), not a hard override.
    cli_cfg = {k: v for k, v in {"provider": provider, "model": model}.items() if v}
    # Load and preflight once, mirroring `yian run` — both failure modes surface as a clean
    # stderr error + exit 1, never a raw traceback:
    #   - a malformed flow raises LoadError (a ValueError) -> a located `.yaml` source frame;
    #   - a missing key with no TTY raises RuntimeError    -> a red key message.
    try:
        loaded = load_flow(text, search_paths=search, source_label=path.name)
    except LoadError as err:
        _render_load_error(err)
        raise typer.Exit(code=1)
    try:
        _ensure_provider_keys(loaded, cli_cfg)
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    # The last completed turn is the restart seed after a reply failure. A failed
    # in-progress turn never reaches its body END, so it cannot alter this mirror.
    state: dict[str, Any] = {"transcript": ""}

    try:
        while True:
            # A node failure sets its run's cooperative stop event to drain siblings.
            # Start each attempt under a fresh broker/controller lifecycle so that a
            # failed run's terminal event never leaks into the next immutable run.
            current_flow = load_flow(
                text,
                search_paths=search,
                source_label=path.name,
            )
            run_id = default_run_id()
            log = _default_run_log_path(run_id)
            _report_run_log(log)

            def controller(stop_event, on_event, ask_user, render):
                """Run one immutable chat attempt on the controller thread."""

                def observe(event: Any) -> None:
                    """Mirror completed turns and queue replies on the terminal FIFO."""
                    if on_event is not None:
                        on_event(event)
                    if not isinstance(event, NodeSucceeded):
                        return
                    segment = _last_segment(event.node_id)
                    if segment == "reply":
                        render(partial(console.print, Markdown(str(event.output))))
                    elif segment == "__end__" and isinstance(event.output, dict):
                        state["transcript"] = event.output.get(
                            "transcript", state["transcript"]
                        )

                with chat_tools._bind_controller_stop(stop_event), chat_tools._bind_ask_user(
                    ask_user
                ):
                    return _run_flow_controlled(
                        current_flow,
                        {"opening": state["transcript"]},
                        log=log,
                        run_id=run_id,
                        on_event=observe,
                        llm_config=cli_cfg or None,
                        stop_event=stop_event,
                        # A tools mapping hides the registry; keep the chat agent's
                        # registered flow-op tools next to this host's handler.
                        tools={**registered_tools(), "ask_user": ask_user},
                    )

            result = _run_with_terminal_controller(
                controller,
                prompt_handler=_prompt_chat_request,
                reporter=None,
                eof_value="/exit",
            )
            if result.status != "failed":
                break
            detail = result.failure.message if result.failure is not None else "(no detail)"
            console.print(f"[red]turn failed:[/red] {detail}")
            console.print("[dim]continuing — your conversation so far is preserved.[/dim]")
    except KeyboardInterrupt:
        err_console.print("[yellow]cancelled by user[/yellow]")
        raise typer.Exit(code=130)
    except (OSError, TypeError, ValueError) as exc:
        message = Text("chat failed: ", style="red bold")
        message.append(str(exc), style="red")
        err_console.print(message)
        raise typer.Exit(code=1)

    console.print("[dim]session ended[/dim]")
