"""Run identity and ambient UTC clock seeding for the typed variable pool."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only (keeps state import-clean)
    from agent_composer.typesys.pool import VariablePool


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def now_utc() -> str:
    """The run's wall-clock 'now' as an ISO-8601 datetime (UTC)."""
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    """A fresh run identifier for when the host doesn't supply one.

    A run id is inherently unique per run, so the boundary mints one (uuid4 hex)
    unless the host injects a meaningful value — mirroring how the clock ambients
    are auto-seeded, so `${system.run_id}` always resolves.
    """
    return uuid.uuid4().hex


def seed_system_clock(pool: "VariablePool") -> None:
    """Seed the host-ambient clock into the pool's `system` namespace, once per run.

    `${system.today}` is a `date`; `${system.now}` a `datetime`. Computed once here and
    inherited by REF/MAP children (they copy `pool.system`), so every node, sub-flow,
    and MAP element observes one consistent clock. The clock is the only seam through
    which a flow may read "now" (the engine never resolves date words).
    """
    from agent_composer.typesys.values import DateValue, DateTimeValue

    pool.system["today"] = DateValue(value=today_utc())
    pool.system["now"] = DateTimeValue(value=now_utc())
