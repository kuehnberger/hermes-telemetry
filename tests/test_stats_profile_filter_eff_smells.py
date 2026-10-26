"""Tests for the /stats --profile filter on the efficiency and smells views.

Milestone 2 of issue #69: extends the profile filter introduced for
summary/providers/models to the two remaining stats views.

Covers:
  - db.efficiency_runs(profile=...) filtering (including the NULL sentinel)
  - smell_detector.detect_all(profile=...) / detect_by_session(profile=...)
    filtering across the five detectors, including the tool_calls-joining
    detectors (loop_trap, tool_thrashing) which filter on r.profile
  - stats.handle('efficiency --profile ...') and ('smells --profile ...')
"""

from __future__ import annotations

import hermes_telemetry.db as db
import hermes_telemetry.smell_detector as sd
import hermes_telemetry.stats as stats_mod
import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    db._local.conn = None
    yield
    if getattr(db._local, "conn", None):
        db._local.conn.close()
        db._local.conn = None


def _seed_two_profiles() -> None:
    """Seed two named-profile sessions plus one unattributed (NULL) session."""
    now = db._utcnow()
    db.start_run("s-faro", model="m", platform="cli", profile="faro")
    db.record_llm_call("s-faro", now, "model-faro", "prov-a", 10, 5, 0.10, 100)
    db.end_run("s-faro", "ok")
    db.start_run("s-nadia", model="m", platform="cli", profile="nadia")
    db.record_llm_call("s-nadia", now, "model-nadia", "prov-b", 20, 10, 0.20, 100)
    db.end_run("s-nadia", "ok")
    # Unattributed: record_llm_call backfills a runs row with a NULL profile.
    db.record_llm_call("s-orphan", now, "model-orphan", "prov-c", 30, 15, 0.30, 100)
    db.end_run("s-orphan", "ok")


# ---------------------------------------------------------------------------
# db.efficiency_runs
# ---------------------------------------------------------------------------


def test_efficiency_runs_filters_by_profile() -> None:
    _seed_two_profiles()

    faro = db.efficiency_runs(window_hours=24, profile="faro")
    nadia = db.efficiency_runs(window_hours=24, profile="nadia")
    all_rows = db.efficiency_runs(window_hours=24)

    assert [r["session_id"] for r in faro] == ["s-faro"]
    assert [r["session_id"] for r in nadia] == ["s-nadia"]
    assert len(all_rows) == 3


def test_efficiency_runs_null_profile_sentinel() -> None:
    _seed_two_profiles()

    orphan = db.efficiency_runs(window_hours=24, profile="-")
    assert [r["session_id"] for r in orphan] == ["s-orphan"]


def test_handle_efficiency_profile_flag() -> None:
    _seed_two_profiles()
    out = stats_mod.handle("efficiency --profile faro")
    assert "s-faro" in out
    assert "s-nadia" not in out
    assert "s-orphan" not in out
    assert "profile faro" in out


def test_handle_efficiency_null_profile_sentinel() -> None:
    _seed_two_profiles()
    out = stats_mod.handle("efficiency --profile -")
    assert "s-orphan" in out
    assert "s-faro" not in out
    assert "unattributed" in out


# ---------------------------------------------------------------------------
# smell_detector (context_rotation is representative of the runs-based
# detectors; loop_trap / tool_thrashing exercise the tool_calls join path)
# ---------------------------------------------------------------------------


def test_smells_filter_by_profile() -> None:
    """A context_rotation smell in one profile must not leak into another."""
    now = db._utcnow()
    db.start_run("s-rot", model="m", platform="cli", profile="faro")
    db.record_llm_call("s-rot", now, "modelA", "openai", 10000, 500, 0.05, 100)
    db.end_run("s-rot", "ok")
    db.start_run("s-nadia", model="m", platform="cli", profile="nadia")
    db.record_llm_call("s-nadia", now, "modelA", "openai", 100, 50, 0.001, 100)
    db.end_run("s-nadia", "ok")

    faro = sd.detect_all(window_hours=24, profile="faro")
    nadia = sd.detect_all(window_hours=24, profile="nadia")

    faro_ids = {s["session_id"] for s in faro}
    nadia_ids = {s["session_id"] for s in nadia}
    assert "s-rot" in faro_ids
    assert "s-nadia" not in faro_ids
    assert nadia_ids == set()


def test_smells_tool_join_filters_by_profile() -> None:
    """loop_trap joins tool_calls to runs; the r.profile filter must apply."""
    now = db._utcnow()
    db.start_run("s-loop", model="m", platform="cli", profile="faro")
    for _ in range(15):
        db.record_tool_call("s-loop", now, "web_search", True, 100)
    db.end_run("s-loop", "ok")
    db.start_run("s-nadia", model="m", platform="cli", profile="nadia")
    for _ in range(15):
        db.record_tool_call("s-nadia", now, "web_search", True, 100)
    db.end_run("s-nadia", "ok")

    faro = sd.detect_all(window_hours=24, profile="faro")
    nadia = sd.detect_all(window_hours=24, profile="nadia")

    assert {s["session_id"] for s in faro} == {"s-loop"}
    assert {s["session_id"] for s in nadia} == {"s-nadia"}


def test_smells_null_profile_sentinel() -> None:
    """The '-' sentinel selects unattributed (NULL-profile) runs only."""
    now = db._utcnow()
    db.start_run("s-rot", model="m", platform="cli", profile="faro")
    db.record_llm_call("s-rot", now, "modelA", "openai", 10000, 500, 0.05, 100)
    db.end_run("s-rot", "ok")
    # Unattributed context_rotation session (no start_run -> NULL profile).
    db.record_llm_call("s-orphan", now, "modelA", "openai", 10000, 500, 0.05, 100)
    db.end_run("s-orphan", "ok")

    orphan = sd.detect_all(window_hours=24, profile="-")
    assert {s["session_id"] for s in orphan} == {"s-orphan"}


def test_detect_by_session_profile() -> None:
    now = db._utcnow()
    db.start_run("s-rot", model="m", platform="cli", profile="faro")
    db.record_llm_call("s-rot", now, "modelA", "openai", 10000, 500, 0.05, 100)
    db.end_run("s-rot", "ok")

    sessions = sd.detect_by_session(window_hours=24, profile="faro")
    assert [s["session_id"] for s in sessions] == ["s-rot"]


def test_handle_smells_profile_flag() -> None:
    now = db._utcnow()
    db.start_run("s-rot", model="m", platform="cli", profile="faro")
    db.record_llm_call("s-rot", now, "modelA", "openai", 10000, 500, 0.05, 100)
    db.end_run("s-rot", "ok")

    out = stats_mod.handle("smells --profile faro")
    assert "s-rot" in out
    assert "profile faro" in out
