"""Tests for the /stats --profile filter on the cron and raw views (issue #69).

cron and raw are runs-based (no llm_calls join), so they reuse
``db._profile_where`` against ``runs.profile``. Covers the db layer, the slash
path (``stats.handle``) and the standalone CLI path (``stats cron``).
"""

from __future__ import annotations

import json

import hermes_telemetry.db as db
import hermes_telemetry.stats as stats_mod
import pytest
from hermes_telemetry.telemetry_cli import main


@pytest.fixture(autouse=True)
def isolated_db():
    db._local.conn = None
    yield
    if getattr(db._local, "conn", None):
        db._local.conn.close()
        db._local.conn = None


def _seed() -> None:
    """Two named profiles plus one NULL-profile run, each a distinct cron job."""
    db.start_run("s-faro", model="m-faro", platform="cron", cron_job_id="job-faro", profile="faro")
    db.start_run(
        "s-nadia", model="m-nadia", platform="cron", cron_job_id="job-nadia", profile="nadia"
    )
    db.start_run("s-none", model="m-none", platform="cron", cron_job_id="job-none")


# ---------------------------------------------------------------------------
# db layer
# ---------------------------------------------------------------------------


def test_cost_by_job_filters_by_profile() -> None:
    _seed()
    assert {r["cron_job_id"] for r in db.cost_by_job(window_hours=24)} == {
        "job-faro",
        "job-nadia",
        "job-none",
    }
    assert [r["cron_job_id"] for r in db.cost_by_job(window_hours=24, profile="faro")] == [
        "job-faro"
    ]
    assert [r["cron_job_id"] for r in db.cost_by_job(window_hours=24, profile="-")] == ["job-none"]


def test_recent_runs_filters_by_profile() -> None:
    _seed()
    assert len(db.recent_runs(limit=10, window_hours=24)) == 3
    nadia = db.recent_runs(limit=10, window_hours=24, profile="nadia")
    assert [r["session_id"] for r in nadia] == ["s-nadia"]
    none = db.recent_runs(limit=10, window_hours=24, profile="-")
    assert [r["session_id"] for r in none] == ["s-none"]


def test_unattributed_child_cost_filters_by_profile() -> None:
    db.start_run("c-faro", model="m", platform="cli", profile="faro")
    db.record_llm_call("c-faro", db._utcnow(), "m", "p", 10, 5, 0.02, 100)
    db.record_subagent_start(child_session_id="c-faro", parent_session_id="ghost-1")
    db.start_run("c-nadia", model="m", platform="cli", profile="nadia")
    db.record_llm_call("c-nadia", db._utcnow(), "m", "p", 10, 5, 0.05, 100)
    db.record_subagent_start(child_session_id="c-nadia", parent_session_id="ghost-2")

    since = "2000-01-01T00:00:00+00:00"
    assert db.unattributed_child_cost(since)["edges"] == 2
    faro = db.unattributed_child_cost(since, profile="faro")
    assert faro["edges"] == 1
    assert faro["unattributed_usd"] == pytest.approx(0.02)
    assert db.unattributed_child_cost(since, profile="-")["edges"] == 0


# ---------------------------------------------------------------------------
# slash path
# ---------------------------------------------------------------------------


def test_handle_cron_profile_flag() -> None:
    _seed()
    out = stats_mod.handle("cron --profile faro")
    assert "job-faro" in out
    assert "job-nadia" not in out
    assert "job-none" not in out
    assert "profile faro" in out


def test_handle_cron_null_profile_sentinel() -> None:
    _seed()
    out = stats_mod.handle("cron month --profile -")
    assert "job-none" in out
    assert "job-faro" not in out
    assert "unattributed" in out


def test_handle_cron_empty_result_mentions_profile() -> None:
    _seed()
    out = stats_mod.handle("cron --profile ghost")
    assert out.startswith("No cron runs")
    assert "profile ghost" in out


def test_handle_cron_default_unchanged() -> None:
    _seed()
    out = stats_mod.handle("cron")
    for job in ("job-faro", "job-nadia", "job-none"):
        assert job in out
    assert "profile" not in out


def test_handle_cron_profile_composes_with_date_range() -> None:
    _seed()
    out = stats_mod.handle("cron --profile nadia --from 2020-01-01")
    assert "job-nadia" in out
    assert "job-faro" not in out


def test_handle_raw_profile_flag() -> None:
    _seed()
    out = stats_mod.handle("raw --profile faro")
    assert "job-faro" in out
    assert "job-nadia" not in out
    assert "job-none" not in out
    assert "profile faro" in out


def test_handle_raw_null_profile_sentinel() -> None:
    _seed()
    out = stats_mod.handle("raw 10 --profile -")
    assert "job-none" in out
    assert "job-faro" not in out
    assert "unattributed" in out


def test_handle_raw_profile_composes_with_date_range() -> None:
    _seed()
    out = stats_mod.handle("raw --profile nadia --from 2020-01-01")
    assert "job-nadia" in out
    assert "job-faro" not in out


def test_handle_raw_default_unchanged() -> None:
    _seed()
    out = stats_mod.handle("raw")
    for job in ("job-faro", "job-nadia", "job-none"):
        assert job in out


def test_handle_raw_empty_result_mentions_profile() -> None:
    _seed()
    out = stats_mod.handle("raw --profile ghost")
    assert out.startswith("No runs")
    assert "profile ghost" in out


# ---------------------------------------------------------------------------
# CLI path
# ---------------------------------------------------------------------------


def test_cli_stats_cron_text_profile(capsys) -> None:
    _seed()
    main(["stats", "cron", "--profile", "faro"])
    out, _ = capsys.readouterr()
    assert "job-faro" in out
    assert "job-nadia" not in out
    assert "profile faro" in out


def test_cli_stats_cron_text_null_profile(capsys) -> None:
    _seed()
    main(["stats", "cron", "--profile", "-"])
    out, _ = capsys.readouterr()
    assert "job-none" in out
    assert "job-faro" not in out


def test_cli_stats_cron_json_profile(capsys) -> None:
    _seed()
    main(["stats", "cron", "--json", "--profile", "nadia"])
    out, _ = capsys.readouterr()
    assert [r["cron_job_id"] for r in json.loads(out)] == ["job-nadia"]
