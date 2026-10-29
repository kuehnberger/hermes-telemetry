"""Tests for the /stats --profile filter (issue #69).

Covers:
  - db.stats_summary / stats_by_provider / stats_by_model profile filtering,
    including join correctness (llm_calls has no profile column; it is joined
    to runs on session_id via a subquery)
  - the NULL-profile bucket via the ``-`` sentinel
  - stats.handle('... --profile ...') slash-command flag parsing
  - telemetry_cli --profile flag wiring (text and JSON paths)
"""

from __future__ import annotations

from unittest.mock import patch

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


def _seed_two_profiles() -> None:
    """Seed runs for two named profiles plus one unattributed (NULL) run."""
    now = db._utcnow()
    db.start_run("s-faro", model="m", platform="cli", profile="faro")
    db.record_llm_call("s-faro", now, "model-faro", "prov-a", 10, 5, 0.10, 100)
    db.start_run("s-nadia", model="m", platform="cli", profile="nadia")
    db.record_llm_call("s-nadia", now, "model-nadia", "prov-b", 20, 10, 0.20, 100)
    # No start_run: record_llm_call backfills a runs row with a NULL profile.
    db.record_llm_call("s-orphan", now, "model-orphan", "prov-c", 30, 15, 0.30, 100)


# ---------------------------------------------------------------------------
# db.stats_summary
# ---------------------------------------------------------------------------


def test_stats_summary_filters_runs_by_profile() -> None:
    _seed_two_profiles()

    all_runs = db.stats_summary(window_hours=24)
    faro = db.stats_summary(window_hours=24, profile="faro")
    nadia = db.stats_summary(window_hours=24, profile="nadia")

    assert all_runs["total_runs"] == 3
    assert faro["total_runs"] == 1
    assert nadia["total_runs"] == 1
    # cost and api_calls isolate to the filtered profile (api_calls comes from
    # llm_calls, so this exercises the session_id subquery join).
    assert abs(faro["cost_usd"] - 0.10) < 1e-9
    assert faro["api_calls"] == 1
    assert abs(nadia["cost_usd"] - 0.20) < 1e-9


def test_stats_summary_exposes_applied_profile_in_result() -> None:
    _seed_two_profiles()
    assert db.stats_summary(window_hours=24, profile="faro")["profile"] == "faro"
    assert db.stats_summary(window_hours=24, profile="-")["profile"] == "-"
    assert db.stats_summary(window_hours=24)["profile"] is None


# ---------------------------------------------------------------------------
# db.stats_by_provider / db.stats_by_model
# ---------------------------------------------------------------------------


def test_stats_by_provider_filters_by_profile() -> None:
    _seed_two_profiles()

    faro = db.stats_by_provider(window_hours=24, profile="faro")
    assert [r["provider"] for r in faro] == ["prov-a"]

    all_rows = db.stats_by_provider(window_hours=24)
    assert len(all_rows) == 3


def test_stats_by_model_filters_by_profile() -> None:
    _seed_two_profiles()

    nadia = db.stats_by_model(window_hours=24, profile="nadia")
    assert len(nadia) == 1
    assert nadia[0]["model"] == "model-nadia"


def test_stats_null_profile_sentinel_selects_unattributed_only() -> None:
    _seed_two_profiles()

    orphan = db.stats_by_model(window_hours=24, profile="-")
    assert len(orphan) == 1
    assert orphan[0]["model"] == "model-orphan"

    summary = db.stats_summary(window_hours=24, profile="-")
    assert summary["total_runs"] == 1
    assert abs(summary["cost_usd"] - 0.30) < 1e-9


# ---------------------------------------------------------------------------
# stats.handle slash-command flag
# ---------------------------------------------------------------------------


def test_handle_models_profile_flag() -> None:
    _seed_two_profiles()
    out = stats_mod.handle("models --profile faro")
    assert "model-faro" in out
    assert "model-nadia" not in out
    assert "model-orphan" not in out
    assert "profile faro" in out


def test_handle_models_null_profile_sentinel() -> None:
    _seed_two_profiles()
    out = stats_mod.handle("models --profile -")
    assert "model-orphan" in out
    assert "model-faro" not in out
    assert "unattributed" in out


def test_handle_summary_profile_flag() -> None:
    _seed_two_profiles()
    out = stats_mod.handle("--profile nadia")
    assert "profile nadia" in out


def test_handle_profile_composes_with_date_range() -> None:
    _seed_two_profiles()
    # A from-date far in the past keeps all rows; profile still applies.
    out = stats_mod.handle("models --profile faro --from 2020-01-01")
    assert "model-faro" in out
    assert "model-nadia" not in out


def test_handle_missing_profile_value_returns_error() -> None:
    out = stats_mod.handle("models --profile")
    assert "Missing value for --profile" in out


# ---------------------------------------------------------------------------
# telemetry_cli --profile flag
# ---------------------------------------------------------------------------


def test_cli_models_profile_text(capsys) -> None:
    with patch("hermes_telemetry.stats._models_block", return_value="M") as m:
        main(["stats", "models", "--profile", "faro"])
    args, kwargs = m.call_args
    assert kwargs.get("profile") == "faro"


def test_cli_models_profile_null_sentinel_text(capsys) -> None:
    with patch("hermes_telemetry.stats._models_block", return_value="M") as m:
        main(["stats", "models", "--profile", "-"])
    args, kwargs = m.call_args
    assert kwargs.get("profile") == "-"


def test_cli_models_profile_json(capsys) -> None:
    fake = [{"model": "x", "cost_usd": 1.0}]
    with patch("hermes_telemetry.telemetry_cli.db.stats_by_model", return_value=fake) as m:
        main(["stats", "models", "--profile", "faro", "--json"])
    args, kwargs = m.call_args
    assert kwargs.get("profile") == "faro"


def test_cli_summary_profile_json(capsys) -> None:
    fake = {"total_runs": 1, "profile": "faro"}
    with patch("hermes_telemetry.telemetry_cli.db.stats_summary", return_value=fake) as m:
        main(["stats", "--profile", "-", "--json"])
    args, kwargs = m.call_args
    assert kwargs.get("profile") == "-"
