from datetime import datetime, timedelta, timezone

import pytest

from core import tiers

CFG = tiers.load_config()
OWNER = CFG["owner"]


def comment(body, login=OWNER, assoc="OWNER"):
    return {"comment": {"body": body, "user": {"login": login}, "author_association": assoc},
            "issue": {"number": 7, "labels": []}}


def pr(labels=(), action="opened", added=None):
    ev = {"action": action, "pull_request": {"number": 7, "labels": [{"name": l} for l in labels]}}
    if added:
        ev["label"] = {"name": added}
    return ev


def test_config_has_three_tiers_and_no_flash_in_djup():
    assert set(CFG["tiers"]) >= {"snabb", "standard", "djup"}
    assert all("flash" not in m for m in CFG["tiers"]["djup"]["models"])
    assert CFG["default_tier"] == "snabb"


def test_default_is_snabb():
    assert tiers.select_tier(CFG, "pull_request", pr()) == tiers.TierChoice("snabb", "default")


def test_label_selects_tier():
    assert tiers.select_tier(CFG, "pull_request", pr(["ai:standard"])).tier == "standard"
    assert tiers.select_tier(CFG, "pull_request", pr(["ai:djup"], "labeled", "ai:djup")) == tiers.TierChoice("djup", "label")


def test_several_labels_take_the_cheapest():
    assert tiers.select_tier(CFG, "pull_request", pr(["ai:djup", "ai:snabb"])).tier == "snabb"


def test_command_beats_label_and_carries_the_question():
    ch = tiers.select_tier(CFG, "issue_comment", comment("/gemini standard Varför låser sig tråden?"))
    assert (ch.tier, ch.source, ch.extra_prompt) == ("standard", "kommando", "Varför låser sig tråden?")


def test_djup_command_only_for_owner():
    ch = tiers.select_tier(CFG, "issue_comment", comment("/gemini djup analysera", login="someone", assoc="COLLABORATOR"))
    assert ch.tier is None and "ägaren" in ch.denied
    assert tiers.select_tier(CFG, "issue_comment", comment("/gemini djup analysera")).tier == "djup"


def test_command_requires_repo_member():
    ch = tiers.select_tier(CFG, "issue_comment", comment("/gemini snabb", login="x", assoc="NONE"))
    assert ch.tier is None


def test_non_command_comment_does_nothing():
    assert tiers.select_tier(CFG, "issue_comment", comment("bra jobbat")).tier is None
    assert tiers.select_tier(CFG, "issue_comment", comment("/gemini expert")).tier is None


def test_djup_rate_limit_per_hour():
    now = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
    body = "...\n- **Nivå:** `djup (kommando)`\n..."
    assert tiers.djup_recently_used([(body, now - timedelta(minutes=30))], CFG, now)
    assert not tiers.djup_recently_used([(body, now - timedelta(minutes=61))], CFG, now)
    assert not tiers.djup_recently_used([("- **Nivå:** `snabb (default)`", now)], CFG, now)


def test_cost_estimate_counts_thinking_as_output():
    usage = {"input": 1_000_000, "output": 0, "thinking": 1_000_000}
    p = CFG["pricing"]["models"]["gemini-4-argon"]
    expected = (p["input"] + p["output"]) * CFG["pricing"]["usd_to_sek"]
    assert tiers.estimate_cost_sek(CFG, "gemini-4-argon", usage) == pytest.approx(expected)
    assert tiers.estimate_cost_sek(CFG, "unknown-model", usage) is None


def test_input_truncation():
    text, cut = tiers.truncate_input("x" * (CFG["max_input_bytes"] + 10), CFG)
    assert cut and len(text.encode()) == CFG["max_input_bytes"]
    assert tiers.truncate_input("kort", CFG) == ("kort", False)


def test_should_run_auto_events_only_once():
    ch = tiers.TierChoice("snabb", "default")
    assert tiers.should_run("pull_request", {"action": "synchronize"}, ch, True, False) == (False, "")
    assert tiers.should_run("pull_request", {"action": "opened"}, ch, False, False) == (True, "")


def test_should_run_active_choices_always_but_djup_once_an_hour():
    lab = tiers.TierChoice("standard", "label")
    ev = {"action": "labeled", "label": {"name": "ai:standard"}}
    assert tiers.should_run("pull_request", ev, lab, True, False) == (True, "")
    djup = tiers.TierChoice("djup", "kommando")
    run, note = tiers.should_run("issue_comment", {}, djup, True, True)
    assert not run and "timmen" in note


def test_should_run_reports_denied_command_but_ignores_plain_comments():
    assert tiers.should_run("issue_comment", {}, tiers.TierChoice(None, "kommando", denied="ingen /gemini-rad"), False, False) == (False, "")
    run, note = tiers.should_run("issue_comment", {}, tiers.TierChoice(None, "kommando", denied="djup via kommando kräver repo-ägaren"), False, False)
    assert not run and "ägaren" in note


def test_unsupported_thinking_level_is_raised_to_nearest_supported():
    assert tiers.supported_thinking_level(CFG, "gemini-3.8-flash", "minimal") == "low"
    assert tiers.supported_thinking_level(CFG, "gemini-4-argon", "high") == "high"
    assert tiers.supported_thinking_level(CFG, "okand-modell", "minimal") == "minimal"
    assert tiers.supported_thinking_level(CFG, "gemini-2.5-pro", "high") == ""     # budget-modell: ingen nivå


def test_tiers_only_use_models_that_answered_on_vertex_global():
    for t in CFG["tiers"].values():
        assert "gemini-2.5-pro" not in t["models"]
