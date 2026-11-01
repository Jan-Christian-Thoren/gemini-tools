from datetime import datetime, timedelta, timezone

import pytest

from core import tiers

CFG = tiers.load_config()
OWNER = CFG["djup_command_users"][0]


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


def test_djup_command_works_for_org_member_listed_in_config():
    # i ett org-repo är ägarens personkonto MEMBER, aldrig OWNER, och org-namnet kommenterar aldrig
    ch = tiers.select_tier(CFG, "issue_comment", comment("/gemini djup x", login=OWNER, assoc="MEMBER"))
    assert ch.tier == "djup"
    ch = tiers.select_tier(CFG, "issue_comment", comment("/gemini djup x", login=CFG["owner"], assoc="MEMBER"))
    assert ch.tier is None or CFG["owner"] in CFG["djup_command_users"]


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


# --- samlad start ---------------------------------------------------------------------------------

def test_settle_only_for_label_and_open_events():
    waits = []
    issue_labeled = {"action": "labeled", "issue": {"labels": []}, "label": {"name": "utredning"}}
    assert tiers.settle(CFG, "issues", issue_labeled, sleep=waits.append) == CFG["settle_s"] > 0
    assert tiers.settle(CFG, "pull_request", pr(action="opened"), sleep=waits.append) == CFG["settle_s"]
    assert tiers.settle(CFG, "issue_comment", comment("/gemini snabb"), sleep=waits.append) == 0
    assert tiers.settle(CFG, "pull_request", pr(action="synchronize"), sleep=waits.append) == 0
    assert waits == [CFG["settle_s"], CFG["settle_s"]]


def test_issue_created_with_both_labels_runs_at_the_chosen_level_from_either_event():
    # issuet skapades med utredning + ai:djup: två händelser, båda ska ge djup efter omläsning
    fresh = ["utredning", "ai:djup"]
    for added in ("utredning", "ai:djup"):
        ev = {"action": "labeled", "issue": {"number": 7, "labels": [{"name": "utredning"}]},
              "label": {"name": added}}
        ch = tiers.select_tier(CFG, "issues", tiers.with_fresh_labels(ev, fresh))
        assert ch.tier == "djup", added


def test_fresh_labels_replace_the_payload_labels_for_prs():
    ev = tiers.with_fresh_labels(pr(labels=["ai:djup"]), ["ai:snabb"])
    assert tiers.select_tier(CFG, "pull_request", ev).tier == "snabb"


# --- månadsbudget ---------------------------------------------------------------------------------

def footer(kr):
    return f"...\n- **Uppskattad kostnad:** `{kr} kr` (uppskattning från pristabellen i gemini-tiers.yml)\n"


def test_spent_sums_footer_costs():
    assert tiers.spent_this_month([footer("1.29"), footer("0.06"), "inget", None]) == pytest.approx(1.35)


def test_over_budget_downgrades_standard_and_djup_only():
    limit = CFG["budget"]["monthly_sek_per_repo"]
    for t in ("standard", "djup"):
        ch = tiers.apply_budget(CFG, tiers.TierChoice(t, "label", "fråga"), limit + 0.01)
        assert ch.tier == "snabb" and "nedgraderad från " + t in ch.source and ch.extra_prompt == "fråga"
    keep = tiers.TierChoice("snabb", "default")
    assert tiers.apply_budget(CFG, keep, limit * 10) is keep
    under = tiers.TierChoice("djup", "kommando")
    assert tiers.apply_budget(CFG, under, limit - 1) is under


def test_downgraded_run_does_not_count_as_djup_for_the_rate_limit():
    now = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
    body = "- **Nivå:** `snabb (label, nedgraderad från djup: månadsbudget 21.00 av 20 kr använd)`"
    assert not tiers.djup_recently_used([(body, now)], CFG, now)


def test_month_start():
    assert tiers.month_start(datetime(2026, 11, 17, 9, 5, tzinfo=timezone.utc)) == \
        datetime(2026, 11, 1, tzinfo=timezone.utc)
