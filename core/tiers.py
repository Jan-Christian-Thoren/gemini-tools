import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "gemini-tiers.yml"
TIER_NAMES = ("snabb", "standard", "djup")
COMMAND_RE = re.compile(r"^\s*/gemini\s+(snabb|standard|djup)\b[ \t]*(.*)", re.IGNORECASE | re.DOTALL)
LEVEL_MARKER = "- **Nivå:** `"


@dataclass(frozen=True)
class Tier:
    name: str
    models: tuple[str, ...]
    thinking_level: str
    max_output_tokens: int
    grounding: bool
    request_timeout_s: int = 600
    run_budget_s: int = 1200


@dataclass(frozen=True)
class TierChoice:
    tier: Optional[str]          # None: ingen körning (nekad eller inget att göra)
    source: str                  # default | label | kommando
    extra_prompt: str = ""
    denied: str = ""             # sanerad orsak när ett val nekas


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for name in TIER_NAMES:
        if name not in cfg["tiers"]:
            raise ValueError(f"nivån {name} saknas i {path.name}")
    return cfg


def get_tier(cfg: dict, name: str) -> Tier:
    name = cfg.get("aliases", {}).get(name, name)
    t = cfg["tiers"][name]
    return Tier(name, tuple(t["models"]), t["thinking_level"], int(t["max_output_tokens"]), bool(t["grounding"]),
                int(t.get("request_timeout_s", 600)), int(t.get("run_budget_s", 1200)))


def _labels(item: dict) -> list[str]:
    return [l.get("name", "") for l in (item or {}).get("labels") or []]


def select_tier(cfg: dict, event_name: str, event: dict) -> TierChoice:
    """Välj nivå för en GitHub-händelse. Kommando > label > default; ingen automatisk eskalering."""
    item = event.get("pull_request") or event.get("issue") or {}

    if event_name == "issue_comment":
        comment = event.get("comment") or {}
        m = COMMAND_RE.match(comment.get("body") or "")
        if not m:
            return TierChoice(None, "kommando", denied="ingen /gemini-rad")
        tier, extra = m.group(1).lower(), m.group(2).strip()
        author = (comment.get("user") or {}).get("login", "")
        if comment.get("author_association") not in cfg.get("command_associations", []):
            return TierChoice(None, "kommando", denied="kommandot får bara användas av repots medlemmar")
        if tier == "djup" and author.lower() not in {u.lower() for u in cfg.get("djup_command_users") or []}:
            return TierChoice(None, "kommando", denied="djup via kommando kräver repo-ägaren (djup_command_users)")
        return TierChoice(tier, "kommando", extra)

    ai = [l.split(":", 1)[1] for l in _labels(item) if l.startswith("ai:") and l.split(":", 1)[1] in TIER_NAMES]
    if event_name in ("pull_request", "issues") and event.get("action") == "labeled":
        added = ((event.get("label") or {}).get("name") or "")
        if added.startswith("ai:") and added.split(":", 1)[1] in TIER_NAMES:
            return TierChoice(added.split(":", 1)[1], "label")
    if ai:
        # flera ai-labels: billigaste vinner, så att en kvarglömd ai:djup inte tar över i onödan
        return TierChoice(min(ai, key=TIER_NAMES.index), "label")
    return TierChoice(cfg.get("default_tier", "snabb"), "default")


def load_github_event() -> tuple[str, dict]:
    name = os.environ.get("GITHUB_EVENT_NAME", "")
    path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not path or not Path(path).is_file():
        return name, {}
    with open(path, encoding="utf-8") as f:
        return name, json.load(f)


def djup_recently_used(comment_bodies_with_time: list[tuple[str, datetime]], cfg: dict,
                       now: Optional[datetime] = None) -> bool:
    """True om djup redan körts för samma PR/issue inom en timme (räknat på bot-kommentarer)."""
    now = now or datetime.now(timezone.utc)
    limit = int(cfg.get("djup_per_item_per_hour", 1))
    recent = [t for body, t in comment_bodies_with_time
              if f"{LEVEL_MARKER}djup" in (body or "") and now - t < timedelta(hours=1)]
    return len(recent) >= limit


def estimate_cost_sek(cfg: dict, model: Optional[str], usage: dict) -> Optional[float]:
    """Uppskattad kostnad i kr. Resonemangstokens debiteras som utdata."""
    prices = (cfg.get("pricing") or {}).get("models", {}).get(model or "")
    if not prices or not usage:
        return None
    usd = (usage.get("input", 0) * prices["input"]
           + (usage.get("output", 0) + usage.get("thinking", 0)) * prices["output"]) / 1_000_000
    return usd * float(cfg["pricing"].get("usd_to_sek", 0))


def truncate_input(text: str, cfg: dict) -> tuple[str, bool]:
    """Kapar text över max_input_bytes. Returnerar (text, trunkerad)."""
    limit = int(cfg.get("max_input_bytes", 200000))
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text, False
    return raw[:limit].decode("utf-8", errors="ignore"), True


def should_run(event_name: str, event: dict, choice: TierChoice, has_previous_report: bool,
               recent_djup: bool) -> tuple[bool, str]:
    """Om en körning ska ske, och annars varför (tom sträng = tyst hoppa över).

    Automatiska händelser (öppnad PR, ny push, utredningsetikett) körs bara en gång per PR/issue,
    så att varje push inte kostar. Ett aktivt val (ai:-label eller /gemini) kör alltid, men djup
    högst en gång per timme."""
    if choice.tier is None:
        return False, "" if choice.denied == "ingen /gemini-rad" else choice.denied
    action = event.get("action", "")
    added = ((event.get("label") or {}).get("name") or "")
    active = event_name == "issue_comment" or (action == "labeled" and added.startswith("ai:"))
    if not active and has_previous_report:
        return False, ""
    if choice.tier == "djup" and recent_djup:
        return False, "djup har redan körts för detta ärende den senaste timmen"
    return True, ""


def command_or_label_note(choice: TierChoice) -> str:
    return f"{choice.tier} ({choice.source})"


LEVEL_ORDER = ("minimal", "low", "medium", "high")


def supported_thinking_level(cfg: dict, model: str, level: str) -> str:
    """Nivåns thinking_level justerad till vad modellen stöder: samma om den stöds, annars
    närmaste stödda nivå uppåt (sist: högsta stödda). Okänd modell: oförändrad."""
    params = (cfg.get("model_params") or {}).get(model)
    if not params or not level:
        return level
    allowed = [l.lower() for l in params.get("thinking_levels") or []]
    if not allowed or level.lower() in allowed:
        return level if allowed else ""
    start = LEVEL_ORDER.index(level.lower()) if level.lower() in LEVEL_ORDER else 0
    for l in LEVEL_ORDER[start:]:
        if l in allowed:
            return l
    return allowed[-1]


# --- Samlad start: flera etikett-händelser i följd ger bara en körning ----------------------------
SETTLE_ACTIONS = ("opened", "labeled", "reopened")


def settle_seconds(cfg: dict, event_name: str, event: dict) -> int:
    """Väntetid före modellanropet för etikett- och öppna-händelser. Workflowens concurrency
    (cancel-in-progress) avbryter då en tidigare körning när nästa händelse kommer, innan någon
    modell anropats, och den sista körningen läser issuet/PR:en på nytt efter väntan."""
    if event_name in ("issues", "pull_request") and event.get("action") in SETTLE_ACTIONS:
        return int(cfg.get("settle_s", 0))
    return 0


def settle(cfg: dict, event_name: str, event: dict, sleep=time.sleep) -> int:
    s = settle_seconds(cfg, event_name, event)
    if s:
        print(f"Väntar {s} s på fler etikett-händelser innan körningen", flush=True)
        sleep(s)
    return s


def with_fresh_labels(event: dict, label_names: list[str]) -> dict:
    """Kopia av händelsen där PR:ens/issuets etiketter ersatts med de aktuella."""
    ev = dict(event)
    key = "pull_request" if "pull_request" in ev and "issue" not in ev else "issue"
    item = dict(ev.get(key) or {})
    item["labels"] = [{"name": n} for n in label_names]
    ev[key] = item
    return ev


# --- Månadsbudget per repo ------------------------------------------------------------------------
COST_RE = re.compile(r"\*\*Uppskattad kostnad:\*\* `([0-9]+(?:\.[0-9]+)?) kr`")


def month_start(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def spent_this_month(bot_comment_bodies: list[str]) -> float:
    """Summan av uppskattade kostnader i bot-kommentarernas sidfötter (anroparen filtrerar på
    författare och månad)."""
    total = 0.0
    for body in bot_comment_bodies:
        for m in COST_RE.finditer(body or ""):
            total += float(m.group(1))
    return total


def apply_budget(cfg: dict, choice: TierChoice, spent_sek: float) -> TierChoice:
    """Över månadsbudgeten körs nivåerna i budget.applies_to som snabb i stället. Sänker bara."""
    b = cfg.get("budget") or {}
    limit = b.get("monthly_sek_per_repo")
    if choice.tier is None or limit is None or choice.tier not in (b.get("applies_to") or []):
        return choice
    if spent_sek < float(limit):
        return choice
    note = (f"{choice.source}, nedgraderad från {choice.tier}: månadsbudget "
            f"{spent_sek:.2f} av {float(limit):.0f} kr använd")
    return TierChoice(b.get("fallback_tier", "snabb"), note, choice.extra_prompt)
