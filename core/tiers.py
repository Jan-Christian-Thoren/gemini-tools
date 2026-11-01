import json
import os
import re
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
        if tier == "djup" and author != cfg["owner"]:
            return TierChoice(None, "kommando", denied="djup via kommando kräver repo-ägaren")
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
