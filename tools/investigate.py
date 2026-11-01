import os
import re
import sys
from pathlib import Path

# Lägg till repots rot så att core/ kan importeras oavsett var skriptet körs ifrån
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import tiers
from core.client import get_github_context, load_system_role, run_resilient_generation

DEFAULT_ROLE = (
    "Du är en teknisk utredare och senior mjukvaruarkitekt. "
    "Ditt uppdrag är att analysera frågeställningen i detta ärende, "
    "granska eventuellt bifogade filer och föreslå robusta åtgärder."
)

REPORT_HEADER = "## 📋 Gemini Utredningsrapport"
STATUS_FAILURE_HEADER = "## ⚠️ Gemini Utredning – Körning avbruten"
STATUS_SKIPPED_HEADER = "## ⏸️ Gemini Utredning – ingen körning"

def main():
    issue_number = int(os.environ["ISSUE_NUMBER"])
    _, repo = get_github_context()
    issue = repo.get_issue(number=issue_number)
    cfg = tiers.load_config()
    event_name, event = tiers.load_github_event()
    choice = tiers.select_tier(cfg, event_name, event)

    print(f"Hanterar Issue #{issue_number}: {issue.title} (händelse {event_name or 'okänd'})")

    comments = list(issue.get_comments())
    has_previous = any(REPORT_HEADER in (c.body or "") for c in comments)
    recent_djup = tiers.djup_recently_used([(c.body, c.created_at) for c in comments], cfg)
    run, note = tiers.should_run(event_name, event, choice, has_previous, recent_djup)
    if not run:
        if note:
            issue.create_comment(f"{STATUS_SKIPPED_HEADER}\n\n{note}. Ingen modell anropades.")
        print(f"Ingen körning: {note or 'redan utredd eller inget val'}")
        sys.exit(0)

    # Läs in och bifoga nämnda filer, inom nivåkonfigurationens storleksgräns
    body_text = issue.body if issue.body else ""
    raw_matches = re.findall(
        r"`([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)`|(?:\s|^)([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)",
        body_text,
    )
    candidate_paths = {m[0] or m[1] for m in raw_matches}

    attached_files = []
    budget = int(cfg["max_input_bytes"])
    skipped = []
    for file_str in sorted(candidate_paths):
        path = Path(file_str.strip("./"))
        if not path.is_file() or str(path).startswith(".git/"):
            continue
        try:
            size = path.stat().st_size
            if size > budget:
                skipped.append(str(path))
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            budget -= size
            attached_files.append(f"### Fil: `{path}`\n```\n{content}\n```\n")
            print(f"Bifogade fil: {path}")
        except Exception as e:
            print(f"Kunde inte läsa {path}: {type(e).__name__}")

    context_block = ("\n---\n### Källkodsfiler från repot\n" + "\n".join(attached_files)) if attached_files else ""
    if skipped:
        context_block += f"\n*(Ej bifogade, över gränsen {cfg['max_input_bytes']} byte: {', '.join(skipped)})*\n"
    extra = f"\n---\n### Extra fråga från {event['comment']['user']['login']}\n{choice.extra_prompt}\n" \
        if choice.extra_prompt else ""

    prompt = f"""
### Uppdragsbeskrivning (GitHub Issue #{issue_number})
**Titel:** {issue.title}

**Beskrivning:**
{body_text if body_text else 'Ingen beskrivning angiven.'}
{context_block}
{extra}
---
Vänligen leverera en utredningsrapport formaterad i ren Markdown.
"""

    role_text, role_source = load_system_role(DEFAULT_ROLE)
    print(f"Rollkälla: {role_source}, nivå: {choice.tier} ({choice.source})")
    result = run_resilient_generation(
        prompt=prompt,
        system_instruction=role_text,
        role_source=role_source,
        profile_name=choice.tier,
        tier_source=choice.source,
    )
    skipped_note = f"\n*Filer över storleksgränsen bifogades inte: {', '.join(skipped)}.*" if skipped else ""

    if not result.success:
        issue.create_comment(f"""{STATUS_FAILURE_HEADER}

Utredningen kunde inte genomföras med godkänd analysmodell. Inget tekniskt resultat har genererats.

{result.audit_md}{skipped_note}
""")
        print("Körning avbruten; statusrapport postad.")
        sys.exit(1)

    attached_summary = f"\n*Bifogade källkodsfiler: {len(attached_files)} st.*" if attached_files else ""
    final_comment = f"""{REPORT_HEADER} ({result.chosen_model})

{result.text}
{result.sources_md}

---
{result.audit_md}{skipped_note}
*Automatiserad rapport genererad via Gemini API.{attached_summary}*"""

    issue.create_comment(final_comment)
    print(f"Rapport postad med {result.chosen_model}.")


if __name__ == "__main__":
    main()
