import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import tiers
from core.client import apply_repo_budget, get_github_context, load_system_role, run_resilient_generation

DEFAULT_ROLE = (
    "Du är en Principal Software Engineer. Utför en noggrann, saklig och konstruktiv "
    "kodgranskning (Code Review) av bifogade ändringar med fokus på buggar, "
    "prestanda, säkerhet och arkitektur."
)

REVIEW_HEADER = "## 🔍 Gemini Code Review"
STATUS_FAILURE_HEADER = "## ⚠️ Gemini Code Review – Körning avbruten"
STATUS_SKIPPED_HEADER = "## ⏸️ Gemini Code Review – ingen körning"

def main():
    pr_number = int(os.environ["PR_NUMBER"])
    _, repo = get_github_context()
    cfg = tiers.load_config()
    event_name, event = tiers.load_github_event()
    tiers.settle(cfg, event_name, event)
    pr = repo.get_pull(pr_number)  # läses efter väntan: aktuell text och aktuella etiketter
    event = tiers.with_fresh_labels(event, [l.name for l in pr.labels])
    choice = tiers.select_tier(cfg, event_name, event)

    print(f"Hanterar PR #{pr_number}: {pr.title} (händelse {event_name or 'okänd'})")

    comments = list(pr.get_issue_comments())
    has_previous = any(REVIEW_HEADER in (c.body or "") for c in comments)
    recent_djup = tiers.djup_recently_used([(c.body, c.created_at) for c in comments], cfg)
    run, note = tiers.should_run(event_name, event, choice, has_previous, recent_djup)
    if run:
        choice = apply_repo_budget(repo, cfg, choice)
    if not run:
        if note:
            pr.create_issue_comment(f"{STATUS_SKIPPED_HEADER}\n\n{note}. Ingen modell anropades.")
        print(f"Ingen körning: {note or 'redan granskad eller inget val'}")
        sys.exit(0)

    # Samla in diffar
    diff_entries = []
    ignored = ["package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock"]
    for f in pr.get_files():
        if f.status == "removed" or any(f.filename.endswith(ign) for ign in ignored):
            continue
        patch = f.patch or ""
        if patch:
            diff_entries.append(f"### Fil: `{f.filename}` ({f.status})\n```diff\n{patch}\n```\n")

    if not diff_entries:
        print("Inga granskningsbara ändringar hittades.")
        sys.exit(0)

    diff_text, truncated = tiers.truncate_input("".join(diff_entries), cfg)
    if truncated:
        diff_text += f"\n\n*(Diff trunkerad vid {cfg['max_input_bytes']} byte)*\n"
    extra = f"\n---\n### Extra fråga från {event['comment']['user']['login']}\n{choice.extra_prompt}\n" \
        if choice.extra_prompt else ""

    prompt = f"""
### Pull Request Granskning: #{pr_number} - {pr.title}

**Beskrivning:**
{pr.body if pr.body else 'Ingen beskrivning angiven.'}

---
### Ändrad källkod (Diff)
{diff_text}
{extra}
---
Vänligen leverera en strukturerad kodgranskningsrapport i ren Markdown.
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
    input_note = f"\n*Indata trunkerad vid {cfg['max_input_bytes']} byte.*" if truncated else ""

    if not result.success:
        pr.create_issue_comment(f"""{STATUS_FAILURE_HEADER}

Kodgranskningen kunde inte slutföras med godkänd modell. Inget tekniskt resultat har genererats.

{result.audit_md}{input_note}
""")
        print("Körning avbruten; statusrapport postad.")
        sys.exit(1)

    final_comment = f"""{REVIEW_HEADER} ({result.chosen_model})

{result.text}
{result.sources_md}

---
{result.audit_md}{input_note}
*Automatiserad granskning genererad via Gemini API ({len(diff_entries)} filer analyserade).*"""

    pr.create_issue_comment(final_comment)
    print(f"Granskning postad med {result.chosen_model}.")


if __name__ == "__main__":
    main()
