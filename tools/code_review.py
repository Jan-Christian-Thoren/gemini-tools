import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.client import get_github_context, load_system_role, run_resilient_generation

DEFAULT_ROLE = (
    "Du är en Principal Software Engineer. Utför en noggrann, saklig och konstruktiv "
    "kodgranskning (Code Review) av bifogade ändringar med fokus på buggar, "
    "prestanda, säkerhet och arkitektur."
)

REVIEW_HEADER = "## 🔍 Gemini Code Review"
STATUS_FAILURE_HEADER = "## ⚠️ Gemini Code Review – Körning avbruten"

def main():
    pr_number = int(os.environ["PR_NUMBER"])
    _, repo = get_github_context()
    pr = repo.get_pull(pr_number)

    print(f"Hanterar PR #{pr_number}: {pr.title}")

    # Dubblettkontroll
    for comment in pr.get_issue_comments():
        if REVIEW_HEADER in comment.body:
            print(f"En granskning finns redan postad på PR #{pr_number}. Avbryter körning.")
            sys.exit(0)

    # Samla in diffar
    diff_entries = []
    total_chars = 0
    max_chars = 64 * 1024
    ignored = ["package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock"]

    for f in pr.get_files():
        if f.status == "removed" or any(f.filename.endswith(ign) for ign in ignored):
            continue
        patch = f.patch or ""
        if not patch:
            continue

        total_chars += len(patch)
        if total_chars > max_chars:
            diff_entries.append(f"### Fil: `{f.filename}`\n*(Diff trunkerad: storleksgräns nådd)*\n")
            break
        diff_entries.append(f"### Fil: `{f.filename}` ({f.status})\n```diff\n{patch}\n```\n")

    if not diff_entries:
        print("Inga granskningsbara ändringar hittades.")
        sys.exit(0)

    prompt = f"""
### Pull Request Granskning: #{pr_number} - {pr.title}

**Beskrivning:**
{pr.body if pr.body else 'Ingen beskrivning angiven.'}

---
### Ändrad källkod (Diff)
{"".join(diff_entries)}

---
Vänligen leverera en strukturerad kodgranskningsrapport i ren Markdown.
"""

    profile_name = os.getenv("GEMINI_PROFILE", "standard_review").strip() or "standard_review"
    role_text, role_source = load_system_role(DEFAULT_ROLE)
    print(f"Rollkälla: {role_source}")
    result = run_resilient_generation(
        prompt=prompt,
        system_instruction=role_text,
        role_source=role_source,
        profile_name=profile_name,
    )

    if not result.success:
        pr.create_issue_comment(f"""{STATUS_FAILURE_HEADER}

Kodgranskningen kunde inte slutföras med godkänd modell. Inget tekniskt resultat har genererats.

{result.audit_md}
""")
        print("Körning avbruten; statusrapport postad.")
        sys.exit(1)

    final_comment = f"""{REVIEW_HEADER} ({result.chosen_model})

{result.text}
{result.sources_md}

---
{result.audit_md}
*Automatiserad granskning genererad via Gemini API ({len(diff_entries)} filer analyserade).*"""

    pr.create_issue_comment(final_comment)
    print(f"Granskning postad med {result.chosen_model}.")

if __name__ == "__main__":
    main()
