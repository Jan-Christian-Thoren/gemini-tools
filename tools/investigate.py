import os
import re
import sys
from pathlib import Path

# Lägg till repots rot så att core/ kan importeras oavsett var skriptet körs ifrån
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.client import get_github_context, load_system_role, run_resilient_generation

DEFAULT_ROLE = (
    "Du är en teknisk utredare och senior mjukvaruarkitekt. "
    "Ditt uppdrag är att analysera frågeställningen i detta ärende, "
    "granska eventuellt bifogade filer och föreslå robusta åtgärder."
)

REPORT_HEADER = "## 📋 Gemini Utredningsrapport"
STATUS_FAILURE_HEADER = "## ⚠️ Gemini Utredning – Körning avbruten"

def main():
    issue_number = int(os.environ["ISSUE_NUMBER"])
    _, repo = get_github_context()
    issue = repo.get_issue(number=issue_number)

    print(f"Hanterar Issue #{issue_number}: {issue.title}")

    # Dubblettkontroll
    for comment in issue.get_comments():
        if REPORT_HEADER in comment.body:
            print(f"En utredningsrapport finns redan i #{issue_number}. Avbryter körning.")
            sys.exit(0)

    # Läs in och bifoga nämnda filer
    body_text = issue.body if issue.body else ""
    raw_matches = re.findall(r"`([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)`|(?:\s|^)([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)", body_text)
    candidate_paths = {m[0] or m[1] for m in raw_matches}

    attached_files = []
    max_size = 64 * 1024

    for file_str in sorted(candidate_paths):
        path = Path(file_str.strip("./"))
        if not path.is_file() or str(path).startswith(".git/"):
            continue
        try:
            if path.stat().st_size > max_size:
                attached_files.append(f"### Fil: `{path}`\n*(Överhoppad: filen är större än 64 KB)*\n")
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            attached_files.append(f"### Fil: `{path}`\n```\n{content}\n```\n")
            print(f"Bifogade fil: {path}")
        except Exception as e:
            print(f"Kunde inte läsa {path}: {e}")

    context_block = ("\n---\n### Källkodsfiler från repot\n" + "\n".join(attached_files)) if attached_files else ""
    
    prompt = f"""
### Uppdragsbeskrivning (GitHub Issue #{issue_number})
**Titel:** {issue.title}

**Beskrivning:**
{body_text if body_text else 'Ingen beskrivning angiven.'}
{context_block}

---
Vänligen leverera en utredningsrapport formaterad i ren Markdown.
"""

    role_text, role_source = load_system_role(DEFAULT_ROLE)
    print(f"Rollkälla: {role_source}")
    result = run_resilient_generation(
        prompt=prompt,
        system_instruction=role_text,
        role_source=role_source,
        profile_name="deep_investigation",
    )

    if not result.success:
        issue.create_comment(f"""{STATUS_FAILURE_HEADER}

Utredningen kunde inte genomföras med godkänd analysmodell. Inget tekniskt resultat har genererats.

{result.audit_md}
""")
        print("Körning avbruten; statusrapport postad.")
        sys.exit(1)

    attached_summary = f"\n*Bifogade källkodsfiler: {len(attached_files)} st.*" if attached_files else ""
    final_comment = f"""{REPORT_HEADER} ({result.chosen_model})

{result.text}
{result.sources_md}

---
{result.audit_md}
*Automatiserad rapport genererad via Gemini API.{attached_summary}*"""

    issue.create_comment(final_comment)
    print(f"Rapport postad med {result.chosen_model}.")

if __name__ == "__main__":
    main()
