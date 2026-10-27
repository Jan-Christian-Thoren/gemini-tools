import os
import sys
import time
from pathlib import Path
from github import Github
from google import genai
from google.genai.errors import APIError
import httpx

def get_github_context():
    """Hämtar och initierar GitHub-klienten och måltrepot."""
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("REPOSITORY")
    if not token or not repo_name:
        raise ValueError("Miljövariablerna GITHUB_TOKEN och REPOSITORY måste vara definierade.")
    gh = Github(token)
    return gh, gh.get_repo(repo_name)

def load_system_role(default_role: str) -> str:
    """
    Läser in .github/gemini_role.md från det anropande repot.
    Faller tillbaka på repots miljövariabel eller angiven default-roll.
    """
    role_candidates = [
        Path(".github/gemini_role.md"),
        Path(".github/gemini-role.md")
    ]
    for path in role_candidates:
        if path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                print(f"Laddade lokal domänroll från: {path}")
                return content

    env_role = os.environ.get("GEMINI_SYSTEM_ROLE", "").strip()
    if env_role:
        print("Laddade roll från miljövariabeln GEMINI_SYSTEM_ROLE.")
        return env_role

    print("Ingen lokal rollfil hittades, använder standardsystemroll.")
    return default_role

def get_gemini_client():
    """Initierar Google GenAI-klienten."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("Miljövariabeln GEMINI_API_KEY saknas.")
    return genai.Client(api_key=key)

def discover_candidate_models(client: genai.Client) -> list[str]:
    """Hämtar tillgängliga modeller i runtime och prioriterar Pro före Flash."""
    print("Identifierar tillgängliga modeller via API:et...")
    available = []
    try:
        for m in client.models.list():
            actions = getattr(m, "supported_actions", []) or getattr(m, "supported_generation_methods", [])
            if "generateContent" in actions:
                available.append(m.name.replace("models/", ""))
    except Exception as err:
        print(f"Kunde inte lista modeller dynamiskt ({err}). Använder nödlista.")
        available = ["gemini-2.5-pro", "gemini-2.5-flash"]

    def sort_key(name: str):
        low = name.lower()
        if "pro" in low:
            return (0, low)
        elif "flash" in low:
            return (1, low)
        return (2, low)

    ordered = sorted(available, key=sort_key)
    print(f"Prioriterad modellordning: {ordered[:3]}")
    return ordered

def run_resilient_generation(prompt: str, system_instruction: str, temperature: float = 0.1):
    """Kör modellförfrågan med Search Grounding, modell-fallback och felhantering."""
    client = get_gemini_client()
    candidate_models = discover_candidate_models(client)

    config = genai.types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        tools=[{"google_search": {}}],
    )

    response = None
    chosen_model = None

    for model_name in candidate_models:
        print(f"Försöker anropa: {model_name}")
        delay = 5
        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                chosen_model = model_name
                break
            except APIError as e:
                print(f"API-fel för {model_name} ({e}) vid försök {attempt}/3.")
                if getattr(e, "code", None) == 404 or "NOT_FOUND" in str(e):
                    break
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                else:
                    break
            except (httpx.HTTPError, httpx.StreamError) as net_err:
                print(f"Nätverksfel mot {model_name} ({net_err}) vid försök {attempt}/3.")
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                else:
                    break

        if response is not None:
            break

    if response is None:
        raise RuntimeError("Samtliga anrop mot Gemini API misslyckades.")

    # Extrahera källor från Google Search om tillgängligt
    sources_md = ""
    try:
        if response.candidates:
            candidate = response.candidates[0]
            metadata = getattr(candidate, "grounding_metadata", None)
            if metadata:
                chunks = getattr(metadata, "grounding_chunks", [])
                links = []
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web and getattr(web, "uri", None) and getattr(web, "title", None):
                        links.append(f"- [{web.title}]({web.uri})")
                if links:
                    sources_md = "\n\n### 🌐 Referenser & Källor (Google Search)\n" + "\n".join(sorted(set(links)))
    except Exception as meta_err:
        print(f"Kunde inte tolka grounding metadata: {meta_err}")

    return response.text, sources_md, chosen_model
