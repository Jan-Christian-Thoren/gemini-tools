import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from github import Github
from google import genai
from google.genai.errors import APIError
import httpx


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    primary_model: str
    allowed_fallbacks: tuple[str, ...]
    thinking_level: str


PROFILES: dict[str, ExecutionProfile] = {
    "deep_investigation": ExecutionProfile(
        name="deep_investigation",
        primary_model="gemini-4-argon",
        allowed_fallbacks=("gemini-3.1-pro-preview", "gemini-2.5-pro"),
        thinking_level="high",
    ),
    "standard_review": ExecutionProfile(
        name="standard_review",
        primary_model="gemini-3.8-flash",
        allowed_fallbacks=("gemini-3.1-pro-preview",),
        thinking_level="medium",
    ),
}

ALL_APPROVED_MODELS = {
    "gemini-4-argon",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-3.8-flash",
}

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

SANITIZED_ERROR_LABELS = {
    400: "400 – ogiltig begäran/konfiguration",
    401: "401 – autentiseringsfel",
    403: "403 – behörighet saknas",
    404: "404 – modellen ej anropbar",
    429: "429 – kapacitetsbegränsning",
    500: "500 – internt serverfel hos API",
    502: "502 – dålig gateway",
    503: "503 – tjänsten tillfälligt otillgänglig",
    504: "504 – timeout mot upstream",
}


class RunConfigError(Exception):
    """Konfigurationsfel med ett eget, redan sanerat meddelande."""


@dataclass
class GenerationResult:
    success: bool
    text: str
    sources_md: str
    audit_md: str
    chosen_model: Optional[str]
    requested_model: str


def get_github_context():
    """Hämtar och initierar GitHub-klienten och måltrepot."""
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("REPOSITORY")
    if not token or not repo_name:
        raise ValueError("Miljövariablerna GITHUB_TOKEN och REPOSITORY måste vara definierade.")
    gh = Github(token)
    return gh, gh.get_repo(repo_name)


def load_system_role(default_role: str) -> tuple[str, str]:
    """
    Läser in .github/gemini_role.md från det anropande repot.
    Returnerar (rolltext, rollkälla).
    """
    role_candidates = [
        Path(".github/gemini_role.md"),
        Path(".github/gemini-role.md"),
    ]
    for path in role_candidates:
        if path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                return content, path.as_posix()

    env_role = os.environ.get("GEMINI_SYSTEM_ROLE", "").strip()
    if env_role:
        return env_role, "env:GEMINI_SYSTEM_ROLE"

    return default_role, "default"


def get_gemini_client() -> genai.Client:
    """Initierar Google GenAI-klienten."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RunConfigError("GEMINI_API_KEY saknas")
    return genai.Client(api_key=key)


def fetch_available_models(client: genai.Client) -> set[str]:
    """Hämtar alla modeller på kontot som stöder generateContent."""
    available = set()
    for model in client.models.list():
        name = (model.name or "").removeprefix("models/")
        actions = (
            getattr(model, "supported_actions", None)
            or getattr(model, "supported_generation_methods", None)
        )
        if actions and "generateContent" not in actions:
            continue
        if name:
            available.add(name)
    return available


def discover_candidate_models(
    client: genai.Client,
    profile_name: str = "deep_investigation",
) -> tuple[list[str], str]:
    """
    Returnerar (kandidatlista, begärd_förstahandsmodell) enligt profil eller GEMINI_MODEL.
    Verifierar alltid att valda modeller är tillgängliga via API:et.
    """
    if profile_name not in PROFILES:
        raise RunConfigError(f"okänd exekveringsprofil: {profile_name}")

    profile = PROFILES[profile_name]
    pinned = os.getenv("GEMINI_MODEL", "").strip()

    if pinned and pinned not in ALL_APPROVED_MODELS:
        raise RunConfigError(f"ej godkänd modell i GEMINI_MODEL: {pinned}")

    requested_primary = pinned if pinned else profile.primary_model
    desired_order = (pinned,) if pinned else (profile.primary_model, *profile.allowed_fallbacks)

    try:
        available = fetch_available_models(client)
    except Exception as exc:
        raise RunConfigError("kunde inte verifiera modellutbud mot Gemini API") from exc

    if pinned and pinned not in available:
        raise RunConfigError(f"fastlåst modell ({pinned}) är inte tillgänglig för detta API-konto")

    candidates = [m for m in desired_order if m in available]
    if not candidates:
        raise RunConfigError("ingen godkänd analysmodell för profilen är tillgänglig")

    return candidates, requested_primary


def build_generation_config(
    model_name: str,
    system_instruction: str,
    temperature: float,
    thinking_level: str,
    grounding: bool = True,
) -> tuple[genai.types.GenerateContentConfig, str]:
    """
    Bygger GenerateContentConfig anpassad efter modellfamilj och installerad SDK-version.
    Returnerar (config, faktiskt_applicerad_resonemangsnivå).
    """
    config_kwargs: dict = {
        "system_instruction": system_instruction,
        "temperature": temperature,
    }
    if grounding:
        config_kwargs["tools"] = [{"google_search": {}}]

    # store=False är opt-in (GEMINI_OPT_OUT_STORAGE=true) och sätts bara om SDK:n har fältet
    gen_config_fields = getattr(genai.types.GenerateContentConfig, "model_fields", {})
    if os.getenv("GEMINI_OPT_OUT_STORAGE", "").strip().lower() == "true" and "store" in gen_config_fields:
        config_kwargs["store"] = False

    applied_thinking = "none (stöds ej av SDK)"
    thinking_cls = getattr(genai.types, "ThinkingConfig", None)

    if thinking_cls and thinking_level:
        thinking_fields = getattr(thinking_cls, "model_fields", {})
        if model_name.startswith("gemini-2.5"):
            if "thinking_level" in thinking_fields:
                config_kwargs["thinking_config"] = thinking_cls(thinking_level=thinking_level)
                applied_thinking = thinking_level
            elif "thinking_budget" in thinking_fields:
                budget = 16384 if thinking_level == "high" else 4096
                config_kwargs["thinking_config"] = thinking_cls(thinking_budget=budget)
                applied_thinking = f"{thinking_level} (budget={budget})"
        elif "thinking_level" in thinking_fields:
            config_kwargs["thinking_config"] = thinking_cls(thinking_level=thinking_level)
            applied_thinking = thinking_level

    return genai.types.GenerateContentConfig(**config_kwargs), applied_thinking


def error_code(err: Exception) -> Optional[int]:
    return getattr(err, "code", None) if isinstance(err, APIError) else None


def classify_error(err: Exception) -> tuple[bool, str]:
    """
    Klassificerar fel i (är_övergående_för_fallback, sanerad_orsak).
    Läcker aldrig råa felsträngar eller nyckeldata.
    """
    if isinstance(err, RunConfigError):
        return False, str(err)

    if isinstance(err, APIError):
        code = getattr(err, "code", None)
        label = SANITIZED_ERROR_LABELS.get(code, f"API-fel ({code or 'okänd kod'})")
        return code in RETRYABLE_STATUS_CODES, label

    if isinstance(err, httpx.TimeoutException):
        return True, "nätverkstimeout"

    if isinstance(err, (httpx.HTTPError, httpx.StreamError)):
        return True, "tillfälligt nätverksfel"

    return False, f"internt körningsfel ({type(err).__name__})"


def validate_response(response) -> tuple[bool, str]:
    """Kontrollerar att svaret har giltig kandidat, STOP-status och icke-tom text."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return False, "tom kandidatlista från modellen"

    finish_reason = getattr(candidates[0], "finish_reason", None)
    finish_str = str(finish_reason).upper() if finish_reason is not None else ""
    if not finish_str.endswith("STOP"):
        return False, f"ofullständigt avslut ({finish_str.split('.')[-1] or 'saknas'})"

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        return False, "tom modellrespons"

    return True, ""


def format_audit_block(
    chosen_model: Optional[str],
    requested_model: str,
    thinking_desc: str,
    fallback_used: bool,
    fallback_reason: str,
    role_source: str,
    grounding_desc: str = "aktiv",
) -> str:
    """Skapar ett konsekvent revisionsblock för både lyckade och misslyckade körningar."""
    lines = [
        "### 🔎 Revisions- & Modellinformation",
        f"- **Modell:** `{chosen_model or 'ingen (körning avbruten)'}`",
        f"- **Begärd modell:** `{requested_model}`",
        f"- **Resonemang:** `{thinking_desc}`",
        f"- **Fallback:** `{'ja' if fallback_used else 'nej'}`",
    ]
    if fallback_reason:
        lines.append(f"- **Orsak:** `{fallback_reason}`")
    lines.append(f"- **Grounding:** `{grounding_desc}`")
    lines.append(f"- **Rollkälla:** `{role_source}`")
    return "\n".join(lines)


def _extract_sources(response) -> str:
    try:
        metadata = getattr(response.candidates[0], "grounding_metadata", None)
        if not metadata:
            return ""
        links = []
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web and getattr(web, "uri", None) and getattr(web, "title", None):
                links.append(f"- [{web.title}]({web.uri})")
        if links:
            return "\n\n### 🌐 Referenser & Källor (Google Search)\n" + "\n".join(sorted(set(links)))
    except Exception:
        pass
    return ""


def run_resilient_generation(
    prompt: str,
    system_instruction: str,
    role_source: str = "default",
    profile_name: str = "deep_investigation",
    temperature: float = 0.1,
) -> GenerationResult:
    """
    Kör modellförfrågan enligt vald profil med strikt felklassning,
    svarvalidering och fullständig revisionsloggning. Kastar aldrig vid
    API-/konfigurationsfel; returnerar istället en misslyckad GenerationResult.
    """
    profile = PROFILES.get(profile_name) or PROFILES["deep_investigation"]
    requested_model = os.getenv("GEMINI_MODEL", "").strip() or profile.primary_model

    grounding_desc = "aktiv"

    def failure(reasons: list[str], thinking: str) -> GenerationResult:
        return GenerationResult(
            success=False,
            text="",
            sources_md="",
            audit_md=format_audit_block(
                chosen_model=None,
                requested_model=requested_model,
                thinking_desc=thinking,
                fallback_used=False,
                fallback_reason="; ".join(reasons),
                role_source=role_source,
                grounding_desc=grounding_desc,
            ),
            chosen_model=None,
            requested_model=requested_model,
        )

    try:
        client = get_gemini_client()
        candidate_models, requested_model = discover_candidate_models(client, profile_name=profile_name)
    except Exception as init_err:
        _, reason = classify_error(init_err)
        return failure([reason], profile.thinking_level)

    fallback_reasons: list[str] = []
    chosen_model: Optional[str] = None
    applied_thinking = profile.thinking_level
    final_response = None

    for model_name in candidate_models:
        grounding = True

        def make_config():
            return build_generation_config(
                model_name=model_name,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking_level=profile.thinking_level,
                grounding=grounding,
            )

        config, applied_thinking = make_config()

        delay = 5
        abort_all = False
        attempt = 0

        while attempt < 3:
            attempt += 1
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
            except Exception as err:
                code = error_code(err)
                _, sanitized_reason = classify_error(err)

                # Nekat Search Grounding (kvot/stöd): ett sista försök utan verktyget på samma modell
                if grounding and code in (400, 429):
                    grounding = False
                    grounding_desc = "avaktiverad (kvot/stöd saknas)"
                    config, applied_thinking = make_config()
                    attempt -= 1
                    continue

                # 404 på en enskild modell: gå vidare till nästa tillåtna modell
                if code == 404:
                    fallback_reasons.append(f"{model_name}: {sanitized_reason}")
                    break

                if not classify_error(err)[0]:
                    fallback_reasons.append(f"{model_name}: {sanitized_reason}")
                    abort_all = True
                    break
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                else:
                    fallback_reasons.append(f"{model_name}: {sanitized_reason}")
                continue

            is_valid, invalid_reason = validate_response(resp)
            if not is_valid:
                # Ett ofullständigt svar blir inte bättre av omförsök; prova nästa tillåtna modell
                fallback_reasons.append(f"{model_name}: {invalid_reason}")
                break

            final_response = resp
            chosen_model = model_name
            break

        if final_response is not None or abort_all:
            break

    if final_response is None:
        return failure(fallback_reasons, applied_thinking)

    return GenerationResult(
        success=True,
        text=final_response.text.strip(),
        sources_md=_extract_sources(final_response),
        audit_md=format_audit_block(
            chosen_model=chosen_model,
            requested_model=requested_model,
            thinking_desc=applied_thinking,
            fallback_used=chosen_model != requested_model,
            fallback_reason="; ".join(fallback_reasons),
            role_source=role_source,
            grounding_desc=grounding_desc,
        ),
        chosen_model=chosen_model,
        requested_model=requested_model,
    )
