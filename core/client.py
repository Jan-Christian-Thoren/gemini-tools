import json
from types import SimpleNamespace
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from github import Github
from google import genai
from google.genai.errors import APIError
from google.oauth2 import service_account
import httpx

from core import tiers


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    primary_model: str
    allowed_fallbacks: tuple[str, ...]
    thinking_level: str
    enable_grounding: bool = True     # Google Search kostar per sökning; kan överstyras med GEMINI_ENABLE_GROUNDING
    max_output_tokens: Optional[int] = None
    request_timeout_s: Optional[int] = None
    run_budget_s: Optional[int] = None


def _profiles_from_tiers() -> dict[str, ExecutionProfile]:
    """Nivåerna i config/gemini-tiers.yml är de enda profilerna; gamla namn är alias."""
    cfg = tiers.load_config()
    out = {}
    for name in list(cfg["tiers"]) + list(cfg.get("aliases", {})):
        t = tiers.get_tier(cfg, name)
        out[name] = ExecutionProfile(name=t.name, primary_model=t.models[0], allowed_fallbacks=t.models[1:],
                                     thinking_level=t.thinking_level, enable_grounding=t.grounding,
                                     max_output_tokens=t.max_output_tokens,
                                     request_timeout_s=t.request_timeout_s, run_budget_s=t.run_budget_s)
    return out


PROFILES: dict[str, ExecutionProfile] = _profiles_from_tiers()

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
    usage: Optional[dict] = None          # {"input", "output", "thinking"} från usage_metadata
    truncated: bool = False


def get_github_context():
    """Hämtar och initierar GitHub-klienten och måltrepot."""
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("REPOSITORY")
    if not token or not repo_name:
        raise ValueError("Miljövariablerna GITHUB_TOKEN och REPOSITORY måste vara definierade.")
    gh = Github(token)
    return gh, gh.get_repo(repo_name)


def repo_spend_this_month(repo) -> float:
    """Uppskattad kostnad i kr i repots bot-kommentarer sedan månadens början."""
    since = tiers.month_start()
    bodies = [c.body for c in repo.get_issues_comments(since=since)
              if getattr(c.user, "type", "") == "Bot" and c.created_at >= since]
    return tiers.spent_this_month(bodies)


def apply_repo_budget(repo, cfg: dict, choice: "tiers.TierChoice") -> "tiers.TierChoice":
    applies = (cfg.get("budget") or {}).get("applies_to") or []
    if choice.tier not in applies:
        return choice
    try:
        spent = repo_spend_this_month(repo)
    except Exception as err:  # budgetkollen får aldrig stoppa en körning
        print(f"Budgetkontroll misslyckades ({type(err).__name__}); kör vald nivå")
        return choice
    new = tiers.apply_budget(cfg, choice, spent)
    print(f"Månadens uppskattade kostnad i repot: {spent:.2f} kr"
          + (f"; {new.source}" if new is not choice else ""))
    return new


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


VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _vertex_location() -> str:
    # Nya preview-/3.x/4.x-modeller exponeras ofta på den globala endpointen i Vertex AI.
    return os.getenv("GOOGLE_CLOUD_LOCATION", "").strip() or "global"


def active_platform() -> str:
    """Plattform för revisionsblocket, enligt samma regel som get_gemini_client()."""
    if os.environ.get("GCP_SA_KEY", "").strip():
        return f"Vertex AI ({_vertex_location()})"
    return "AI Studio"


def log(msg: str):
    """Löpande rad i Actions-loggen (aldrig nyckelinnehåll), skrivs direkt."""
    print(f"[gemini {time.strftime('%H:%M:%S')}] {msg}", flush=True)
    sys.stdout.flush()


def _http_options(timeout_s: Optional[int]) -> dict:
    return {"http_options": genai.types.HttpOptions(timeout=int(timeout_s * 1000))} if timeout_s else {}


def get_gemini_client(timeout_s: Optional[int] = None) -> genai.Client:
    """Initierar Google GenAI-klienten.

    Primärt Vertex AI med ett Service Account i GCP_SA_KEY (rå JSON). Annars AI Studio med
    GEMINI_API_KEY. Felmeddelanden innehåller aldrig nyckelns innehåll."""
    raw = os.environ.get("GCP_SA_KEY", "").strip()
    if raw:
        try:
            info = json.loads(raw)
        except (ValueError, TypeError):
            raise RunConfigError("GCP_SA_KEY är inte giltig JSON") from None
        if not isinstance(info, dict):
            raise RunConfigError("GCP_SA_KEY är inte ett JSON-objekt")
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or info.get("project_id")
        if not project:
            raise RunConfigError("projekt saknas i GCP_SA_KEY och GOOGLE_CLOUD_PROJECT")
        try:
            credentials = service_account.Credentials.from_service_account_info(info, scopes=VERTEX_SCOPES)
        except Exception:
            raise RunConfigError("GCP_SA_KEY kunde inte användas som service account") from None
        return genai.Client(vertexai=True, project=project, location=_vertex_location(),
                            credentials=credentials, **_http_options(timeout_s))

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RunConfigError("varken GCP_SA_KEY eller GEMINI_API_KEY finns")
    return genai.Client(api_key=key, **_http_options(timeout_s))


def grounding_enabled(profile: "ExecutionProfile") -> tuple[bool, str]:
    """(påslaget, beskrivning). GEMINI_ENABLE_GROUNDING=true/false överstyr profilen."""
    override = os.getenv("GEMINI_ENABLE_GROUNDING", "").strip().lower()
    if override in ("true", "false"):
        on = override == "true"
        return on, ("aktiv" if on else "av") + " (GEMINI_ENABLE_GROUNDING)"
    return profile.enable_grounding, "aktiv" if profile.enable_grounding else "av (profil)"


def fetch_available_models(client: genai.Client) -> set[str]:
    """Hämtar alla modeller på kontot som stöder generateContent."""
    available = set()
    for model in client.models.list():
        # AI Studio: "models/<namn>"; Vertex AI: "publishers/google/models/<namn>" eller
        # "projects/.../models/<namn>". Sista ledet matchas mot ALL_APPROVED_MODELS.
        name = (model.name or "").split("/")[-1]
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
    max_output_tokens: Optional[int] = None,
    request_timeout_s: Optional[int] = None,
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
    if max_output_tokens:
        config_kwargs["max_output_tokens"] = int(max_output_tokens)

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

    if request_timeout_s:
        # Per anrop, ersätter klientens tidsgräns (millisekunder i google-genai)
        config_kwargs["http_options"] = genai.types.HttpOptions(timeout=int(request_timeout_s * 1000))

    return genai.types.GenerateContentConfig(**config_kwargs), applied_thinking


def api_error_detail(err: Exception, limit: int = 200) -> str:
    """API:ts egen status och felmeddelande för loggen (aldrig för GitHub-kommentarer).
    Meddelandet kapas och rensas från sådant som kan likna nycklar eller tokens."""
    if not isinstance(err, APIError):
        return type(err).__name__
    msg = " ".join(str(getattr(err, "message", "") or "").split())
    msg = re.sub(r"(AIza[0-9A-Za-z_\-]{10,}|ya29\.[0-9A-Za-z_\-\.]+|-----BEGIN[^-]*-----)", "[borttaget]", msg)
    return f"{getattr(err, 'code', '?')} {getattr(err, 'status', '') or ''}: {msg}"[:limit]


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

    # TimeoutException är en underklass till HTTPError och måste prövas först
    if isinstance(err, httpx.TimeoutException):
        return True, f"nätverkstimeout ({type(err).__name__})"

    if isinstance(err, (httpx.HTTPError, httpx.StreamError)):
        return True, f"tillfälligt nätverksfel ({type(err).__name__})"

    return False, f"internt körningsfel ({type(err).__name__})"


def validate_response(response) -> tuple[bool, str]:
    """Kontrollerar att svaret har giltig kandidat, STOP-status och icke-tom text."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return False, "tom kandidatlista från modellen"

    finish_reason = getattr(candidates[0], "finish_reason", None)
    finish_str = str(finish_reason).upper() if finish_reason is not None else ""
    text = (getattr(response, "text", None) or "").strip()
    # Nivåns utdatagräns: ett avkapat men icke-tomt svar används och märks som trunkerat.
    if finish_str.endswith("MAX_TOKENS") and text:
        return True, "MAX_TOKENS"
    if not finish_str.endswith("STOP"):
        return False, f"ofullständigt avslut ({finish_str.split('.')[-1] or 'saknas'})"

    if not text:
        return False, "tom modellrespons"

    return True, ""


def extract_usage(response) -> Optional[dict]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    def n(field):
        v = getattr(meta, field, None)
        return int(v) if isinstance(v, (int, float)) else 0
    return {"input": n("prompt_token_count") + n("tool_use_prompt_token_count"),
            "output": n("candidates_token_count"), "thinking": n("thoughts_token_count")}


def format_audit_block(
    chosen_model: Optional[str],
    requested_model: str,
    thinking_desc: str,
    fallback_used: bool,
    fallback_reason: str,
    role_source: str,
    grounding_desc: str = "aktiv",
    platform: str = "AI Studio",
    tier_desc: Optional[str] = None,
    usage: Optional[dict] = None,
    cost_sek: Optional[float] = None,
    available: Optional[list] = None,
    truncated: bool = False,
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
    lines.append(f"- **Plattform:** `{platform}`")
    if tier_desc:
        lines.append(f"{tiers.LEVEL_MARKER}{tier_desc}`")
    if usage is not None:
        lines.append(f"- **Tokens:** in {usage['input']} · ut {usage['output']} · varav thinking {usage['thinking']}"
                     + (" · svaret trunkerat vid nivåns gräns" if truncated else ""))
        lines.append("- **Uppskattad kostnad:** "
                     + (f"`{cost_sek:.2f} kr` (uppskattning från pristabellen i gemini-tiers.yml)"
                        if cost_sek is not None else "`okänd` (modell saknas i pristabellen)"))
    if available is not None:
        lines.append(f"- **Godkända modeller hos plattformen:** `{', '.join(sorted(available)) or 'inga'}`")
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


NO_GROUNDING_NOTE = (
    "\n\n---\nWebbsökning (grounding) är avstängd i den här körningen. Hänvisa inte till källor, "
    "länkar, dokumenttitlar eller utgåvor som inte finns i underlaget du fått. Bygger ett påstående "
    "på allmän kunskap, skriv det som allmän kunskap och markera det som overifierat."
)


def effective_system_instruction(system_instruction: str, grounding: bool) -> str:
    return system_instruction if grounding else system_instruction + NO_GROUNDING_NOTE


def merge_stream(chunks) -> SimpleNamespace:
    """Slår ihop strömmade delsvar till ett svar med samma fält som ett vanligt anrop:
    text (utan resonemangsdelar), sista finish_reason, grounding_metadata och usage_metadata."""
    texts, finish, grounding, usage = [], None, None, None
    for ch in chunks:
        try:
            t = ch.text
        except Exception:
            t = None
        if t:
            texts.append(t)
        cand = (getattr(ch, "candidates", None) or [None])[0]
        if cand is not None:
            finish = getattr(cand, "finish_reason", None) or finish
            grounding = getattr(cand, "grounding_metadata", None) or grounding
        usage = getattr(ch, "usage_metadata", None) or usage
    candidates = [SimpleNamespace(finish_reason=finish, grounding_metadata=grounding)] if finish or texts else []
    return SimpleNamespace(text="".join(texts), candidates=candidates, usage_metadata=usage)


def _bounded(chunks, max_total_s: Optional[int]):
    t0 = time.monotonic()
    for ch in chunks:
        yield ch
        if max_total_s and time.monotonic() - t0 > max_total_s:
            raise httpx.ReadTimeout(f"strömmat svar längre än {max_total_s} s")


def _call_model_streaming(client, model_name: str, contents, config, max_total_s: Optional[int] = None):
    """Strömmat anrop (Google Clouds rekommendation för långa svar): delsvar kommer löpande,
    också resonemangssammanfattningar, så förbindelsen tystnar aldrig. Returnerar ett samlat svar
    som validate_response() och källutvinningen läser precis som ett vanligt svar."""
    if getattr(config, "thinking_config", None) is not None:
        config.thinking_config.include_thoughts = True   # håller strömmen vid liv under resonemanget
    stream_iter = client.models.generate_content_stream(model=model_name, contents=contents, config=config)
    return merge_stream(_bounded(stream_iter, max_total_s))


def generate(client, model: str, prompt: str, config, stream: bool = True, max_total_s: Optional[int] = None):
    # Vanligt anrop bara om strömning är avslagen för modellen, eller klienten saknar strömning
    if not stream or not hasattr(client.models, "generate_content_stream"):
        return client.models.generate_content(model=model, contents=prompt, config=config)
    return _call_model_streaming(client, model, prompt, config, max_total_s)


def run_resilient_generation(
    prompt: str,
    system_instruction: str,
    role_source: str = "default",
    profile_name: str = "deep_investigation",
    temperature: float = 0.1,
    tier_source: str = "default",
) -> GenerationResult:
    """
    Kör modellförfrågan enligt vald profil med strikt felklassning,
    svarvalidering och fullständig revisionsloggning. Kastar aldrig vid
    API-/konfigurationsfel; returnerar istället en misslyckad GenerationResult.
    """
    profile = PROFILES.get(profile_name) or PROFILES["deep_investigation"]
    requested_model = os.getenv("GEMINI_MODEL", "").strip() or profile.primary_model

    profile_grounding, grounding_desc = grounding_enabled(profile)
    platform = active_platform()
    cfg = tiers.load_config()
    tier_desc = f"{profile.name} ({tier_source})"
    available_approved: Optional[list] = None

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
                platform=platform,
                tier_desc=tier_desc,
                available=available_approved,
            ),
            chosen_model=None,
            requested_model=requested_model,
        )

    started = time.monotonic()
    budget = profile.run_budget_s
    heartbeat_s = int(cfg.get("heartbeat_s", 60))
    log(f"start: nivå {tier_desc}, plattform {platform}, tidsgräns per anrop "
        f"{profile.request_timeout_s or 'ingen'} s, budget {budget or 'ingen'} s")
    try:
        client = get_gemini_client(timeout_s=profile.request_timeout_s)
        try:
            available_approved = sorted(fetch_available_models(client) & ALL_APPROVED_MODELS)
            log(f"godkända modeller hos plattformen: {', '.join(available_approved) or 'inga'}")
        except Exception:
            available_approved = None
        candidate_models, requested_model = discover_candidate_models(client, profile_name=profile_name)
    except Exception as init_err:
        _, reason = classify_error(init_err)
        return failure([reason], profile.thinking_level)

    fallback_reasons: list[str] = []
    truncated = False
    chosen_model: Optional[str] = None
    applied_thinking = profile.thinking_level
    final_response = None

    for model_name in candidate_models:
        grounding = profile_grounding
        policy = tiers.model_policy(cfg, model_name, profile.request_timeout_s)
        thinking = tiers.supported_thinking_level(cfg, model_name, profile.thinking_level)

        def make_config():
            return build_generation_config(
                model_name=model_name,
                system_instruction=effective_system_instruction(system_instruction, grounding),
                temperature=temperature,
                thinking_level=thinking,
                grounding=grounding,
                max_output_tokens=profile.max_output_tokens,
                request_timeout_s=policy.request_timeout_s,
            )

        config, applied_thinking = make_config()

        delay = 5
        abort_all = False
        attempt = 0
        max_attempts = policy.max_attempts

        while attempt < max_attempts:
            attempt += 1
            elapsed = time.monotonic() - started
            # Ett nytt anrop kräver minst en hel tystnadsgräns kvar; strömmen kapas vid budgeten
            if budget and elapsed + (policy.request_timeout_s or 0) > budget:
                fallback_reasons.append(f"{model_name}: tidsbudget {budget} s räcker inte för ett nytt anrop")
                log(f"stopp: {int(elapsed)} s använt, budgeten {budget} s räcker inte för ett nytt anrop")
                abort_all = True
                break
            log(f"anrop {attempt}/{max_attempts} till {model_name} (grounding {'på' if grounding else 'av'}, "
                f"resonemang {applied_thinking or 'standard'}, "
                f"{'strömmat, tystnadsgräns' if policy.stream else 'tidsgräns'} {policy.request_timeout_s or 'ingen'} s)")
            t0 = time.monotonic()
            done = threading.Event()

            def heartbeat(model=model_name, t_start=t0):
                while not done.wait(heartbeat_s):
                    log(f"väntar på svar från {model}: {int(time.monotonic() - t_start)} s")
            threading.Thread(target=heartbeat, daemon=True).start()
            try:
                cap = policy.stream_total_s
                if budget:
                    left = int(budget - (time.monotonic() - started))
                    cap = min(cap, left) if cap else left
                resp = generate(client, model_name, prompt, config, stream=policy.stream, max_total_s=cap)
                log(f"svar från {model_name} efter {int(time.monotonic() - t0)} s")
            except Exception as err:
                log(f"fel från {model_name} efter {int(time.monotonic() - t0)} s: {classify_error(err)[1]}"
                    f" | API: {api_error_detail(err)}")
                done.set()
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
                took = time.monotonic() - t0
                rest = candidate_models[candidate_models.index(model_name) + 1:]
                if rest and policy.slow_attempt_s and took > policy.slow_attempt_s:
                    # Ett långsamt misslyckande upprepas inte: nästa modell får budgeten
                    fallback_reasons.append(f"{model_name}: {sanitized_reason} efter {int(took)} s")
                    log(f"{model_name}: {sanitized_reason} efter {int(took)} s (> {policy.slow_attempt_s} s), "
                        f"direkt till {rest[0]}")
                    break
                # Omförsök bara om reservmodellen ändå får hela sin tidsgräns efteråt
                reserve_s = tiers.model_policy(cfg, rest[0], profile.request_timeout_s).max_call_s if rest else 0
                if (attempt < max_attempts and budget and rest
                        and time.monotonic() - started + delay + policy.max_call_s + reserve_s > budget):
                    fallback_reasons.append(f"{model_name}: {sanitized_reason}; inget omförsök, tiden behövs för {rest[0]}")
                    log(f"{model_name}: hoppar över omförsök så att {rest[0]} får hela sin tidsgräns")
                    break
                if attempt < max_attempts:
                    if policy.retry_thinking_level and policy.retry_thinking_level != thinking:
                        thinking = tiers.supported_thinking_level(cfg, model_name, policy.retry_thinking_level)
                        config, applied_thinking = make_config()
                        log(f"omförsök på {model_name} med resonemang {applied_thinking}")
                    time.sleep(delay)
                    delay *= 2
                else:
                    fallback_reasons.append(f"{model_name}: {sanitized_reason}")
                continue

            done.set()
            is_valid, invalid_reason = validate_response(resp)
            if not is_valid:
                # Ett ofullständigt svar blir inte bättre av omförsök; prova nästa tillåtna modell
                fallback_reasons.append(f"{model_name}: {invalid_reason}")
                break

            final_response = resp
            chosen_model = model_name
            truncated = invalid_reason == "MAX_TOKENS"
            break

        if final_response is not None or abort_all:
            break

    if final_response is None:
        log(f"klart utan resultat efter {int(time.monotonic() - started)} s: {'; '.join(fallback_reasons)}")
        return failure(fallback_reasons, applied_thinking)
    log(f"klart med {chosen_model} efter {int(time.monotonic() - started)} s")

    usage = extract_usage(final_response)
    cost = tiers.estimate_cost_sek(cfg, chosen_model, usage) if usage else None

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
            platform=platform,
            tier_desc=tier_desc,
            usage=usage or {"input": 0, "output": 0, "thinking": 0},
            cost_sek=cost,
            available=available_approved,
            truncated=truncated,
        ),
        chosen_model=chosen_model,
        requested_model=requested_model,
        usage=usage,
        truncated=truncated,
    )
