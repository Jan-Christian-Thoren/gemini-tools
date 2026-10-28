import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from google.genai.errors import APIError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import client as c


class FakeModels:
    def __init__(self, names, outcomes):
        self._names = names
        self._outcomes = outcomes
        self.calls = []

    def list(self):
        return [SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"]) for n in self._names]

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        outcome = self._outcomes[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, names, outcomes):
        self.models = FakeModels(names, outcomes)


def ok_response(text="Rapport", finish="STOP"):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=f"FinishReason.{finish}", grounding_metadata=None)],
    )


def api_error(code):
    return APIError(code, {"error": {"message": "SECRET-RAW-TEXT", "status": "X"}})


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(c.time, "sleep", lambda s: None)


def install(monkeypatch, names, outcomes):
    fake = FakeClient(names, outcomes)
    monkeypatch.setattr(c, "get_gemini_client", lambda: fake)
    return fake


def run(profile="deep_investigation"):
    return c.run_resilient_generation("p", "sys", role_source="default", profile_name=profile)


def test_profile_model_selection(monkeypatch):
    fake = install(monkeypatch, list(c.ALL_APPROVED_MODELS), {m: ok_response() for m in c.ALL_APPROVED_MODELS})
    assert run("deep_investigation").chosen_model == "gemini-4-argon"
    assert run("standard_review").chosen_model == "gemini-3.8-flash"


def test_deep_investigation_never_falls_back_to_flash():
    assert not any("flash" in m for m in (c.PROFILES["deep_investigation"].primary_model, *c.PROFILES["deep_investigation"].allowed_fallbacks))


def test_pinned_model_must_be_approved(monkeypatch):
    install(monkeypatch, ["gemini-2.5-flash"], {})
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    result = run()
    assert not result.success
    assert "ej godkänd" in result.audit_md


def test_pinned_model_must_be_available(monkeypatch):
    fake = install(monkeypatch, ["gemini-2.5-pro"], {})
    monkeypatch.setenv("GEMINI_MODEL", "gemini-4-argon")
    result = run()
    assert not result.success
    assert fake.models.calls == []
    assert "inte tillgänglig" in result.audit_md


@pytest.mark.parametrize("code", [401, 403])
def test_non_retryable_errors_stop_without_fallback(monkeypatch, code):
    fake = install(monkeypatch, list(c.ALL_APPROVED_MODELS), {m: api_error(code) for m in c.ALL_APPROVED_MODELS})
    result = run()
    assert not result.success
    assert fake.models.calls == ["gemini-4-argon"]
    assert "SECRET-RAW-TEXT" not in result.audit_md
    assert str(code) in result.audit_md


def test_429_triggers_allowed_fallback_with_sanitized_reason(monkeypatch):
    outcomes = {
        "gemini-4-argon": api_error(429),
        "gemini-3.1-pro-preview": ok_response(),
        "gemini-2.5-pro": ok_response(),
    }
    fake = install(monkeypatch, list(outcomes), outcomes)
    result = run()
    assert result.success
    assert result.chosen_model == "gemini-3.1-pro-preview"
    assert "`ja`" in result.audit_md
    assert "429 – kapacitetsbegränsning" in result.audit_md
    assert "SECRET-RAW-TEXT" not in result.audit_md


def test_network_error_is_retryable():
    assert c.classify_error(httpx.ReadTimeout("x")) == (True, "nätverkstimeout")
    assert c.classify_error(httpx.ConnectError("x"))[0] is True


@pytest.mark.parametrize(
    "resp",
    [
        SimpleNamespace(text="x", candidates=[]),
        ok_response(finish="MAX_TOKENS"),
        ok_response(finish="SAFETY"),
        ok_response(text="   "),
    ],
)
def test_incomplete_or_empty_responses_rejected(resp):
    assert c.validate_response(resp)[0] is False


def test_valid_response_accepted():
    assert c.validate_response(ok_response()) == (True, "")


def test_audit_block_always_created(monkeypatch):
    install(monkeypatch, list(c.ALL_APPROVED_MODELS), {m: ok_response() for m in c.ALL_APPROVED_MODELS})
    success = run()
    install(monkeypatch, list(c.ALL_APPROVED_MODELS), {m: api_error(401) for m in c.ALL_APPROVED_MODELS})
    failure = run()
    for r in (success, failure):
        for label in ("Modell", "Begärd modell", "Resonemang", "Fallback", "Rollkälla"):
            assert f"**{label}:**" in r.audit_md
    assert not failure.success and failure.text == ""


def test_missing_api_key_returns_failure(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = run()
    assert not result.success
    assert "GEMINI_API_KEY" in result.audit_md


def test_load_system_role_sources(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_SYSTEM_ROLE", raising=False)
    assert c.load_system_role("d") == ("d", "default")
    monkeypatch.setenv("GEMINI_SYSTEM_ROLE", "envrole")
    assert c.load_system_role("d") == ("envrole", "env:GEMINI_SYSTEM_ROLE")
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "gemini_role.md").write_text("filrole")
    assert c.load_system_role("d") == ("filrole", ".github/gemini_role.md")


def test_400_without_grounding_still_aborts(monkeypatch):
    fake = install(monkeypatch, list(c.ALL_APPROVED_MODELS), {m: api_error(400) for m in c.ALL_APPROVED_MODELS})
    result = run()
    assert not result.success
    assert fake.models.calls == ["gemini-4-argon", "gemini-4-argon"]  # med och utan grounding
    assert "avaktiverad" in result.audit_md


def test_404_falls_back_to_next_model(monkeypatch):
    outcomes = {
        "gemini-4-argon": api_error(404),
        "gemini-3.1-pro-preview": ok_response(),
        "gemini-2.5-pro": ok_response(),
    }
    install(monkeypatch, list(outcomes), outcomes)
    result = run()
    assert result.success and result.chosen_model == "gemini-3.1-pro-preview"
    assert "404 – modellen ej anropbar" in result.audit_md


def test_grounding_dropped_on_429_then_succeeds(monkeypatch):
    class Models(FakeModels):
        def generate_content(self, model, contents, config):
            self.calls.append((model, bool(getattr(config, "tools", None))))
            if getattr(config, "tools", None):
                raise api_error(429)
            return ok_response()

    fake = install(monkeypatch, list(c.ALL_APPROVED_MODELS), {})
    fake.models = Models(list(c.ALL_APPROVED_MODELS), {})
    result = run()
    assert result.success and result.chosen_model == "gemini-4-argon"
    assert fake.models.calls == [("gemini-4-argon", True), ("gemini-4-argon", False)]
    assert "Grounding:** `avaktiverad (kvot/stöd saknas)`" in result.audit_md


def test_store_is_opt_in(monkeypatch):
    monkeypatch.delenv("GEMINI_OPT_OUT_STORAGE", raising=False)
    cfg, _ = c.build_generation_config("gemini-4-argon", "s", 0.1, "high")
    assert getattr(cfg, "store", None) is None


def test_standard_review_never_sends_google_search(monkeypatch):
    class Models(FakeModels):
        def generate_content(self, model, contents, config):
            self.calls.append(bool(getattr(config, "tools", None)))
            return ok_response()

    fake = install(monkeypatch, list(c.ALL_APPROVED_MODELS), {})
    fake.models = Models(list(c.ALL_APPROVED_MODELS), {})
    result = run("standard_review")
    assert result.success and fake.models.calls == [False]
    assert "Grounding:** `av (profil)`" in result.audit_md


def test_deep_investigation_keeps_grounding(monkeypatch):
    class Models(FakeModels):
        def generate_content(self, model, contents, config):
            self.calls.append(bool(getattr(config, "tools", None)))
            return ok_response()

    fake = install(monkeypatch, list(c.ALL_APPROVED_MODELS), {})
    fake.models = Models(list(c.ALL_APPROVED_MODELS), {})
    result = run("deep_investigation")
    assert result.success and fake.models.calls == [True]
    assert "Grounding:** `aktiv`" in result.audit_md
