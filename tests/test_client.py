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

    def generate_content_stream(self, model, contents, config):
        # samma utfall som ett vanligt anrop, levererat som en enda del
        self.streamed = getattr(self, "streamed", []) + [model]
        yield self.generate_content(model, contents, config)


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
    for v in ("GCP_SA_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GEMINI_ENABLE_GROUNDING"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(c.time, "sleep", lambda s: None)


def install(monkeypatch, names, outcomes):
    fake = FakeClient(names, outcomes)
    monkeypatch.setattr(c, "get_gemini_client", lambda **kw: fake)
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
    assert c.classify_error(httpx.ReadTimeout("x")) == (True, "nätverkstimeout (ReadTimeout)")
    assert c.classify_error(httpx.ConnectError("x")) == (True, "tillfälligt nätverksfel (ConnectError)")
    assert c.classify_error(httpx.RemoteProtocolError("x")) == \
        (True, "tillfälligt nätverksfel (RemoteProtocolError)")


@pytest.mark.parametrize(
    "resp",
    [
        SimpleNamespace(text="x", candidates=[]),
        ok_response(text="  ", finish="MAX_TOKENS"),
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


# --- Vertex AI -----------------------------------------------------------------------------------

SA_JSON = '{"type": "service_account", "project_id": "proj-from-key", "client_email": "x@y", "private_key": "SECRET"}'


def _capture_vertex(monkeypatch):
    seen = {}

    def fake_creds(info, scopes=None):
        seen["info"], seen["scopes"] = info, scopes
        return "CREDS"

    class FakeGenaiClient:
        def __init__(self, **kw):
            seen["client_kwargs"] = kw

    monkeypatch.setattr(c.service_account.Credentials, "from_service_account_info", staticmethod(fake_creds))
    monkeypatch.setattr(c.genai, "Client", FakeGenaiClient)
    return seen


def test_vertex_client_from_service_account(monkeypatch):
    seen = _capture_vertex(monkeypatch)
    monkeypatch.setenv("GCP_SA_KEY", SA_JSON)
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-be-used")
    c.get_gemini_client()
    assert seen["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
    assert seen["client_kwargs"] == {"vertexai": True, "project": "proj-from-key", "location": "global",
                                     "credentials": "CREDS"}
    assert c.active_platform() == "Vertex AI (global)"


def test_vertex_project_and_location_overrides(monkeypatch):
    seen = _capture_vertex(monkeypatch)
    monkeypatch.setenv("GCP_SA_KEY", SA_JSON)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-override")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
    c.get_gemini_client()
    assert (seen["client_kwargs"]["project"], seen["client_kwargs"]["location"]) == ("proj-override", "europe-west1")


def test_invalid_service_account_key_never_leaks(monkeypatch):
    _capture_vertex(monkeypatch)
    monkeypatch.setenv("GCP_SA_KEY", '{"private_key": "SECRET"')          # broken JSON
    result = c.run_resilient_generation("p", "sys")
    assert not result.success
    assert "SECRET" not in result.audit_md
    assert "GCP_SA_KEY" in result.audit_md
    assert "Vertex AI" in result.audit_md


def test_ai_studio_fallback_without_service_account(monkeypatch):
    seen = _capture_vertex(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    c.get_gemini_client()
    assert seen["client_kwargs"] == {"api_key": "k"}
    assert c.active_platform() == "AI Studio"


def test_vertex_model_names_are_normalised(monkeypatch):
    class M:
        def __init__(self, name):
            self.name, self.supported_actions = name, None

    class L:
        class models:
            @staticmethod
            def list():
                return [M("publishers/google/models/gemini-3.1-pro-preview"),
                        M("projects/p/locations/global/models/gemini-4-argon"),
                        M("models/gemini-2.5-pro")]
    assert c.fetch_available_models(L()) == {"gemini-3.1-pro-preview", "gemini-4-argon", "gemini-2.5-pro"}


def test_standard_review_has_no_search_by_default():
    assert c.PROFILES["standard_review"].enable_grounding is False
    assert c.PROFILES["deep_investigation"].enable_grounding is True
    on, desc = c.grounding_enabled(c.PROFILES["standard_review"])
    assert (on, desc) == (False, "av (profil)")


def test_grounding_override_by_env(monkeypatch):
    monkeypatch.setenv("GEMINI_ENABLE_GROUNDING", "true")
    assert c.grounding_enabled(c.PROFILES["standard_review"]) == (True, "aktiv (GEMINI_ENABLE_GROUNDING)")
    monkeypatch.setenv("GEMINI_ENABLE_GROUNDING", "false")
    assert c.grounding_enabled(c.PROFILES["deep_investigation"]) == (False, "av (GEMINI_ENABLE_GROUNDING)")


def test_audit_shows_platform(monkeypatch):
    install(monkeypatch, list(c.ALL_APPROVED_MODELS), {m: ok_response() for m in c.ALL_APPROVED_MODELS})
    monkeypatch.setenv("GCP_SA_KEY", SA_JSON)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
    assert "Plattform:** `Vertex AI (europe-west1)`" in run().audit_md


def test_max_tokens_with_text_is_accepted_as_truncated():
    assert c.validate_response(ok_response(finish="MAX_TOKENS")) == (True, "MAX_TOKENS")


def test_tier_limits_reach_the_config():
    cfg, _ = c.build_generation_config("gemini-3.8-flash", "s", 0.1, "minimal", grounding=False, max_output_tokens=2000)
    assert cfg.max_output_tokens == 2000
    assert c.PROFILES["snabb"].max_output_tokens == 2000
    assert c.PROFILES["djup"].primary_model == "gemini-4-argon"
    assert "flash" not in " ".join(c.PROFILES["djup"].allowed_fallbacks)


def test_audit_reports_tier_tokens_and_cost(monkeypatch):
    resp = ok_response()
    resp.usage_metadata = SimpleNamespace(prompt_token_count=1000, candidates_token_count=500,
                                          thoughts_token_count=1500, tool_use_prompt_token_count=0)
    install(monkeypatch, list(c.ALL_APPROVED_MODELS), {m: resp for m in c.ALL_APPROVED_MODELS})
    r = c.run_resilient_generation("p", "sys", profile_name="snabb", tier_source="label")
    assert "Nivå:** `snabb (label)`" in r.audit_md
    assert "Resonemang:** `low`" in r.audit_md         # minimal stöds inte av Flash
    assert "Tokens:** in 1000 · ut 500 · varav thinking 1500" in r.audit_md
    assert "Uppskattad kostnad:** `" in r.audit_md and "uppskattning" in r.audit_md
    assert r.usage == {"input": 1000, "output": 500, "thinking": 1500}


def test_request_timeout_reaches_the_client(monkeypatch):
    seen = _capture_vertex(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    c.get_gemini_client(timeout_s=180)
    assert seen["client_kwargs"]["http_options"].timeout == 180000


def test_run_budget_stops_before_the_job_timeout(monkeypatch):
    install(monkeypatch, list(c.ALL_APPROVED_MODELS), {m: api_error(503) for m in c.ALL_APPROVED_MODELS})
    clock = {"t": 0.0}
    monkeypatch.setattr(c.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(c.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 200))
    r = c.run_resilient_generation("p", "sys", profile_name="snabb")
    assert not r.success
    assert "tidsbudget" in r.audit_md


def test_progress_is_logged(monkeypatch, capsys):
    install(monkeypatch, list(c.ALL_APPROVED_MODELS), {m: ok_response() for m in c.ALL_APPROVED_MODELS})
    c.run_resilient_generation("p", "sys", profile_name="snabb")
    out = capsys.readouterr().out
    assert "start: nivå snabb" in out and "anrop 1/3 till gemini-3.8-flash" in out and "klart med" in out


def test_no_grounding_forbids_invented_sources(monkeypatch):
    class Models(FakeModels):
        def generate_content(self, model, contents, config):
            self.calls.append((bool(getattr(config, "tools", None)), config.system_instruction))
            return ok_response()

    for profile, grounded in (("standard_review", False), ("deep_investigation", True)):
        fake = install(monkeypatch, list(c.ALL_APPROVED_MODELS), {})
        fake.models = Models(list(c.ALL_APPROVED_MODELS), {})
        assert run(profile).success
        tools_on, instruction = fake.models.calls[0]
        assert tools_on is grounded
        assert (c.NO_GROUNDING_NOTE in instruction) is (not grounded)


def test_budget_check_failure_never_blocks(monkeypatch):
    from core import tiers

    class Repo:
        def get_issues_comments(self, since):
            raise RuntimeError("nät")

    ch = tiers.TierChoice("djup", "label")
    assert c.apply_repo_budget(Repo(), tiers.load_config(), ch) is ch


def test_budget_counts_only_bot_comments_this_month(monkeypatch):
    from datetime import timedelta
    from core import tiers
    start = tiers.month_start()
    cost = "- **Uppskattad kostnad:** `15.00 kr` (x)"
    comments = [SimpleNamespace(body=cost, user=SimpleNamespace(type="Bot"), created_at=start + timedelta(hours=1)),
                SimpleNamespace(body=cost, user=SimpleNamespace(type="Bot"), created_at=start + timedelta(hours=2)),
                SimpleNamespace(body=cost, user=SimpleNamespace(type="User"), created_at=start + timedelta(hours=3))]

    class Repo:
        def get_issues_comments(self, since):
            return comments

    assert c.repo_spend_this_month(Repo()) == 30.0
    ch = c.apply_repo_budget(Repo(), tiers.load_config(), tiers.TierChoice("standard", "label"))
    assert ch.tier == "snabb"



# --- tung modell: timeout, omförsök och reserv -----------------------------------------------------

class RecordingModels(FakeModels):
    """Spelar upp en lista utfall per modell och sparar config för varje anrop."""
    def __init__(self, names, script, clock=None, durations=None):
        super().__init__(names, {})
        self.script = {m: list(v) for m, v in script.items()}
        self.configs = []
        self.clock = clock
        self.durations = durations or {}

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        self.configs.append((model, config))
        if self.clock is not None:
            self.clock["t"] += self.durations.get(model, 1)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def install_script(monkeypatch, script, durations=None):
    clock = {"t": 0.0}
    monkeypatch.setattr(c.time, "monotonic", lambda: clock["t"])
    fake = FakeClient(list(c.ALL_APPROVED_MODELS), {})
    fake.models = RecordingModels(list(c.ALL_APPROVED_MODELS), script, clock, durations)
    monkeypatch.setattr(c, "get_gemini_client", lambda **kw: fake)
    return fake


def test_argon_gets_its_own_300_s_request_timeout(monkeypatch):
    fake = install_script(monkeypatch, {"gemini-4-argon": [ok_response()]})
    assert c.run_resilient_generation("p", "sys", profile_name="djup").success
    _, cfg = fake.models.configs[0]
    assert cfg.http_options.timeout == 300_000


def test_every_model_gets_a_300_s_timeout_in_milliseconds(monkeypatch):
    fake = install_script(monkeypatch, {"gemini-4-argon": [api_error(404)],
                                        "gemini-3.1-pro-preview": [ok_response()]})
    assert c.run_resilient_generation("p", "sys", profile_name="djup").success
    fake2 = install_script(monkeypatch, {"gemini-3.8-flash": [ok_response()]})
    assert c.run_resilient_generation("p", "sys", profile_name="snabb").success
    assert [cfg.http_options.timeout for _, cfg in fake.models.configs + fake2.models.configs] == [300_000] * 3


def test_argon_timeout_retried_once_with_lower_thinking_then_reserve(monkeypatch):
    fake = install_script(monkeypatch, {
        "gemini-4-argon": [httpx.ReadTimeout("x"), httpx.ReadTimeout("x")],
        "gemini-3.1-pro-preview": [ok_response()],
    }, durations={"gemini-4-argon": 100})
    r = c.run_resilient_generation("p", "sys", profile_name="djup")
    assert r.success and r.chosen_model == "gemini-3.1-pro-preview"
    assert fake.models.calls == ["gemini-4-argon", "gemini-4-argon", "gemini-3.1-pro-preview"]
    levels = [str(cfg.thinking_config.thinking_level).lower() for m, cfg in fake.models.configs if m == "gemini-4-argon"]
    assert "high" in levels[0] and "medium" in levels[1]
    assert "nätverkstimeout (ReadTimeout)" in r.audit_md


def test_slow_argon_failure_goes_straight_to_reserve(monkeypatch):
    fake = install_script(monkeypatch, {
        "gemini-4-argon": [httpx.RemoteProtocolError("x")],
        "gemini-3.1-pro-preview": [ok_response()],
    }, durations={"gemini-4-argon": 270})
    r = c.run_resilient_generation("p", "sys", profile_name="djup")
    assert r.success and fake.models.calls == ["gemini-4-argon", "gemini-3.1-pro-preview"]
    assert "RemoteProtocolError" in r.audit_md and "efter 270 s" in r.audit_md


def test_worst_case_argon_leaves_budget_for_the_reserve(monkeypatch):
    # två fulla timeouts på argon (2 x 300 s + väntan) ska lämna plats för reservens hela tidsgräns
    install_script(monkeypatch, {
        "gemini-4-argon": [httpx.ReadTimeout("x"), httpx.ReadTimeout("x")],
        "gemini-3.1-pro-preview": [ok_response()],
    }, durations={"gemini-4-argon": 179})
    r = c.run_resilient_generation("p", "sys", profile_name="djup")
    assert r.success and r.chosen_model == "gemini-3.1-pro-preview"
    djup = c.PROFILES["djup"]
    policy = c.tiers.model_policy(c.tiers.load_config(), "gemini-4-argon", djup.request_timeout_s)
    # ett långt försök (> slow_attempt_s) går direkt till reserven, som då får hela sin gräns
    assert policy.max_call_s + 5 + djup.request_timeout_s <= djup.run_budget_s


def test_other_models_keep_three_attempts(monkeypatch):
    fake = install_script(monkeypatch, {"gemini-3.8-flash": [httpx.ReadTimeout("x"), httpx.ReadTimeout("x"),
                                                             ok_response()]})
    assert c.run_resilient_generation("p", "sys", profile_name="snabb").success
    assert fake.models.calls == ["gemini-3.8-flash"] * 3



# --- strömning ------------------------------------------------------------------------------------

def chunk(text=None, finish=None, grounding=None, usage=None, raise_text=False):
    class C:
        candidates = [SimpleNamespace(finish_reason=finish, grounding_metadata=grounding)]
        usage_metadata = usage

        @property
        def text(self):
            if raise_text:
                raise ValueError("inga textdelar")
            return text
    return C()


def test_merge_stream_joins_text_and_keeps_last_metadata():
    g = SimpleNamespace(grounding_chunks=[])
    u = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=7,
                        tool_use_prompt_token_count=0)
    r = c.merge_stream([chunk(None), chunk(raise_text=True), chunk("Hej "), chunk("världen", grounding=g),
                        chunk(None, finish="FinishReason.STOP", usage=u)])
    assert r.text == "Hej världen"
    assert r.candidates[0].finish_reason == "FinishReason.STOP" and r.candidates[0].grounding_metadata is g
    assert c.validate_response(r)[0] and c.extract_usage(r)["thinking"] == 7


def test_merge_stream_without_content_is_rejected():
    assert not c.validate_response(c.merge_stream([]))[0]


def test_all_levels_are_streamed_with_thought_summaries(monkeypatch):
    for tier, model in (("djup", "gemini-4-argon"), ("standard", "gemini-3.1-pro-preview"),
                        ("snabb", "gemini-3.8-flash")):
        fake = install_script(monkeypatch, {model: [ok_response()]})
        assert c.run_resilient_generation("p", "sys", profile_name=tier).success
        assert fake.models.streamed == [model]
        _, cfg = fake.models.configs[0]
        assert cfg.thinking_config.include_thoughts is True


def test_client_without_streaming_uses_a_plain_call():
    class Plain:
        def generate_content(self, model, contents, config):
            return ok_response("vanligt")
    r = c.generate(SimpleNamespace(models=Plain()), "gemini-3.8-flash", "p", SimpleNamespace(thinking_config=None))
    assert r.text == "vanligt"


def test_merge_stream_over_many_chunks_keeps_text_finish_and_sources():
    web = SimpleNamespace(web=SimpleNamespace(uri="https://example.org/a", title="Källa A"))
    g = SimpleNamespace(grounding_chunks=[web])
    r = c.merge_stream([chunk("Del 1. "), chunk(None, finish=None), chunk("Del 2. "),
                        chunk("Del 3.", finish="FinishReason.STOP"), chunk(None, grounding=g)])
    assert r.text == "Del 1. Del 2. Del 3."
    assert r.candidates[0].finish_reason == "FinishReason.STOP"
    assert c.validate_response(r)[0] is True
    assert "[Källa A](https://example.org/a)" in c._extract_sources(r)


def test_timeout_after_more_than_120_s_goes_straight_to_the_next_model(monkeypatch):
    fake = install_script(monkeypatch, {
        "gemini-4-argon": [httpx.ReadTimeout("x"), ok_response()],
        "gemini-3.1-pro-preview": [ok_response()],
    }, durations={"gemini-4-argon": 121})
    r = c.run_resilient_generation("p", "sys", profile_name="djup")
    assert r.success and fake.models.calls == ["gemini-4-argon", "gemini-3.1-pro-preview"]
    assert "nätverkstimeout (ReadTimeout) efter 121 s" in r.audit_md


def test_error_inside_the_stream_is_handled_like_a_call_error(monkeypatch):
    fake = install_script(monkeypatch, {"gemini-4-argon": [httpx.RemoteProtocolError("x")],
                                        "gemini-3.1-pro-preview": [ok_response()]},
                          durations={"gemini-4-argon": 300})
    r = c.run_resilient_generation("p", "sys", profile_name="djup")
    assert r.success and r.chosen_model == "gemini-3.1-pro-preview"


def test_stream_longer_than_its_cap_is_a_timeout(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(c.time, "monotonic", lambda: clock["t"])

    def endless():
        while True:
            clock["t"] += 100
            yield chunk("x")

    class M:
        def generate_content_stream(self, model, contents, config):
            return endless()

    with pytest.raises(httpx.ReadTimeout):
        c.generate(SimpleNamespace(models=M()), "gemini-4-argon", "p", SimpleNamespace(thinking_config=None),
                   stream=True, max_total_s=600)


def test_no_retry_on_argon_when_it_would_starve_the_reserve(monkeypatch):
    # utan 120 s-regeln: 200 s in ryms inte ett nytt argon-försök (600 s) plus reservens 600 s i 1380 s
    from core import tiers
    cfg = tiers.load_config()
    cfg["model_params"]["gemini-4-argon"]["slow_attempt_s"] = 1000
    monkeypatch.setattr(c.tiers, "load_config", lambda *a, **k: cfg)
    fake = install_script(monkeypatch, {
        "gemini-4-argon": [httpx.ConnectError("x"), ok_response()],
        "gemini-3.1-pro-preview": [ok_response()],
    }, durations={"gemini-4-argon": 200})
    r = c.run_resilient_generation("p", "sys", profile_name="djup")
    assert r.success and fake.models.calls == ["gemini-4-argon", "gemini-3.1-pro-preview"]
    assert "inget omförsök" in r.audit_md
