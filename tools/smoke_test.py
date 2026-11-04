"""Röktest: ett minimalt anrop per nivå och modell med exakt nivåns konfiguration.

Kör innan en PR med ändrad nivåkonfiguration mergas. Skriver API:ts felmeddelande vid fel och,
vid 400, provar varianter utan en parameter i taget för att visa vilken parameter modellen
avvisar. Avslutar med kod 1 om någon nivås förstahandsmodell inte svarar.
Loggar aldrig nyckel eller prompt utöver den fasta testfrågan.
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import tiers
from core.client import (active_platform, api_error_detail, build_generation_config, error_code, generate,
                         get_gemini_client, log)

PROMPT = "Svara bara med ordet OK."


def call(client, model, stream=False, **kw):
    cfg, applied = build_generation_config(model_name=model, system_instruction="Röktest.", temperature=0.0, **kw)
    t0 = time.monotonic()
    try:
        r = generate(client, model, PROMPT, cfg, stream=stream)
        u = getattr(r, "usage_metadata", None)
        toks = f"in {getattr(u, 'prompt_token_count', '?')} ut {getattr(u, 'candidates_token_count', '?')} " \
               f"thinking {getattr(u, 'thoughts_token_count', '?')}" if u else "usage saknas"
        return True, f"200 OK på {time.monotonic() - t0:.1f} s, resonemang {applied}{', strömmat' if stream else ''}, {toks}"
    except Exception as err:
        return False, api_error_detail(err), error_code(err)


def main() -> int:
    cfg = tiers.load_config()
    log(f"röktest, plattform {active_platform()}")
    failures = 0
    for name in tiers.TIER_NAMES:
        t = tiers.get_tier(cfg, name)
        client = get_gemini_client(timeout_s=t.request_timeout_s)
        for i, model in enumerate(t.models):
            level = tiers.supported_thinking_level(cfg, model, t.thinking_level)
            policy = tiers.model_policy(cfg, model, t.request_timeout_s)
            full = dict(thinking_level=level, grounding=t.grounding, max_output_tokens=t.max_output_tokens,
                        request_timeout_s=policy.request_timeout_s, stream=policy.stream)
            res = call(client, model, **full)
            log(f"{name}/{model}: {'OK' if res[0] else 'FEL'} – {res[1]}")
            if res[0]:
                continue
            if i == 0:
                failures += 1
            if len(res) > 2 and res[2] == 400:
                for label, variant in (("utan thinking", dict(full, thinking_level="")),
                                       ("thinking low", dict(full, thinking_level="low")),
                                       ("utan max_output_tokens", dict(full, max_output_tokens=None)),
                                       ("utan grounding", dict(full, grounding=False))):
                    v = call(client, model, **variant)
                    log(f"  diagnos {label}: {'OK' if v[0] else 'FEL'} – {v[1]}")
    log(f"röktest klart: {failures} nivå(er) där förstahandsmodellen inte svarar")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
