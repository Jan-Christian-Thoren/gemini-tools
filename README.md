# gemini-tools

Delade skript för Gemini i GitHub Actions: kodgranskning av PR:er (`tools/code_review.py`) och utredningar av issues (`tools/investigate.py`). Workflows i de anslutna repona checkar ut det här repot och kör skripten.

## Välja AI-nivå

Alla anrop körs på en av tre nivåer. Modeller, resonemang, utdatagränser, tidsgränser och pristabell står **bara** i [`config/gemini-tiers.yml`](config/gemini-tiers.yml).

| Nivå | Används för | Modell (enligt konfigurationen) | Resonemang | Max utdata |
| --- | --- | --- | --- | --- |
| `snabb` | standard: PR-sammanfattning, lint-förklaring, statusrapport | Flash | minimal | ~2 000 tokens |
| `standard` | vanlig kodgranskning | Pro | låg | ~6 000 tokens |
| `djup` | arkitektur, svår felsökning, säkerhet | Argon | hög | ~16 000 tokens |

**Så väljs nivån** (högst prioritet först):

1. **Kommentar** i PR:en eller issuet: `/gemini snabb|standard|djup [fri fråga]`. Den fria texten skickas med som extra fråga. Kommandot får användas av repots medlemmar; `djup` bara av kontona i `djup_command_users` (ägarens personkonto; organisationen själv kan inte kommentera).
2. **Label:** `ai:snabb`, `ai:standard` eller `ai:djup`. Att sätta labeln startar en körning. Har en PR flera ai-labels gäller den billigaste.
3. **Default:** `snabb`.

Ingenting eskalerar av sig självt: en modell som inte svarar ersätts bara av samma eller en billigare modell inom nivån.

**När körs det?**
- PR öppnas: en granskning. Nya pushar körs bara om PR:en inte redan har en granskning, och en ny push avbryter en pågående körning.
- En ai-label sätts eller ett `/gemini`-kommando skrivs: alltid en ny körning.
- Issues: etiketten `utredning` startar en utredning; `/gemini …` i en kommentar kör om den.
- `djup` körs högst en gång per PR eller issue och timme.

**Skydd:** diff eller bilagor över 200 kB trunkeras och det står i rapporten. Varje nivå har en tidsgräns per anrop och en total tidsbudget, så att en statusrapport hinner postas innan jobbet når sin egen tidsgräns. Loggen visar löpande vad som händer, med ett livstecken varje minut under långa anrop.

**Sidfoten i varje bot-kommentar** visar plattform (Vertex AI eller AI Studio), nivå och hur den valdes, modell, tokens (in, ut och varav thinking) och en uppskattad kostnad i kronor från pristabellen i konfigurationen. Priserna där är uppskattningar och ska stämmas av mot fakturan.

## Plattform

`GCP_SA_KEY` (service account-JSON) ger Vertex AI; utan den används `GEMINI_API_KEY` mot AI Studio. Nyckelns innehåll loggas aldrig.
