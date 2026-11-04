# gemini-tools

Delade skript för Gemini i GitHub Actions: kodgranskning av PR:er (`tools/code_review.py`) och utredningar av issues (`tools/investigate.py`). Workflows i de anslutna repona checkar ut det här repot och kör skripten.

## Välja AI-nivå

Alla anrop körs på en av tre nivåer. Modeller, resonemang, utdatagränser, tidsgränser och pristabell står **bara** i [`config/gemini-tiers.yml`](config/gemini-tiers.yml).

| Nivå | Används för | Modell (enligt konfigurationen) | Resonemang | Max utdata |
| --- | --- | --- | --- | --- |
| `snabb` | standard: PR-sammanfattning, lint-förklaring, statusrapport | Flash | låg (Flash stöder inte minimal) | ~2 000 tokens |
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

**Starta en utredning i rätt ordning.** Annars kan flera workflows starta samtidigt, med halvfärdig text eller fel nivå:
1. Skapa issuet **utan etiketter** och skriv klart titel och beskrivning: frågan, sammanhanget och filer som ska bifogas (sökväg i backticks, t.ex. `` `docs/EV.md` ``).
2. Sätt en eventuell nivåetikett (`ai:standard` eller `ai:djup`). Utan `utredning` startar den ingenting.
3. Sätt `utredning` **sist**, som ett eget steg.

Skapa aldrig issuet med etiketterna direkt (`gh issue create --label utredning,ai:djup` eller `labels` i API-anropet). GitHub skickar då en händelse per etikett, och två körningar startar. Har beskrivningen ändrats efter starten: kör om med `/gemini <nivå>`. Att ta bort och sätta tillbaka `utredning` startar ingen ny körning.

**Skydd:** diff eller bilagor över 200 kB trunkeras och det står i rapporten. Tunga modeller kan ha egna gränser under `model_params`: `gemini-4-argon` har 300 s per anrop, högst ett omförsök (med resonemang `medium`), och går direkt till reserven om ett misslyckat försök tagit mer än 180 s. Argon anropas dessutom strömmat med resonemangssammanfattningar (`stream: true`), så att förbindelsen aldrig är tyst under ett långt resonemang; utan strömning stängdes den efter ~320 s. Tidsgränsen blir då en gräns för tystnad, inte för hela svaret. Då finns alltid tid kvar för reservmodellen. Orsaken i sidfoten anger felklassen, t.ex. `nätverkstimeout (ReadTimeout)` eller `tillfälligt nätverksfel (RemoteProtocolError)`. Varje nivå har en tidsgräns per anrop och en total tidsbudget, så att en statusrapport hinner postas innan jobbet når sin egen tidsgräns. Loggen visar löpande vad som händer, med ett livstecken varje minut under långa anrop.

**Flera etiketter på en gång** ger ändå bara en körning. Vid etikett- och öppna-händelser väntar körningen `settle_s` (20 s) innan modellen anropas. Kommer en till händelse under tiden avbryts den första körningen (workflowens concurrency), och den sista läser issuet eller PR:en på nytt: aktuell text och aktuella etiketter. Ordningen ovan gäller ändå, så att texten är klar när körningen startar.

**Månadsbudget per repo:** botens sidfötter sedan den 1:a summeras. Över `budget.monthly_sek_per_repo` körs `standard` och `djup` som `snabb`, och det står i sidfoten (`Nivå: snabb (…, nedgraderad från djup: månadsbudget …)`). Snabb påverkas inte. Gränsen räknas på uppskattningarna, inte på fakturan.

**Källor:** utan webbsökning (snabb och standard) instrueras modellen att inte hänvisa till källor som inte finns i underlaget. Riktiga länkar finns bara när grounding är på, under "Referenser & Källor".

**Sidfoten i varje bot-kommentar** visar plattform (Vertex AI eller AI Studio), nivå och hur den valdes, modell, tokens (in, ut och varav thinking) och en uppskattad kostnad i kronor från pristabellen i konfigurationen. Priserna där är uppskattningar och ska stämmas av mot fakturan.

## Plattform

`GCP_SA_KEY` (service account-JSON) ger Vertex AI; utan den används `GEMINI_API_KEY` mot AI Studio. Nyckelns innehåll loggas aldrig.
