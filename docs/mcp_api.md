# HsH RAG MCP Server — Nutzungsanleitung für Projektpartner

**Projekt:** RAGHSH — Retrieval-Augmented Generation für die Hochschule Hannover
**MCP-Spezifikation:** Streamable HTTP (MCP Spec 2025-03-26)
**Stand:** 11. November 2026 (Prüfungsordnung, Rückfrage, Kontingent, Erkennung neu; Laufzeit- und Modellmessungen vom 17.08.2026)

> Alle Beispiele, Modellnamen, Antwortstrukturen und Laufzeiten in diesem Dokument
> wurden am 17.08.2026 gegen den laufenden Server ausgeführt und gemessen.
> Wer gegen die Fassung vom Juli 2026 programmiert hat: die Abweichungen stehen
> gesammelt am Ende unter [Änderungen gegenüber Juli 2026](#änderungen-gegenüber-juli-2026).

---

## Überblick

Der HsH RAG MCP Server stellt eine zweistufige KI-Pipeline als standardisiertes MCP-Interface bereit. Clients stellen Fragen zur Hochschule Hannover und erhalten Antworten, die mit wörtlichen Zitaten aus offiziellen HsH-Dokumenten belegt sind.

**Öffentlicher Endpunkt:** `https://nemo.christianwill.de/hsh/mcp`
**Lokaler Endpunkt (auf dem Server selbst):** `http://127.0.0.1:8765/mcp`

Die Pipeline arbeitet in fünf Schritten, zwei davon mit LLM:

1. **Hybrid-Suche** — zwei Dense-Arme (Chunktext, generierte Fragen) und ein Sparse-Arm (BM25), per DBSF fusioniert, ergeben Chunk-Treffer. Nennt die Frage eine Fakultät, einen Studiengang oder eine Prüfungsordnung, wird die Suche darauf ausgerichtet (siehe [Prüfungsordnung und Rückfrage](#prüfungsordnung-und-rückfrage)).
2. **Kandidaten** — die Treffer werden auf Dokumente (URLs) reduziert.
3. **Dokumentauswahl (LLM 1)** — ein LLM wählt aus der Kandidatenliste bis zu 4 Dokumente.
4. **Volltext laden** — die gewählten Dokumente werden vollständig aus Qdrant zusammengesetzt.
5. **Antwortgenerierung (LLM 2)** — ein LLM formuliert die Antwort mit wörtlichen Zitaten.

Beide LLM-Runden nutzen dasselbe Modell — das, was im Parameter `modell` steht.

---

## Protokoll: Streamable HTTP

Jede Anfrage ist ein HTTP-POST mit JSON-RPC-2.0-Body. Der Server antwortet als `text/event-stream` (SSE).

### Pflicht-Header

```http
Content-Type: application/json
Accept: application/json, text/event-stream
mcp-session-id: <session-id>   ← ab dem zweiten Request
```

**Der `Accept`-Header muss beide Typen nennen.** Fehlt einer, antwortet der Server mit
`-32600 Not Acceptable: Client must accept both application/json and text/event-stream`.
Das ist der häufigste Anfängerfehler.

### Session starten

```bash
curl -si -X POST https://nemo.christianwill.de/hsh/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-03-26",
      "capabilities": {},
      "clientInfo": {"name": "mein-client", "version": "1.0"}
    }
  }'
```

Die Session-ID steht im **Response-Header** `mcp-session-id` und gehört in jede Folge-Anfrage:

```
mcp-session-id: 449635f7a1c341fb81952530361c98e7
```

Die Spezifikation sieht danach eine `notifications/initialized`-Nachricht vor. Dieser Server
verlangt sie nicht (er ist nach der `initialize`-Antwort bereit), quittiert sie aber
spezifikationsgemäß mit `202 Accepted`. Wer einen fertigen MCP-Client benutzt, muss sich
darum ohnehin nicht kümmern.

Eine Session lässt sich mit `DELETE` auf denselben Endpunkt beenden (Antwort `200`). Nötig ist
das nicht — offene Sessions kosten den Server nichts, die Modelle bleiben ohnehin prozessweit
geladen.

---

## Resource: verfügbare Modelle

```bash
curl -s -X POST https://nemo.christianwill.de/hsh/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: <session-id>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "resources/read",
    "params": {"uri": "hsh://models"}
  }'
```

**Antwort** — das Element in `contents` hat die Felder `uri`, `mimeType` und `text`;
ein Feld `type` gibt es hier **nicht** (anders als bei Tool-Ergebnissen):

```json
{
  "result": {
    "contents": [{
      "uri": "hsh://models",
      "mimeType": "text/plain",
      "text": "[\"qwen3.5-122b-a10b\", \"glm-4.7\", \"meta-llama-3.1-8b-instruct\", ...]"
    }]
  }
}
```

Das `text`-Feld enthält eine JSON-Liste von Modell-IDs als String — also zweimal parsen.
Die Liste wird serverseitig 5 Minuten gecacht.

> **Wichtig:** `hsh://models` liefert ausschließlich die **kostenlosen GWDG-Modelle**.
> Die kostenpflichtigen Modelle (Claude, Gemini, ChatGPT) stehen dort nicht drin,
> funktionieren aber — siehe [Bezahlmodelle](#bezahlmodelle-claude-gemini-chatgpt).

---

## Resource: GWDG-Kontingent

`hsh://kontingent` liefert die verbleibenden Anfragen an die kostenlosen GWDG-Modelle, Stand
der letzten GWDG-Antwort: `{"zeit", "rest": {"minute", "stunde", "tag", "monat"}, "limit": {…}}`.
Die Grenzen gelten für alle Nutzer des Servers gemeinsam. Wer viele Anfragen stellt, prüft das
vorher.

---

## Modelle

### Kostenlose Modelle (GWDG ChatAI)

Die Liste ändert sich, wenn die GWDG ihr Angebot umstellt — **immer `hsh://models` abfragen und
niemals eine Modell-ID fest verdrahten.** Genau daran ist die Vorgängerfassung dieses Dokuments
gescheitert: die dort empfohlenen Modelle existieren heute nicht mehr.

Stand 17.08.2026 sind 16 Modelle verfügbar. Sinnvoll für diese Anwendung:

| Modell | Wofür | Gesamt* | davon Antwort-LLM |
|---|---|---|---|
| `glm-4.7` | Schnellste brauchbare Wahl — guter Ausgangspunkt | **4,9 s** | 2,5 s |
| `mistral-medium-3.5-128b` | Solides Deutsch, mittlere Größe | 7,0 s | 4,5 s |
| `openai-gpt-oss-120b` | Große Alternative, trotzdem flott | 9,1 s | 5,6 s |
| `qwen3.5-122b-a10b` | Größtes frei verfügbares Modell, für schwierige Fragen | 98,9 s | 32,1 s |
| `qwen3.6-27b` | Kleineres Reasoning-Modell | 90,6 s | 47,9 s |
| `meta-llama-3.1-8b-instruct` | Nur für Verbindungstests, schwächste Antworten | 12,2 s | 1,0 s |

\* Eine vollständige Anfrage („Welche Fristen gelten für die Rückmeldung zum nächsten Semester?",
Kontext ≈ 18.000 Token), gemessen am 17.08.2026 nacheinander auf dem Produktivsystem.
**Gemessen wurde die Laufzeit, nicht die Antwortqualität** — die Spalte „Wofür" ist Einschätzung,
keine Messung. Die Werte hängen an der Auslastung der GWDG und vor allem an der Größe der
ausgewählten Dokumente (siehe [Laufzeiten](#laufzeiten-und-timeouts)).

> **Reasoning-Modelle zahlen doppelt.** Die beiden Qwen-Modelle erzeugen intern einen
> Denkprozess. Weil die Pipeline **zwei** LLM-Runden fährt, fällt der zweimal an — bei
> `qwen3.5-122b-a10b` entfielen von 99 Sekunden rund 60 auf die Dokumentauswahl, also auf einen
> Schritt, der nur eine Handvoll Zahlen ausgeben soll. Wer Tempo braucht, nimmt ein Modell ohne
> Denkprozess; wer Qualität bei schwierigen Fragen braucht, nimmt die Wartezeit in Kauf.

Nicht für diesen Zweck gedacht, obwohl in der Liste: `qwen3-coder-next` und
`devstral-2-123b-instruct-2512` (Code), `medgemma-27b-it` (Medizin),
`qwen3-omni-30b-a3b-instruct` (Audio/Bild).

### Bezahlmodelle (Claude, Gemini, ChatGPT)

Über OpenRouter stehen zusätzlich drei kommerzielle Modelle bereit. Sie kosten pro Anfrage
echtes Geld vom Konto der Hochschule und sind deshalb **passwortgeschützt**: Ohne den Parameter
`passwort` lehnt der Server sie ab. Die Prüfung passiert serverseitig, ein direkter Zugriff auf
den MCP-Endpunkt hilft also nicht daran vorbei.

| Modell-ID | Anbieter | Preis Eingabe | Preis Ausgabe | Kosten je Anfrage** |
|---|---|---|---|---|
| **`openai/gpt-5.6-luna`** | OpenAI | $0,20 / Mio Token | $1,20 / Mio Token | **≈ 0,7–2 Cent** |
| `google/gemini-2.5-flash` | Google | $0,30 / Mio Token | $2,50 / Mio Token | ≈ 1–3,5 Cent |
| `anthropic/claude-sonnet-4.5` | Anthropic | $3,00 / Mio Token | $15,00 / Mio Token | ≈ 10–33 Cent |

Preise laut OpenRouter-API am 17.08.2026.
\*\* Gerechnet auf einen typischen Kontext von 15.000–30.000 Token plus den Ausreißer nach oben,
wenn das Tokenbudget von 100.000 voll ausgeschöpft wird. Die Auswahlrunde kommt mit wenigen
tausend Token noch hinzu.

> ### Empfehlung: `openai/gpt-5.6-luna` (ChatGPT)
>
> **Wer ein Bezahlmodell nutzt, sollte ChatGPT nehmen.** Es ist mit Abstand das günstigste der
> drei — pro Anfrage rund **ein Fünfzehntel der Kosten von Claude Sonnet** und immer noch
> spürbar billiger als Gemini Flash, bei einem Kontextfenster von über einer Million Token, das
> für diese Pipeline mehr als ausreicht. Claude Sonnet ist die teuerste Option und lohnt allenfalls
> für einzelne, besonders schwierige Fragen; als Standardeinstellung ist es Geldverbrennung.
>
> Und es ist nicht nur billig: Dieselbe Testfrage wie oben lief am 17.08.2026 mit
> `openai/gpt-5.6-luna` in **8,5 Sekunden** durch (Auswahl 3,8 s, Antwort 2,9 s, Kontext 18.000
> Token) und lieferte belegte Zitate — schneller als jedes große GWDG-Modell und im Bereich der
> schnellsten kostenlosen Modelle.
>
> Für den Regelbetrieb gilt trotzdem: Die kostenlosen GWDG-Modelle liefern für die allermeisten
> Fragen brauchbare Antworten. Ein Bezahlmodell ist die Ausnahme, nicht die Voreinstellung.

**So ruft man ein Bezahlmodell auf** — die Modell-ID in `modell`, das Passwort in `passwort`:

```bash
curl -s --no-buffer -X POST https://nemo.christianwill.de/hsh/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: <session-id>" \
  -d '{
    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
    "params": {
      "_meta": {"progressToken": "fortschritt-1"},
      "name": "frag_hsh",
      "arguments": {
        "frage": "Welche Fristen gelten für die Rückmeldung?",
        "modell": "openai/gpt-5.6-luna",
        "passwort": "<Passwort — wird separat übergeben>"
      }
    }
  }'
```

Bei den kostenlosen GWDG-Modellen wird `passwort` ignoriert und kann weggelassen werden.

Zwei Fehlerfälle, beide als `fehler`-Feld im Ergebnis:

- falsches oder fehlendes Passwort → *„Dieses Premium-Modell ist passwortgeschützt. …"*
- auf dem Server ist kein OpenRouter-Key hinterlegt → *„Premium-Modell nicht verfügbar …"*

---

## Tool: `frag_hsh`

### Parameter

| Parameter | Typ | Pflicht | Beschreibung |
|---|---|---|---|
| `frage` | string | ja | Frage auf Deutsch |
| `modell` | string | ja | Modell-ID aus `hsh://models` **oder** eine der Bezahlmodell-IDs |
| `fakultaet` | string | nein | `f1`…`f5`, leer = alle. Kein harter Filter, siehe unten |
| `speichern` | bool | nein | DSGVO-Einwilligung zur Speicherung (Standard `false`) |
| `passwort` | string | nein | Nur für Bezahlmodelle erforderlich, sonst ignoriert |
| `mit_chunks` | bool | nein | Liefert zusätzlich die gefundenen Chunks zurück (Standard `false`) |
| `nur_retrieval` | bool | nein | Bricht nach der Kandidatenliste ab, kein LLM; `modell` wird ignoriert (Standard `false`) |
| `po` | object | nein | Prüfungsordnung, für die die Antwort gelten soll: `{"studiengang": "BJO", "fassung": 2019}`, oder nur `{"studiengang": "BBA"}` als Antwort auf eine Rückfrage nach dem Studiengang. Siehe unten |
| `experiment` | object | nein | Nur für Messungen: überschreibt die Suchkonfiguration je Aufruf (`dense_arms`, `sparse_arms`, `fusion`, `top_docs`, …) |

**Fakultäts-Kürzel:**

| Kürzel | Fakultät |
|---|---|
| `f1` | Elektro- und Informationstechnik |
| `f2` | Maschinenbau und Bioverfahrenstechnik |
| `f3` | Medien, Information und Design |
| `f4` | Wirtschaft und Informatik |
| `f5` | Diakonie, Gesundheit und Soziales |

> **`fakultaet` ist ein Vorzug, kein Ausschluss.** Die Suche läuft in zwei Armen: einer auf die
> gewählte Fakultät gefiltert, einer auf die zentralen Seiten ohne Fakultätszuordnung. Beide
> Ergebnismengen kommen in die Kandidatenliste. Das ist Absicht — rund 39.600 Seiten
> (Immatrikulation, Prüfungsamt, Semesterticket, Bibliothek) tragen gar keine Fakultät und
> wären bei einem harten Filter verloren. Ergebnisse ohne oder mit anderer Fakultät im Feld
> `kandidaten[].fakultaet` sind also **kein Fehler**.

### Fortschrittsanzeige — `progressToken` nicht vergessen

Der Server meldet jeden der fünf Pipeline-Schritte als `notifications/progress` im SSE-Stream
derselben POST-Antwort. **Das passiert nur, wenn der Client in `params._meta.progressToken`
einen Token mitschickt.** Ohne Token sendet der Server keine einzige Fortschrittsmeldung —
ohne Fehler, ohne Hinweis. Wer eine Fortschrittsanzeige baut und den Token vergisst, sucht den
Fehler sehr lange bei sich.

Der Token ist eine frei wählbare Zeichenkette oder Zahl; er kommt in jeder Meldung als
`params.progressToken` zurück, damit sich parallele Aufrufe auseinanderhalten lassen.

### Aufruf (curl)

```bash
curl -s --no-buffer -X POST https://nemo.christianwill.de/hsh/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: <session-id>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "_meta": {"progressToken": "fortschritt-1"},
      "name": "frag_hsh",
      "arguments": {
        "frage": "Welche Studiengänge gibt es in der Fakultät I?",
        "modell": "glm-4.7",
        "fakultaet": "f1",
        "speichern": false
      }
    }
  }'
```

### SSE-Antwort

Fünf Fortschrittsmeldungen, dann das Ergebnis — alles im selben Stream, in dieser Reihenfolge.
Beachte: `progress` und `total` sind **Fließkommazahlen** (`1.0`, nicht `1`).

```
event: message
data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"fortschritt-1","progress":1.0,"total":5.0,"message":"Hybrid-Suche läuft…"}}

event: message
data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"fortschritt-1","progress":2.0,"total":5.0,"message":"120 Chunks → 24 Kandidaten"}}

event: message
data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"fortschritt-1","progress":3.0,"total":5.0,"message":"LLM wählt Dokumente aus (glm-4.7)…"}}

event: message
data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"fortschritt-1","progress":4.0,"total":5.0,"message":"Lade vollständige Dokumente aus Qdrant…"}}

event: message
data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"fortschritt-1","progress":5.0,"total":5.0,"message":"Generiere Antwort…"}}

event: message
data: {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{...}"}],"isError":false}}
```

### Ergebnis-Struktur

Das eigentliche Ergebnis steckt als JSON-String in `result.content[0].text` — also erneut parsen.

```json
{
  "fehler": null,
  "kandidaten": [
    {
      "nr": 1,
      "titel": "Bachelor Elektrotechnik",
      "url": "https://www.hs-hannover.de/...",
      "fakultaet": "f1",
      "struktur": "# Bachelor Elektrotechnik\n## Studieninhalte\n## Zulassung"
    }
  ],
  "auswahl_begruendung": "{\"selected\": [1, 3]}",
  "ausgewaehlte_docs": [
    {"nr": 1, "titel": "Bachelor Elektrotechnik", "url": "https://..."},
    {"nr": 3, "titel": "Studienordnung EIT",      "url": "https://..."}
  ],
  "token_budget": {"verwendet": 18432, "maximum": 100000},
  "antwort": "In der Fakultät I gibt es folgende Studiengänge: ...",
  "quellen": [
    {
      "titel": "Bachelor Elektrotechnik",
      "url": "https://www.hs-hannover.de/...",
      "crawl_date": "2026-08-01",
      "fremde_fundstellen": [],
      "abschnitt": "Studienangebot",
      "snippets": ["...relevanter Originaltext..."],
      "zitate": ["Regelstudienzeit beträgt sechs Semester"],
      "verifiziert": true
    }
  ],
  "zeiten": {
    "hybrid_suche_s": 2.23, "kandidaten_s": 0.02, "llm_auswahl_s": 0.33,
    "docs_laden_s": 0.11, "quellen_s": 2.68, "antwort_llm_s": 3.34, "gesamt_s": 6.02
  }
}
```

| Feld | Beschreibung |
|---|---|
| `fehler` | `null` bei Erfolg, sonst Fehlermeldung als Text |
| `kandidaten` | Dokumente aus der Hybrid-Suche, mit Inhaltsstruktur |
| `auswahl_begruendung` | **Roh**antwort des LLM aus Runde 1 — meist `{"selected": [...]}`, oft aber in einen Markdown-Codeblock verpackt. Nur zur Nachvollziehbarkeit gedacht; wer sie auswertet, muss beides vertragen |
| `ausgewaehlte_docs` | Vom LLM gewählte Dokumente (höchstens 4) |
| `token_budget` | Genutztes Kontextfenster gegen Maximum (100.000 Token) |
| `antwort` | Generierte Antwort mit wörtlichen Zitaten in `„…" [N]` |
| `quellen` | Quelldokumente inklusive Belegstellen |
| `quellen[].verifiziert` | `true`, wenn Zitate aus der Antwort im Originaltext gefunden wurden |
| `quellen[].snippets` | Originalabsätze, aus denen zitiert wurde |
| `quellen[].zitate` | Die wörtlichen Zitate aus der Antwort |
| `quellen[].abschnitt` | Überschrift des Abschnitts, aus dem der Treffer stammt |
| `quellen[].fremde_fundstellen` | **Siehe unten** — URLs, aus denen einzelne Absätze wortgleich übernommen wurden |
| `zeiten` | Dauer der Pipeline-Phasen in Sekunden |
| `chunks` | Nur bei `mit_chunks: true` — die gefundenen Chunks mit Rang, Score, Textauszug |
| `fakultaet_erkannt` | Fakultät, die aus der Frage gelesen wurde (`f1`…`f5`), sonst leer |
| `po_erkannt` | Prüfungsordnung, die aus der Frage gelesen wurde (`{studiengang, fassung}`), sonst `null` |
| `po_vielfalt` | mehrere geltende Fassungen desselben Studiengangs: `{studiengang, fassungen, …}`, sonst `null` |
| `po_entscheidung` | bei Vielfalt ohne PO-Angabe: `{urteil, quelle, …}` — warum zurückgefragt wurde oder nicht |
| `rueckfrage` | statt einer Antwort: `{frage, optionen: [...]}`, `antwort` ist dann leer |
| `studiengang_mehrdeutig` | Kürzel zu einer mehrdeutigen Kurzform („BWL im Bachelor“) |

Das Schema ist über alle Ausgänge gleich: Auch eine Fehlerantwort enthält sämtliche Felder,
nur eben leer (`kandidaten: []`, `zeiten: {}`, `antwort: ""`).

> **`fremde_fundstellen` gehört in eure Quellenanzeige.** Wortgleiche Textabschnitte liegen im
> Index nur einmal, unter einer von mehreren URLs. Steht in diesem Feld eine URL, stammt ein Teil
> des zitierten Wortlauts aus einem gleichlautenden Dokument — oft einer anderen Fakultät oder
> einem anderen Jahrgang. Wer das Feld ignoriert, zeigt eine Quellenangabe an, unter der der
> zitierte Satz so nicht zu finden ist.

### Prüfungsordnung und Rückfrage

Viele Studiengänge haben mehrere gleichzeitig geltende Fassungen ihrer Prüfungsordnung.
Hängt die Antwort davon ab, antwortet der Server **nicht**, sondern fragt zurück:

```json
{
  "antwort": "",
  "rueckfrage": {
    "frage": "Die Antwort hängt davon ab, nach welcher Prüfungsordnung Sie im Studiengang Journalistik (BJO) studieren. Welche Fassung gilt für Sie? Vorhanden: PO 2022 (aktuelle Fassung), PO 2019 — alle noch gültig, je nach Studienbeginn.",
    "optionen": [
      {"studiengang": "BJO", "name": "Journalistik", "fassung": 2019, "aktuell": false},
      {"studiengang": "BJO", "name": "Journalistik", "fassung": 2022, "aktuell": true}
    ]
  },
  "po_entscheidung": {"urteil": "verschieden", "quelle": "tabelle", "verglichen": [2019, 2022]}
}
```

Der Client zeigt die Optionen an und stellt **dieselbe Frage erneut**, jetzt mit
`po: {"studiengang": "BJO", "fassung": 2019}`. Die übrigen Fassungen werden dann aus der Suche
ausgeschlossen. Der Server merkt sich nichts — den Gesprächszustand hält der Client. Wer die
gewählte Fassung für Folgefragen zum selben Studiengang mitschickt, erspart der Nutzerin
wiederholte Rückfragen.

- **Rückfrage nach dem Studiengang:** Ist eine Kurzform mehrdeutig („BWL im Bachelor“), sind
  die Optionen Studiengänge ohne Fassung (`{studiengang, name}`). Erneut fragen mit
  `po: {"studiengang": "BBA"}`; die Fassung kommt dann aus der Frage oder per PO-Rückfrage.
- **Nennt die Frage die Fassung** („nach der PO 2018“) oder einen Studienbeginn („ich fange
  jetzt an“ → jüngste Fassung), gilt sie ohne Rückfrage; sie steht dann in `po_erkannt`.
- **Unbekannter Studiengang oder unbekannte Fassung in `po`** → `fehler` mit den vorhandenen
  Studiengängen bzw. Fassungen.
- `po_entscheidung.urteil`: `verschieden` (Rückfrage), `gleich`, `zusatzbedingung`, `unklar`,
  `eine_geltende`, `schranke` (alle ohne Rückfrage). Zur Diagnose, nicht zur Anzeige.

### Fehlerbehandlung

Fachliche Fehler kommen **nicht** als JSON-RPC-Error, sondern als reguläres Ergebnis mit
`isError: false` und gefülltem `fehler`-Feld. Immer zuerst `fehler` prüfen, nicht den
HTTP-Status:

| Situation | `fehler` beginnt mit |
|---|---|
| Modell existiert nicht | `Unbekanntes Modell '…'. Verfügbar: [...]` |
| Bezahlmodell ohne Passwort | `Dieses Premium-Modell ist passwortgeschützt.` |
| Nichts gefunden | `Keine passenden Dokumente in der Wissensdatenbank.` |
| LLM-Runde fehlgeschlagen | `Fehler bei der Dokumentauswahl (LLM Runde 1): …` bzw. `… (LLM Runde 2): …` |

---

## Vollständiges Bash-Beispiel

```bash
#!/usr/bin/env bash
set -euo pipefail

MCP_URL="https://nemo.christianwill.de/hsh/mcp"
HDR=(-H "Content-Type: application/json" -H "Accept: application/json, text/event-stream")

# 1. Session starten, Session-ID aus dem Response-Header ziehen
SESSION=$(curl -si -X POST "$MCP_URL" "${HDR[@]}" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' \
  | grep -i "^mcp-session-id:" | awk '{print $2}' | tr -d '\r')

echo "Session-ID: $SESSION"

# 2. Verfügbare Modelle abfragen
curl -s -X POST "$MCP_URL" "${HDR[@]}" -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":2,"method":"resources/read","params":{"uri":"hsh://models"}}' \
  | grep "^data:" | cut -c6- \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.loads(d['result']['contents'][0]['text']))"

# 3. Frage stellen — Fortschritt und Ergebnis im selben SSE-Stream
curl -s --no-buffer -X POST "$MCP_URL" "${HDR[@]}" -H "mcp-session-id: $SESSION" \
  -d '{
    "jsonrpc":"2.0","id":3,"method":"tools/call",
    "params":{
      "_meta":{"progressToken":"fortschritt-1"},
      "name":"frag_hsh",
      "arguments":{
        "frage":"Wie bewerbe ich mich für einen Studiengang?",
        "modell":"glm-4.7",
        "fakultaet":"",
        "speichern":false
      }
    }
  }'
```

---

## Python-Client

Eine vollständige, erprobte Implementierung liegt in `src/raghsh/web/app.py`
(Klasse `MCPClient`) und kann direkt übernommen werden. Hier die eigenständige Minimalvariante:

```python
import json
import requests


class MCPClient:
    """Minimaler MCP-Client für Streamable HTTP (MCP Spec 2025-03-26)."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.session_id: str | None = None
        self._req_id = 0

    def _id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            # Beide Typen sind Pflicht — sonst -32600 "Not Acceptable"
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            h["mcp-session-id"] = self.session_id
        return h

    @staticmethod
    def _sse(resp) -> "list[dict]":
        """Liefert alle JSON-RPC-Nachrichten eines SSE-Streams der Reihe nach."""
        for raw in resp.iter_lines():
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if line.startswith("data:"):
                yield json.loads(line[5:].lstrip())

    def initialize(self) -> None:
        resp = requests.post(self.url, headers=self._headers(), timeout=15, json={
            "jsonrpc": "2.0", "id": self._id(), "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "my-client", "version": "1.0"},
            },
        })
        resp.raise_for_status()
        for k, v in resp.headers.items():
            if k.lower() == "mcp-session-id":
                self.session_id = v

    def get_models(self) -> list[str]:
        """Die kostenlosen GWDG-Modelle. Bezahlmodelle stehen hier NICHT drin."""
        resp = requests.post(self.url, headers=self._headers(), stream=True, timeout=30, json={
            "jsonrpc": "2.0", "id": self._id(),
            "method": "resources/read", "params": {"uri": "hsh://models"},
        })
        resp.raise_for_status()
        for msg in self._sse(resp):
            contents = msg.get("result", {}).get("contents", [])
            if contents:
                # contents[0] hat uri/mimeType/text — kein "type"-Feld
                return sorted(json.loads(contents[0]["text"]))
        return []

    def frag_hsh(
        self,
        frage: str,
        modell: str,
        fakultaet: str = "",
        speichern: bool = False,
        passwort: str = "",      # nur für Bezahlmodelle nötig
        mit_chunks: bool = False,
    ) -> dict:
        req_id = self._id()
        body = {
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {
                # OHNE progressToken sendet der Server KEINE Fortschrittsmeldungen —
                # kommentarlos. Das ist der häufigste Stolperstein.
                "_meta": {"progressToken": f"frag-{req_id}"},
                "name": "frag_hsh",
                "arguments": {
                    "frage": frage,
                    "modell": modell,
                    "fakultaet": fakultaet,
                    "speichern": speichern,
                    "passwort": passwort,
                    "mit_chunks": mit_chunks,
                },
            },
        }
        # Timeout großzügig: einzelne Anfragen können mehrere Minuten dauern.
        with requests.post(self.url, headers=self._headers(), stream=True,
                           timeout=600, json=body) as resp:
            resp.raise_for_status()
            for msg in self._sse(resp):
                if msg.get("method") == "notifications/progress":
                    p = msg["params"]
                    print(f"[{p['progress']:.0f}/{p['total']:.0f}] {p.get('message', '')}")
                elif "result" in msg:
                    content = msg["result"].get("content", [])
                    if content and content[0].get("type") == "text":
                        return json.loads(content[0]["text"])
                elif "error" in msg:
                    raise RuntimeError(f"MCP-Fehler: {msg['error']}")
        return {}


if __name__ == "__main__":
    client = MCPClient("https://nemo.christianwill.de/hsh/mcp")
    client.initialize()

    modelle = client.get_models()
    print("Verfügbare GWDG-Modelle:", modelle)

    result = client.frag_hsh(
        frage="Welche Prüfungsleistungen gibt es im Bachelor Elektrotechnik?",
        modell="glm-4.7",
        fakultaet="f1",
    )

    # Bezahlmodell — deutlich günstigste Wahl ist ChatGPT:
    # result = client.frag_hsh(
    #     frage="...",
    #     modell="openai/gpt-5.6-luna",
    #     passwort="<Passwort — wird separat übergeben>",
    # )

    if result.get("fehler"):
        print("Fehler:", result["fehler"])
    else:
        print("\n=== Antwort ===")
        print(result["antwort"])
        print("\n=== Quellen ===")
        for q in result["quellen"]:
            print(f"- {q['titel']}: {q['url']}")
            for fremd in q.get("fremde_fundstellen", []):
                print(f"    Teile wortgleich aus: {fremd}")
        print(f"\nDauer: {result['zeiten'].get('gesamt_s')} s")
```

---

## Laufzeiten und Timeouts

Eine gewöhnliche Anfrage dauerte im Test **5 bis 30 Sekunden**, mit Reasoning-Modellen rund
anderthalb Minuten (siehe die Tabelle unter [Modelle](#kostenlose-modelle-gwdg-chatai)). Die
Vektorsuche selbst braucht davon nur 2–7 Sekunden; alles Weitere sind die beiden LLM-Runden.

**Nach oben ist das offen.** Gemessen am 17.08.2026: eine Frage, deren gewähltes Dokument das
volle Tokenbudget von 100.000 füllte, brauchte **608 Sekunden** — davon 606 Sekunden allein die
Antwortgenerierung. Das ist kein Fehler, sondern ein sehr großes Dokument (etwa ein
Vorlesungsverzeichnis) in einem kleinen Modell. Solche Ausreißer sind selten, aber sie kommen vor,
und die Pipeline gibt vorher keinen Hinweis darauf.

Daraus folgt für die Client-Seite:

- **Client-Timeout auf mindestens 600 Sekunden** setzen, besser höher.
- **Der öffentliche Endpunkt kappt bei 300 Sekunden** (nginx `proxy_read_timeout`). Anfragen,
  die länger brauchen, brechen dort ab, obwohl der Server weiterrechnet. Wer solche Ausreißer
  verlässlich braucht, sollte das vorher ansprechen — der Wert lässt sich serverseitig anheben.
- Große Modelle sind bei großen Kontexten oft die *schnellere* Wahl: Der 608-Sekunden-Fall
  entstand mit dem kleinsten Modell der Liste.
- Ein `progressToken` hilft doppelt: Er zeigt der Nutzerin, dass etwas passiert, und über das Feld
  `zeiten` im Ergebnis seht ihr hinterher, in welchem Schritt die Zeit verbrannt ist.

---

## Weitere Hinweise

**Zitate prüfen.** `quellen[].verifiziert` sagt, ob die Zitate der Antwort wörtlich im
Originaltext gefunden wurden. Bei `false` hat das Modell paraphrasiert oder halluziniert — die
Antwort dann nicht ungeprüft weiterreichen.

**Modelle nie fest verdrahten.** Immer `hsh://models` abfragen und die Auswahl daraus aufbauen.
Bei unbekannter ID antwortet das Tool sofort mit `fehler` und der aktuellen Liste — was auch ein
brauchbarer Weg ist, den Bestand zu prüfen.

**`antwort` kann einen Denkprozess enthalten.** Manche Modelle schreiben ihre Überlegungen mit in
den Antworttext, abgegrenzt durch `<think>…</think>` — gelegentlich steht dort auch nur das
schließende `</think>` ohne öffnendes Gegenstück. Beobachtet bei `glm-4.7`, aber nicht bei jeder
Frage. Wer den Text direkt anzeigt, sollte den Block vorher entfernen (oder einklappbar
darstellen), sonst liest die Nutzerin die Selbstgespräche des Modells mit:
`re.sub(r"(?s).*?</think>", "", antwort, count=1).strip()`.

**DSGVO.** `speichern: true` setzt eine aktive Einwilligung der Nutzerin voraus. Nur dann werden
Frage, Antwort, Modell und Quellen protokolliert; die Einträge werden nach 12 Monaten
automatisch gelöscht. Ohne Einwilligung wird nichts gespeichert.

**`mit_chunks` ist für die Auswertung.** Es liefert die tatsächlichen Treffer der Vektorsuche mit
Rang, Score und Textauszug — nützlich, um Retrieval-Qualität zu messen, aber deutlich größer in
der Übertragung. Für den Normalbetrieb weglassen.

**`nur_retrieval` ebenfalls.** Der Server führt Hybrid-Suche und Kandidatenbildung aus (Schritte
1–2, derselbe Code wie im Volllauf) und kehrt dann zurück: `kandidaten`, `chunks` und `zeiten`
sind gefüllt, `antwort`, `quellen`, `ausgewaehlte_docs`, `auswahl_begruendung` bleiben leer,
`fehler` ist `null`. `modell` muss syntaktisch mitgegeben werden, wird aber nicht geprüft;
`speichern` und `passwort` sind wirkungslos. Eine Anfrage dauert zwei bis drei Sekunden statt
dreißig bis sechzig. Gedacht für `auswertung/gold/lauf.py --nur-retrieval`, nicht für die
Weboberfläche.

---

## Änderungen gegenüber August 2026

| Was | Jetzt |
|---|---|
| **Suche** | DBSF statt RRF, zusätzlicher Dense-Arm über generierte Fragen; kein Cross-Encoder |
| **`po`** | neuer Parameter; Rückfrage statt Antwort, wenn die Antwort von der Fassung abhängt |
| **Ergebnis** | neue Felder `fakultaet_erkannt`, `po_erkannt`, `po_vielfalt`, `po_entscheidung`, `rueckfrage`, `studiengang_mehrdeutig` |
| **`fakultaet`** | wird aus der Frage gelesen, wenn leer („Fakultät IV“, „F4“) |
| **Ressource** | `hsh://kontingent` |
| **Fakultätskürzel** | `f4` ist Wirtschaft und Informatik (war hier falsch angegeben) |

---

## Änderungen gegenüber Juli 2026

Wer gegen die alte Fassung programmiert hat, prüft am besten diese Punkte:

| Was | Alt | Jetzt |
|---|---|---|
| **Modelle** | `llama-3.3-70b-instruct`, `qwen3.5-27b`, `deepseek-r1-distill-llama-70b` empfohlen | **Alle drei existieren nicht mehr.** Aktuelle Liste über `hsh://models` |
| **Fortschritt** | „Der Server sendet 5 Progress-Events" | Nur mit `params._meta.progressToken` — ohne Token kommt nichts |
| **`resources/read`** | Element mit `type: "text"` | Felder sind `uri`, `mimeType`, `text` — kein `type` |
| **Progress-Werte** | `1`, `5` | `1.0`, `5.0` (Fließkomma) |
| **`passwort`** | nicht dokumentiert | Pflicht für die drei Bezahlmodelle |
| **`mit_chunks`** | nicht dokumentiert | optionaler Diagnose-Parameter |
| **`zeiten`** | nicht dokumentiert | neues Feld im Ergebnis, Phasendauer in Sekunden |
| **`quellen[].fremde_fundstellen`** | nicht dokumentiert | neues Feld, gehört in die Quellenanzeige |
| **`fakultaet`** | als Filter beschrieben | Vorzug statt Filter — zentrale Seiten bleiben immer im Ergebnis |
| **Timeout** | „nicht unter 300 Sekunden" | mindestens 600 Sekunden; öffentlicher Endpunkt kappt bei 300 |

Endpunkte, Header, Session-Handshake, Protokollversion `2025-03-26`, Fakultätskürzel,
`speichern`-Semantik und die übrigen Ergebnisfelder sind unverändert.
