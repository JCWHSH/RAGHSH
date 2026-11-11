# Betrieb

Stand: 11.11.2026.

## Rechner

| Rolle | Rechner | Dienste |
|---|---|---|
| Produktion | Jetson im Heimnetz (seit 19.10.2026) | MCP-Server 8765, Web-App 8766, eigene Qdrant |
| Reverse-Proxy | Entwicklungsrechner | nginx 443: `/hsh/` → MCP, `/hsh-app/` → Web-App ([betrieb/nginx/hsh.conf](../betrieb/nginx/hsh.conf)) |
| Entwicklung, Messläufe, Agenten | Entwicklungsrechner | Qdrant 6333 (Kopie der Produktionsdaten), Agenten-Qdrant 6343; MCP/Web-App nur bei Bedarf |

Der Deploy-Weg zum Jetson ist noch nicht beschrieben (offen).

## Dienste

Vorlagen in `betrieb/systemd/`:

- `hsh-mcp.service` — User-Unit, `python -m raghsh.server`. Legt beim Start
  `/tmp/hsh-mcp.ready` an.
- `hsh-webapp.service` — System-Unit, `streamlit run src/raghsh/web/app.py`,
  wartet bis zu 10 min auf die Ready-Datei.

Zweite Instanz für Messungen: `HSH_MCP_PORT=8775 python -m raghsh.server`
(Ready-Datei `/tmp/hsh-mcp-8775.ready`); `HSH_COLLECTION` wählt eine andere
Collection.

## Umgebung

| Variable | Wirkung |
|---|---|
| `RAGHSH_VAR` | Ort der Laufzeitdaten (Standard `<repo>/var`) |
| `RAGHSH_ENV`, `RAGHSH_OPENROUTER_KEY` | Ort von `.env` bzw. `openrouter_key.txt` |
| `RAGHSH_KORPUS`, `RAGHSH_DOC_INDEX`, `RAGHSH_QA_LOG`, `RAGHSH_PO_DATEN` | einzelne Orte |
| `HSH_MCP_PORT`, `HSH_QDRANT_URL`, `HSH_COLLECTION` | Port, Qdrant, Collection |
| `HSH_GWDG_STAND` | Datei mit dem GWDG-Restkontingent (Standard `/var/lib/raghsh/gwdg_kontingent.json`) |
| `HSH_PO_STICHTAG`, `HSH_PO_VERGLEICH_GRENZE_S` | Stichtag der PO-Geltung, Zeitgrenze des Fassungsvergleichs |
| `HSH_ANSWER_ZUSATZ_DATEI` | Text, der an den Antwortprompt angehängt wird (Prompt-Versuche) |

Geheimnisse in `.env` (Vorlage `.env.example`): `GWDG_API_KEY`,
`GWDG_API_BASE`, `FULL_ACCESS_PASSWORD` (Premium-Modelle). Der
OpenRouter-Schlüssel steht in `openrouter_key.txt`.

## Laufzeitdaten (`var/`)

| Ort | Inhalt | Sicherung |
|---|---|---|
| `korpus/` | gecrawlte Seiten (Markdown) | ja (Neucrawl dauert Tage) |
| `doc_index.db` | aus Qdrant abgeleitet | nein, `build_doc_index --rebuild` |
| `qa_log.jsonl` | Fragen mit Einwilligung, 12 Monate | ja, personenbezogen |
| `vektoren/` | Parquet vom HPC | ja (GPU-Lauf) |
| `qdrant/` | Qdrant-Speicher (docker-compose) | Snapshot |
| `cache/`, `logs/`, `daten/`, `backup/` | Modelle, Logs, Delta-Archive | nein |

## GWDG-Kontingent

Produktion und Agenten teilen einen Schlüssel (30/min, 200/h, 1000/Tag,
3000/Monat). Jede Antwort trägt die Restwerte; sie landen in `HSH_GWDG_STAND`
und sind über die Ressource `hsh://kontingent`, die Web-App und das Kommando
`gwdg-kontingent` sichtbar.

## Veröffentlichung

Das öffentliche Repository bekommt einen bereinigten Schnappschuss, nie die
private Historie: Nutzerfragen (`auswertung/fragen/fragen_nutzer.csv` (nicht veröffentlicht),
Fälle mit `herkunft = nutzer_gold`) und studentische Adressen werden
entfernt bzw. geschwärzt. Das Werkzeug liegt bewusst außerhalb des
Repositorys.
