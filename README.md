# RAGHSH — RAG-System für die Hochschule Hannover

RAGHSH beantwortet Fragen zur Hochschule Hannover ausschließlich aus deren
Webauftritt (rund 5.600 Seiten und PDFs) und belegt jede Aussage mit einem
wörtlichen Zitat. Prüfungsordnungen werden nach Studiengang und Fassung
unterschieden; ist die Antwort fassungsabhängig, fragt das System nach.

```
Crawl ──► Markdown ──► Chunking + Vektoren (HPC) ──► Qdrant
                                                       │
Frage ──► Studiengang/Fakultät/PO erkennen ──► Hybrid-Suche (2× Dense, 1× Sparse, DBSF)
      ──► LLM wählt Quellen ──► LLM antwortet mit Zitaten ──► Zitat- und Markenprüfung
```

Zugriff über einen MCP-Server (Tool `frag_hsh`) und eine Streamlit-Oberfläche,
die nur über MCP mit dem Server spricht.

## Schnellstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[alles]"            # Kern + Crawler + Web + Auswertung
cp .env.example .env                 # GWDG_API_KEY eintragen
docker compose up -d                 # Qdrant auf 6333 (Speicher: var/qdrant)

python -m raghsh.server              # MCP-Server, http://127.0.0.1:8765/mcp
streamlit run src/raghsh/web/app.py  # Oberfläche, http://localhost:8501
```

Ohne Korpus und Vektoren antwortet der Server nicht sinnvoll; wie beides
entsteht, steht in [docs/pipeline.md](docs/pipeline.md).

## Aufbau

| Verzeichnis | Inhalt |
|---|---|
| `src/raghsh/` | das System als Paket: `crawl/`, `index/`, `suche/`, `po/`, `server/`, `web/`, `konfig.py` |
| `stammdaten/po/` | Studiengänge, PO-Zuordnung, Geltung, Wertetabelle, Module (versioniert) |
| `auswertung/` | Gold-Fallbasis und Messläufe (`gold/`, `laeufe/`), Fragenlisten (`fragen/`), Testindex |
| `werkzeuge/` | Wartungsskripte: Recrawl einzelner URLs, Korpusprüfung, Collection löschen, PO-Tabellen erzeugen; `werkzeuge/hpc/` Anreicherung und Vektorisierung auf KISSKI |
| `tests/` | `pytest` |
| `betrieb/` | systemd-Units, nginx-Ausschnitt |
| `docs/` | Dokumentation; `docs/archiv/` enthält überholte Pläne und Befundberichte mit Datum |
| `var/` | Laufzeitdaten, nicht versioniert: Korpus, `doc_index.db`, Vektoren, Logs, Caches, Qdrant-Speicher |

Alle Orte stehen in `src/raghsh/konfig.py` und lassen sich per
Umgebungsvariable umlenken (`RAGHSH_VAR`, `RAGHSH_ENV`, …).

## Dokumentation

| Thema | Datei |
|---|---|
| Architektur und Entscheidungen | [docs/architektur.md](docs/architektur.md) |
| Suche: Arme, Fusion, Fakultätsfilter | [docs/retrieval.md](docs/retrieval.md) |
| Prüfungsordnungen: Erkennung, Fassungen, Rückfrage | [docs/po_logik.md](docs/po_logik.md) |
| Schnittstelle `frag_hsh` für Projektpartner | [docs/mcp_api.md](docs/mcp_api.md) |
| Pipeline Crawl → HPC → Import, Delta-Betrieb | [docs/pipeline.md](docs/pipeline.md) |
| HPC (KISSKI): Zugang, Anreicherung, Vektorisierung, Delta | [docs/hpc.md](docs/hpc.md) |
| Parquet-Schema und Qdrant-Import (Vertrag) | [docs/parquet_schema.md](docs/parquet_schema.md) |
| Betrieb: Dienste, Produktion, Veröffentlichung | [docs/betrieb.md](docs/betrieb.md) |
| Auswertung: Fallbasis, Messläufe, Holdout | [docs/auswertung.md](docs/auswertung.md) |
| Entwicklungsagenten | [docs/agenten.md](docs/agenten.md) |

## Lizenz

Code: [MIT](LICENSE). Die Seitenauszüge in `auswertung/laeufe/` stammen von
hs-hannover.de; die Rechte daran liegen bei der Hochschule bzw. den jeweiligen
Urhebern, sie dienen nur der Nachvollziehbarkeit der Messungen. Die 20
Ausgangsfälle in `auswertung/gold/quelle_chargy_beidawi_validation_cases.json`
stammen aus dem Studienprojekt von Mohamed Salem Charghy und Taleb El Beidawi
([Repository](https://github.com/MohamedSalemCharghy/RAGHSH)).
Der Crawler nutzt `pymupdf4llm` (PyMuPDF, AGPL-3.0); wer ihn weitergibt,
beachtet deren Bedingungen.
