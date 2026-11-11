# CLAUDE.md

Hinweise für Claude Code in diesem Repository. Code-Kommentare und Doku sind
deutsch.

## Was das ist

RAG-System für die Hochschule Hannover: Crawl von hs-hannover.de → Chunks mit
Dense- und Sparse-Vektoren in Qdrant → Hybrid-Suche → zwei LLM-Runden
(Quellenauswahl, Antwort mit wörtlichen Zitaten) über den MCP-Server
`frag_hsh`. Überblick: `README.md`, Einzelheiten: `docs/`.

## Struktur

- `src/raghsh/` installierbares Paket (`pip install -e ".[alles]"`):
  `crawl/`, `index/`, `suche/`, `po/` (Prüfungsordnungen), `server/`, `web/`.
- `src/raghsh/konfig.py` ist die einzige Stelle für Pfade. Keine Pfade relativ
  zu `__file__` oder zum Arbeitsverzeichnis bauen — `konfig` benutzen.
- `var/` Laufzeitdaten (Korpus, `doc_index.db`, `qa_log.jsonl`, Parquet,
  Qdrant-Speicher), nie versioniert. `.env` und `openrouter_key.txt` im
  Wurzelverzeichnis, nie versioniert.
- `stammdaten/po/` versionierte PO-Tabellen; `werkzeuge/po/` erzeugt sie.
- `werkzeuge/hpc/` läuft nur auf KISSKI (`docs/hpc.md`), neben dem dortigen
  `chunking.py`. Der Index vom 14.08.2026 stammt aus `werkzeuge/hpc/stand_2026-08-14/`,
  nicht aus `src/raghsh/index/` (anderer `chunking_config_hash`).
- `auswertung/` Fallbasis und Messläufe; `auswertung/laeufe/*/manifest.json`
  und `faelle/` sind Belege und werden nicht verändert.

## Befehle

```bash
python -m raghsh.server                          # MCP-Server :8765 (HSH_MCP_PORT)
streamlit run src/raghsh/web/app.py              # Oberfläche
python -m raghsh.crawl.spider                    # Crawl → var/korpus
python -m raghsh.index.local_importer            # Parquet → Qdrant
python -m raghsh.index.build_doc_index --rebuild # nach jedem Import Pflicht
python auswertung/gold/lauf.py --nur-retrieval --mcp-url http://127.0.0.1:8775/mcp
pytest
```

## Fallen

- Dense-Anfragen: `embedder.query_embed(...)`; Sparse:
  `SparseTextEmbedding(..., language="german").query_embed(...)`. Andere Wege
  liefern stillschweigend falsche Vektoren (siehe `docs/architektur.md`).
- Sparse-Felder in Qdrant nur mit `modifier=Modifier.IDF`.
- `questions` (doc2query) ist Suchsignal, nie LLM-Kontext.
- `faculty` (`Fakultät IV`) und `doc_index.db` (`f4`) schreiben Fakultäten
  verschieden; `_faculty_key()` übersetzt. Kein harter Fakultätsfilter.
- DBSF hängt an `CANDIDATE_LIMIT`.
- `doc_index.db` nach jedem Import neu bauen; `po_werte.tsv` trägt Punkt-IDs
  und muss nach einem Neuimport mit `werkzeuge/po/po_werte_pruefen.py` geprüft
  werden.
- Nutzerfragen (`qa_log.jsonl`, `fragen_nutzer.csv`, Fälle `nutzer_gold`)
  sind personenbezogen: nie in Commits für das öffentliche Repository, nie in
  Ausgaben an Dritte. Holdout-Fragen nie in Dateien, die die Agenten lesen.

## Arbeitsweise

- LLM: GWDG ChatAI über den OpenAI-Client, `temperature=0.0`. Kontingent wird
  mit den Agenten geteilt — Messläufe sparsam.
- Verhaltensänderungen am Antwortpfad mit einem Retrieval-Lauf vorher/nachher
  belegen (`auswertung/gold/lauf.py --nur-retrieval`).
- Doku in `docs/` mit dem Code ändern; Zahlen nicht abschreiben, sondern auf
  die Konstante verweisen.
