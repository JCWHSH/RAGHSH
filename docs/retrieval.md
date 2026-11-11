# Suche (Retrieval)

Stand: 11.11.2026. Werte: `src/raghsh/suche/hybrid_search.py` und Kopf von
`src/raghsh/server/mcp_server.py`. Die Begründungen mit Messungen stehen dort
an den Konstanten.

## Qdrant-Collection `hsh_knowledge`

| Feld | Art | Inhalt |
|---|---|---|
| `text` | dense, Jina v3 (`retrieval.passage`) | Chunktext |
| `questions` | dense, Jina v3 (`retrieval.query`) | doc2query-Fragen zum Chunk (nicht auf jedem Punkt) |
| `sparse` | BM25, IDF-Modifier | Chunktext |
| `sparse_expanded` | BM25, IDF-Modifier | Chunktext + Fragen |

Payload u. a. `source_url`, `title`, `faculty`, `organisation`, `chunk_index`,
`content_type`, Datum. Vollständiger Vertrag: [parquet_schema.md](parquet_schema.md).

## Ablauf einer Suche

1. **Arme:** `DENSE_ARMS = ("text", "questions")`, `SPARSE_ARMS = ("sparse_expanded",)`.
   Nur ein Sparse-Arm, weil zwei gleichartige Arme in der Fusion doppelt
   zählen (eval_gold: vier Arme 44/73, drei Arme 53/73).
2. **Fusion:** `FUSION = DBSF`. RRF wertet nur Plätze; bei der flachen Spitze
   dieses Index entscheidet dann Rauschen.
3. **Kandidaten:** `CANDIDATE_LIMIT` je Arm, URL-Dedup (`MAX_PER_URL`),
   Nachbar-Chunks für die besten `AUGMENT_TOP_N`.
4. **Fakultät:** Ist eine Fakultät gesetzt (Parameter, aus der Frage oder aus
   dem erkannten Studiengang), laufen zwei Suchen — eine nur in der Fakultät
   (plus die URLs des Studiengangs), eine über alles — und
   `_mische_kandidaten` verzahnt sie. Ein harter Filter verlöre die zentralen
   Seiten ohne Fakultät.
5. **Prüfungsordnung:** Ist eine Fassung bekannt, schließt ein Qdrant-Filter
   die anderen Fassungen desselben Studiengangs aus (`raghsh.po.filter`).
6. **Server:** `RAG_CANDIDATE_K` Kandidaten, davon `TOP_DOCS` in die
   Auswahlrunde, das LLM wählt bis zu `SELECT_N`.

## Experimente ohne Codeänderung

`frag_hsh(experiment={...})` überschreibt je Aufruf `dense_arms`,
`sparse_arms`, `fusion`, `candidate_limit`, `dedup_buffer`, `max_per_url`,
`top_docs`, `keyword_boost`, `po`. `auswertung/gold/lauf.py --nur-retrieval`
misst eine Konfiguration in Minuten ohne LLM, siehe [auswertung.md](auswertung.md).
