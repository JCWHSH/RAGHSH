# Architektur

Stand: 11.11.2026. Zahlen und Parameter stehen im Code; hier steht, wo, und
warum es so ist.

## Datenfluss

```
             ┌──────────── offline ────────────┐
hs-hannover.de ─► crawl/spider.py ─► var/korpus/*.md (YAML-Kopf, content_hash)
                                          │
                     index/hpc_vectorizer.py (HPC, GPU) ─► var/vektoren/*.parquet
                                          │
                     index/local_importer.py ─► Qdrant „hsh_knowledge“
                     index/build_doc_index.py ─► var/doc_index.db
             └─────────────────────────────────┘

             ┌──────────── je Frage (server/mcp_server.py: frag_hsh) ───────────┐
Frage ─► po/pfad.py: Fakultät, Studiengang, PO-Fassung aus der Frage
      ─► suche/hybrid_search.py: Dense „text“ + Dense „questions“ + Sparse „sparse_expanded“, DBSF
         (Fakultät bekannt → zweiter Arm nur in der Fakultät, Kandidaten gemischt)
      ─► po/pfad.py: Fassungsvielfalt → Wertetabelle → Schranke → LLM-Fassungsvergleich
         → ggf. Rückfrage statt Antwort
      ─► LLM-Runde 1: wählt bis zu SELECT_N Dokumente aus TOP_DOCS Kandidaten
      ─► LLM-Runde 2: Antwort mit wörtlichen Zitaten „…“ [N]
      ─► Zitatbereinigung und Markenkorrektur (deterministisch)
             └──────────────────────────────────────────────────────────────────┘
```

## Module

| Modul | Aufgabe |
|---|---|
| `raghsh.konfig` | alle Orte (Code, `var/`, Stammdaten, `.env`), per Umgebung umlenkbar |
| `raghsh.crawl` | Spider, Fortsetzen, Change Detection (`crawler_utils`), Delta-Liste |
| `raghsh.index` | Chunking, Vektorisierung (HPC), Import, Dokumentindex, Collection anlegen |
| `raghsh.suche.hybrid_search` | Suche je Arm, Fusion, URL-Dedup, Nachbar-Chunks |
| `raghsh.po` | `filter` (Fassungen je Studiengang, Qdrant-Filter, Geltung), `werte` (Wertetabelle), `pfad` (der PO-Teil von `frag_hsh`) |
| `raghsh.server.mcp_server` | MCP-Server: `frag_hsh`, Ressourcen `hsh://models`, `hsh://kontingent`, LLM-Aufrufe, Prompts, Belegprüfung |
| `raghsh.web.app` | Streamlit-Oberfläche, reiner MCP-Client |

## Entscheidungen

- **Ein Einstiegspunkt.** Jede Antwort entsteht in `frag_hsh`; Oberfläche,
  Auswertung und Projektpartner sprechen nur MCP. So verhalten sich alle
  Clients gleich.
- **Zwei LLM-Runden.** Runde 1 wählt aus den Kandidaten die Dokumente, Runde 2
  antwortet nur daraus. Die Kandidatenliste ist breit (Recall), die Auswahl
  schmal (Präzision, Kontextgröße).
- **Belegpflicht.** Jede Aussage braucht ein wörtliches Zitat mit Marke.
  Zitate, die nicht in den Quellen stehen, werden entfernt bzw. auf die
  richtige Quelle umgesetzt (deterministisch, ohne LLM).
- **`questions` ist Suchsignal, kein Inhalt.** Die doc2query-Fragen liegen im
  Payload, kommen aber nie in den LLM-Kontext — sie stehen nicht auf der Seite.
- **`faculty` und `organisation` sind getrennt.** Zentrale Seiten haben keine
  Fakultät; ein harter Fakultätsfilter verlöre rund 39.600 Punkte
  (Immatrikulation, Prüfungsamt). Deshalb zwei Suchen, siehe
  [retrieval.md](retrieval.md). Qdrant führt `Fakultät IV`, `doc_index.db`
  `f4`; `_faculty_key()` im Server übersetzt.
- **`doc_index.db` ist aus Qdrant abgeleitet.** Nach jedem Import
  `python -m raghsh.index.build_doc_index --rebuild`, sonst lädt der Server
  Chunks über veraltete IDs.
- **Deterministische IDs.** UUID5 über `source_url + chunk_index`; jede URL
  wird beim ersten Auftreten in einem Importlauf per Filter geleert und neu
  geschrieben.
- **Zustandsloser Server.** Die gewählte Prüfungsordnung schickt der Client
  mit (`po`); der Server merkt sich nichts zwischen Aufrufen.
- **Kein Cross-Encoder.** Seit 21.09.2026 aus, seit S075 entfernt: auf dieser
  Hardware ≥ 17× zu langsam, Nutzen nicht belegt.

## Fallen

- Dense-Anfragen mit `embedder.query_embed(...)`, Sparse mit
  `SparseTextEmbedding(..., language="german").query_embed(...)`.
  `embed(..., task="retrieval.query")` schluckt fastembed 0.8 stillschweigend
  (dann Passage-Adapter), und `language` steht sonst auf Englisch — die
  Token-IDs gibt es im Index nicht.
- Beide Sparse-Felder brauchen `modifier=Modifier.IDF`, sonst rankt Qdrant
  stillschweigend nach roher Termfrequenz.
- DBSF normalisiert über die zurückgelieferte Kandidatenmenge: Wer
  `CANDIDATE_LIMIT` ändert, verschiebt das Ranking.
- Gleiche Slugs auf verschiedenen Hosts (`index`): Cache, Aufräumen und Löschen
  gehen immer über `source_url` im Kopf, nie über den Dateinamen.
