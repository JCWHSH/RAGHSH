> **Archiv.** Plan vom Mai 2026, Umsetzungsstand August. Überholt: doc2query-Anreicherung ist umgesetzt (Arm `questions`), mehrere HPC-Skripte wurden nie gebaut. Aktueller Stand: docs/architektur.md, docs/retrieval.md.

# RAGHSH — Empfohlene Systemverbesserungen

Basierend auf der Architekturanalyse vom Mai 2026. Die Verbesserungen sind nach
Aufwand und Abhängigkeiten geordnet — spätere Punkte bauen teilweise auf
früheren auf.

---

## Umsetzungsstand (Stand: 2026-08-05)

Legende: ✅ umgesetzt · 🟡 teilweise umgesetzt · ❌ offen

| # | Maßnahme | Status |
|---|----------|--------|
| 1a | Content-Hash im Crawler | ✅ |
| 1b | Sitemap.xml als Vorstufe | ❌ |
| 1c | HTTP HEAD + ETag/Last-Modified | ❌ |
| 2 | Pipeline-Zustandsverfolgung in `doc_index.db` | 🟡 |
| 3 | LLM-Anreicherung (Keywords + Fragen pro Chunk) | ❌ |
| 4 | Deutsche Textvorverarbeitung für BM25 (spaCy) | ❌ |
| 5a | HyDE | ❌ |
| 5b | Query Expansion via Embeddings | ❌ |
| 5c | Pseudo-Relevance Feedback | ❌ |
| 6 | Evaluation mit RAGAS | ✅ |
| 8 | Automatisierter HPC-Job nach jedem Crawl | ❌ |

---

## 1. Change Detection

**Problem:** Der Crawler prüft heute nur das Alter einer gecachten Datei
(`MAX_AGE_DAYS`). Nach Ablauf der Frist werden alle Seiten neu heruntergeladen,
konvertiert und vektorisiert — unabhängig davon ob sich der Inhalt geändert hat.

### 1a. Content-Hash im Crawler (Kern-Maßnahme) — ✅ UMGESETZT

> **Status:** Vollständig umgesetzt. `crawler_utils.compute_hash()` bildet einen
> SHA-256 (auf 16 Hex-Zeichen gekürzt) über den Markdown-Body. `main.py` und
> `resume_crawler.py` schreiben ihn als `content_hash` ins Frontmatter
> (`build_yaml_header()`), vergleichen ihn beim Re-Crawl gegen
> `read_stored_hash()` und schreiben die Datei nur bei Abweichung neu
> (`save_markdown_if_changed()`). Unveränderte Seiten werden per
> `touch_cached_file()` nur re-datiert — der Body bleibt byteweise erhalten,
> der Hash also gültig.

Beim Schreiben einer `.md`-Datei einen SHA-256 über den Markdown-Body (ohne
YAML-Header) berechnen und als `content_hash` im Frontmatter speichern:

```yaml
---
source_url: "https://..."
content_hash: "a3f9c2d1..."
crawl_date: "2026-05-06"
---
```

Beim nächsten Crawl-Durchlauf: Seite herunterladen → neuen Hash berechnen →
mit gespeichertem Hash vergleichen. Nur bei Abweichung die Datei neu schreiben.
Da die Datei nur bei echtem Inhaltsunterschied angefasst wird, ist `file_mtime`
danach ein zuverlässiger Proxy für Inhaltsänderungen.

**Einsparung:** Verhindert Re-Vektorisierung und Qdrant-Import für unveränderte
Seiten. Größter Gewinn bei häufigen Crawl-Durchläufen.

### 1b. Sitemap.xml als Vorstufe (optional, effizient) — ❌ OFFEN

> **Status:** Nicht umgesetzt. Weder `main.py` noch `resume_crawler.py` oder
> `crawler_utils.py` rufen eine Sitemap ab; die Queue wird ausschließlich aus
> den beim Crawlen gefundenen Links aufgebaut.

`https://www.hs-hannover.de/sitemap.xml` einmalig zu Beginn des Crawls parsen.
Seiten deren `<lastmod>` ≤ `crawl_date` in der vorhandenen `.md`-Datei sind,
werden gar nicht erst in die Crawl-Queue aufgenommen — ein einziger HTTP-Request
statt hunderten.

Einschränkung: Nicht alle Seiten stehen in der Sitemap. Als Ergänzung zu 1a,
nicht als Ersatz.

### 1c. HTTP HEAD + ETag/Last-Modified (optional, aufwändig) — ❌ OFFEN

> **Status:** Nicht umgesetzt. Ein `HEAD`-Request existiert zwar
> (`main.py:is_pdf_url()`, `resume_crawler.py`), dient aber nur der
> Content-Type-Erkennung für PDFs. `ETag` und `Last-Modified` werden weder
> ausgewertet noch gespeichert.

Vor dem eigentlichen Playwright-Rendering einen `HEAD`-Request schicken und
`ETag` / `Last-Modified` gegen gespeicherte Werte prüfen. Bei unverändertem
ETag entfällt das teure JS-Rendering komplett.

Einschränkung: TYPO3 liefert diese Header oft unzuverlässig. Lohnt sich erst
wenn Playwright-Rendering nachweislich der Engpass ist.

---

## 2. Pipeline-Zustandsverfolgung (`doc_index.db` erweitern) — 🟡 TEILWEISE

> **Status:** Schema fertig, Befüllung fehlt.
> `build_doc_index.py:open_db()` legt alle vier Spalten an und migriert
> bestehende Datenbanken per `ALTER TABLE`. `upsert_document()` nutzt
> `ON CONFLICT DO UPDATE`, damit die Zeitstempel beim Re-Index nicht
> zurückgesetzt werden — das ist bereits korrekt vorbereitet.
> **Es fehlt:** `content_hash` wird geschrieben, aber `vectorized_at`,
> `imported_at` und `enriched_at` werden von keinem Skript je gesetzt.
> `hpc_vectorizer.py` und `local_importer.py` öffnen `doc_index.db` gar nicht.
> Die im Abschnitt gezeigten Fortschritts-Queries liefern damit noch keine
> brauchbaren Ergebnisse.

**Problem:** Es gibt keinen persistenten Ort der den vollständigen Zustand einer
URL durch alle Pipeline-Stufen kennt. Die Frage „welche URLs sind gecrawlt aber
noch nicht vektorisiert?" lässt sich nur durch Ad-hoc-Vergleich von Dateisystem
und Qdrant beantworten.

### Neue Spalten in der `documents`-Tabelle

```sql
ALTER TABLE documents ADD COLUMN content_hash   TEXT;
ALTER TABLE documents ADD COLUMN vectorized_at  TEXT;  -- NULL = noch ausstehend
ALTER TABLE documents ADD COLUMN imported_at    TEXT;  -- NULL = noch ausstehend
ALTER TABLE documents ADD COLUMN enriched_at    TEXT;  -- NULL = LLM noch nicht gelaufen
```

| Spalte | Bedeutung |
|--------|-----------|
| `content_hash` | SHA-256 des MD-Bodys; Änderung triggert Re-Vektorisierung |
| `vectorized_at` | Zeitstempel wann Vektoren berechnet wurden; `NULL` = steht noch aus |
| `imported_at` | Zeitstempel wann nach Qdrant importiert wurde; `NULL` = steht noch aus |
| `enriched_at` | Zeitstempel wann Keywords + Fragen per LLM generiert wurden |

Damit lässt sich der Pipeline-Fortschritt jederzeit per SQL abfragen:

```sql
-- Gecrawlt aber noch nicht vektorisiert:
SELECT source_url FROM documents WHERE vectorized_at IS NULL;

-- Vektorisiert aber noch nicht in Qdrant:
SELECT source_url FROM documents WHERE vectorized_at IS NOT NULL
                                   AND imported_at IS NULL;

-- Inhalt geändert seit letzter Vektorisierung (wenn hash sich unterscheidet):
SELECT source_url FROM documents WHERE content_hash != last_vectorized_hash;
```

`doc_index.db` bleibt die einzige Datei für Pipeline-Zustand. Das Parquet wird
zum flüchtigen HPC-Transferformat degradiert — kein dauerhafter Artefakt mehr.

---

## 3. LLM-Anreicherung: Keywords und Fragen pro Chunk — ❌ OFFEN

> **Status:** Nicht umgesetzt. Es gibt keinen Anreicherungsschritt zwischen
> Crawlen und Vektorisieren. Das Parquet-Schema in `hpc_vectorizer.py` kennt
> weder `text_original` noch `keywords` oder `questions`; `local_importer.py`
> schreibt diese Felder folglich auch nicht ins Qdrant-Payload. Vektorisiert
> wird weiterhin der rohe Chunk-Text.

**Ziel:** Die BM25-Sparse-Suche trifft Chunks präziser, weil für jeden Chunk
vorher per LLM typische Nutzerfragen und Stichworte generiert werden. Diese
werden dem Chunk-Text vorangestellt bevor der BM25-Vektor berechnet wird.

### Neuer Pipeline-Schritt: zwischen Crawlen und Vektorisieren

```
Crawlen → [NEU: LLM-Anreicherung] → Vektorisieren → Qdrant-Import
```

Der neue Schritt läuft pro `.md`-Datei (oder besser: pro Chunk nach dem
Chunking-Schritt in `hpc_vectorizer.py`):

1. LLM liest den Chunk-Text
2. Generiert 3–5 Fragen, die dieser Chunk beantworten kann
3. Generiert 5–10 Stichworte / Keyphrases
4. Fragen + Stichworte werden dem Text **vorangestellt**:

```
[Mögliche Fragen: Wie bewerbe ich mich für den Studiengang? Gibt es einen NC?
Stichworte: Bewerbung, Zulassung, Numerus Clausus, Einschreibung]

<ursprünglicher Chunk-Text>
```

5. Dieser angereicherte Text wird vektorisiert (Dense + BM25)
6. Die Originaltexte bleiben im Payload erhalten (`text_original`)

**Warum BM25 davon profitiert:** Wenn ein Nutzer fragt „Wie bewerbe ich mich?",
trifft der Sparse-Arm exakt die Chunks für die das LLM genau diese Frage
generiert hat — auch wenn der Originaltext diese Formulierung nicht enthält
(z.B. wenn er nur „Zulassungsverfahren" als Überschrift hat).

### Qdrant-Payload-Erweiterung

Die Rohlisten zusätzlich im Payload speichern (ohne in den Vektortext
einzufließen), damit `build_rag_context()` sie dem LLM als Kontext mitgeben
kann:

```python
payload={
    ...,                         # bestehende Felder
    "keywords":  ["Bewerbung", "NC", "Zulassung"],
    "questions": ["Wie bewerbe ich mich?", "Gibt es einen NC?"],
    "text_original": "<ursprünglicher Chunk-Text ohne Anreicherung>",
}
```

Qdrant erlaubt Payload-Updates ohne Neuvektorisierung (`client.set_payload()`),
falls die Anreicherung nachträglich ergänzt oder korrigiert werden soll.

### Inkrementelle Verarbeitung

`enriched_at` in `doc_index.db` steuert welche Dokumente bereits angereichert
wurden. Nur Dokumente mit `enriched_at IS NULL` oder `content_hash` geändert
werden neu durch das LLM geschickt. Bei einem vollständigen Crawl der HSH-Website
(~3.000–5.000 Seiten × ~10 Chunks) ist das ein nicht-trivialer LLM-Aufwand —
inkrementelle Verarbeitung ist daher Pflicht.

---

## 4. Deutsche Textvorverarbeitung für BM25 — ❌ OFFEN

> **Status:** Nicht umgesetzt — gilt für alle drei Unterpunkte (4a
> Lemmatisierung, 4b Kompositum-Zerlegung, 4c POS-Filterung). spaCy, HanTa,
> CharSplit o.ä. kommen im gesamten Projektcode nicht vor und stehen auch
> nicht in `requirements.txt`. `hpc_vectorizer.py` übergibt den Rohtext direkt
> an FastEmbeds `Qdrant/bm25`, `hybrid_search.embed_query_sparse()` ebenso die
> rohe Nutzerfrage.

**Problem:** FastEmbed's `Qdrant/bm25` verwendet einfaches Whitespace-Tokenizing
ohne deutsches Sprachprocessing. Das führt dazu, dass verschiedene Wortformen
desselben Konzepts als unterschiedliche Terme behandelt werden, und dass lange
deutsche Komposita nicht auf ihre Bestandteile durchsucht werden können.

**Wichtig:** Diese Vorverarbeitung muss konsistent an zwei Stellen angewendet
werden — beim Aufbau des BM25-Index (Vektorisierungsschritt) und zur Laufzeit
bei der Suchanfrage (`hybrid_search.py`).

### 4a. Lemmatisierung

Wörter werden auf ihre Grundform zurückgeführt, bevor der BM25-Vektor berechnet
wird:

| Wortform | Lemma |
|----------|-------|
| Studierende, Studierenden, Studierender | Studierender |
| beantragen, beantragt, Beantragung | beantragen |
| Prüfungen, Prüfung, Prüfungsleistung | Prüfung |

Empfehlung: **spaCy** mit `de_core_news_lg` oder **HanTa** (Hanover Tagger,
speziell für Deutsch entwickelt). Beide sind als Python-Pakete verfügbar.

### 4b. Kompositum-Zerlegung

Das Deutsche bildet lange zusammengesetzte Wörter. Ohne Zerlegung findet eine
Suche nach „Zulassung" keinen Chunk der nur „Zulassungsvoraussetzungen" enthält:

```
"Zulassungsvoraussetzungen" → "Zulassung", "Voraussetzung"
"Prüfungsordnung"           → "Prüfung", "Ordnung"
"Studienangebot"            → "Studium", "Angebot"
"Immatrikulationsbescheinigung" → "Immatrikulation", "Bescheinigung"
```

Heuristik: Wörter über ~15 Zeichen auf bekannte Wortstämme aufteilen. spaCy's
`de_core_news_lg` erkennt viele Komposita. Alternativ das spezialisierte Paket
`CharSplit` oder `compound-split`.

### 4c. POS-Filterung

Grammatische Funktionswörter (Artikel, Präpositionen, Konjunktionen, Pronomen)
tragen keine inhaltliche Bedeutung und verrauschen den BM25-Index. Nur
inhaltstragende Wortarten behalten:

- **Behalten:** Nomen, Verben (außer Hilfsverben), Adjektive, Adverbien
- **Entfernen:** Artikel, Präpositionen, Konjunktionen, Pronomen, Interpunktion

Dieser Schritt baut auf der POS-Erkennung von spaCy auf — wenn spaCy ohnehin
für Lemmatisierung eingesetzt wird, ist POS-Filterung kostenlos dazu verfügbar.

---

## 5. Suchanfragen-Verbesserungen zur Laufzeit

### 5a. HyDE — Hypothetical Document Embeddings — ❌ OFFEN

> **Status:** Nicht umgesetzt. `perform_hybrid_search()` embeddet in
> `embed_query_dense()` unverändert die rohe Nutzerfrage; vor dem
> `query_points()`-Aufruf findet kein LLM-Call statt.

**Idee:** Statt die rohe Nutzerfrage zu embedden, lässt man das LLM zuerst eine
kurze hypothetische Antwort generieren und bettet *diese* ein. Das verbessert
den Dense-Arm der Hybrid-Suche erheblich, weil die Hypothese sprachlich den
echten Dokumenten ähnelt — kurze Fragen und lange Dokumentpassagen liegen im
Embedding-Raum weit auseinander.

```
Nutzerfrage:  "Wie bewerbe ich mich für Informatik?"
       ↓ LLM
Hypothese: "Für den Bachelorstudiengang Informatik an der HsH bewerben
            Sie sich über das Online-Bewerbungsportal. Die Bewerbungsfrist
            für das Wintersemester endet am 15. Juli..."
       ↓ Dense-Embedding der Hypothese
→ Vektorsuche findet Chunks die dieser Antwort ähneln
```

Implementierung: ein kurzer LLM-Aufruf (GWDG-API) vor dem `query_points()`-
Aufruf in `hybrid_search.py`. Der originale BM25-Arm bleibt unverändert —
nur der Dense-Vektor wird durch den Hypothesen-Vektor ersetzt.

### 5b. Query Expansion via Embeddings — ❌ OFFEN

> **Status:** Nicht umgesetzt. Ein Vokabular-Embedding-Index existiert nicht,
> `build_doc_index.py` erzeugt keinen.
> **Verwandt, aber etwas anderes:** `hsh_mcp_server._keyword_boost_urls()`
> holt zusätzliche Quellen über einen exakten `MatchText`-Scroll auf
> Eigennamen aus der Frage (Bindestrich-/Großbuchstaben-Heuristik). Das ist
> ein lexikalischer Recall-Boost ohne semantische Termerweiterung — die hier
> beschriebene Expansion ersetzt es nicht.

Aus dem Korpusvokabular werden zur Queryzeit semantisch ähnliche Terme ergänzt,
bevor die BM25-Suche läuft:

```
Query:   "Stipendium"
→ Expansion findet im HSH-Korpus: "Förderung", "Deutschlandstipendium", "BAföG"
→ BM25 sucht nach allen vier Termen
```

Erfordert einen vorberechneten Vokabular-Embedding-Index aus dem Korpus (einmalig
beim Setup, passt gut in `build_doc_index.py`). Zur Laufzeit: Query-Term embedden,
Top-K ähnliche Vokabular-Terme per Cosine-Similarity finden, gewichtet mit
niedrigerem Faktor (0.2) der BM25-Suche hinzufügen.

### 5c. Pseudo-Relevance Feedback (PRF) — ❌ OFFEN

> **Status:** Nicht umgesetzt. Die Suche läuft in einem einzigen
> `query_points()`-Aufruf; es gibt keinen zweiten Durchlauf mit erweiterter
> Query.

Zweistufige Suche: ein erster BM25-Durchlauf liefert Top-8-Kandidaten. Aus
deren Text werden häufige, informative Terme extrahiert (TF×IDF-gewichtet) und
mit niedrigem Gewicht zur ursprünglichen Query addiert. Dann zweiter, finaler
Suchdurchlauf.

```
1. Suche nach "Pflegegrad beantragen"
2. Top-8 Treffer enthalten häufig: "MDK", "Begutachtung", "Pflegekasse"
3. Neue Suche: "Pflegegrad beantragen" + 0.2×"MDK" + 0.2×"Begutachtung"
```

**Trade-off:** Verdoppelt die Suchlatenz. Sinnvoll wenn Antwortqualität Vorrang
vor Geschwindigkeit hat.

---

## 6. Evaluation mit RAGAS — ✅ UMGESETZT

> **Status:** Umgesetzt in `eval_ragas.py` — sogar in einer weiterentwickelten
> Form gegenüber dem Vorschlag: statt `qa_log.jsonl` dient ein synthetisch
> erzeugtes Test-Set (`eval_results/testset.json`) mit bekannter Ground-Truth-
> `source_url` als Grundlage. Gemessen werden **Source Hit Rate**,
> **Faithfulness** und **Answer Relevancy** (GWDG-LLM `llama-3.3-70b-instruct`
> + lokale Jina-Embeddings). Läufe werden per `--label` versioniert abgelegt.
> Die Baseline liegt vor: `eval_results/2026-05-06_baseline.json`.
> **Abweichung vom Vorschlag:** Context Precision und Context Recall werden
> nicht berechnet — die Source Hit Rate tritt an ihre Stelle.

**Problem:** Ohne systematische Messung ist unbekannt ob die implementierten
Verbesserungen (Lemmatisierung, LLM-Anreicherung, HyDE etc.) die Antwortqualität
tatsächlich verbessern — oder ob sie nur Aufwand erzeugen.

**RAGAS** (RAG Assessment) ist ein Evaluationsframework das automatisch vier
Metriken berechnet:

| Metrik | Frage |
|--------|-------|
| **Faithfulness** | Hält sich die Antwort an die gefundenen Quellen? (keine Halluzinationen) |
| **Answer Relevancy** | Beantwortet die Antwort die gestellte Frage? |
| **Context Precision** | Sind die gefundenen Chunks tatsächlich relevant? |
| **Context Recall** | Wurden alle relevanten Chunks gefunden? |

Das bestehende `qa_log.jsonl` (enthält gestellte Fragen und gegebene Antworten)
ist die direkte Grundlage dafür. Ergänzt um die genutzten Quell-Chunks pro
Antwort lässt sich RAGAS ohne zusätzliche manuelle Arbeit betreiben.

**Praktische Anwendung:** Vor und nach jeder größeren Änderung (z.B. nach
Aktivierung der Lemmatisierung) einen RAGAS-Lauf über das `qa_log.jsonl`
durchführen und die vier Metriken vergleichen.

---

## 7. Empfohlene Umsetzungsreihenfolge

| Schritt | Maßnahme | Aufwand | Erwarteter Gewinn | Status |
|---------|----------|---------|-------------------|--------|
| 1 | `doc_index.db`: Spalten `content_hash`, `vectorized_at`, `imported_at`, `enriched_at` | gering | Betriebseffizienz | 🟡 Spalten da, Zeitstempel werden nie gesetzt |
| 2 | Crawler: `content_hash` berechnen, Datei nur bei Änderung neu schreiben | gering | Betriebseffizienz | ✅ |
| 3 | RAGAS-Baseline messen (vor allen Änderungen) | gering | Messbarkeit | ✅ `2026-05-06_baseline.json` |
| 4 | spaCy-Vorverarbeitung: Lemmatisierung + Komposita + POS für BM25 | mittel | Suchqualität hoch | ❌ |
| 5 | LLM-Anreicherungsskript: Fragen + Keywords pro Chunk | mittel | Suchqualität hoch | ❌ |
| 6 | `hpc_vectorizer.py`: angereicherter Text als Vektorisierungsbasis | mittel | Suchqualität hoch | ❌ |
| 7 | HyDE in `hybrid_search.py` | gering | Suchqualität mittel | ❌ |
| 8 | Query Expansion (Vokabular-Index + Laufzeit-Expansion) | mittel | Suchqualität mittel | ❌ |
| 9 | Sitemap-Pre-Check im Crawler | gering | Betriebseffizienz | ❌ |
| 10 | PRF (zweistufige Suche) | mittel | Suchqualität mittel, Latenz ↑ | ❌ |

Schritt 3 (RAGAS-Baseline) sollte vor Schritt 4 erfolgen, damit Verbesserungen
messbar sind. Schritte 4–6 sind die wirkungsvollsten Einzel-Maßnahmen und haben
untereinander keine Abhängigkeiten — sie können parallel angegangen werden.

**Nächster sinnvoller Schritt:** Die Schritte 1–3 sind faktisch erledigt
(Betriebseffizienz + Messbarkeit stehen), Schritt 1 bräuchte nur noch das
Setzen der drei Zeitstempel durch `hpc_vectorizer.py` bzw. `local_importer.py`.
Inhaltlich beginnt die eigentliche Arbeit damit bei Schritt 4.

---

## 8. Automatisierter HPC-Job nach jedem Crawl (KISSKI) — ❌ OFFEN

> **Status:** Nicht umgesetzt. Von den beschriebenen Bausteinen existiert nur
> `hpc_vectorizer.py` in seiner alten Form: es liest weiterhin die
> `.md`-Dateien direkt (inkl. Chunking), kennt keinen `--input`-Parameter für
> ein Chunk-Parquet und erzeugt kein Delta. `hpc_preprocess.py`,
> `hpc_enrich.py`, `raghsh_pipeline.slurm` und `submit_hpc_job.py` gibt es
> nicht; Transfer und Job-Einreichung laufen unverändert manuell per `scp`.

**Ziel:** Nach jedem Crawl-Durchlauf wird automatisch ein SLURM-Job auf dem
KISSKI-HPC eingereicht, der Lemmatisierung, LLM-Anreicherung und Vektorisierung
für alle geänderten Seiten übernimmt.

### Gesamtübersicht

```
Lokal                          HPC KISSKI
──────                         ──────────
Crawl (geänderte Seiten)
       │
       ▼
Geänderte .md-Dateien
+ doc_index.db (Zustand)
       │
       └─── rsync ───────────► data/ingested_delta/
                                       │
                                       ▼
                               [Step 1 — CPU] hpc_preprocess.py
                                 Chunking (wie bisher)
                                 spaCy: Lemmatisierung
                                 Kompositum-Zerlegung
                                 POS-Filterung
                                       │
                                       ▼ chunks_preprocessed.parquet
                               [Step 2 — CPU/API] hpc_enrich.py
                                 GWDG ChatAI API
                                 Pro Chunk: Keywords + Fragen
                                 Batch-Calls (20 Chunks/Call)
                                       │
                                       ▼ chunks_enriched.parquet
                               [Step 3 — GPU H100] hpc_vectorizer.py
                                 Dense: Jina v3 auf angereichertem Text
                                 BM25: auf lemmatisiertem Text
                                       │
                                       ▼ hsh_vectors_delta.parquet
       ┌─── scp ◄──────────────────────┘
       │
       ▼
local_importer.py → Qdrant (Upsert)
doc_index.db aktualisieren
```

### Step 1 — `hpc_preprocess.py` (CPU, ~10 Min)

Liest die `.md`-Dateien, chunked wie bisher (`MarkdownHeaderTextSplitter` +
`RecursiveCharacterTextSplitter`) und erzeugt zwei Textvarianten pro Chunk:

| Feld | Inhalt | Verwendung |
|------|--------|------------|
| `text_original` | Originaltext des Chunks | Anzeige im Chatbot |
| `text_lemmatized` | spaCy-lemmatisiert, Komposita zerlegt, POS-gefiltert | BM25-Basis |

Ausgabe: `chunks_preprocessed.parquet` (keine Vektoren, nur Text + Metadaten).
spaCy-Modell: `de_core_news_lg` (150 MB, deckt Lemmatisierung + POS + Komposita ab).

### Step 2 — `hpc_enrich.py` (API-Calls, ~30–90 Min)

Liest `chunks_preprocessed.parquet` und schickt Chunks in Batches an die GWDG
ChatAI API. Pro Batch (~20 Chunks) ein einziger API-Call mit JSON-Mode-Output:

```
Ein API-Call → 20 Chunks rein → 20 × {keywords, questions} raus (JSON)
```

Pro Chunk werden ergänzt:
- `keywords` — 5–10 deutsche Stichworte als JSON-Array
- `questions` — 3–5 typische Nutzerfragen als JSON-Array
- `text_enriched` — `[Fragen: ...]\n[Stichwörter: ...]\n\n<text_original>`

`text_enriched` wird später als Basis für den Dense-Vektor verwendet.
Für BM25 wird `text_lemmatized` mit den (ebenfalls lemmatisierten) Keywords und
Fragen vorangestellt.

Modell: ein günstiges, schnelles Modell reicht für diese strukturierte
Extraktionsaufgabe aus (`meta-llama/Llama-3.1-8B-Instruct` o.ä.) — kein
leistungsstarkes Modell nötig.

Ausgabe: `chunks_enriched.parquet`

### Step 3 — `hpc_vectorizer.py` (GPU H100, ~15 Min)

Liest `chunks_enriched.parquet` statt der `.md`-Dateien direkt (Chunking und
Preprocessing sind bereits erledigt):
- Dense-Embedding auf `text_enriched`
- BM25-Embedding auf lemmatisiertem Text + Keywords/Fragen

Ausgabe: `hsh_vectors_delta.parquet` — nur die neu verarbeiteten Chunks, nicht
alle (Upsert-sicher durch deterministische UUID5-IDs).

### Parquet-Schema-Erweiterung

```python
pa.field("text_original",  pa.string()),  # Originaltext für Anzeige
pa.field("keywords",       pa.string()),  # JSON-Array der Stichworte
pa.field("questions",      pa.string()),  # JSON-Array der typischen Fragen
# "text" bleibt, enthält jetzt text_enriched (Dense-Basis)
```

### SLURM-Job-Script (`raghsh_pipeline.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=raghsh_pipeline
#SBATCH --time=03:00:00
#SBATCH --output=raghsh_%j.log
#SBATCH --mail-type=END,FAIL

# Step 1: Preprocessing (CPU)
srun --nodes=1 --cpus-per-task=8 --mem=16G \
    python hpc_preprocess.py --input data/ingested_delta/

# Step 2: LLM-Anreicherung (CPU + API)
srun --nodes=1 --cpus-per-task=4 --mem=8G \
    python hpc_enrich.py

# Step 3: Vektorisierung (GPU)
srun --nodes=1 --gres=gpu:h100:1 --mem=32G \
    python hpc_vectorizer.py --input chunks_enriched.parquet
```

### Lokaler Trigger — `submit_hpc_job.py`

Neues Skript das nach jedem Crawl-Durchlauf aufgerufen wird:

1. `doc_index.db` abfragen: alle URLs mit `vectorized_at IS NULL` oder
   geändertem `content_hash`
2. Nur diese `.md`-Dateien per `rsync` nach KISSKI übertragen
   (`data/ingested_delta/`)
3. SLURM-Job per SSH einreichen:
   `ssh kisski sbatch raghsh_pipeline.slurm`
4. Job-ID speichern, auf Fertigstellung pollen:
   `ssh kisski squeue -j <job_id>`
5. Delta-Parquet zurückholen:
   `scp kisski:~/raghsh/hsh_vectors_delta.parquet .`
6. `local_importer.py` ausführen (Upsert — nur geänderte Chunks)
7. `doc_index.db` aktualisieren: `vectorized_at`, `enriched_at`,
   `imported_at` setzen

### Inkrementelle Verarbeitung

`enriched_at` in `doc_index.db` steuert welche Dokumente das LLM bereits
gesehen hat. Nur Dokumente mit `enriched_at IS NULL` oder geändertem
`content_hash` werden durch Step 2 geschickt. Beim ersten Vollauf
(~3.000–5.000 Seiten × ~10 Chunks = ~50.000 Chunks) entstehen bei 20
Chunks/Batch etwa 2.500 API-Calls. Danach wächst der Aufwand nur mit den
tatsächlich geänderten Seiten.
