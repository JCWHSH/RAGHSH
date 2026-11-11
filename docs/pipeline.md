# RAGHSH — Pipeline-Ablauf

Wie Inhalte von der Website in den Chatbot gelangen — und wie man nach einem
Teil-Crawl nur die geänderten Seiten nachzieht, statt jedes Mal alles neu zu
rechnen.

Alle Befehle laufen aus dem Wurzelverzeichnis des Repositorys, mit aktivem venv und installiertem Paket (`pip install -e .`).

---

## Die zwei Betriebsarten

| | Volldurchlauf | Delta-Durchlauf |
|---|---|---|
| Wann | Erstaufbau, geändertes Chunking, neues Embedding-Modell | Normalfall nach jedem Crawl |
| Umfang | alle 5.300 Dokumente | nur die geänderten |
| Transfer zum HPC | ~1,2 GB | wenige MB |
| Dauer | über eine Stunde | wenige Minuten |

Der Delta-Durchlauf ist der Regelfall. Wann er **nicht** genügt, steht unten
unter „Wann ein Volldurchlauf nötig ist".

---

## Warum der HPC überhaupt gebraucht wird

Die Vektorisierung besteht aus zwei Teilen mit sehr unterschiedlichen Kosten:

- **Dense** (Jina Embeddings v3, 1024 Dimensionen) — braucht eine GPU.
  Auf dem Server (Intel i5-2450M, keine GPU) dauert ein Chunk rund 12 Sekunden.
  Die 62 Seiten eines kleinen Nachcrawls entsprechen etwa 2.400 Chunks, also
  ungefähr **8 Stunden**. Auf einer H100 sind es Minuten.
- **Sparse** (BM25) — reine CPU-Statistik, 202.000 Chunks in 7 Minuten.

- **Anreicherung** (doc2query, Qwen3.5-9B über vLLM): drei Fragen je Chunk,
  die den vierten Retrieval-Arm (`questions`, `sparse_expanded_*`,
  `vector_questions`) speisen. Ebenfalls GPU.

Deshalb laufen Anreicherung und Vektorisierung immer über KISSKI, auch bei
kleinen Deltas. **Die Anreicherung gehört jedes Mal dazu:** Punkte ohne Fragen
haben nur drei von vier Armen und verlieren in der Fusion (gemessen im
September an g11, g34, g38).

> **Zugang, Umgebung, Skripte und Befehle auf dem Cluster: `docs/hpc.md`.**
> Die HPC-Skripte liegen in `werkzeuge/hpc/`.
>
> **Welcher Code auf dem HPC liegt, entscheidet über die Punkt-IDs.** Der Index
> vom 14.08.2026 (`chunking_config_hash 5d4df3ab6a51ec37`) entstand mit
> `werkzeuge/hpc/stand_2026-08-14/`. `src/raghsh/index/chunking.py` ist seit dem
> 22.09.2026 ein anderer Stand (`e73a521098f93fc3`, Tabellen und Kontextkopf).
> Ein Delta gegen den heutigen Index geht nur mit dem alten Stand. Den neuen
> übernimmt man nur über einen Volldurchlauf (`docs/hpc.md`, Abschnitt 3).

---

## Delta-Durchlauf

### 1. Crawlen

```bash
python -m raghsh.crawl.spider                      # vollständiger Crawl
python -m raghsh.crawl.resume_crawler            # unterbrochenen Crawl fortsetzen
python werkzeuge/crawl/recrawl_urls.py urls.txt     # gezielt einzelne URLs
```

Jeder dieser Läufe schreibt am Ende ein Archiv mit **genau den Seiten, deren
Inhalt sich geändert hat**:

```
var/daten/delta_2026-08-05.tar.xz
```

Unveränderte Seiten kommen nicht hinein — sie werden nur umdatiert. Die
Dateien liegen flach im Archiv, ohne Verzeichnisanteil. Markdown komprimiert
sich auf etwa ein Viertel.

Gab es keine Änderungen, wird kein Archiv angelegt.

### 2. Delta-Liste erzeugen

```bash
python -m raghsh.crawl.delta --stats             # Überblick
python -m raghsh.crawl.delta -o delta.txt        # URL-Liste schreiben
```

`delta.py` vergleicht den `content_hash` jeder Markdown-Datei mit dem
`vectorized_hash` in `doc_index.db` — dem Textstand, der zuletzt vektorisiert
wurde. Laufen beide auseinander oder fehlt der zweite, ist das Dokument offen.

Der Vergleich läuft gegen den Hash **in der Datei**, nicht gegen den in der
Datenbank. Das Ergebnis stimmt deshalb auch dann, wenn `build_doc_index.py`
seit dem Crawl noch nicht gelaufen ist.

### 3. Zum HPC übertragen

Auf den Cluster kommen der **vollständige** aktuelle Korpus (nach
`data/ingested/`) und `delta.txt`. Ein Verzeichnis nur mit den Delta-Seiten
reicht nicht. Die Deduplizierung wählt ihre Gruppenvertreter über alle
Dateien und schränkt erst danach auf `--urls` ein. Über eine Teilmenge
gerechnet fiele die Wahl anders aus, und Fragen und Vektoren hingen an anderen
Punkten als im Bestand. Befehle: `docs/hpc.md`, Abschnitt 4. Der Schlüssel
liegt nur auf dem Laptop, die Übertragung läuft daher über ihn.

### 4. Anreichern und vektorisieren (HPC)

Vier Schritte, im Einzelnen in `docs/hpc.md`, Abschnitt 6:

1. `enrich_vllm.sbatch` mit `URLS=delta.txt` erzeugt Fragen nur für die
   Delta-Seiten.
2. `fragen_zusammenfuehren.py` setzt sie in den ungefilterten Bestand
   (`enrichment_v2.parquet`) ein.
3. `filter_fragen.py` filtert den ganzen Bestand neu.
4. `vectorize.sbatch` mit `URLS=delta.txt` und dem `avg_len` des Bestands
   (`AVG_LEN=88.3`, `AVG_LEN_EXPANDED=107.1`).

`--output` ist beim Delta-Betrieb Pflicht. Ohne das Flag überschriebe der Lauf
die vollständige Datei mit einem Bruchstück. Beide Skripte brechen in dem Fall ab.

### 5. Zurückholen und importieren

```bash
# Rückweg über den Laptop, siehe docs/hpc.md Abschnitt 4
python -m raghsh.index.local_importer --input var/vektoren/hsh_vectors_delta.parquet
```

Der Import ist überschreibungssicher: Punkt-IDs sind UUID5 über
`source_url + chunk_index`, und jedes Dokument wird beim ersten Auftreten per
Filter vollständig gelöscht, bevor die neuen Chunks geschrieben werden. Wird
eine Seite kürzer, bleiben keine überzähligen Chunks zurück.

### 6. Index nachziehen

```bash
python -m raghsh.index.build_doc_index --urls delta.txt
```

Das aktualisiert `chunk_ids`, `content_hash` und die Zustandsfelder
(`vectorized_hash`, `vectorized_at`, `imported_at`) — aber nur für die
gelisteten URLs. Ohne `--urls` würde der Lauf sämtliche Dokumente als
vektorisiert markieren, auch solche, die nie beim HPC waren, und die
Delta-Erkennung damit dauerhaft blind machen.

**Kein Neustart des MCP-Servers nötig.** Er hält `doc_index.db` zwar seit dem
Start geöffnet, aber read-only — und eine solche SQLite-Verbindung sieht
Commits eines fremden Prozesses sofort. Ein Neustart würde nur die
Embedding-Modelle aus dem Speicher werfen und die erste Anfrage danach
verlangsamen.

---

## Einmalige Initialisierung

Die Zustandsverfolgung muss einmal aufgesetzt werden, sonst hält `delta.py`
jedes Dokument für unbearbeitet:

```bash
python -m raghsh.index.build_doc_index --rebuild
```

Das legt die Spalte `vectorized_hash` an und markiert alle Dokumente, die
bereits Chunks in Qdrant haben, als vektorisiert. Danach bleiben nur noch die
tatsächlich offenen Dokumente in der Delta-Liste.

`--rebuild` ist hier wichtig: Der inkrementelle Lauf überspringt Dateien mit
unveränderter mtime und würde die meisten Dokumente ungemarkt lassen.

---

## Volldurchlauf

```bash
python -m raghsh.crawl.spider
# auf dem HPC (docs/hpc.md, Abschnitt 5):
#   enrich_vllm.sbatch → enricher.py --merge → filter_fragen.py → vectorize.sbatch
# Parquet nach var/vektoren/ holen, dann:
python werkzeuge/delete_qdrant.py                        # nur bei Schema-/Modellwechsel
python -m raghsh.index.local_importer --input var/vektoren/<datei>.parquet
python -m raghsh.index.build_doc_index --rebuild
```

`delete_qdrant.py` löscht die Collection und legt sie **nicht** neu an — das
übernimmt `local_importer.ensure_collection()`. Das ist Absicht: Existiert die
Collection bereits, läuft der Importer im Upsert-Modus und feuert für jedes
Dokument einen überflüssigen delete-by-filter gegen eine leere Collection.

### Wann ein Volldurchlauf nötig ist

Die Punkt-IDs sind UUID5 über `source_url + chunk_index`. Alles, was die
Chunk-Grenzen verschiebt, verschiebt auch die IDs — ein Delta-Import erzeugt
dann Karteileichen statt Überschreibungen:

- `chunking_config_hash` geändert, also `CHUNK_SIZE`, `CHUNK_OVERLAP`,
  `HEADERS_TO_SPLIT`, Tabellenregel oder Bereinigungsmuster in `chunking.py`.
  Das gilt auch für den Wechsel vom Stand 14.08.2026 auf `src/raghsh/index/`.
- Prompt oder Modell der Anreicherung gewechselt. Sonst stünden alte und neue
  Fragen nebeneinander im Index.
- Embedding-Modell gewechselt (auch die Dimension in `delete_qdrant.py` und
  `local_importer.py` anpassen)

In beiden Fällen: `delete_qdrant.py`, dann alles neu.

---

## Skripte im Überblick

| Aufruf | Aufgabe | Wichtige Flags |
|---|---|---|
| `python -m raghsh.crawl.spider` | Vollständiger Crawl ab Startseite | — |
| `python -m raghsh.crawl.resume_crawler` | Unterbrochenen Crawl fortsetzen | `--dry-run` |
| `python werkzeuge/crawl/recrawl_urls.py` | Gezielt URLs aus einer Liste crawlen | `--dry-run` |
| `python -m raghsh.crawl.delta` | Offene Dokumente ermitteln | `--stats`, `--count`, `-o` |
| `enricher.py` (auf dem HPC) | doc2query-Fragen je Chunk | `--urls`, `--output`, `--shard`, `--merge` |
| `fragen_zusammenfuehren.py` (HPC) | Delta-Fragen in den Bestand einfügen | `--bestand`, `--delta`, `--urls` |
| `filter_fragen.py` (HPC) | zu allgemeine Fragen verwerfen | `--max-wiederholung`, `--schreiben` |
| `hpc_vectorizer.py` (auf dem HPC) | Chunking + Dense + BM25 + Fragen-Arme → Parquet | `--urls`, `--output`, `--enrichment`, `--avg-len` |
| `python -m raghsh.index.local_importer` | Parquet → Qdrant | `--input` |
| `python -m raghsh.index.build_doc_index` | Metadaten + Chunk-IDs + Zustand | `--urls`, `--rebuild` |
| `python werkzeuge/delete_qdrant.py` | Collection löschen | — |

---

## Zustandsfelder in `doc_index.db`

| Spalte | Bedeutung |
|---|---|
| `content_hash` | Textstand der Datei beim letzten Indexlauf |
| `vectorized_hash` | Textstand, der zuletzt vektorisiert und importiert wurde |
| `vectorized_at` | Zeitpunkt der Vektorisierung |
| `imported_at` | Zeitpunkt des Qdrant-Imports |
| `enriched_at` | reserviert für die geplante LLM-Anreicherung |

`content_hash` schreibt `build_doc_index.py` bei jedem Lauf fort.
`vectorized_hash` dagegen ausschließlich `mark_vectorized()`, und auch nur für
Dokumente, die tatsächlich Chunks in Qdrant haben. Die Differenz der beiden
ist das Delta.

Bricht ein Transfer oder Import ab, bleibt `vectorized_hash` auf dem alten
Stand — das Dokument taucht beim nächsten Lauf wieder auf. Eine separate
Zustandsdatei gibt es bewusst nicht.

---

## Noch offen

**Verwaiste Chunks.** Verschwindet eine Seite von der Website, löscht der
Crawler die Markdown-Datei — ihre Chunks bleiben aber in Qdrant, weil
`local_importer.py` nur URLs aufräumt, die im Parquet vorkommen. Im
Delta-Betrieb sammelt sich das langsam an. `build_doc_index.py` berechnet die
betroffene Menge bereits für seine Statistik („Nur in Qdrant"); ein
`--purge-orphans`-Flag wäre der nächste Schritt.

**Automatisierung.** Schritt 3 bis 5 sind noch Handarbeit, und nur der
Laptop hat einen Schlüssel für KISSKI. Ein Skript, das
`rsync`, `sbatch` und den Rücktransfer übernimmt, ist in `docs/archiv/2026-05_verbesserungen.md`
unter Punkt 8 beschrieben.
