# `hsh_vectors_*.parquet` — Aufbau und Verarbeitung im RAG-System

**Stand:** 14. August 2026
**Erzeuger:** `hpc_vectorizer.py` auf KISSKI
**Gilt für:** `chunking_config_hash = 5d4df3ab6a51ec37`,
`selection_hash = 40fbfb36ee010cf8` — **121.134 Punkte**

| | |
|---|---|
| Datei | `hsh_vectors_wdh10.parquet` (915,7 MB) |
| Sidecar | `hsh_vectors_wdh10_sources.txt` (5.306 URLs) |
| erzeugt | Slurm-Job 15277868, 14.08.2026, 20 Minuten auf einer A100 |

Dieses Dokument beschreibt alles, was die Empfängerseite braucht: Herkunft der
Datei, Schema, Bedeutung jeder Spalte, Anlegen der Qdrant-Collection, Import,
Query-Seite und Selbsttest. Es ist so geschrieben, dass sich das RAG-System
allein damit aufbauen lässt.

**Der Variantenname im Dateinamen ist Absicht.** `wdh10` bezeichnet die
Einstellung des Fragenfilters (`--max-wiederholung 10`). Ob diese Einstellung
die beste ist, ist eine offene, gegen ein Goldset zu messende Frage — siehe
`Messung_Fragenfilter_2026-08-14.md`. Weitere Läufe können folgen; sie
unterscheiden sich **ausschließlich** in `questions`, `sparse_expanded_*` und
`vector_questions`. Alles andere — IDs, Text, Payload, `vector`, `sparse_*` —
entsteht aus identischen Eingaben und identischen Parametern; ein Vergleich
zweier Varianten misst deshalb wirklich den Filter und nichts sonst. Zwei
Varianten gehören in **zwei Collections**, nicht in eine: Sie teilen sich die
Punkt-IDs und würden einander beim Upsert überschreiben.

---

## 0. Woher die Datei kommt

Sechs Schritte auf dem HPC, jeder mit einem Artefakt. Wer nur importiert,
braucht das nicht — wer einen Wert erklären oder einen Lauf wiederholen muss,
schon. Die Skripte liegen in `werkzeuge/hpc/` (Chunking und Vektorisierer dieses
Laufs in `werkzeuge/hpc/stand_2026-08-14/`), die Anleitung in `docs/hpc.md`.
Die Berichte `Messung_*.md` und `Uebergabe_*.md` liegen in `docs/archiv/2026-08_hpc/`.

```
var/korpus/*.md              5.616 Markdown-Dateien aus dem Crawl
   │                            (5.604 mit lesbarem Frontmatter)
   │  chunking.py
   ▼
196.934 Chunks                  Überschriften-Split, dann Größenbegrenzung
   │                            (1000 Zeichen, 200 Überlappung)
   │  Deduplizierung: je Gruppe wortgleicher Chunks EIN Vertreter,
   │  zusätzlich einer je vorkommender Fakultät und Organisation
   ▼
121.134 Punkte                  = die Zeilen dieser Parquet-Datei
   │
   │  enricher.py — doc2query mit Qwen/Qwen3.5-9B über vLLM, 1,7 GPU-Stunden
   ▼
enrichment.parquet              110.333 Zeilen (Chunks ab 200 Zeichen)
   │
   │  erbe_fragen.py — Fragen an die 1.343 zuordnungsbedingten Zusatzpunkte
   │  vererben; wortgleicher Text, also dieselben Fragen. Keine GPU.
   ▼
enrichment_v2.parquet           111.456 Zeilen, 329.506 Fragen
   │
   │  filter_fragen.py --max-wiederholung 10
   ▼
enrichment_v2_wdh10.parquet     303.716 Fragen (92,2 %)
   │
   │  hpc_vectorizer.py --require-gpu --enrichment …
   ▼
hsh_vectors_wdh10.parquet       121.134 Punkte, vier Arme je Punkt
```

Die Deduplizierung ist der Grund für die Differenz 196.934 → 121.134: 39 % der
Chunks sind wortgleiche Wiederholungen, vor allem gespiegelte
Prüfungsordnungen und Modulhandbücher. Was das für Filter und Quellenangabe
bedeutet, steht in Abschnitt 9.

---

## 1. Was ankommt

| Datei | Inhalt |
|---|---|
| `hsh_vectors_wdh10.parquet` | ein Datensatz je Chunk, alle Vektoren und Payload-Felder — 121.134 Zeilen |
| `hsh_vectors_wdh10_sources.txt` | eine `source_url` je Zeile — **alle 5.604 verarbeiteten Seiten**, auch die 298 ohne eigenen Punkt |
| `chunk_vertretung.parquet` | 196.934 Zeilen: welcher Punkt vertritt welchen Chunk. Nur nötig, wenn vollständige Dokumente rekonstruiert werden sollen — siehe Abschnitt 4.1 |

Die Sidecar-Datei ist kein Beiwerk. Sie wird gebraucht, um verwaiste Punkte zu
löschen (Abschnitt 7.2).

Format: Parquet, zstd-komprimiert, in Row-Groups zu `FLUSH_CHUNKS` Zeilen.
Mit `pyarrow` lesbar, zeilenweise oder gruppenweise streambar.

---

## 2. Schema

20 Spalten. Typen sind Arrow-Typen; `list<float>` ist `float32`.

| Spalte | Typ | Bedeutung |
|---|---|---|
| `id` | `string` | **Qdrant-Punkt-ID.** UUID5, deterministisch aus `source_url#chunk_index` |
| `vector` | `list<float32>` | Dense über den Chunktext, 1024 Dim., L2-normalisiert |
| `source_url` | `string` | Quellseite |
| `title` | `string` | Seitentitel; bei PDFs oft der Dateiname |
| `crawl_date` | `string` | `YYYY-MM-DD` |
| `content_type` | `string` | `html` oder `pdf` |
| `faculty` | `string` | `Fakultät I` … `Fakultät V` oder **leer** |
| `organisation` | `string` | `Bibliothek`, `Karriere`, `Projektwebseite`, `International`, … oder **leer** |
| `section_heading` | `string` | Überschrift des Abschnitts, in dem der Chunk steht |
| `text` | `string` | Volltext des Chunks — **der Text, der ins LLM geht** |
| `chunk_index` | `int32` | Position innerhalb des Dokuments, 0-basiert |
| `total_chunks` | `int32` | Anzahl Chunks des Dokuments |
| `sparse_indices` | `list<int32>` | BM25-Token-IDs über den reinen Text |
| `sparse_values` | `list<float32>` | zugehörige Gewichte, **ohne IDF** |
| `sparse_expanded_indices` | `list<int32>` | BM25 über Text **+ Fragen** |
| `sparse_expanded_values` | `list<float32>` | zugehörige Gewichte, **ohne IDF** |
| `questions` | `list<string>` | generierte Fragen; leere Liste möglich |
| `vector_questions` | `list<float32>` | Dense über die Fragen, 1024 Dim.; **leere Liste möglich** |
| `chunking_config_hash` | `string` | Fingerabdruck der Chunk-**Grenzen** |
| `selection_hash` | `string` | Fingerabdruck der Dedup-**Auswahlregel** (neu am 14.08.2026) |

> `faculty` und `organisation` sind getrennt. Früher landeten `Bibliothek`,
> `Karriere` und `International` im Feld `faculty` — das sind keine Fakultäten
> und machten jeden Fakultätsfilter unsauber. **Beide Felder in die Payload
> übernehmen.**
>
> Beide Felder sind **einwertig** und bleiben es. Ein Punkt trägt genau eine
> Fakultät und eine Organisation; ein Filter `MatchValue` genügt, `MatchAny`
> über Arrays wird nicht gebraucht. Damit das trotz Deduplizierung stimmt,
> behält `select_unique_ids()` je Gruppe wortgleicher Chunks einen Vertreter
> **pro vorkommender Fakultät und Organisation** — sonst wäre ein Text, der in
> Dokumenten zweier Fakultäten steht, nur unter einer von beiden filterbar.
> Kosten: 1.343 Punkte (+1,12 %).

---

## 3. Die vier Retrieval-Arme

Jeder Chunk trägt vier unabhängige Repräsentationen. Sie stehen bewusst
nebeneinander, damit die Query entscheidet, welche mitspielen — ein
A/B-Vergleich braucht keinen zweiten Vektorisierungslauf.

| Arm | Spalte | Modell | Besonderheit |
|---|---|---|---|
| Dense Text | `vector` | jina-embeddings-v3, Task `retrieval.passage` | 1024 Dim., L2-normalisiert |
| Dense Fragen | `vector_questions` | jina-embeddings-v3, Task **`retrieval.query`** | kann leer sein |
| Sparse Text | `sparse_*` | `Qdrant/bm25`, `language="german"` | Gewichte ohne IDF |
| Sparse erweitert | `sparse_expanded_*` | `Qdrant/bm25`, `language="german"` | Gewichte ohne IDF |

### 3.1 Warum der Fragenvektor mit dem QUERY-Adapter erzeugt wurde

jina-v3 nutzt aufgabenspezifische LoRA-Adapter. Der Chunktext ist ein Passage,
die generierten Fragen sind Fragen — sie sollen im selben Raum liegen wie die
Nutzeranfrage. Deshalb `retrieval.query` für `vector_questions` und
`retrieval.passage` für `vector`.

Folge für die Suche: Die Nutzeranfrage (ebenfalls `retrieval.query`) wird gegen
**beide** Dense-Arme geprüft, aber `vector_questions` ist der Arm, in dem
Frageformulierung auf Frageformulierung trifft.

### 3.2 Wann welche Spalte leer ist

Hier sind **zwei Fälle sauber zu trennen** — sie sehen ähnlich aus und
bedeuten Verschiedenes.

**Fall A — der ganze Lauf ohne `--enrichment`** (siehe Abschnitt 3.4). Dann
sind `sparse_expanded_*`, `questions` und `vector_questions` in **jeder Zeile**
leer. Die Datei hat trotzdem alle 20 Spalten und ist regulär lesbar.

**Fall B — Lauf mit `--enrichment`, aber dieser eine Chunk hat keine Fragen.**
Dann sind nur `questions` und `vector_questions` leer. **`sparse_expanded_*`
ist belegt** — mit denselben Termen wie `sparse`, aber **anderen Gewichten**,
weil die beiden Arme mit unterschiedlichem `avg_len` normalisiert wurden:

```
sparse_values          [1.5063, 1.5063, …]   avg_len des Textarms
sparse_expanded_values [1.5386, 1.5386, …]   avg_len des erweiterten Arms
```

**Konsequenz für den Importer:** Ob ein Chunk angereichert ist, entscheidet
allein `questions`. `sparse_expanded` taugt nicht als Signal — es ist im Fall B
belegt, obwohl keine Fragen vorliegen, und im Fall A leer, obwohl der Lauf
technisch fehlerfrei war.

**Die Gewichte enthalten kein IDF.** `Qdrant/bm25` liefert nur die
Dokumentkomponente; das IDF entsteht erst zur Query-Zeit aus den
Collection-Statistiken — aber **nur, wenn der Sparse-Index mit
`modifier=Modifier.IDF` angelegt wurde.** Fehlt das, ist das Ranking still
falsch: Es rankt dann nach reiner Termfrequenz, häufige Allerweltswörter
gewinnen.

### 3.4 Betrieb ohne Anreicherung

Die Anreicherung ist optional. Drei Abstufungen:

| Aufruf | `vector` | `sparse` | `sparse_expanded` | `questions` | `vector_questions` |
|---|---|---|---|---|---|
| ohne `--enrichment` | ✓ | ✓ | leer | leer | leer |
| `--enrichment` + `--no-expanded-sparse` + `--no-question-vector` | ✓ | ✓ | leer | ✓ | leer |
| `--enrichment` + `--no-expanded-sparse` | ✓ | ✓ | leer | ✓ | ✓ |
| `--enrichment` + `--no-question-vector` | ✓ | ✓ | ✓ | ✓ | leer |
| `--enrichment` (voll) | ✓ | ✓ | ✓ | ✓ | ✓ |

`vectorize.sbatch` wählt automatisch: Liegt die als `ENRICHMENT` eingestellte
Datei vor (Vorgabe `enrichment_v2_wdh10.parquet`), wird sie durchgereicht;
fehlt sie, läuft der Job ohne Fragen weiter statt daran zu sterben. Beide
Namen sind von außen setzbar:

```bash
sbatch --export=ALL,ENRICHMENT=enrichment_v2.parquet,\
AUSGABE=hsh_vectors_ungefiltert.parquet vectorize.sbatch
```

**Auf der Qdrant-Seite** dann nur die belegten Arme anlegen — also `text`
(dense) und `sparse`. Eine Collection mit vier Vektoren funktioniert auch, die
beiden ungenutzten bleiben schlicht unbelegt; ein Prefetch darauf liefert nur
keine Treffer.

**Unabhängig davon bleibt die Deduplizierung aktiv.** Sie sitzt in
`chunking.py`, nicht in der Anreicherung. Die Punktzahl ist in allen Varianten
dieselbe.

**Die Gewichte enthalten kein IDF.** `Qdrant/bm25` liefert nur die
Dokumentkomponente; das IDF entsteht erst zur Query-Zeit aus den
Collection-Statistiken — aber **nur, wenn der Sparse-Index mit
`modifier=Modifier.IDF` angelegt wurde.** Fehlt das, ist das Ranking still
falsch: Es rankt dann nach reiner Termfrequenz, häufige Allerweltswörter
gewinnen.

### 3.3 Zahlen zur Größenordnung

Aus dem Produktionslauf (`hsh_vectors_wdh10.parquet`, alle 121.134 Punkte
nachgezählt):

```
Punkte gesamt                    121.134   (= eindeutige IDs, keine Kollision)
davon mit Fragen                 109.955   (90,8 %)
   dabei vector_questions belegt 109.955   (ausnahmslos)
   dabei sparse_expanded belegt  109.955   (ausnahmslos)
davon ohne Fragen                 11.179
   dabei vector_questions leer    11.179   (ausnahmslos)
Row-Groups                            29
```

Es gibt also **keinen halb gefüllten Punkt**: Entweder ein Punkt hat Fragen,
dann sind beide fragenabhängigen Arme da, oder er hat keine, dann ist der
Fragenvektor leer. Der Importer darf sich darauf verlassen — muss den leeren
Fall aber trotzdem behandeln (Abschnitt 7.1).

Termzahlen je Punkt, aus einem Testlauf über 524 Punkte:

```
vector                1024 Dim., |v| = 1.0000  (durchweg L2-normalisiert)
sparse                Terme je Punkt: min 6, Median 52, max 99
sparse_expanded       Terme je Punkt: min 6, Median 57, max 104
```

Weil die Dense-Vektoren normalisiert sind, liefern **Cosine und Dot identische
Rangfolgen**. Dot ist minimal billiger; Cosine ist die sichere Wahl, falls je
unnormalisierte Vektoren dazukommen.

### 3.5 `avg_len` — womit der Index gebaut wurde

Die BM25-Gewichte hängen von der mittleren Dokumentlänge ab. Gemessen an einer
Stichprobe von 400 Dateien, protokolliert im Job-Log:

```
avg_len Textarm            88,3 Tokens
avg_len erweiterter Arm   107,1 Tokens     (fastembed-Vorgabe wäre 256,0)
```

**Diese Werte sind Index-Seite, nicht Query-Seite.** Sie stecken bereits in den
gespeicherten `sparse_values` und `sparse_expanded_values`. Auf der Query-Seite
haben sie keine Wirkung — Abschnitt 8.2 zeigt das gemessen. Hier stehen sie,
damit ein späterer Delta-Lauf dieselben Werte setzen kann (`--avg-len 88.3
--avg-len-expanded 107.1`): Nachgelieferte Punkte, die mit anderer
Normalisierung berechnet wurden, konkurrieren sonst unfair mit den
vorhandenen.

Der zweite Arm hat bewusst einen eigenen Wert. Beiden denselben zu geben wäre
der Fehler, den die Messung gerade vermeidet: Die expandierten Dokumente sind
systematisch länger und bekämen sonst einen unverdienten Vorteil, den ein
A/B-Vergleich als doc2query-Effekt fehldeutet.

---

## 4. Die Punkt-ID

```python
id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_url}#{chunk_index}"))
```

Deterministisch. Derselbe Chunk derselben Seite bekommt bei jedem Lauf
dieselbe ID. Qdrant akzeptiert UUID-Strings direkt als Punkt-ID.

Das hat zwei Konsequenzen:

* **Ein erneuter Import überschreibt** statt zu duplizieren.
* Die ID ist zugleich der Join-Schlüssel zur Anreicherung. Sie ist bereits
  aufgelöst — `questions` und `vector_questions` stehen in derselben Zeile.

### 4.1 `chunk_index` hat Lücken, `total_chunks` zählt mehr — mit Absicht

**Das ist der Punkt, an dem eine Empfängerseite am ehesten stolpert.** Beide
Felder beziehen sich auf das Dokument **vor** der Deduplizierung:

```
chunk_index    Position im ungekürzten Dokument. Lückenhaft, sobald einzelne
               Chunks wortgleich anderswo stehen.
total_chunks   Chunkzahl des ungekürzten Dokuments. IMMER >= der Zahl der
               Punkte, die dieses Dokument in der Datei hat.
```

Gemessen an der gelieferten Datei:

```
Dokumente                                  5.306
davon mit Lücken (total_chunks != Punkte)  1.559   (29,4 %)
fehlende Positionen insgesamt             65.763
Dokumente mit unter 50 % ihrer Chunks        457
Dokumente mit unter 20 % ihrer Chunks        180
Extremfall                                 3 von 242 Chunks
```

**Der Text ist nicht verloren** — er steht wortgleich unter einer anderen
`source_url` im Index. Verloren ist nur die Möglichkeit, ein Dokument aus
*einer* URL zusammenzusetzen. Drei Konsequenzen, alle unbedingt beachten:

1. **Kein Dokument aus `source_url` rekonstruieren und dem LLM als
   vollständigen Beleg vorlegen.** Bei 3 von 242 Absätzen antwortet ein Modell
   sonst „laut Prüfungsordnung …" auf Basis von einem Prozent des Textes und
   merkt nichts davon. Das ist der gefährlichste Fehler, weil das Ergebnis wie
   ein vollständiger Beleg aussieht.
2. **`chunk_index < total_chunks - 1` ist keine gültige Grenzprüfung.** Für
   Nachbarschaftszugriffe die tatsächlich vorhandenen Indizes des Dokuments
   abfragen, statt zu rechnen.
3. **Nachbar-Chunks können fehlen.** `chunk_index ± 1` liefert dann nichts.
   Das ist normal und kein Fehler.

**Vollständigkeit lässt sich beim Import ausrechnen**, ganz ohne Zusatzdatei —
je `source_url` die Punkte zählen und gegen `total_chunks` halten:

```python
from collections import Counter
vorhanden = Counter()          # source_url -> Punkte in der Datei
total     = {}                 # source_url -> total_chunks
# … beim Durchlaufen der Row-Groups füllen …

payload["chunks_vorhanden"]  = vorhanden[url]
payload["chunks_dokument"]   = total[url]
payload["text_vollstaendig"] = vorhanden[url] == total[url]
```

Damit kann die Auswahlrunde Fragmente abwerten oder kennzeichnen, statt sie für
ganze Dokumente zu halten.

**Vollständige Dokumente rekonstruieren** — dafür gibt es
`chunk_vertretung.parquet`:

| Spalte | Bedeutung |
|---|---|
| `chunk_id` | Punkt-ID dieses Chunks (auch wenn er nicht im Index ist) |
| `source_url`, `chunk_index` | seine Stelle im Dokument |
| `vertreten_durch` | Punkt-ID, unter der sein Text im Index steht |
| `selbst_im_index` | `True`, wenn er sein eigener Vertreter ist |

Geprüft: 196.934 Zeilen, **0 Chunks ohne auffindbaren Vertreter**, danach
**0 Dokumente mit Lücken**. Jeder Text ist erreichbar, nur eben teils unter
fremder Adresse. Erzeugt von `Backup/analyse_skripte/pruefe_reparaturen.py`
auf dem Login-Node in zwei Minuten — **keine GPU, keine Neuvektorisierung.**

```python
# alle Chunks eines Dokuments in Reihenfolge, Löcher aufgefüllt
zeilen = vertretung[vertretung.source_url == url].sort_values("chunk_index")
punkte = client.retrieve("hsh", ids=list(zeilen.vertreten_durch))
volltext = "\n\n".join(p.payload["text"] for p in punkte)
```

Wer so zusammensetzt, sollte die fremden Fundstellen in der Quellenangabe
nennen — der Absatz stammt dann aus einem gleichlautenden Dokument einer
anderen Fakultät.

---

## 5. `chunking_config_hash` und `selection_hash` — die Sicherung

Zwei Fingerabdrücke mit verschiedenem Gewicht. Bis zum 13.08.2026 war es
einer, der beides vermischte.

`chunking_config_hash` — Chunkgröße, Überlappung, Verschmelzungsgrenzen,
Überschriftenebenen, Bereinigungsmuster. Alles, was **Chunk-Grenzen**
bestimmt. Ändert es sich, zeigen dieselben IDs auf andere Textstellen; der
Vektorisierer bricht beim Join ab.

`selection_hash` — welche Chunks die Deduplizierung **behält**. Ändert es
sich, zeigen die IDs weiterhin auf denselben Text, es decken sich nur die
Punktmengen nicht: für hinzugekommene Chunks fehlen dann Fragen. Der
Vektorisierer warnt und rechnet weiter, die Trefferquote des Joins im Log
zeigt das Ausmaß.

Beide Werte sind einheitlich über die ganze Datei und gehören in die Payload,
damit sich später feststellen lässt, mit welchem Stand ein Punkt erzeugt wurde.

Aktuell: **`5d4df3ab6a51ec37`** (Grenzen) und **`40fbfb36ee010cf8`** (Auswahl)

Mischt sich in einer Collection mehr als ein Hash, stammen die Punkte aus
verschiedenen Chunkungen und sind nicht vergleichbar. Das ist ein Grund, die
Collection neu aufzubauen, nicht zu ergänzen.

---

## 6. Collection anlegen

Vier benannte Vektoren: zwei dense, zwei sparse.

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.recreate_collection(
    collection_name="hsh",
    vectors_config={
        "text":      models.VectorParams(size=1024, distance=models.Distance.COSINE),
        "questions": models.VectorParams(size=1024, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        # modifier=IDF ist PFLICHT — die gespeicherten Gewichte enthalten kein IDF
        "sparse":          models.SparseVectorParams(modifier=models.Modifier.IDF),
        "sparse_expanded": models.SparseVectorParams(modifier=models.Modifier.IDF),
    },
)
```

> **Benannte Vektoren sind zwingend.** Qdrant erlaubt entweder *einen*
> unbenannten oder *mehrere benannte* Dense-Vektoren. Eine bestehende
> Collection mit unbenanntem Vektor lässt sich nicht um `questions` erweitern —
> sie muss neu angelegt werden.

Nützliche Payload-Indizes für Filter:

```python
for feld in ("source_url", "faculty", "organisation", "content_type"):
    client.create_payload_index("hsh", feld, models.PayloadSchemaType.KEYWORD)
```

---

## 7. Import

### 7.1 Punkte bauen

```python
import pyarrow.parquet as pq
from qdrant_client import models

BATCH = 512
pf = pq.ParquetFile("hsh_vectors_wdh10.parquet")

for rg in range(pf.num_row_groups):
    zeilen = pf.read_row_group(rg).to_pylist()
    punkte = []
    for r in zeilen:
        vektoren = {
            "text": r["vector"],
            "sparse": models.SparseVector(
                indices=r["sparse_indices"], values=r["sparse_values"]),
            "sparse_expanded": models.SparseVector(
                indices=r["sparse_expanded_indices"],
                values=r["sparse_expanded_values"]),
        }
        # Leere Liste bedeutet: kein Fragenvektor. Nicht mit Nullen füllen —
        # ein Nullvektor hat keine definierte Cosine-Ähnlichkeit und würde
        # den Punkt entweder nie oder immer treffen.
        if r["vector_questions"]:
            vektoren["questions"] = r["vector_questions"]

        punkte.append(models.PointStruct(
            id=r["id"],
            vector=vektoren,
            payload={
                "source_url":      r["source_url"],
                "title":           r["title"],
                "crawl_date":      r["crawl_date"],
                "content_type":    r["content_type"],
                "faculty":         r["faculty"],
                "organisation":    r["organisation"],
                "section_heading": r["section_heading"],
                "text":            r["text"],
                "chunk_index":     r["chunk_index"],
                "total_chunks":    r["total_chunks"],
                "questions":       r["questions"],
                "chunking_config_hash": r["chunking_config_hash"],
                "selection_hash":  r["selection_hash"],
            },
        ))
    for i in range(0, len(punkte), BATCH):
        client.upsert("hsh", points=punkte[i:i + BATCH], wait=False)
```

### 7.2 Verwaiste Punkte löschen — vor dem Upsert

`chunk_index` hängt von der Chunkung des **gesamten** Dokuments ab. Liefert
eine neu gecrawlte Seite 7 statt vorher 12 Chunks, werden 0–6 überschrieben —
**7–11 bleiben mit veraltetem Text liegen** und tauchen weiterhin in Treffern
auf.

Dagegen die Sidecar-Datei:

```python
urls = [z.strip() for z in open("hsh_vectors_wdh10_sources.txt") if z.strip()]

for i in range(0, len(urls), 200):
    client.delete("hsh", points_selector=models.FilterSelector(
        filter=models.Filter(must=[models.FieldCondition(
            key="source_url",
            match=models.MatchAny(any=urls[i:i + 200]))])))
```

Bei einem vollständigen Neuaufbau ist `recreate_collection` einfacher und
sicherer. Der Löschweg ist für Delta-Läufe (`--urls`) gedacht.

> **Achtung bei Delta-Läufen mit dedupliziertem Korpus:** 298 Seiten enthalten
> keinen eigenen Chunk mehr, weil ihr Inhalt wortgleich anderswo steht. Sie
> stehen trotzdem in `sources.txt` — die Löschung greift, ein Upsert folgt
> nicht. Genau so muss es sein: Ihr Altbestand aus einem früheren Import
> gehört weg, sonst bleiben Punkte mit veraltetem Text liegen.
>
> **Bis zum 17.08.2026 stimmte das nicht.** `hpc_vectorizer.py` vermerkte eine
> Seite erst nach dem Dedup-Filter; die 298 fehlten in der Liste (5.306 statt
> 5.604 Zeilen). Behoben, und die ausgelieferte Datei wurde ohne neuen GPU-Lauf
> nachgezogen (`Backup/analyse_skripte/sidecar_reparieren.py`). Wer eine
> `sources.txt` mit 5.306 Zeilen hat, benutzt die alte.

---

## 8. Query-Seite

### 8.1 Dense — der Task-Adapter muss wechseln

```python
from fastembed import TextEmbedding

dense = TextEmbedding("jinaai/jina-embeddings-v3", cache_dir=CACHE)

# retrieval.query, NICHT retrieval.passage
q_vec = list(dense.embed([frage], task="retrieval.query"))[0]
```

Der Parametername für den Task hängt von der fastembed-Version ab
(`task`, `task_type` oder `task_id`). `hpc_vectorizer.py` ermittelt ihn
empirisch in `resolve_task_kwarg()` und **protokolliert ihn**. Den Wert aus dem
Log übernehmen.

### 8.2 Sparse — `language="german"`, aber **kein** `avg_len`

```python
from fastembed import SparseTextEmbedding

sparse = SparseTextEmbedding("Qdrant/bm25", cache_dir=CACHE, language="german")

# query_embed(), NICHT embed()
q_sparse = list(sparse.query_embed([frage]))[0]
```

Zwei Dinge, gemessen und wichtig:

**`language="german"` ist zwingend.** Stemmer und Stoppwortliste bestimmen die
Token-IDs. Dieselbe Frage liefert:

```
language="german"  → 2 Token-IDs   [282458459, 628880358]
language="english" → 4 Token-IDs   [1109731834, 282458459, 628880358, 99212807]
```

Mit `english` bleiben deutsche Stoppwörter stehen und die Stämme weichen ab —
die IDs passen dann nicht zu denen im Index, und die Treffer bleiben aus oder
sind zufällig.

**`avg_len` spielt auf der Query-Seite keine Rolle.** Es normalisiert die
Dokumentlänge und steckt bereits in den gespeicherten `sparse_values` — die
Werte aus Abschnitt 3.5 gehören dorthin, nicht hierher. Gemessen:

```
query_embed()  → Gewichte 1.0     bei avg_len 90 UND bei avg_len 400
embed()        → Gewichte 1.6667  bzw. 1.6865
```

> **Korrektur gegenüber dem Sitzungsbericht vom 12.08.2026.** Dort steht,
> die Query-Seite brauche „`language='german'` und die gemessenen
> `avg_len`-Werte". Der erste Teil stimmt, der zweite nicht. `avg_len` auf der
> Query-Seite zu setzen ist harmlos, aber wirkungslos.
>
> **Wirklich schädlich wäre**, auf der Query-Seite `embed()` statt
> `query_embed()` zu benutzen: Dann bekommt die Anfrage Dokumentgewichte
> (1.6667 statt 1.0), die mit dem IDF multipliziert die Scores verzerren.

### 8.3 Fusion

Vier Arme, per RRF zusammengeführt. Welche mitspielen, entscheidet die Query —
genau dafür liegen sie nebeneinander.

```python
treffer = client.query_points(
    "hsh",
    prefetch=[
        models.Prefetch(query=q_vec, using="text",      limit=50),
        models.Prefetch(query=q_vec, using="questions", limit=50),
        models.Prefetch(
            query=models.SparseVector(indices=q_sparse.indices.tolist(),
                                      values=q_sparse.values.tolist()),
            using="sparse", limit=50),
        models.Prefetch(
            query=models.SparseVector(indices=q_sparse.indices.tolist(),
                                      values=q_sparse.values.tolist()),
            using="sparse_expanded", limit=50),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
    with_payload=True,
)
```

Für den A/B-Vergleich einzelne `Prefetch`-Einträge weglassen:

| Variante | aktive Arme |
|---|---|
| Baseline | `text`, `sparse` |
| nur doc2query sparse | `text`, `sparse_expanded` |
| nur doc2query dense | `text`, `questions`, `sparse` |
| alles | alle vier |

**Der Fragenvektor existiert nicht für jeden Punkt.** Punkte ohne
`questions`-Vektor werden vom `questions`-Prefetch schlicht nicht gefunden.
RRF fängt das ab: Ein Punkt, der nur über `text` und `sparse` kommt, bleibt im
Ergebnis — er bekommt nur keinen Beitrag aus dem vierten Arm.

---

## 9. Weiterverarbeitung im RAG

**In den Kontext des LLM gehört `text`**, nicht `questions`. Die Fragen sind
ein Retrieval-Signal, kein Inhalt — sie stehen so nicht auf der Webseite und
würden als Kontext Aussagen suggerieren, die nicht belegt sind.

**Als Quellenangabe** `source_url` plus `title`. `section_heading` hilft bei
langen Seiten, die Fundstelle zu benennen.

> **`title` ist bei PDFs der Dateiname — für eine LLM-Auswahlrunde untauglich.**
> Gemessen: 95.841 von 98.451 PDF-Chunks (97,3 %) und 4.709 von 22.683
> HTML-Chunks (20,8 %) tragen einen Titel ohne Aussage. Ein Modell soll dann
> zwischen `01_F_I_PO_EIT_2019`, `20181912_BPO_MAT_verk` und `F_I_PO_MEC_2021`
> wählen — das ist nicht entscheidbar.
>
> **Der URL-Pfad trägt die fehlende Information bereits.** Er steht im Payload,
> es braucht keinen neuen Crawl und keine Neuvektorisierung:
>
> ```
> 01_F_I_PO_EIT_2019   →  f1 › 10 Bachelorstdg › EIT › 01 F I PO EIT 2019
> 20193009_BPO_EWI_verk →  f1 › 10 Bachelorstdg › EWI › 20193009 BPO EWI verk
> ```
>
> Fakultät, Abschlussart und Studiengangskürzel werden damit sichtbar. Rezept:
> Pfadsegmente nehmen, technische Segmente verwerfen (`fileadmin`, `HsH`,
> `Forms`, `Dokumente`, `wp-content`, `uploads`, reine Zahlen), die letzten
> zwei Ordner plus Dateiname entkoppeln, `_`/`-` durch Leerzeichen ersetzen,
> Host-Label voranstellen. Beispielcode in
> `Backup/analyse_skripte/pruefe_reparaturen.py`.
>
> Ergänzend: Bei 2.016 der 2.880 PDF-Dokumente ohne brauchbaren Titel (70 %)
> liefert das `section_heading` des ersten Chunks eine echte Überschrift
> (`Baker_Hughes_Fragebogen` → `Baker Hughes > Steckbrief`). Bei den
> Prüfungsordnungen greift eher der Pfad, bei Flyern und Steckbriefen eher die
> Überschrift — beides kombinieren.
>
> **Nicht** stattdessen den Titel im Korpus reparieren: `title` geht über
> `informative_title()` in den Dedup-Schlüssel ein. Andere Titel heißen andere
> Gruppen, andere Punktmenge — und damit eine ungültige Anreicherung. Ein
> Anzeigetitel gehört auf die Empfängerseite, nicht ins Chunking.

**Nachbarschaft nachladen:** Mit `source_url` und `chunk_index` lassen sich
angrenzende Chunks holen, wenn ein Treffer mitten in einem Zusammenhang
liegt — `total_chunks` gibt die Obergrenze.

```python
nachbarn = client.scroll("hsh", scroll_filter=models.Filter(must=[
    models.FieldCondition(key="source_url", match=models.MatchValue(value=url)),
    models.FieldCondition(key="chunk_index",
                          range=models.Range(gte=idx - 1, lte=idx + 1)),
]), limit=3, with_payload=True)
```

**Filter** über `faculty`, `organisation` und `content_type`. Alle drei sind
einwertige Zeichenketten, `MatchValue` genügt:

```python
nur_f2 = models.Filter(must=[models.FieldCondition(
    key="faculty", match=models.MatchValue(value="Fakultät II"))])

treffer = client.query_points("hsh", prefetch=[…], query=…,
                              query_filter=nur_f2, limit=10)
```

> **Schreibweise:** Die Payload führt `Fakultät IV`, das Markdown-Frontmatter
> und `doc_index.db` führen `f4`. Umrechnung ist eine feste Tabelle —
> `f1 … f5` ↔ `Fakultät I … Fakultät V` —, die Reihenfolge ist die der
> römischen Ziffern. Sie steht als `FACULTY_MAP` in `chunking.py`. Wer beides
> braucht, legt beim Import ein zweites Feld `faculty_key` an; ein künftiger
> Vektorisierungslauf kann es direkt mitschreiben.

`faculty` und `organisation` können **leer** sein. Ein harter Filter auf
`faculty` schließt damit 39.600 Punkte aus — alle zentralen Seiten unter
`www.hs-hannover.de`, darunter Immatrikulation, Prüfungsamt und Semesterticket.
Das ist selten gewollt. Sinnvoller ist meist, die Fakultät als **Gewichtung**
zu verwenden (zwei Abfragen, die gefilterte bevorzugt) oder als
`should`-Bedingung, statt fremde Fakultäten hart auszuschließen.

Für „nur PDFs" bzw. „nur Webseiten" ist `content_type` zuverlässig und
vollständig belegt.

**Deduplizierte Seiten:** 298 Seiten des Korpus haben keinen eigenen Punkt, weil
ihr Inhalt anderswo wortgleich vorhanden ist. Eine Anfrage findet dann die
andere Seite. Inhaltlich gleichwertig, aber die Quellen-URL kann von der
erwarteten abweichen.

Was dabei **nicht** mehr passiert: dass ein Text unter einer Fakultät
verschwindet, die ihn im Bestand hat. Jede vorkommende (Text, Fakultät)- und
(Text, Organisation)-Kombination ist im Index vertreten — gemessen 84.778 von
84.778 Paaren, Prüfung in `Backup/analyse_skripte/konsistenz.py`, Abschnitt 5.
Offen bleibt die **Zitation**: Bei 29.258 Textgruppen steht derselbe Wortlaut
auf mehreren Seiten, und zitiert wird eine davon. Wer das schließen will,
schreibt beim Import Alias-Listen (`source_urls`) ins Payload; die Vektoren
bleiben davon unberührt.

---

## 10. Prüfliste vor dem ersten Produktivlauf

| | Prüfung |
|---|---|
| ☐ | Beide Sparse-Indizes mit `modifier=Modifier.IDF` angelegt |
| ☐ | Dense-Vektoren **benannt** (`text`, `questions`), nicht unbenannt |
| ☐ | Payload enthält `organisation` **und** `selection_hash` |
| ☐ | Query-Dense mit `retrieval.query` eingebettet |
| ☐ | Query-Sparse mit `language="german"` **und** `query_embed()` |
| ☐ | Leere `vector_questions` werden übersprungen, nicht mit Nullen gefüllt |
| ☐ | Vor Delta-Upserts Punkte je `source_url` gelöscht |
| ☐ | `chunking_config_hash` und `selection_hash` in der Collection sind einheitlich |
| ☐ | Nur **eine** Filtervariante je Collection (siehe Kopf) |
| ☐ | `total_chunks` **nicht** als Zahl der vorhandenen Chunks gelesen (Abschnitt 4.1) |
| ☐ | Dokumente werden nicht aus einer `source_url` als vollständig zusammengesetzt |
| ☐ | Anzeigetitel aus dem URL-Pfad abgeleitet, nicht `title` roh verwendet |
| ☐ | `sources.txt` hat 5.604 Zeilen, nicht 5.306 |

### Selbsttest nach dem Import — mit Sollwerten

Diese Zahlen stammen aus der gelieferten Datei, nicht aus einer Schätzung. Wer
sie reproduziert, hat vollständig und ohne Verlust importiert.

```python
info = client.get_collection("hsh")
assert info.points_count == 121134
print(sorted(info.config.params.vectors.keys()))          # ['questions', 'text']
print(sorted(info.config.params.sparse_vectors.keys()))   # ['sparse', 'sparse_expanded']
```

Verteilung der Payload-Felder — je Wert die erwartete Punktzahl:

| `faculty` | Punkte | | `organisation` | Punkte |
|---|---|---|---|---|
| *(leer)* | 39.600 | | *(leer)* | 117.890 |
| Fakultät V | 32.636 | | Projektwebseite | 1.891 |
| Fakultät IV | 15.772 | | MINT | 1.014 |
| Fakultät II | 14.949 | | Karriere | 233 |
| Fakultät III | 10.401 | | IT-Service | 75 |
| Fakultät I | 7.776 | | Bibliothek | 16 |
| | | | Forschungscluster | 15 |

```python
for wert, soll in [("Fakultät V", 32636), ("Fakultät II", 14949)]:
    n = client.count("hsh", count_filter=models.Filter(must=[
        models.FieldCondition(key="faculty",
                              match=models.MatchValue(value=wert))]), exact=True)
    assert n.count == soll, (wert, n.count, soll)
```

Die fünf Fakultätszahlen sind zugleich die Zusage aus Abschnitt 9: Sie
entsprechen **exakt** der Anzahl verschiedener Texte, die jede Fakultät im
Bestand führt (unabhängig nachgezählt in
`Backup/analyse_skripte/messung_wege.py`). Weicht eine ab, ist beim Import
etwas verloren gegangen.

Für die leere Fakultät gilt das nicht: 39.600 Punkte gegenüber 46.096
verschiedenen Texten ohne Fakultät. Der Unterschied ist gewollt — ein Text
ohne Fakultät bekommt nur dann einen eigenen Punkt, wenn er der Vertreter
seiner Gruppe ist. Sonst wären es 6.931 Punkte mehr, die kein Fakultätsfilter
je zurückgibt.

Ein Punkt mit Fragen — sind alle vier Arme belegt?

```python
p = client.retrieve("hsh", ids=[irgendeine_id], with_vectors=True)[0]
print(sorted(p.vector.keys()))   # sparse, sparse_expanded, text (+ questions)
```

### Wenn etwas nicht funktioniert

| Symptom | fast immer die Ursache |
|---|---|
| Deutsche Anfrage liefert über `sparse` keine Treffer | `language` steht nicht auf `german`, oder der Index wurde ohne `modifier=IDF` angelegt |
| Ranking wirkt zufällig, häufige Wörter gewinnen | `modifier=IDF` fehlt — die gespeicherten Gewichte enthalten keines |
| `questions`-Arm liefert nie etwas | Query mit `retrieval.passage` statt `retrieval.query` eingebettet |
| Scores unplausibel hoch | Query-Sparse mit `embed()` statt `query_embed()` erzeugt |
| Punktzahl kleiner als 121.134 | leere `vector_questions` als Nullvektor eingetragen und von Qdrant abgewiesen, oder Upsert-Fehler verschluckt |
| Treffer zeigt eine fremde Fakultät | kein Fehler — der Text steht dort wortgleich, siehe Abschnitt 9 |
| Nachbar-Chunk `idx ± 1` liefert nichts | normal, das Dokument hat Lücken — Abschnitt 4.1 |
| Dokument wirkt beim Zusammensetzen unvollständig | ist es auch: 29 % der Dokumente, siehe Abschnitt 4.1 |
| Das LLM wählt sichtbar das falsche PDF | `title` ist der Dateiname; Anzeigetitel aus dem Pfad bauen, Abschnitt 9 |
