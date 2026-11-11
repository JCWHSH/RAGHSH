# Übergabe an die Empfängerseite — Antwort auf die Prüfmeldung

**Datum:** 17. August 2026
**Betrifft:** `hsh_vectors_wdh10.parquet` und die fünf Befunde der Prüfmeldung
**Referenz:** `Parquet-Schema_und_Qdrant-Import.md` (überarbeitet, maßgeblich)

Alle fünf Befunde wurden gegen die Datei nachgezählt. Vier sind bestätigt, bei
zweien ist der vorgeschlagene Reparaturweg falsch, einer ist gegenstandslos.
**Nichts davon erfordert eine Neuvektorisierung.**

---

## 1. Dateien

| Datei | Größe | md5 |
|---|---|---|
| `hsh_vectors_wdh10.parquet` | 916 MB | `25f73aef2694c910be54b74d4b6e8e5e` |
| `hsh_vectors_wdh10_sources.txt` | 5.604 Zeilen | `36324b7f57a1e02e8ef6472beba90b05` |
| `chunk_vertretung.parquet` | 6,9 MB | `efd9ee68265553e93305efe026f22dad` |

```bash
scp HPC_KENNUNG@glogin9.hlrn.de:/mnt/vast-kisski/projects/PROJEKTKONTO/users/\
NUTZERNAME/hsh_scraper/{hsh_vectors_wdh10.parquet,\
hsh_vectors_wdh10_sources.txt,chunk_vertretung.parquet} .
```

> **`hsh_vectors_wdh10_sources.txt` neu ziehen.** Die erste Auslieferung hatte
> 5.306 Zeilen und war unvollständig — der Befund war richtig. Wer die alte
> Datei benutzt, räumt bei Delta-Läufen den Altbestand von 298 Seiten nicht ab.
> Die Vektordatei ist unverändert; ihr md5 dient nur der Kontrolle.

`chunk_vertretung.parquet` ist neu und **optional** — nur nötig, wenn
vollständige Dokumente rekonstruiert werden sollen (Punkt 3 unten).

---

## 2. Was in der Erzeugung korrigiert wurde

| | |
|---|---|
| `hpc_vectorizer.py` | Sidecar-Liste wird jetzt **vor** dem Dedup-Filter befüllt |
| `hsh_vectors_wdh10_sources.txt` | ohne GPU nachgezogen, 5.306 → 5.604 Zeilen |
| `Parquet-Schema_und_Qdrant-Import.md` | Abschnitt 4.1 neu, 7.2 korrigiert, 9 erweitert, Prüfliste ergänzt |

Die Behauptung in der alten Fassung von Abschnitt 7.2, die 298
wegdeduplizierten Seiten stünden in `sources.txt`, war falsch. Sie stimmt jetzt.

---

## 3. Vier Dinge, die auf der Empfängerseite zu bauen sind

Alle vier sind Import-Zeit-Arbeit. Keine GPU, kein neuer Lauf.

### 3.1 Vollständigkeitskennzeichen — höchste Priorität

`chunk_index` und `total_chunks` beziehen sich auf das Dokument **vor** der
Deduplizierung. Gemessen: 1.559 der 5.306 Dokumente (29,4 %) haben Lücken,
65.763 Positionen fehlen, 457 Dokumente unter 50 %, 180 unter 20 %,
Extremfall 3 von 242.

Der Text ist **nicht verloren** — er steht wortgleich unter einer anderen
`source_url`. Verloren ist nur die Möglichkeit, ein Dokument aus *einer* URL
zusammenzusetzen.

Beim Import je `source_url` mitzählen und ins Payload schreiben:

```python
payload["chunks_vorhanden"]  = vorhanden[url]      # Punkte dieser URL in der Datei
payload["chunks_dokument"]   = r["total_chunks"]   # Chunks des ganzen Dokuments
payload["text_vollstaendig"] = vorhanden[url] == r["total_chunks"]
```

Damit kann die Auswahlrunde Fragmente abwerten oder kennzeichnen.

**Zwingend zusätzlich:**

* `_load_chunks_from_qdrant()` darf ein Dokument **nicht** aus einer
  `source_url` als vollständigen Beleg an das LLM geben. Bei 3 von 242
  Absätzen antwortet das Modell sonst „laut Prüfungsordnung …" auf Basis von
  einem Prozent des Textes — und merkt nichts davon. Das ist der einzige der
  fünf Befunde, der eine falsche, aber überzeugend aussehende Antwort erzeugt.
* `chunk_index < total_chunks - 1` ist **keine** gültige Grenzprüfung. Die
  tatsächlich vorhandenen Indizes abfragen statt zu rechnen.
* `_fetch_neighbor_chunk(url, i±1)` liefert bei Lücken nichts. Das ist normal.

### 3.2 Anzeigetitel aus dem URL-Pfad

`title` ist bei PDFs der Dateiname: 95.841 von 98.451 PDF-Chunks (97,3 %) und
4.709 von 22.683 HTML-Chunks (20,8 %) tragen keinen aussagekräftigen Titel.

Der URL-Pfad trägt die fehlende Information und steht bereits im Payload:

```
01_F_I_PO_EIT_2019     →  f1 › 10 Bachelorstdg › EIT › 01 F I PO EIT 2019
20193009_BPO_EWI_verk  →  f1 › 10 Bachelorstdg › EWI › 20193009 BPO EWI verk
```

Rezept: Host-Label voranstellen, Pfadsegmente nehmen, technische verwerfen
(`fileadmin`, `HsH`, `Forms`, `Dokumente`, `Bilder_Datein`, `wp-content`,
`uploads`, reine Zahlen), die letzten zwei Ordner plus Dateiname behalten,
Dateiendung ab, `_` und `-` zu Leerzeichen, `%`-Kodierung auflösen.
Beispielcode in `Backup/analyse_skripte/pruefe_reparaturen.py`.

Ergänzend: Bei 2.016 von 2.880 PDF-Dokumenten ohne brauchbaren Titel (70 %)
liefert das `section_heading` des ersten Chunks eine echte Überschrift
(`Baker_Hughes_Fragebogen` → `Baker Hughes > Steckbrief`). Bei Ordnungen trägt
eher der Pfad, bei Flyern und Steckbriefen eher die Überschrift — beides
kombinieren.

### 3.3 Dokumente rekonstruieren (nur wenn nötig)

`chunk_vertretung.parquet`, 196.934 Zeilen:

| Spalte | Bedeutung |
|---|---|
| `chunk_id` | Punkt-ID dieses Chunks, auch wenn er nicht im Index ist |
| `source_url`, `chunk_index` | seine Stelle im Dokument |
| `vertreten_durch` | Punkt-ID, unter der sein Text im Index steht |
| `selbst_im_index` | `True`, wenn er sein eigener Vertreter ist |

Geprüft: 0 Chunks ohne auffindbaren Vertreter, danach 0 Dokumente mit Lücken.

```python
zeilen = vertretung[vertretung.source_url == url].sort_values("chunk_index")
punkte = client.retrieve("hsh", ids=list(zeilen.vertreten_durch))
volltext = "\n\n".join(p.payload["text"] for p in punkte)
```

Wer so zusammensetzt, muss die fremden Fundstellen in der Quellenangabe nennen —
der Absatz stammt dann aus einem gleichlautenden Dokument einer anderen
Fakultät.

### 3.4 `faculty_key`

Payload führt `Fakultät IV`, der Korpus führt `f4`. Die Tabelle ist fest
(`FACULTY_MAP` in `chunking.py`), die Reihenfolge die der römischen Ziffern.
Beim Import ein zweites Feld `faculty_key` anlegen; die bestehende
`_faculty_key()`-Übersetzung ist völlig in Ordnung. Ein künftiger
Vektorisierungslauf kann das Feld direkt mitschreiben.

---

## 4. Drei Dinge, die **nicht** gemacht werden sollten

**Titel im Korpus reparieren (PDF-`/Title` im Crawler auslesen).** In dieser
Pipeline der teuerste aller Wege: `title` geht über `informative_title()` in
den Dedup-Schlüssel ein. Andere Titel heißen andere Gruppen, andere Punktmenge
und damit eine **ungültige Anreicherung** — Crawl, 1,7 GPU-Stunden Anreicherung
und Vektorisierung müssten alle wiederholt werden. Ein Anzeigetitel gehört auf
die Empfängerseite (3.2).

**Mindestlänge beim Chunking.** Die 2.530 Chunks unter 100 Zeichen sind nicht
durchweg wertlos:

```
[69] ## Fachgebiete * Pflege * Rehabilitation * Ernährung * Soziale Arbeit
[76] ### Vorbereitung Für die heutige Stunde ist keine Vorbereitung erforderlich.
[92] **Kontakt** Hochschule Hannover BSKE Experimentelle Gestaltung Expo Plaza 2
```

Kurze, aber vollständige Aussagen — teils exakt die Antwort auf eine kurze
Frage. Eine Mindestlänge änderte die Punktmenge und erzwänge eine
Neuvektorisierung.

**Wegen `avg_len` neu vektorisieren.** Exakt über alle 121.134 Punkte gemessen:
88,4 statt 88,3 (Text) und 107,6 statt 107,1 (erweitert), Abweichung +0,1 %
und +0,5 %. Wirkung auf die BM25-Längennormalisierung rund **0,1 %** bei jeder
Chunklänge — zwei Größenordnungen unter allem, was ein Ranking bewegt. Für die
nächste Erzeugung wird die Stichprobe hochgesetzt.

---

## 5. Einordnung: mehrfach vorhandene Texte

1.825 Texte kommen auf 5.038 Punkten mehrfach vor. Zusammensetzung:

* **967 Gruppen (2.310 Punkte)** — die gewollte Ausnahme der Deduplizierung:
  je Gruppe ein Vertreter pro vorkommender Fakultät und Organisation, damit
  jeder Filter jeden Text findet.
* **858 Texte** — unterscheiden sich **nur im Titel**: derselbe Absatz unter
  „Material Monday 2026: Asbest" und „… Kunststoffe". Verschiedene Titel
  erzeugen verschiedene Fragen, deshalb bleiben sie getrennt.

Kein Fehler, aber beim Lesen von Trefferlisten zu wissen.

---

## 6. Sollwerte nach dem Import

```
points_count                        121.134
benannte Dense-Vektoren             text, questions
benannte Sparse-Vektoren            sparse, sparse_expanded
Punkte mit questions                109.955   (alle mit Fragenvektor UND
                                               erweitertem Sparse-Arm)
Punkte ohne questions                11.179   (alle mit leerem Fragenvektor)
```

| `faculty` | Punkte | | `organisation` | Punkte |
|---|---|---|---|---|
| *(leer)* | 39.600 | | *(leer)* | 117.890 |
| Fakultät V | 32.636 | | Projektwebseite | 1.891 |
| Fakultät IV | 15.772 | | MINT | 1.014 |
| Fakultät II | 14.949 | | Karriere | 233 |
| Fakultät III | 10.401 | | IT-Service | 75 |
| Fakultät I | 7.776 | | Bibliothek | 16 |
| | | | Forschungscluster | 15 |

Die fünf Fakultätszahlen sind zugleich die Anzahl verschiedener Texte, die
jede Fakultät im Bestand führt. Weicht eine ab, ist beim Import etwas verloren
gegangen. Für die leere Fakultät gilt das nicht — Begründung in Abschnitt 10
des Schema-Dokuments.

---

## 7. Offen, bewusst

Bei 29.258 Textgruppen steht derselbe Wortlaut auf mehreren Seiten; zitiert
wird eine davon. Alias-Listen (`source_urls` als Array im Payload) würden das
schließen und ließen sich aus `chunk_vertretung.parquet` beim Import erzeugen —
die Vektoren bleiben unberührt. Bisher nicht gebaut, weil noch keine
Retrieval-Messung zeigt, dass die willkürliche Quellenangabe stört.

Ebenfalls offen: ob `--max-wiederholung 10` beim Fragenfilter die richtige
Einstellung ist. Das entscheidet der Vergleich gegen das Goldset. Weitere
Varianten sind in 20 Minuten GPU erzeugbar; sie unterscheiden sich
ausschließlich in `questions`, `sparse_expanded_*` und `vector_questions` und
gehören in eine **eigene Collection**, weil sie sich die Punkt-IDs teilen.
