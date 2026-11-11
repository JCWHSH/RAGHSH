# Auftrag: doc2query-Prompt v2 — Pilot auf den Golddokumenten

Zweiter Auftrag für denselben Agenten wie `AUFTRAG.md` (Zugänge, Pfade und
Regeln gelten unverändert). Er ist **unabhängig** vom ersten und kann davor,
danach oder parallel laufen; er berührt weder Qdrant noch den Zielserver,
außer beim Abholen und Zurücklegen von Dateien.

## Worum es geht

`enricher.py` erzeugt mit Qwen3.5-9B zu jedem Chunk Fragen („doc2query"). Sie
liegen im Index als vierter Arm: `questions` (Dense-Vektor über die
zusammengefügten Fragen, Task-Adapter `retrieval.query`) und `sparse_expanded`
(BM25 über Text + Fragen). Seit 22.09.2026 ist `questions` in der Produktion
aktiv.

Die Auswertung der 303.716 vorhandenen Fragen (22.09.2026) zeigt fünf
Schwächen — sie sind der Grund für diesen Auftrag:

1. **9,4 % der Indexpunkte haben gar keine Fragen**, fast alle, weil der
   Enricher erst ab 200 Zeichen arbeitet. Kurze Chunks tragen aber oft die
   eigentliche Antwort (Raumnummer, Sprechzeit, E-Mail).
2. **Immer genau drei Fragen**, unabhängig vom Inhalt (107.074 von 111.456
   Chunks). Eine Tabelle mit 15 Modulzeilen bekommt so viele wie ein
   Fließtextabsatz.
3. **Der Chunk kennt seinen Kontext nicht, die Frage deshalb auch nicht.**
   Echte Beispiele: „Wie melde ich mich zur Weiterbildung an?", „Wo finde ich
   den Link zur Anmeldung?" — solche Fragen passen zu hunderten Dokumenten
   und trennen nichts.
4. **Die Fragen benutzen die Wörter des Dokuments, Nutzer umschreiben.** Der
   Nutzer fragt „Welche Weiterbildung richtet sich an Menschen, die selbst
   Seminare geben?", generiert wurde „Wer darf sich für den
   Train-the-Trainer-Kurs anmelden?". Wer den Fachbegriff kennt, muss nicht
   fragen. Alle vier Evaluationsfälle, die der Fragen-Arm gegenüber dem
   Textarm verliert (g16, g23, g27, g28), sind von diesem Typ.
5. **17,7 % Dubletten** trotz Filter auf maximal zehn Vorkommen.

Deine Aufgabe: den Prompt austauschen, die Mechanik in den Punkten 1 und 2
anpassen, **nur die 108 Golddokumente** neu anreichern und messen, ob die
neuen Fragen näher an den echten Nutzerfragen liegen als die alten. Der volle
Lauf über den Korpus wird erst danach entschieden.

## Eingabedateien

Auf dem Zielserver unter /mnt/nemo/RAGHSH/hsh_scraper/eval_gold/testindex/hpc/

| Datei | Zweck | md5 (Anfang) |
|---|---|---|
| `urls.txt` | die 108 Golddokumente | `072e6138` |
| `goldfragen.json` | 75 echte Nutzerfragen mit Goldquelle und Belegstelle | `8ed4a6e8` |
| `hsh_korpus_2026-09-22.tar.xz` | `data/ingested/` (5.619 Dateien) | `a993b983` |
| `chunking.py`, `titel.py` | Chunking v2 mit Kontextkopf | `0454e450`, `efd415c8` |

Auf dem HPC liegen `enricher.py`, das venv mit vLLM und das Qwen-Modell aus
dem Lauf vom August.

## Der neue Prompt

Behalte das **Ausgabeformat bei**: ein JSON-Array von Strings, nichts
davor, nichts danach. Der vorhandene Parser und die Gültigkeitsprüfung in
`enricher.py` bleiben unverändert (sie verwerfen unter anderem Ja/Nein-Fragen
und Antworten wie „Keine Fragen möglich" — das ist richtig so).

**System-Nachricht:**

```
Du erzeugst Suchfragen für die Volltextsuche der Hochschule Hannover.

Zu einem Abschnitt einer Hochschulseite schreibst du Fragen, die genau dieser
Abschnitt beantwortet. Sie werden nicht angezeigt; sie dienen nur dazu, den
Abschnitt auffindbar zu machen, wenn jemand mit eigenen Worten sucht.

Regeln:
1. Jede Frage muss der Abschnitt vollständig beantworten. Was dort nicht
   steht, wird nicht gefragt — auch nicht, wenn es naheliegt.
2. Jede Frage muss für sich allein verständlich sein. Benenne den Gegenstand
   konkret: den Studiengang, die Ordnung, den Kurs, die Person, das Modul, die
   Fakultät. „Wie beantrage ich eine Pause?" ist unbrauchbar, „Wie beantrage
   ich eine Beurlaubung vom Studium an der HsH?" ist gut. Die Herkunft des
   Abschnitts steht im Feld KONTEXT — benutze sie.
   Für BEDARF-Fragen (Regel 3) gilt das eingeschränkt: Dort nennst du nur, was
   der Suchende selbst kennt und schreiben würde — die Hochschule, die
   Bibliothek, den Studiengang, das Prüfungsamt. Interne Stellenbezeichnungen
   und Namen von Beschäftigten (Justiziariat, Abteilung D4, Frau Meier) gehören
   in die FACH-Frage; in der BEDARF-Frage machen sie sie unauffindbar.
3. Schreibe zwei Sorten Fragen:
   FACH — mit den Begriffen des Abschnitts, für Leute, die wissen, wonach sie
   suchen.
   BEDARF — aus der Lage heraus, in der jemand sucht, und ohne den Fachbegriff,
   den er noch nicht kennt. Beschreibe stattdessen, was die Person will, hat
   oder nicht hat. Beispiele:
     Abschnitt über die Beurlaubung vom Studium
       FACH:   Welche Gründe erkennt die HsH für eine Beurlaubung an?
       BEDARF: Was passiert mit meinem Studienplatz, wenn ich wegen Krankheit aussetze?
     Abschnitt über die Fernleihe der Bibliothek
       FACH:   Über welche Datenbanken läuft die Fernleihbestellung an der HsH?
       BEDARF: Wie komme ich an ein Buch, das die Bibliothek nicht im Katalog hat?
     Abschnitt über die Richtlinie zum Deutschlandstipendium
       FACH:   Wer vergibt die Deutschlandstipendien nach der Richtlinie der HsH?
       BEDARF: Welche Förderung gibt es für begabte Studierende unabhängig vom Einkommen?
4. Keine Ja/Nein-Fragen. Keine Fragen nach dem Dokument selbst („Was steht in
   diesem Abschnitt?"). Keine zwei Fragen, die dasselbe meinen.
5. Bei Tabellen: je Zeile oder Zeilengruppe eine Frage, die den Eintrag aus der
   Kopfzeile heraus benennt — „In welchem Raum findet MDT-341 Cyber-Sicherheit
   statt?" statt „Welche Veranstaltungen stehen im Stundenplan?".
6. Deutsch, eine Zeile je Frage, höchstens 15 Wörter.

Antworte ausschließlich mit einem JSON-Array von Strings.
```

**Nutzer-Nachricht (Template):**

```
KONTEXT: {kontext}
ABSCHNITT:
{text}

Schreibe {n} Fragen: etwa zur Hälfte FACH, zur Hälfte BEDARF.
```

- `{kontext}` ist das neue Chunk-Feld aus `chunking.py` (Anzeigetitel ›
  Überschriftenkette). Es steht nach `chunk_document()` in jedem Chunk unter
  dem Schlüssel `kontext`. Fällt es leer aus, schreibe `KONTEXT: (unbekannt)`.
- `{text}` ist der Chunktext wie bisher.
- `{n}` berechnest du im Skript, siehe unten.

## Mechanik: zwei Änderungen in `enricher.py`

1. **Mindestlänge** von 200 auf **80 Zeichen** senken (die Konstante heißt im
   Skript vermutlich `MIN_CHARS` oder ähnlich — such sie, ändere nur den Wert).
2. **Fragenzahl je Chunk** statt fest 3:

   ```python
   def fragenzahl(text: str) -> int:
       zeilen = [z for z in text.splitlines() if z.lstrip().startswith("|")]
       if len(zeilen) >= 4:                    # Tabellenchunk
           return max(4, min(8, len(zeilen) - 2))   # Kopf- und Trennzeile abziehen
       return max(2, min(6, len(text) // 250))
   ```

   Setz diesen Wert als `{n}` in den Prompt ein und lass die vorhandene
   Prüfung („mindestens eine gültige Frage") unverändert.

Sonst nichts ändern — insbesondere nicht das Ausgabeformat, nicht die
Batchgröße, nicht die ID-Bildung (`make_point_id`), nicht die
Fingerabdruck-Prüfungen.

## Ablauf auf dem HPC

Arbeitsverzeichnis `…/hsh_scraper/fragen_v2/`, getrennt von `testindex_v2/`.

1. Korpus entpacken (oder das schon entpackte aus `testindex_v2/ingested/`
   mitbenutzen), `chunking.py` und `titel.py` aus dem Paket dorthin, die
   angepasste `enricher.py` ebenfalls **als Kopie** — das Original im
   Projektverzeichnis bleibt unangetastet.
2. Anreicherung **nur für die 108 URLs** aus `urls.txt` laufen lassen. Hat
   `enricher.py` keine `--urls`-Option, filtere die Dateiliste vor der
   Schleife auf diese source_urls; beschreibe im Bericht, wie du es gelöst
   hast. Erwartung: rund 2.800 Chunks, mit vLLM wenige Minuten GPU.
   Ergebnis: `enrichment_gold_v2.parquet` im gewohnten Schema
   (`id, source_url, chunk_index, questions, raw, model, ok, …`).
3. **Messen** — das ist der eigentliche Zweck. Schreib ein kleines Skript
   (`vergleich_fragen.py`), das für jeden der 75 Fälle aus `goldfragen.json`:
   - den Chunk des Golddokuments sucht, der die `belegstelle` enthält
     (Vergleich auf Kleinschreibung und zusammengezogenen Leerraum; wenn keiner
     passt, den Chunk mit der größten Wortüberschneidung zur Belegstelle);
   - dessen alte Fragen (aus `enrichment_v2_wdh10.parquet` im
     Projektverzeichnis, Join über `id`) und neue Fragen (aus
     `enrichment_gold_v2.parquet`) nimmt;
   - **die Kosinus-Ähnlichkeit** zwischen der echten Nutzerfrage und jeder
     Fragenmenge berechnet. Wichtig: dasselbe Modell und dieselben
     Task-Adapter wie im Index — Jina v3 (`jinaai/jina-embeddings-v3`),
     beide Seiten mit `retrieval.query`, die Fragen eines Chunks mit `" "`
     zusammengefügt, genau wie `hpc_vectorizer.py` es für den
     `questions`-Vektor tut (dort nachsehen, nicht raten).

   Ausgabe als Tabelle: Fall-ID, Ähnlichkeit alt, Ähnlichkeit neu, Differenz,
   Nutzerfrage, beste alte und beste neue Einzelfrage. Dazu Median und
   Mittelwert über alle 75 Fälle sowie die Zahl der Fälle mit Verbesserung
   bzw. Verschlechterung. **Besonders interessant: g16, g23, g27, g28** (dort
   verliert der Arm heute) und **g30, g34, g39, q04, q19** (dort gewinnt er
   schon jetzt — sie dürfen nicht schlechter werden).
4. Zurücklegen auf den Zielserver nach
   `/home/christian/RAGHSH/hsh_scraper/eval_gold/testindex/`:
   `enrichment_gold_v2.parquet`, `vergleich_fragen.py`, die Messtabelle als
   `fragen_v2_messung.tsv` und die angepasste `enricher.py` als
   `enricher_v2.py` (damit der spätere Volllauf denselben Prompt benutzt).

Wenn beide Aufträge laufen: Der Testindex aus `AUFTRAG.md` benutzt die
**alten** Fragen (`enrichment_teil.parquet`) — das ist Absicht, damit die
Wirkung des Chunkings allein messbar bleibt. Misch die neuen Fragen nicht
hinein.

## Bericht

Kurz und in Zahlen: Zahl der angereicherten Chunks, GPU-Zeit, Anteil
ungültiger Antworten (`ok=False`) alt/neu, durchschnittliche Fragenzahl je
Chunk alt/neu, die Ähnlichkeitstabelle (Median, Mittelwert, besser/schlechter)
und je drei Beispiele, wo die neuen Fragen deutlich besser bzw. schlechter
sind — mit Wortlaut. Bewerte nicht, ob der volle Lauf kommen soll; nenne aber
die hochgerechnete GPU-Zeit für alle 121.134 Chunks.

Wenn der Prompt in der Praxis scheitert (Modell hält das Format nicht ein,
liefert leere Arrays, mischt Englisch hinein), ändere ihn so wenig wie
möglich, dokumentiere jede Änderung wörtlich im Bericht und miss mit der
geänderten Fassung weiter — eine Messung mit dokumentiertem Prompt ist
brauchbar, eine mit unbekanntem nicht.
