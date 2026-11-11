# Messung — `filter_fragen.py`

**Datum:** 14. August 2026
**Grundlage:** `enrichment_v2.parquet` (111.456 Zeilen, 329.506 Fragen),
`chunk_cache.parquet`
**Zustand:** gemessen, Filter unverändert, nicht angewandt

| Skript | Frage |
|---|---|
| `messung_filter.py` | Wortregel, Korpus, Vererbung — je einzeln |
| `messung_wdh.py` | Wiederholungskriterium, Schwellenkurve, Herkunft |
| `messung_filter_grid.py` | Entscheidungstabelle |
| `beispiele_filter.txt`, `beispiele_wiederholung.txt` | Volltexte zum Lesen |

---

## 1. Der Filter trifft fast nur mit einem seiner beiden Kriterien

```
Fragen gesamt                       329.506
verworfen wegen Wiederholung         60.057   (18,2 %)
verworfen wegen Dokumentfrequenz      5.681   ( 1,7 %)
Überschneidung                        3.205
behalten                            266.973   (81,0 %)
Chunks, die ALLE Fragen verlieren     7.322   ( 6,6 %)
```

Die Diskussion um die Wortregel betraf damit das kleinere der beiden
Kriterien. Sie ist trotzdem berechtigt — aber sie ist nicht die
Hauptentscheidung.

---

## 2. Die Wiederholungsschwelle ist nie gemessen worden

`--max-wiederholung 3` verwirft eine Frage, sobald sie an mehr als drei Chunks
hängt. Die Verteilung:

| Vorkommen | Fragen |
|---|---|
| 1 | 211.286 |
| 2 | 40.982 |
| 3 | 17.181 |
| 4–5 | 18.295 |
| 6–10 | 17.326 |
| 11–50 | 17.955 |
| über 50 | 6.481 |

Schwellenkurve (nur dieses Kriterium):

| Schwelle | behalten | |
|---|---|---|
| 3 | 269.449 | 81,8 % |
| 5 | 287.744 | 87,3 % |
| 10 | 305.070 | 92,6 % |
| 20 | 315.434 | 95,7 % |

### Der Widerspruch zwischen den beiden Kriterien

```
Wiederholung 3  verwirft ab      4 Chunks  =  0,0033 % des Index
Frequenz 2 %    erlaubt bis  2.422 Chunks  =  2,0000 % des Index
```

Ein Faktor **600**. Dasselbe Skript nennt eine Frage „zu allgemein", wenn sie
auf 4 von 121.134 Chunks passt — und „spezifisch genug", wenn ihr seltenstes
Wort auf 2.422 passt. Beide Kriterien sollen dieselbe Eigenschaft messen.

### Was bei Schwelle 3 wegfällt

Zu Recht — die häufigsten Verworfenen sind Modulhandbuch-Boilerplate:

```
364x  Welche Module gehören zum zweiten Studienabschnitt?
284x  Welche Pflichtmodule gehören zum ersten Studienabschnitt?
247x  Welche Anforderungen gelten für das Selbststudium?
241x  Welche Anforderungen gelten für die Präsenzzeit?
```

91 % der Verworfenen stammen aus PDFs, überwiegend aus Vorlesungsverzeichnissen
und Modulhandbüchern (`BRS-PO-2020`, `BSW-PO-2018`, `MAB-AM-WS2010`).

Zu Unrecht — der 4er-Bereich, der heute knapp herausfällt:

```
Was gehört zum Modul SBW-223?
Wo finde ich das Vorlesungsverzeichnis für BSW-215?
Wie werden Datenstrukturen im Modul MAB-106 formuliert?
Welche Partnerhochschulen gibt es in Frankreich?
Wie viel kostet der 16-wöchige Kurs?
Wann muss die letzte Prüfungsleistung erbracht sein?
Welche Pflichtmodule gehören zum Schwerpunkt Verfahrenstechnik?
```

Eine Frage, die auf 4 von 121.134 Chunks passt, grenzt den Korpus auf
0,003 % ein. Die Annahme „hängt an mehr als drei Chunks, also unterscheidet
sie nichts" hält der Stichprobe nicht stand. Erst ab etwa 10 Vorkommen wird
das Bild gemischt, ab 20 überwiegt Allgemeines.

---

## 3. Die drei Fehler in den Details

### 3.1 Wortregel — 934 Fragen

`_WORT` verlangt vier Kleinbuchstaben. Studiengangskürzel sind damit keine
Wörter, und die Frage wird nach ihrem allgemeinsten Wort beurteilt. Gerettet
würden unter anderem:

```
Welche Prüfungsordnung gilt für den Bachelor IBS?
Wo finde ich das Modulhandbuch MAB-PS?
Wie erfolgt die Zulassung zum Master MKO?
Wo finde ich den besonderen Teil der Prüfungsordnung KTD?
Was lernen Studierende zur HGÜ?
```

Das sind wörtlich die Anfragen, die Studierende tippen. Die Reparatur ist
nicht `{4,}` → `{2,}` — das machte „der", „die", „und" zu Inhaltswörtern und
zerstörte das Kriterium. Sie ist: **Großschreibung als Signal nutzen**, bevor
`text.lower()` sie wegwirft.

```python
ABK = re.compile(r"\b[A-ZÄÖÜ]{2,6}(?:-[A-ZÄÖÜ0-9]{1,4})?\b")
```

Die im Übergabedokument genannten 2.477 Fragen kann ich nicht reproduzieren;
nach dieser Definition — Fragen, die allein durch die Abkürzungsregel das
Frequenzkriterium bestehen — sind es 934, mit Zahlenregel 1.089.

### 3.2 Korpus — 1.461 Fragen

`dokumentfrequenz()` zählt über **alle 196.934** Chunks, auch die 75.800
wortgleichen Kopien, die nie in den Index kommen. Eine 15-fach gespiegelte
Prüfungsordnung zählt 15-mal; ihre Wörter erscheinen dadurch häufiger, als sie
im Index sind. Betroffen ist damit ausgerechnet die Dokumentklasse, deren
Auffindbarkeit heute Vormittag repariert wurde.

### 3.3 Vererbung — 826 Fragen

Die 1.123 geerbten Zeilen tragen dieselben Fragen wie ihr Vertreter. Das
Wiederholungskriterium zählt sie als eigene Chunks. Beispiele, die dadurch
kippen:

```
2 → 6   Wie lautet der Familienname des Inhabers?
1 → 5   Welche Matrikelnummer hat der Studierende?
        Wer hat bei der Bewertung nur eine beratende Stimme?   (§ 3 ATPO)
```

Die letzte ist die Pointe: Sie fällt heraus, weil derselbe Paragraph jetzt
unter drei Fakultäten auffindbar ist. Richtig gezählt wird **je Gruppe
wortgleicher Chunks**, nicht je Chunk.

---

## 4. Entscheidungstabelle

| Einstellung | behalten | | verw. Wdh | verw. DF | Chunks ohne Frage |
|---|---|---|---|---|---|
| heute (Wdh 3, je Chunk, roh, alle) | 266.973 | 81,0 % | 60.057 | 2.476 | 7.322 |
| nur Wortregel + Korpus repariert | 270.313 | 82,0 % | 57.880 | 1.313 | 6.483 |
| Wdh 5, sonst repariert | 287.151 | 87,1 % | 40.770 | 1.585 | 3.720 |
| **Wdh 10, sonst repariert** | **303.716** | **92,2 %** | 23.953 | 1.837 | **1.452** |
| Wdh 20, sonst repariert | 313.275 | 95,1 % | 13.864 | 2.367 | 552 |
| nur Frequenzkriterium | 325.802 | 98,9 % | 0 | 3.704 | 20 |
| gar nicht filtern | 329.506 | 100 % | 0 | 0 | 0 |

---

## 5. Empfehlung

**Filtern — aber mit `--max-wiederholung 10`, je Gruppe gezählt, und den
beiden Reparaturen an Wortregel und Korpus.**

1. Die Boilerplate, um die es geht, verschwindet auch bei 10: Die
   Spitzenreiter kommen 364-, 284-, 247-mal vor. Was Schwelle 3 zusätzlich
   trifft, sind nachweislich Fragen mit Modulcodes.
2. Selbst 10 bleibt 250-mal strenger als das Frequenzkriterium desselben
   Skripts. Es ist keine Aufweichung, es ist eine Annäherung der beiden
   Maßstäbe.
3. 1.452 statt 7.322 Chunks verlieren alle ihre Fragen. Diese Chunks fallen im
   doc2query-Arm auf den reinen Text zurück — für Abschnitte aus
   Modulhandbüchern, die ohnehin schwer zu treffen sind, ist das der teuerste
   Einzelposten.

**Die Asymmetrie, die den Ausschlag gibt:** Zu viel Filtern vernichtet
Information, die nur ein neuer Lauf zurückholt. Zu wenig Filtern hinterlässt
Rauschen, das der BM25-Arm zur Query-Zeit ohnehin abwertet — die häufig
wiederholten Fragen bestehen aus häufigen Termen, und der Sparse-Index läuft
mit `modifier=IDF`. Im Zweifel ist die lockerere Schwelle die billigere
Richtung.

**Was diese Empfehlung nicht ist:** eine Messung der Retrieval-Güte. Ohne
Goldset lässt sich `Recall@10` nicht bestimmen; das Skript selbst schreibt,
die Schwelle gehöre dagegen kalibriert. Die Zahlen hier sagen, *was* der
Filter trifft, und die Volltexte, *ob* das plausibel ist. Der Irrtum kostet
einen Vektorisierungslauf, nicht die Anreicherung.

### Umsetzung, falls freigegeben

| Schritt | Datei |
|---|---|
| 1 | `filter_fragen.py`: `ABK`-Zweig in `inhaltswoerter()`, Dokumentfrequenz über die Behaltemenge, Wiederholung je `dedupe_key` statt je Zeile |
| 2 | Voreinstellung `--max-wiederholung` 3 → 10, mit Begründung im Docstring |
| 3 | `filter_fragen.py enrichment_v2.parquet --schreiben` → `enrichment_v2_gefiltert.parquet` |
| 4 | `vectorize.sbatch` auf die gefilterte Datei zeigen |
| 5 | `sbatch vectorize.sbatch` |
