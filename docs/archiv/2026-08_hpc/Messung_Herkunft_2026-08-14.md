# Messung — Herkunft nach der Deduplizierung

**Datum:** 14. August 2026
**Anlass:** `Uebergabe_Herkunft-nach-Deduplizierung.md`, Abschnitte 3–5
**Zustand:** gemessen, noch nichts geändert

Alle Zahlen aus einem Chunking-Durchlauf über `data/ingested` (5.616 Dateien,
Login-Node, 73 s). Zwischenstand in
`Backup/analyse_skripte/chunk_cache.parquet` (196.934 Zeilen inkl. Volltext),
damit jede Auswertung dieselbe Grundlage hat.

| Skript | Frage |
|---|---|
| `messung_cache.py` | erzeugt den Cache |
| `messung_wege.py` | Nachprüfung + Wirkung der Wege |
| `messung_variante.py` | scharfe Variante + Erbbarkeit der Fragen |
| `messung_inhalt.py` | was die betroffenen Gruppen inhaltlich sind |
| `messung_folgen.py` | Obermenge, Determinismus, Seiten-Auffindbarkeit |
| `messung_konkurrenz.py` | wortgleiche Punkte im Index |
| `messung_grenzen.py` | sitzen die Fragen noch an derselben Stelle? |

---

## 1. Nachprüfung — was stimmt und was nicht

Der Grundbestand stimmt auf den Chunk genau:

```
Chunks gesamt      196.934
Gruppen gesamt     119.791      Ersparnis 39,2 %
geteilte Gruppen    31.708
```

Drei Angaben aus Abschnitt 3 der Übergabe halten der Nachprüfung **nicht**
stand.

### 1.1 Die 465 Gruppen sind unvollständig

`fakultaet_verlust.py` zählt in Abschnitt 1 nur Gruppen mit mehr als einer
**nicht-leeren** Fakultät (`faks = {f … if f}`). Damit fällt der Fall durch,
in dem der Vertreter *gar keine* Fakultät hat — eine zentrale Seite auf
`www.hs-hannover.de` — und die verworfene Kopie eine. Für einen Filter
`faculty = "Fakultät V"` ist diese Gruppe genauso unsichtbar.

```
Gruppen mit >1 nicht-leerer Fakultät       465     (die gemeldete Zahl)
Vertreter ohne Fakultät, Mitglied mit       219     (übersehen)
------------------------------------------------
betroffene Gruppen tatsächlich              684
```

### 1.2 1.781 und 2.000 sind dieselbe Größe, zweimal verschieden gezählt

Die Zahl 1.781 stammt aus Abschnitt 1 des Skripts (nur innerhalb der 465
Gruppen), die Tabelle darunter aus Abschnitt 2 (über alle Gruppen) — deren
Spalte summiert sich auf 2.000. Beide zählen **Chunk-Kopien**: Liegt derselbe
Text fünfmal in Fakultät V, zählt das fünffach, obwohl es *ein* unsichtbarer
Text ist.

Was der Qdrant-Filter tatsächlich nicht mehr findet, sind verlorene
**(Gruppe, Fakultät)-Paare**:

```
Z1  Gruppen mit >1 nicht-leerer Fakultät       465
Z2  fremdvertretene Chunk-Kopien             2.000   (die gemeldete Zählweise)
Z3  verlorene (Gruppe, Fakultät)-Paare         908   ← der harte Schaden
```

Verlust je Fakultät, in Gruppen statt Kopien:

| Fakultät | Chunks | Gruppen | unsichtbar | Anteil |
|---|---|---|---|---|
| Fakultät V | 61.684 | 32.636 | 302 | 0,9 % |
| Fakultät II | 20.366 | 14.949 | 316 | **2,1 %** |
| Fakultät IV | 18.308 | 15.772 | 190 | 1,2 % |
| Fakultät III | 12.826 | 10.401 | 93 | 0,9 % |
| Fakultät I | 13.634 | 7.776 | 7 | 0,1 % |
| (ohne) | 70.116 | 46.096 | 0 | 0,0 % |

Fakultät I ist kaum betroffen, weil sie in der Dateireihenfolge meist den
Vertreter stellt — sie ist die Gewinnerin der Sortierung, nicht die
robusteste Fakultät.

### 1.3 Die Organisationsachse ist betroffen, nicht unberührt

Die Aussage „**keine** Gruppe umspannt mehrere Organisationen … das Problem
betrifft ausschließlich die Fakultätsachse" ist ein Artefakt derselben Zeile.
Gruppen mit zwei *nicht-leeren* Organisationen gibt es tatsächlich keine —
aber 435 Gruppen verlieren ihre Organisation an einen Vertreter ohne:

```
Projektwebseite      424 Gruppen unsichtbar   (itr.wp.hs-hannover.de)
Forschungscluster     11 Gruppen unsichtbar
```

Insgesamt sind es **1.343 verlorene (Gruppe, Zuordnung)-Paare**: 908 auf der
Fakultäts-, 435 auf der Organisationsachse.

---

## 2. Was die betroffenen Gruppen inhaltlich sind

Betrachtet werden die 1.343 Chunks, die je verlorenem Paar wieder in den Index
kämen — nicht die Gruppen, sondern der konkrete Schaden.

| Einordnung | Anzahl | Anteil |
|---|---|---|
| sonstiger Fließtext (Modulbeschreibungen, Kursverzeichnisse) | 576 | 42,9 % |
| Ordnungstext (§, Prüfung, Modul, Praxis, ECTS) | 526 | 39,2 % |
| Fragment unter 200 Zeichen | 200 | 14,9 % |
| Navigation / Boilerplate | 41 | 3,1 % |

Häufigste Herkunftsdokumente:

```
264x  Modulhandbuch_1202_75_ITR_WS2025_final          (itr.wp)
145x  01-ATPO_2015_mit_Anlagen_A1-A3                  (F II)
145x  Allgemeiner_Teil_der_PO_2015                    (itr.wp)
143x  Pruefungsordnung_HP_allg_Teil_01-ATPO_2015      (F V)
 63x  Engl._Lectures-FI-II_WS_26-27                   (F II)
 59x  Grundordnung_der_HsH_7_Aenderung                (F IV)
 36x  04_FIV_ATPO_2025_mit_Anhang_VB                  (F IV)
 20x  Richtlinie_zum_Nachteilsausgleich_(RNTA)
```

Volltext-Stichprobe in `Backup/analyse_skripte/beispiele_volltext.txt`. Drei
Befunde daraus:

* **Es sind echte Ordnungstexte.** `§ 3 Prüfungsausschuss` steht wortgleich in
  der ATPO 2015 von Fakultät I, II und V; `§ 17 Widerspruchsverfahren`
  wortgleich in der ATPO 2025 von Fakultät III, IV und V. Wer aus Fakultät II
  nach dem Prüfungsausschuss fragt, bekommt heute die Fassung aus Fakultät I
  zitiert — inhaltlich dasselbe Dokument, aber unter fremder Adresse.
* **Der Rauschanteil ist benennbar.** Die 200 Fragmente sind fast alle
  Verkündungsblatt-Zeilen (`Nr. 04/2019 vom 31.08.2019`, 69 Zeichen, in 22
  Gruppen über alle fünf Fakultäten). Sie tragen keine Antwort.
* **Der ITR-Block ist ein Sonderfall.** 424 der 435 Organisationsverluste sind
  Kopien der PO und des Modulhandbuchs auf einer WordPress-Projektseite.
  Ob jemand je nach `organisation = "Projektwebseite"` filtert, ist offen.

Gruppengröße der betroffenen Texte: Median 4, Mittel 9,6, Maximum 137.

---

## 3. Die Wege, gemessen

| Weg | Punkte im Index | gegenüber heute |
|---|---|---|
| heute (Text + aussagekräftiger Titel) | 119.791 | — |
| **A** Alias-Listen im Payload | 119.791 | ±0 |
| **C** eng — nur die 465 Gruppen ausnehmen | 124.050 | +3,56 % |
| **C** weit — auch leere Fakultät als Unterschied | 139.517 | +16,5 % |
| **D** stumpf — `faculty` in den Schlüssel | 127.630 | +6,54 % |
| **D-scharf** — nur echte Fakultäten | 120.699 | **+0,76 %** |
| **D-scharf + Organisation** | 121.134 | **+1,12 %** |
| **B** Fan-out | 196.934 | +64,4 % |

### Warum Weg D stumpf sechsmal so teuer ist wie nötig

`dedupe_key = (Text, Titel, Fakultät)` als reine Schlüsselerweiterung kostet
7.839 zusätzliche Punkte — aber **6.931 davon tragen gar keine Fakultät**. Es
sind die zentralen Kopien, die sich von ihrer Fakultätsfassung nur dadurch
unterscheiden, dass ihr Feld leer ist. Ein Filter `faculty = "Fakultät I"`
liefert sie nie. Man bezahlt sechsmal für einen Effekt, den man nicht bekommt.

### Die scharfe Regel

Statt die leere Fakultät als eigenen Wert zu behandeln, erbt sie:

```
leere Fakultät   → behalten, wenn der Schlüssel noch nie vorkam
volle Fakultät   → behalten, wenn (Schlüssel, Fakultät) noch nie vorkam
```

Streaming-fähig, ein Durchlauf, dieselbe Dateireihenfolge — passt ohne
Zweitdurchlauf in `select_unique_ids()`. Dieselbe Regel mit der
Organisation als zweiter Achse kostet 435 Punkte mehr.

**Nachweis der Wirkung:** Unter D-scharf + Organisation sind **84.778 von
84.778** (Gruppe, Zuordnung)-Paaren im Index vertreten — 0 fehlend. Heute
fehlen 1.343. Der harte Schaden ist damit vollständig geschlossen, nicht
teilweise.

### Warum Weg C ausscheidet

Weg C ist in beiden Lesarten von D-scharf dominiert: Die enge Lesart kostet
4,7-mal so viel (+3,56 % statt +0,76 %) und lässt die 219 Gruppen mit leerem
Vertreter sowie die Organisationsachse ungelöst; die weite Lesart kostet das
15-fache, weil sie ganze Gruppen mit bis zu 137 Mitgliedern unangetastet
lässt, statt je Fakultät einen Vertreter zu behalten.

### Konkurrenz wortgleicher Punkte

Der Haupteinwand gegen Weg B — nahezu identische Vektoren konkurrieren um
Trefferplätze — trifft die scharfe Regel kaum:

| | Gruppen mit mehr als einem Punkt | größte Gruppe |
|---|---|---|
| D-scharf + Org | 967 (0,81 %) | 6 Punkte |
| Weg B | 31.708 (26,5 %) | 228 Punkte |

---

## 4. Kostet das GPU-Zeit? Nein.

### 4.1 Die Anreicherung bleibt gültig

Die neue Behaltemenge ist eine **echte Obermenge** der heutigen:

```
heute 119.791 → neu 121.134,  Zuwachs 1.343,  verlorene IDs 0
```

Kein bisher angereicherter Chunk fällt weg. Punkt-IDs sind
`UUID5(source_url#chunk_index)` — der Dedup-Schlüssel geht nicht ein, keine ID
verschiebt sich.

Zusätzlich nachgewiesen statt behauptet: Für alle **110.333** Zeilen in
`enrichment.parquet` stimmt das gespeicherte `section_heading` Zeichen für
Zeichen mit dem überein, was das heutige Chunking für dieselbe ID liefert. Die
Chunk-Grenzen haben sich nicht bewegt.

### 4.2 Die neuen Punkte erben ihre Fragen

| | |
|---|---|
| Zusatzpunkte | 1.343 |
| davon unter 200 Zeichen (nie angereichert, `--min-chars`) | 219 |
| verbleibend | 1.124 |
| davon Fragen vom Gruppenvertreter erbbar | **1.123** (99,9 %) |
| neu zu erzeugen | **1** |

Der eine Ausreißer ist eine kaputt konvertierte PDF-Seite
(`05-26_Praktikum-bei-Airbus.pdf`, unlesbare Zeichenfolge); ihr Vertreter steht
mit `ok = False` in der Anreicherung, hat also selbst keine Fragen. Es geht
nichts verloren.

**Es ist keine GPU-Stunde nötig.** Die Mehrkosten sind 1,12 % Einbettungszeit
im Vektorisierer.

### 4.3 Seiten werden wieder zitierbar

```
Seiten ohne eigenen Punkt   heute 323 (5,8 %)  →  neu 298 (5,3 %)
```

25 Seiten kommen zurück, darunter genau die erwarteten:
`f2…/Ordnungen/01-ATPO_2015_mit_Anlagen_A1-A3.pdf`,
`f2…/Internationales/Engl._Lectures-FI-II_WS_26-27.pdf`,
`itr.wp…/Allgemeiner_Teil_der_PO_2015.pdf`.

---

## 5. Was auch danach offen bleibt

Der **weiche Schaden** wird von keinem der Wege außer B behoben:

```
Gruppen über mehr als eine source_url    29.258
betroffene Chunks                       102.533
```

Für diese Texte bleibt die Zitation eine willkürlich gewählte URL, und der
Leser erfährt nicht, dass dieselbe Aussage auch anderswo steht. Das ist
zehnmal häufiger als der harte Schaden — aber es ist ein **Payload-Problem,
kein Vektor-Problem**: Alias-Listen (Weg A) lassen sich später aus einem
Mapping `dedupe_key → alle Mitglieder` erzeugen, das auf dem Login-Node in
einer Minute entsteht, und beim Qdrant-Import als zusätzliche Felder
mitschreiben. Ohne die Vektoren anzufassen.

Deshalb ist A **kein Gegenspieler von D**, sondern ein späterer, rein additiver
Schritt. Jetzt zu entscheiden ist nur, **welche Punkte überhaupt existieren** —
das ist die einbettungsrelevante Frage.

---

## 6. Empfehlung

**Weg D-scharf, mit Fakultät und Organisation als Achsen.**

1. Er schließt den harten Schaden nachweislich vollständig (0 von 84.778
   Paaren fehlen), nicht überwiegend.
2. Er kostet 1.343 Punkte (+1,12 %) und **keine** GPU-Stunde für die
   Anreicherung — 1.123 der 1.124 anreicherbaren Zusatzpunkte erben ihre
   Fragen vom Gruppenvertreter.
3. Das Payload bleibt einwertig: ein Punkt, eine `faculty`, eine
   `source_url`. Importer, Filter und Zitation aus
   `Parquet-Schema_und_Qdrant-Import.md` bleiben, wie sie dokumentiert sind.
   Weg A machte `faculty` zur Liste und zöge `MatchAny` durch jeden künftigen
   Konsumenten.
4. Weg A bleibt danach möglich und kostet keine Neuberechnung — er ist die
   richtige Antwort auf den weichen Schaden, nur später.
5. Weg B kostet das 57-fache an Zusatzpunkten und bringt für die
   Filterbarkeit nichts, was D-scharf nicht auch bringt.

**Gegenrechnung, ehrlich:** 219 der Zusatzpunkte sind
Verkündungsblatt-Fragmente ohne Aussagewert, 424 sind Kopien auf einer
WordPress-Projektseite. Beides ließe sich mit einer Längenschwelle bzw. ohne
die Organisationsachse vermeiden (dann 1.124 bzw. 908 Zusatzpunkte). Ich rate
davon ab: Die Regel „jede vorhandene Zuordnung ist im Index vertreten" ist als
Invariante prüfbar — `messung_variante.py` prüft sie —, jede Ausnahme macht sie
zu einer Ermessensfrage. Der Preis der Reinheit sind 0,4 % Punkte.

---

## 7. Umsetzung nach Freigabe

| Schritt | Datei | Wirkung |
|---|---|---|
| 1 | `chunking.py` — `select_unique_ids()` auf die scharfe Regel, `DEDUPE_KEY_VERSION` → `…+fakultaet-organisation/v2` | 121.134 statt 119.791 IDs |
| 2 | `chunking.py` — Fingerabdruck trennen: `config_hash()` nur noch Chunk-Grenzen, neu `selection_hash()` für die Behalteregel | Grenzverschiebung bricht weiter hart ab; eine geänderte Behalteregel ist kein stiller Fehler mehr und muss den Join nicht sprengen |
| 3 | `enricher.py`, `hpc_vectorizer.py` | beide Hashes schreiben bzw. prüfen: Grenzen → Abbruch, Auswahl → Warnung mit Erklärung |
| 4 | neu `erbe_fragen.py` (Login-Node) | `enrichment_v2.parquet` = 110.333 alte Zeilen + 1.123 geerbte, mit Herkunftsvermerk; alte Datei nach `Backup/` |
| 5 | `konsistenz.py` erweitern | Obermengen-Eigenschaft und Paar-Vollständigkeit als Dauerprüfung |
| 6 | `Parquet-Schema_und_Qdrant-Import.md` | 121.134 Punkte, 298 statt 323 Seiten ohne Punkt, Hinweis auf die zweite Hash-Spalte |

Schritt 2 ist der Grund, warum das hier sauber geht: Der heutige `config_hash`
vermischt zwei Dinge — *wo die Chunk-Grenzen liegen* (verschieben sich die,
hängen Fragen an fremdem Text, das muss abbrechen) und *welche Chunks behalten
werden* (ändert sich das, bleiben alle IDs gültig, es kommen nur welche hinzu).
Getrennt braucht es kein `--ignore-enrichment-hash`, das den echten Schutz mit
abschaltet.

Nicht mehr gebrauchtes wandert nach `Backup/`, nichts wird gelöscht.

---

## 8. Umgesetzt am 14.08.2026

Freigegeben wurde D-scharf mit beiden Achsen. Was sich geändert hat:

| Datei | Änderung |
|---|---|
| `chunking.py` | `select_unique_ids()` behält je Gruppe zusätzlich einen Vertreter pro Fakultät und Organisation (`DEDUPE_AXES`); `config_hash()` deckt nur noch die Grenzen ab, neu `selection_hash()` für die Auswahlregel |
| `enricher.py` | schreibt beide Fingerabdrücke |
| `hpc_vectorizer.py` | schreibt beide; Grenzen-Abweichung bricht ab, Auswahl-Abweichung warnt mit Erklärung |
| `erbe_fragen.py` | **neu** — erzeugt `enrichment_v2.parquet` aus der bestehenden Anreicherung |
| `konsistenz.py` | prüft zusätzlich Vollständigkeit der Zuordnungen, Kosten und den Join gegen die Anreicherung |
| `vectorize.sbatch` | zeigt auf `enrichment_v2.parquet` |
| `Parquet-Schema_und_Qdrant-Import.md` | neue Punktzahl, neue Spalte, Filterzusage |

Neue Fingerabdrücke: Grenzen **`5d4df3ab6a51ec37`**, Auswahl
**`40fbfb36ee010cf8`**. Die alte `enrichment.parquet` trägt noch
`8e7e7ce74f631963` und wird vom Vektorisierer abgewiesen — die Meldung nennt
jetzt `enrichment_v2.parquet` als richtige Antwort, damit niemand daraus auf
verschobene Grenzen schließt und die Anreicherung neu rechnet.

**`erbe_fragen.py`, Produktionslauf** (Login-Node, 3 min, keine GPU):

```
Behaltemenge                      121.134 Punkte
Grenzen geprüft                   110.333 von 110.333 Zeilen mit
                                  identischem section_heading
Obermengen-Eigenschaft            alle 110.333 alten IDs weiterhin behalten
neue Punkte ohne eigene Zeile      10.801
  davon geerbt                      1.123
  ohne verwertbare Quelle           9.678   (9.458 alte + 219 neue Chunks
                                             unter --min-chars, + 1 kaputte
                                             PDF-Seite — alle hatten auch
                                             vorher keine Fragen)
geschrieben                       enrichment_v2.parquet, 111.456 Zeilen
```

**`konsistenz.py` nach der Änderung:**

```
1. Determinismus                                  True
2. Anreicherer == Vektorisierer                   True, 121.134 IDs
3. Teilmenge [Präfix, erste 1000]                 0 abweichende Vertreter
3. Teilmenge [jede dritte Datei]             11.298 abweichende Vertreter
5. (Gruppe, Zuordnung)-Paare        84.778 vorhanden, 84.778 vertreten, 0 fehlend
6. Zusatzpunkte                              1.343 (+1,12 %)
7. Join gegen enrichment_v2.parquet   Trefferquote 92,0 %, 0 IDs zu viel
```

Der zweite Teilmengen-Test ist neu. Vorher wurde nur ein Präfix der sortierten
Dateiliste geprüft — bei dem kann per Konstruktion nichts abweichen, der Test
war dekorativ. `--urls` und `--shard` wählen aber beliebige Dateien, und dann
sind es 11.298 andere Vertreter. Die Reihenfolge in beiden Skripten (erst
`select_unique_ids()` über alles, dann filtern) ist damit belegt nötig, nicht
nur behauptet.

### Was noch zu tun ist

1. `sbatch vectorize.sbatch` — der Produktionslauf über 121.134 Punkte.
2. `filter_fragen.py` ist weiterhin nicht angewandt (offener Punkt aus der
   Übergabe, Abschnitt 8). Es wirkt auf `questions`, nicht auf die Punktmenge,
   und lässt sich vor der Vektorisierung noch entscheiden.
3. Alias-Listen (Weg A) für den weichen Schaden — später, beim Import,
   ohne Neuberechnung.
