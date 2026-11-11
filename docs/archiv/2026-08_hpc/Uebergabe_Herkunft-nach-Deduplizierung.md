# Übergabe — offener Punkt: Herkunft nach der Deduplizierung

**Datum:** 14. August 2026
**Vorgänger:** `Sitzungsbericht_2026-08-13.md`, `Parquet-Schema_und_Qdrant-Import.md`
**Zustand:** Anreicherung produktiv abgeschlossen, Vektorisierung steht aus

> **ERLEDIGT am 14.08.2026 — dieses Dokument ist Vorgeschichte.**
> Entschieden und umgesetzt wurde Weg D in einer schärferen Fassung: je Gruppe
> ein Vertreter pro vorkommender Fakultät **und Organisation**, +1.343 Punkte
> (+1,12 %), ohne neue GPU-Rechnung. Messungen, Bewertung aller vier Wege und
> Umsetzung stehen in **`Messung_Herkunft_2026-08-14.md`**.
>
> Drei Zahlen in Abschnitt 3 unten haben der Nachprüfung nicht standgehalten:
> Es sind 684 statt 465 betroffene Gruppen, der harte Schaden sind 908
> verlorene (Gruppe, Fakultät)-Paare statt 1.781 Chunk-Kopien, und die
> Organisationsachse ist mit 435 Paaren **doch** betroffen. Begründung im
> neuen Dokument, Abschnitt 1.

---

## 1. Wo wir stehen

| Schritt | Zustand |
|---|---|
| vLLM auf KISSKI | läuft, 18,5 Chunks/s |
| Deduplizierung | eingebaut, `config_hash = 8e7e7ce74f631963` |
| Anreicherung (Produktionslauf) | **fertig** — `enrichment.parquet`, 110.333 Zeilen, 0,04 % ohne Fragen |
| `filter_fragen.py` | geprüft, noch **nicht** angewandt (81,7 % würden bleiben) |
| Vektorisierung | **steht aus** |
| Qdrant-Import | steht aus |

Alles Weitere in den beiden Vorgängerdokumenten. **Dieses Dokument beschreibt
nur den einen offenen Punkt, der vor der Vektorisierung zu entscheiden ist.**

---

## 2. Das Problem

Die Deduplizierung behält je Gruppe wortgleicher Chunks **einen** Vertreter.
Die anderen verschwinden — samt ihrer `source_url`, `faculty` und
`organisation`.

Der Fall, an dem es auffällt:

> Jemand aus Fakultät I sucht die Dauer der Praxisphase. Der Treffer stammt aus
> einem Dokument der Fakultät V. Der Text ist wortgleich, die Antwort also
> sachlich richtig — aber die suchende Person kann nicht erkennen, ob die
> Regelung für ihre Fakultät gilt, und müsste es selbst nachschlagen.

Zwei getrennte Schäden:

**Hart — der Fakultätsfilter greift ins Leere.** Ein Filter
`faculty = "Fakultät I"` findet nur Chunks, deren *Vertreter* zu Fakultät I
gehört. Inhalte, die in einem Dokument der Fakultät I wortgleich existieren,
aber von einer anderen Fakultät vertreten werden, sind für den Filter
unsichtbar.

**Weich — die Quellenangabe ist willkürlich.** Zitiert wird die Seite, die in
Dateireihenfolge zuerst kam. Für den Leser ist nicht erkennbar, dass dieselbe
Aussage auch anderswo steht.

### Warum es entstanden ist

Die Alias-URLs waren früh vorgeschlagen („je Gruppe eine behalten, die
Alias-URLs ins Payload schreiben"). Die Entscheidung, sie fallenzulassen,
stützte sich auf die Annahme, es sei „beim RAG letztlich egal, auf welcher
Seite man die Information findet, wenn sie exakt gleich ist". Für die
**Richtigkeit** der Antwort stimmt das. Für **Geltungsbereich** und
**Filterbarkeit** nicht.

---

## 3. Das Ausmaß — gemessen

```
Chunks gesamt                        196.934
Gruppen gesamt                       119.791
geteilte Gruppen                      31.708

Gruppen über >1 Fakultät                 465   (1,5 % der geteilten)
Gruppen über >1 Organisation               0
Chunks, deren Fakultät verschwindet    1.781   (0,9 % des Bestands)
```

Verlust je Fakultät:

| Fakultät | Bestand | fremdvertreten | Anteil |
|---|---|---|---|
| Fakultät V | 61.684 | 1.233 | 2,0 % |
| Fakultät II | 20.366 | 392 | 1,9 % |
| Fakultät IV | 18.308 | 221 | 1,2 % |
| Fakultät III | 12.826 | 143 | 1,1 % |
| Fakultät I | 13.634 | 11 | 0,1 % |

Häufigste Kombinationen — es sind die gespiegelten Ordnungen:

```
133x  Fakultät I + Fakultät II + Fakultät V
 91x  Fakultät I + Fakultät II
 64x  Fakultät I + Fakultät IV
```

**Einordnung:** Unter 1 % des Bestands. Aber die betroffene Klasse ist
ausgerechnet die regulatorische — Prüfungsordnungen, Modulhandbücher,
Praxisphasen. Genau dort ist „gilt das für mich?" die entscheidende Frage.
Die Zahl ist klein, die Relevanz je Treffer hoch.

Dass **keine** Gruppe mehrere Organisationen umspannt, ist ein Nebenbefund: Das
Problem betrifft ausschließlich die Fakultätsachse.

---

## 4. Zwei Lösungswege

### Weg A — Deduplizierung behalten, Herkunft als Liste ins Payload

Zusätzliche Felder je Punkt:

```
source_urls    list<string>   alle URLs der Gruppe
faculties      list<string>   alle Fakultäten der Gruppe
organisations  list<string>
```

Der Filter wird zum Array-Enthält-Filter (`MatchAny` über `faculties`), die
Zitation kann alle Fundstellen nennen.

* **Vorteil:** Indexgröße bleibt bei 119.791 Punkten.
* **Nachteil:** Filterlogik und Importer müssen umgestellt werden; das
  Einzelfeld `faculty` verliert seine Bedeutung.

### Weg B — zurück zum Fan-out

Alle 196.934 Chunks kommen in den Index; die Fragen werden **einmal** erzeugt
und auf alle Gruppenmitglieder verteilt.

* **Vorteil:** `faculty`, `organisation` und `source_url` bleiben je Punkt
  korrekt. Filter und Zitation funktionieren ohne Sonderlogik. Das war der
  ursprüngliche Vorschlag.
* **Nachteil:** 64 % mehr Punkte im Index (196.934 statt 119.791), 64 % mehr
  Embedding-Rechenzeit. Nahezu identische Vektoren konkurrieren wieder um
  Trefferplätze.

### Was in **beiden** Fällen gilt

**Die Anreicherung muss nicht wiederholt werden.** `enrichment.parquet` ist in
beiden Wegen verwendbar:

* Weg A braucht sie unverändert.
* Weg B braucht eine Abbildung `dedupe_key → Vertreter-ID`, um die Fragen des
  Vertreters auf die Gruppenmitglieder zu verteilen. `select_unique_ids()`
  berechnet die Gruppen ohnehin und müsste statt einer Menge ein Mapping
  liefern.

**Beides sind Änderungen am Vektorisierer, nicht am Anreicherer.** Die
Produktionsstunde von gestern ist in keinem Fall verloren.

---

## 5. Weitere Varianten, die zu prüfen wären

* **Weg C — nur fakultätsübergreifende Gruppen ausnehmen.** Deduplizieren wie
  bisher, aber Gruppen mit mehr als einer Fakultät unangetastet lassen. Kostet
  1.781 zusätzliche Punkte (+0,9 %) und löst den harten Schaden vollständig.
  Der weiche Schaden bleibt für die restlichen Gruppen.
* **Weg D — `faculty` in den Dedup-Schlüssel aufnehmen.** Wie C, nur
  allgemeiner formuliert: `dedupe_key = (Text, aussagekräftiger Titel,
  Fakultät)`. Simpel, weil es nur eine Zeile in `dedupe_key()` ist. Wirkung
  müsste gemessen werden.

Weg D ist vermutlich der beste Kompromiss und war bisher nicht betrachtet.

---

## 6. Vor der Entscheidung zu klären

1. Soll der Fakultätsfilter überhaupt eine harte Einschränkung sein, oder eher
   eine Gewichtung? Bei einer Gewichtung ist Weg A unproblematisch.
2. Wie oft kommen fakultätsgebundene Fragen im echten Betrieb vor? Ohne
   Goldset ist das nicht zu beantworten — es ist derselbe offene Punkt 3 aus
   dem Sitzungsbericht.
3. Ist die Indexgröße überhaupt eine Einschränkung? 196.934 Punkte à 1024 Dim.
   sind rund 800 MB Dense-Vektoren. Wenn das lokale Qdrant das trägt, verliert
   Weg B seinen Hauptnachteil.

---

## 7. Dateien

| Datei | Inhalt |
|---|---|
| `Sitzungsbericht_2026-08-13.md` | vLLM, Deduplizierung, Messungen, Betriebsablauf |
| `Parquet-Schema_und_Qdrant-Import.md` | Schema, Collection, Import, Query-Seite |
| `enrichment.parquet` | 110.333 Zeilen, fertig |
| `Backup/analyse_skripte/fakultaet_verlust.py` | erzeugt die Zahlen aus Abschnitt 3 |
| `Backup/analyse_skripte/delta_risiko.py` | Verflechtung der Seiten (43,4 %) |
| `Backup/analyse_skripte/abnahme.py` | Abnahmeprüfung der Anreicherung |

`chunking.py` enthält `dedupe_key()`, `informative_title()` und
`select_unique_ids()` — dort liegt der Hebel für alle vier Wege.

---

## 8. Nicht vergessen

* `filter_fragen.py` ist noch nicht angewandt (22 % wortgleiche Fragen aus
  PDF-Modulhandbüchern).
* Dessen Wortregel erkennt Abkürzungen (`ISO`, `PO`, `SWS`) und alleinstehende
  Zahlen nicht und verwirft dadurch 2.477 spezifische Fragen.
* Nachcrawls sind konzeptionell ungelöst — der Anreicherer hat keine
  Delta-Fähigkeit, und die Deduplizierung verflicht 43,4 % der Seiten.
