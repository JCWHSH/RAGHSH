# eval_gold — Diagnostische Auswertung des RAG

Misst nicht *ob* das RAG richtig lag, sondern **wo** es falsch abbog.

## Warum ein zweites Eval-Verzeichnis

`eval/run_rag.py` prüft das Golddokument gegen die Quellenliste — und die
entsteht **nach** der LLM-Dokumentauswahl. Eine Trefferquote von dort sagt nur
„gefunden UND ausgewählt". Sie kann nicht unterscheiden, ob ein Fehlschlag am
Retrieval lag oder daran, dass Runde 1 das richtige Dokument in der Hand hatte
und es verwarf. Das sind zwei völlig verschiedene Reparaturen.

Der Aufbau folgt dem Validierungssystem von Mohamed Salem Charghy und Taleb
El Beidawi ([Repository](https://github.com/MohamedSalemCharghy/RAGHSH),
Studienprojekt Fakultät I, Mai 2026), das denselben Gedanken mit zwei
LLM-Judges umsetzt. Hier ist der deterministische Teil davon gebaut: Er kostet
nichts, rauscht nicht und ist reproduzierbar. Ein Judge kann später aufsetzen.

## Was wo liegt

```
auswertung/gold/
├── baue_fallbasis.py      erzeugt den Vertrag
├── fallbasis.json         70 Fälle — was gefragt wird
├── lauf.py                sammelt Rohdaten gegen den MCP-Server
├── vergleich.py           toleranter Begriffsvergleich (einzige Stelle dafür)
├── test_vergleich.py      dessen Tests — pytest auswertung/gold/test_vergleich.py
├── auswertung.py          diagnostiziert, deterministisch
├── bericht.py             HTML-Bericht eines Volllaufs, CLI für alle drei Berichte
├── bericht_retrieval.py   HTML-Bericht eines Retrieval-Laufs (Ränge, Arme)
├── bericht_vergleich.py   HTML-Vergleich zweier Läufe (B gegen A)
├── bericht_teile.py       gemeinsame Bausteine der drei Berichte
├── bericht.css            deren Stylesheet
├── begriffe_vorschlag.py  Begriffsvorschläge für g01–g50, --pruefe für die kuratierte Datei
├── begriffe_g.json        kuratierte Pflichtbegriffe g01–g50 (liest der Builder)
├── vertrag.py             Fingerabdruck der Fallbasis (fallbasis.sha256)
├── fallbasis_excel.py     fallbasis.json ⇄ fallbasis.xlsx — die volle Sicht, zum Kuratieren
├── beitraege.py           Vorlage, Prüfbericht und Übernahme neuer Fälle von Beitragenden
└── laeufe/
    ├── <lauf-id>/
    │   ├── manifest.json      Beleg: Konfiguration, gegen die gemessen wurde
    │   ├── faelle/qXX.json    Beleg: Rohdaten je Fall
    │   ├── lauf.log           Erzeugnis: Protokoll des Durchlaufs
    │   ├── zusammenfassung.json  Erzeugnis: maschinenlesbarer Export
    │   ├── bericht.html       Erzeugnis: die lesbare Auswertung
    │   └── vergleich_gegen_<lauf-a>.html   Erzeugnis: dieser Lauf gegen einen anderen
    └── archiv/<lauf-id>/      überholte Läufe, aus der Übersicht genommen
```

**Beleg bleibt, Erzeugnis darf jederzeit neu entstehen.** `manifest.json` und
`faelle/` werden nach dem Lauf nicht mehr angefasst — sie sind der Beweis. Alles
andere im Laufordner lässt sich aus ihnen wieder herstellen.

`bericht.html` und `vergleich_gegen_*.html` stehen in `.gitignore` — sie
entstehen in Sekunden neu. Die Rohdaten daneben bleiben versioniert.

### Archiv

`laeufe/archiv/` nimmt Läufe auf, die nicht mehr in die Übersicht gehören:
Probeläufe, Abnahmen, und Läufe, deren Fallbasis inzwischen eine andere ist.
Verschieben genügt — Auswertung und Bericht übergehen `archiv/` bei der
Laufsuche und bei `--letzter`, lesen einen Lauf dort aber weiterhin, wenn man
ihn ausdrücklich nennt:

```bash
mv auswertung/laeufe/<lauf-id> auswertung/laeufe/archiv/
python auswertung/gold/auswertung.py archiv/<lauf-id>
```

Gelöscht wird nichts. Ein Lauf ist ein Messergebnis; wenn er nicht mehr gilt,
gehört der Grund dazu und nicht der Papierkorb.

`zusammenfassung.json` liest kein Code dieses Projekts; sie existiert für die
Weitergabe, damit jemand unsere Zahlen nachrechnen kann, ohne unseren Code
auszuführen. Das Feld `stand` hält Zeitpunkt und Git-Commit der Auswertung fest
— ändert sich die Auswertungslogik, ist eine ältere Datei daran erkennbar.

## Die Fallbasis

`fallbasis.json`, 70 Fälle (Stand 11.11.2026; die Läufe vom September maßen noch 75 — Vertrag in `fallbasis.sha256`):

| Herkunft | Anzahl | Quelle |
|---|---|---|
| `chargy_beidawi` | 17 | ihre `validation_cases.json`, Felder unverändert übernommen (q16 gestrichen, s. u.) |
| `nutzer_gold` | 8 | unsere `auswertung/fragen/fragen_nutzer.csv` (nicht veröffentlicht) (echte Nutzerfragen — nicht öffentlich) |
| `eval_gold_50` | 45 | `auswertung/fragen/fragen.csv` (Frage, Goldquelle, Belegstelle, Referenzantwort); Pflichtbegriffe aus `begriffe_g.json` (s. u.) |

Ihre dreizehn Vertragsfelder behalten ihre Originalnamen (`question`,
`reference_answer`, `source_url`, `accepted_source_urls`, `source_policy`,
`required_facts`, `optional_facts`, `forbidden_claims`, `case_type`,
`expected_behavior`, `case_difficulty`, `fairness_risk`, `evaluation_notes`).
Vier Felder kommen dazu:

| Feld | Bedeutung |
|---|---|
| `herkunft` | woher der Fall stammt |
| `gold_im_korpus` | liegt die Goldquelle überhaupt in unserem Index? |
| `pflicht_begriffe` | kurze **wörtliche** Zeichenketten für die maschinelle Prüfung |
| `fakultaet` | Filter, den der Lauf setzt |

Bei drei präzisierten Fragen (siehe unten) kommen `question_original` und
`anpassung` hinzu; bei nachgeprüften Quellen (siehe unten) `quellen_geprueft`,
`quellen_grund`, ggf. `source_url_original`, `source_policy_original` und
`vertreter_von`.

`required_facts` sind ganze Sätze und maschinell nicht prüfbar — eine korrekte
Antwort formuliert anders. `pflicht_begriffe` ist die prüfbare Kurzform
desselben Fakts. Beide stehen nebeneinander: die Sätze für einen späteren
Judge, die Begriffe für heute. **Eine leere Begriffsliste bedeutet „nicht
prüfbar", nicht „erfüllt"** (betrifft q18, wo mehrere offizielle Zahlen
gleichermaßen gültig sind). Ein Eintrag ist ein String oder eine Liste von
Alternativen, von denen eine genügt (q05: „Prüfungsverwaltung" oder
„Anmeldung Studienprojekt").

### Pflichtbegriffe für g01–g50 (22.09.2026)

Die 50 Fälle aus `eval/fragen.csv` kamen am 21.09. ohne Begriffe in die
Fallbasis — für Retrieval-Experimente reichte die Quellenlage. Im Volllauf
hieß das aber: 16 Fälle „nicht prüfbar", und bei den übrigen 34 bedeutete
„Treffer" nur „richtiges Dokument im Kontext". Zugleich zeigte P1 gegen V2,
dass die Antwortrunde bei gleichen Quellen um 3 von 24 prüfbaren Fällen
streut — mit 24 Fällen ist das nicht von Wirkung zu trennen.

`begriffe_vorschlag.py` erzeugt aus **Belegstelle** (der Passage, die beim
Anlegen der Frage aus dem Dokument kopiert wurde), Referenzantwort (Muster:
Raum, Datum, E-Mail, Menge, Name) und den **gemeinsamen Wortfolgen** beider
je Fall Kandidaten und prüft sie nach der Selbsttest-Regel — in der
Referenzantwort *und* im Goldtext, mit dem toleranten Vergleich. Das
Ergebnis (`begriffe_g_vorschlag.md`) ist Arbeitsmaterial; die kuratierte
Auswahl steht in **`begriffe_g.json`** und wird vom Builder gelesen
(`--pruefe` prüft sie vor dem Bau). Regeln wie im übrigen Vertrag: ein bis
drei Begriffe, so eng wie die Frage und nicht enger — g47 fragt nach dem
Einzelfach, also „Mathematik", nicht auch „40 Prozent"; wo die Sprache
variiert, Alternativen („amtsärztlich" statt „amtsärztliches Attest",
„Multicopter | MAB-279"). Nur g50 steht im Gold als nackte Tabellenzelle
„9"; der Begriff verlangt eine Einheit und bleibt eine Warnung.

Nebenbei fanden sich beim Prüfen vier Lücken in `vergleich.py`, die auch
laufende Bewertungen betrafen: Raumnummern vor dem Satzpunkt („Raum
2A.1.11.") wurden zum Datum, `<br>` aus Tabellen-PDFs klebte am Token,
`|` trennte keine Zellen, „40%" traf „40 Prozent" nicht. Alle vier sind
behoben und getestet (52 Tests); die Bewertungen von V0–P1 unter ihrem
Lauf-Vertrag änderten sich dadurch nicht.

**Vertrag 529351a → 8443370.** Unter dem neuen Vertrag (`--vertrag aktuell`),
73 mögliche Fälle, `unbestimmt 0`:

| Lauf | Treffer | davon aus Gold | Antwortproblem | Retrieval |
|---|---|---|---|---|
| V0 alte Produktion | 54 | 48 | 3 | 17 |
| V1 t+q td12 | 58 | 53 | — | — |
| V2 Wiederholung | 59 | 53 | 6 | 10 |
| **P1 neue Produktion** | **57** | 51 | 7 | 10 |

Ab jetzt gilt: Ein Unterschied von drei Treffern zwischen zwei Läufen ist
Rauschen der Antwortrunde; die Suche (Kandidaten, Quellen) bleibt das
stabile Maß.

### Neue Fälle beitragen

Wer eine Frage beisteuert, weiß drei Dinge: die Frage, die richtige Antwort,
und wo sie steht. Alles andere im Vertrag ist entweder ableitbar oder eine
Kuratierungsentscheidung — und die soll nicht beim Beitragenden liegen. Deshalb
zwei Tabellen:

- **`beitraege.xlsx`** (aus `python auswertung/gold/beitraege.py --vorlage`): fünf
  Spalten, drei Pflicht — Frage, Antwort, Quelle; dazu „Wörtlich erwartet“
  (wird `pflicht_begriffe`) und „Anmerkung“ (wird `evaluation_notes`). Keine
  Schema-Begriffe, keine Dropdowns. Eine Datei je Person oder ein Kürzel in
  „Von“ — daraus wird `herkunft = beitrag_<kürzel>`.
- **`fallbasis.xlsx`** (aus `python auswertung/gold/fallbasis_excel.py`): alle
  Felder, eine Spalte je Feld, Listen mit einem Eintrag je Zeile. Zum
  Kuratieren; `--nach-json` schreibt zurück und erneuert den Hash.

```bash
python auswertung/gold/beitraege.py beitraege_cw.xlsx                # Prüfbericht, schreibt nichts
python auswertung/gold/beitraege.py beitraege_cw.xlsx --uebernehmen  # hängt an fallbasis.json an
```

Der Import leitet ab, was das System besser weiß als der Mensch:
`gold_im_korpus` gegen `doc_index.db`, Vertreter-URLs aus `chunk_vertretung`,
`source_policy` aus der Zahl der Quellen, die `id` fortlaufend (`bNN`). Die
Kuratierungsfelder bekommen Defaults (`case_difficulty=mittel`,
`fairness_risk=niedrig`, `required_facts=[Antwort]`); `quellen_geprueft`
bleibt leer, weil Beitragen keine Prüfung ist.

Der Prüfbericht kennt drei Stufen. **FEHLER** blockiert die Übernahme: Pflicht-
feld leer, Frage schon vorhanden, „Wörtlich erwartet“ steht nicht in der
eigenen Antwort. **warn** ist eine Entscheidung für die Sichtung: Quelle nicht
im Korpus (der Fall misst dann „Gold fehlt“, nicht das Retrieval — das kann
gewollt sein), Quelle trägt den Begriff nicht, Frage wiederholt die Antwort
(Stichwortsuche genügt). **hinweis** ist Information: ähnliche Frage vorhanden,
kein Begriff (nur die Quelle prüfbar), Vertreter-URL ergänzt.

Nach der Übernahme ist der Vertrag ein anderer — `lauf.py --fortsetzen`
verweigert dann zu Recht, und ein Vergleich mit älteren Läufen muss die Fälle
nach `herkunft` trennen.

### Der Begriffsvergleich

`vergleich.py` ist die einzige Stelle, die entscheidet, ob ein Begriff in einem
Text steht — Auswertung und Selbsttest des Builders benutzen dieselbe
Funktion. Er toleriert Zahlwörter („vier" ≡ „4"), Datumsformen („1. September
2026" ≡ „01.09.2026"), Interpunktion am Tokenrand („01.01." ≡ „01.01"),
Bindestrich-Komposita, Flexion per Präfix („Zentrums" für „Zentrum"),
„(at)" für „@" und bis zu drei eingeschobene Wörter. Er toleriert nicht:
andere Zahlen, vertauschte Reihenfolge, große Abstände. Die Grenzen stehen in
`test_vergleich.py` als Positiv- und Negativfälle — die Positivfälle sind
wörtlich die Fehlbewertungen des Laufs vom 18.08.2026.

Vor diesem Modul verglich die Auswertung wörtlich (`begriff in text`). Im
Lauf vom 18.08. waren dadurch alle vier ausgewiesenen „Antwortprobleme"
Messfehler: `01.01.` gegen „01.01 - 31.01.", `11 Bachelor` gegen „11
praxisnahe Bachelor-Studiengänge", `support-it@` gegen „Support-it(at)".

### Drei präzisierte Fragen

Drei Fragen waren so gestellt, dass keine Quelle sie eindeutig beantworten
kann — gemessen wurde damit nicht das System, sondern das Raten der
wahrscheinlichsten Lesart:

| Fall | vorher | jetzt |
|---|---|---|
| n02 | [Nutzerfrage – nicht veröffentlicht] | … **an der Fakultät I**? |
| n03 | [Nutzerfrage – nicht veröffentlicht] | … **an der Fakultät I**? |
| q04 | Wie stelle ich einen Online-Antrag? | … **für die Einschreibung**? |

Die Vorlesungszeiten gelten je Fakultät, die Praxisphasenordnung ebenso; die
Goldquellen sind in beiden Fällen die für Fakultät I gültigen. Bei q04 war die
Mehrdeutigkeit zwischen Bewerbung und Antrag eingeschriebener Studierender bei
Charghy und El Beidawi der Prüfgegenstand — mit der Präzisierung entfällt sie,
deshalb verlangt der Vertrag dort keine Rückfrage mehr (`case_type` und
`expected_behavior` sind nachgezogen, `clarification` entfernt).

Die Originalfragen bleiben als `question_original` erhalten und stehen in der
Einzelfallkarte des Berichts. **Läufe von vor und nach dieser Änderung sind
für diese drei Fälle nicht vergleichbar**, ebensowenig q04 mit ihrem Lauf.
`eval/fragen_nutzer.csv` bleibt unverändert; die Präzisierung steht in
`PRAEZISIERT` in `baue_fallbasis.py`.

Neu aufbauen:

```bash
python auswertung/gold/baue_fallbasis.py --quelle auswertung/gold/quelle_chargy_beidawi_validation_cases.json
```

### Quellen nachgeprüft (14.09.2026)

Alle 26 Referenzantworten wurden gegen die Live-Seiten der HsH geprüft — kein
Fakt hat sich geändert. Zugleich wurde je Fall gesucht, welche weiteren
HsH-Dokumente dieselbe Antwort tragen (Web und Korpus-Volltext). Das Ergebnis
steht in `WEITERE_QUELLEN` in `baue_fallbasis.py`, je Fall mit Begründung:
18 Fälle haben jetzt akzeptierte Zweitquellen, `source_policy` wechselt dort
auf `accepted_sources`. Warum das nötig war: Charghy/Beidawi kennen fast
überall nur eine Goldquelle, dieselben Fakten stehen an der HsH aber regelmäßig
auf mehreren offiziellen Seiten. q09 (IT-ServiceDesk) wurde vollständig
richtig beantwortet und trotzdem als Korpuslücke gestrichen — die gecrawlte
Unterseite war schlicht nicht als Quelle eingetragen.

Zwei inhaltliche Korrekturen:

- **n03**: Goldquelle von der Praxisphasenordnung 2012 auf die 2. Änderung
  (Verkündungsblatt 07/2024) umgestellt. Dauer weiterhin 10 Wochen; „18
  Credits" steht in der neuen Fassung nicht mehr und ist kein Pflichtbegriff.
- **q16 gestrichen** (15.09.2026): `f3.hs-hannover.de/` sagt live 15
  Bachelorstudiengänge, `f3.hs-hannover.de/studium` sagt live 14. Die Website
  widerspricht sich selbst; der Fall misst dann nicht das RAG, sondern welche
  Seite es zuerst liest. Bleibt in der Quelldatei, steht in `AUSGESCHLOSSEN`;
  `--vertrag aktuell` lässt ihn bei alten Läufen weg.

**Vertreter-URLs.** Die Deduplizierung beim Import behält je Gruppe
wortgleicher Chunks einen Vertreter; 310 Dokumente haben keinen eigenen Punkt
im Index und erscheinen im Retrieval nur unter der URL ihres Vertreters
(`chunk_vertretung` in `doc_index.db`). `baue_fallbasis.py` trägt deshalb
Hauptvertreter (≥ 50 % der Positionen) automatisch als akzeptierte Quelle ein
und prüft `gold_im_korpus` über sie.

**Selbsttest.** Der Bau prüft den Vertrag gegen sich selbst und den Korpus:
Enthält die Referenzantwort ihre Pflichtbegriffe? Liegt jede Quelle im Korpus
(oder ist vertreten)? Trägt ihr Text die Pflichtbegriffe? Verstöße werden
gemeldet; fünf „nicht im Korpus" sind bewusst (Live-Quellen als
Crawl-Kandidaten, kommentiert im Code).

### Zwei Fälle sind nicht gewinnbar

Bei q10 und q20 fehlt die Goldquelle in unserem Korpus. Sie bleiben in der
Fallbasis, werden aber als `korpus_luecke` ausgewiesen und aus der Quote
herausgerechnet. Bemerkenswert: Dieselben Fälle stehen in ihrem Bericht unter
„Retrieval debuggen" — das ist die falsche Ebene, die Dokumente wurden in
beiden Projekten nie gecrawlt. (q09 stand bis 14.09.2026 ebenfalls hier — zu
Unrecht, siehe oben.)

## Läufe

```bash
python auswertung/gold/lauf.py --modell meta-llama-3.1-8b-instruct --auswerten --bericht
python auswertung/gold/lauf.py --faelle chargy_beidawi --limit 5
python auswertung/gold/lauf.py --faelle q06,q11,q19
python auswertung/gold/lauf.py --fortsetzen laeufe/2026-08-17_1637_meta-llama-3.1-8b-instruct
python auswertung/gold/lauf.py --nur-retrieval --auswerten          # kein LLM, ~70 s für 25 Fälle
```

### Zwei Modi: Volllauf und Retrieval-only

`--nur-retrieval` ruft `frag_hsh` mit `nur_retrieval=true` auf: Der Server
führt Hybrid-Suche und Kandidatenbildung aus — **denselben Code wie im
Volllauf**, dafür wurden die Schritte 1–2 in `_kandidaten_bestimmen()`
herausgelöst — und kehrt vor der LLM-Auswahl zurück. 25 Fälle brauchen 72
Sekunden statt fünf Minuten, ohne einen einzigen GWDG-Aufruf.

Warum durch den Server und nicht direkt über `hybrid_search`: Die
Kandidatenliste entsteht erst aus Fakultätsarmen, Dedup auf Dokumente,
Keyword-Boost und Fakultäts-Sicherheitsnetz. Eine Nachbildung sähe die
Dedup/Limit-Verluste nicht — n02 (Chunk-Rang 13) und q04 (Chunk-Rang 20)
gehen genau dort verloren. Und ein zweiter Prozess müsste Jina, BM25 und den
Reranker ein zweites Mal in den RAM laden.

Ein Retrieval-Lauf steht im Manifest als `modus: retrieval`, heißt
`<datum>_retrieval`, hat kein Modell und keine Antworten. Die Auswertung
zeigt den Trichter bis „in der Kandidatenliste" und dann, **wo das Gold
steht** (Rang 1 / 2–3 / 4–8 / > 8, Median), statt der Quadranten; Diagnosen
sind `gefunden`, `kandidatenlimit` (unter den Chunks, aber nicht in der
Liste), `retrieval_fehler`, `korpus_luecke`. `--vergleich` verweigert den
Vergleich von Retrieval- mit Volllauf und zählt innerhalb des Modus
„neu in Kandidaten" statt „neu getroffen", dazu Rangverschiebungen.
`bericht.py` erkennt den Modus am Manifest und erzeugt für Retrieval-Läufe den
Bericht aus `bericht_retrieval.py` (siehe „Die HTML-Berichte").

Der erste Retrieval-Lauf (21.09.2026, `2026-09-21_1118_retrieval`) liefert
exakt die Retrieval-Stufe des Volllaufs vom 18.08.: 17/23 in der Kandidaten-
liste, Median Rang 1, dieselben sechs Verluste.

### Experimente: Suchkonfiguration je Lauf

Seit 21.09.2026 lässt sich die Suche je Aufruf umkonfigurieren, ohne
Serverneustart — `frag_hsh(experiment={…})`, aus `lauf.py`:

```bash
python auswertung/gold/lauf.py --nur-retrieval --dense text,questions --fusion rrf
python auswertung/gold/lauf.py --nur-retrieval --dense text --sparse ""        # ein Arm allein
python auswertung/gold/lauf.py --nur-retrieval --dedup-buffer 8 --top-docs 12
python auswertung/gold/lauf.py --nur-retrieval --kein-boost
```

`--dense` aus `text`, `questions`; `--sparse` aus `sparse`, `sparse_expanded`
(leer = keiner, mindestens ein Arm insgesamt); `--fusion dbsf|rrf`;
`--candidate-limit`, `--dedup-buffer`, `--max-per-url`, `--top-docs`,
`--kein-boost`. Ungültige Werte scheitern vor dem ersten Fall — im Client
mit derselben Prüfung wie im Server (`hybrid_search.wirksame_konfiguration`).

**Das Manifest sagt, was wirksam war, nicht was in den Konstanten steht.**
Der Server meldet die wirksame Konfiguration im Ergebnis zurück; `lauf.py`
schreibt sie nach dem ersten Fall ins Manifest (`_wirksam: true`). Anlass:
Bis zum 21.09. stand in jedem Manifest `USE_RERANKER True`, weil `lauf.py`
die Modulkonstante ablas — der Server hat aber nie einen Reranker
aufgerufen. Alle Läufe waren reine Fusion. Die Konstante steht jetzt auf
`False`, das Manifest trägt `reranker_aktiv`. Läufe von vor dem 21.09. zeigt
die Auswertung mit „Reranker unbekannt" und dem Hinweis.

Die Lauf-ID trägt ein Kürzel des Experiments (`t+q_se_rrf`, `db8_td12`,
`noboost`), die Übersicht eine Spalte `Suche` (`t_se/dbsf`).

**Arm-Diagnose.** Im Retrieval-Modus stellt der Server jede Frage zusätzlich
jedem der vier Arme einzeln (Top-50, ohne Fusion; `arme` in den Rohdaten,
~1 s je Fall). Die Auswertung zeigt daraus je Fall den Goldrang je Arm und
je Arm: in Top-10, in Top-50, als einziger Arm. Dazu zwei Listen — Fälle, die
kein Arm sieht (dort hilft keine Fusion), und Fälle, die ein Arm in seinen
Top-10 hatte, die Kandidatenliste aber nicht (dort hat die Fusion oder die
Kandidatenstufe verloren).

**Was nicht als Testfragen taugt:** die doc2query-Fragen aus dem Index — sie
*sind* der `questions`-Arm — und das RAGAS-Testset (LLM-generierte Fragen aus
Dokumenten, dieselbe Verzerrung). Die 50 handgebauten Fragen mit
Überlappungsmedian 0,000 sind der Prüfstein.

### Punktgenaue Ziele: `gold_punkte`, `beleg_punkte`, `beleg_urls`

Die Deduplizierung verteilt den Text eines Dokuments über mehrere URLs, und
die Vertreter-Regel (≥ 50 %) greift nicht, wenn die Vertretung zerstreut ist.
Fall g46: Die Richtlinie zu Lehraufträgen 2024 hat 17 von 18 Chunks an vier
Dokumente abgegeben (Verkündungsblatt 10/2024, Revision 2025, zwei
Infopakete); keines erreichte die Hälfte, also gab es keine Zweitquelle. Die
Suche fand den Text auf Rang 1 und 3 — unter fremden URLs — und die
Auswertung zählte „kein Arm sieht das Gold".

Seit 21.09.2026 trägt jeder Fall deshalb (aus `chunk_vertretung` und Qdrant,
im Builder berechnet): `gold_punkte` — alle Punkt-IDs, die eine Position der
Gold- oder akzeptierten Quellen vertreten; `beleg_punkte` — davon die, deren
Text einen `required_fact` enthält (toleranter Vergleich); `beleg_urls` — die
URLs, unter denen diese Belegpunkte im Index stehen. Die Auswertung zählt
einen **Chunk-Treffer über die Punkt-ID** und eine **Kandidaten-URL, wenn sie
einen Belegpunkt trägt** — nicht, wenn sie nur eine Kopfzeile vertritt. Wo
keine Belegstelle gefunden wird (Charghy/Beidawi-Fälle mit frei formulierten
`required_facts`), gilt die URL-Regel wie zuvor. B0 steigt damit von 46 auf
47/73 (g46); Vertrag adc76e8 → 1b5cc7f.

`--gitter BASIS --vertrag aktuell` bewertet alle Retrieval-Läufe mit
derselben Fallmenge nach dem heutigen Vertrag, unabhängig vom Hash der Läufe
— nötig, wenn sich der Vertrag wie hier mitten in einer Serie um Felder
erweitert, ohne dass sich Fragen oder Gold ändern.

### Baseline B0 (21.09.2026, `2026-09-21_1514_retrieval`, Vertrag adc76e8 → 1b5cc7f punktgenau)

75 Fälle, 73 möglich, Produktion (`text` + `sparse_expanded`, DBSF, kein
Reranker): **47/73 in der Kandidatenliste** (punktgenau; 46 nach URL),
Median Rang 2, Rang 1 20/73. Verluste: 6 im Kandidatenlimit (Chunk-Rang
13–25), 20 nicht unter den Chunks.

| Arm allein | Gold in Top-10 | in Top-50 | als einziger Arm |
|---|---|---|---|
| `text` (aktiv) | 43/73 | 58/73 | 5 |
| `questions` (inaktiv) | **45/73** | 56/73 | 3 |
| `sparse` (inaktiv) | 32/73 | 45/73 | 0 |
| `sparse_expanded` (aktiv) | 36/73 | 51/73 | 1 |

### Phase 2 — Arme × Fusion (21.09.2026, 10 Läufe, `--gitter 2026-09-21_1514_retrieval --vertrag aktuell`)

| Suche | Kandidat | Rang 1 | gegen B0 | davon `eval_gold_50` |
|---|---|---|---|---|
| **`t+q_se/dbsf`** | **49/73** | 20 | **+3 −1** | +3 −0 |
| `t+q_s/dbsf` | 49/73 | 21 | +4 −2 | +3 −1 |
| `t+q_s/rrf` | 49/73 | 26 | +5 −3 | +3 −1 |
| `t+q_se/rrf` | 47/73 | 24 | +5 −5 | +3 −3 |
| `t+q_s+se/rrf` | 47/73 | 23 | +4 −4 | +3 −2 |
| `t_s/dbsf` | 47/73 | 24 | +1 −1 | ±0 |
| `t_se/dbsf` (B0) | 47/73 | 20 | — | — |
| `t+q_s+se/dbsf` | 44/73 | 20 | +2 −5 | +1 −3 |
| `t_s/rrf`, `t_se/rrf` | 43/73 | 25 / 19 | +0 −4 | −2 |

Befunde: (1) `questions` gewinnt g21 und g30 in jeder Kombination — beide hat
er allein auf Rang 1. (2) Vier Arme unter DBSF sind die schlechteste
Kombination; `sparse` und `sparse_expanded` sind Alternativen, keine
Ergänzung. (3) RRF und DBSF sehen verschiedene Fälle: RRF holt q04, q19, g43
(Fragen-Arm Rang 1–3), verliert q18, g45, g16, g36 (unter DBSF Rang 1–6).
DBSF normalisiert die Scoreverteilung und lässt den Fragen-Arm nicht durch,
wenn die anderen einig sind — Beleg für eine gewichtete Fusion (Phase 5).
(4) q06 stand in B0 auf Kandidat 8 und fällt in jeder Variante heraus: die
Kante `TOP_DOCS`, nicht die Arme.

### Phase 3 — Kandidatenstufe auf `t+q_se/dbsf` (21.09.2026, 4 Läufe)

| Variante | Kandidat | gegen B0 | davon `eval_gold_50` |
|---|---|---|---|
| **`--top-docs 12`** | **54/73** | **+8 −1** | **+6 −0** |
| `--top-docs 12 --dedup-buffer 8` | 54/73 | +8 −1 | +6 −0 |
| `--dedup-buffer 8` | 49/73 | +3 −1 | +3 −0 |
| `--kein-boost` | 48/73 | +3 −2 | +3 −0 |

`TOP_DOCS` 8 → 12 holt g10, g20, g34, g42, n02, q04 zusätzlich — die
Kandidatenlimit-Verluste aus B0. `DEDUP_BUFFER` ändert nichts (die 30 Chunks
vor der Dokument-Dedup sind bereits genug). Ohne Keyword-Boost geht q17
verloren (das ist sein Fall). Einziger Verlust in allen Varianten bleibt q06:
Die Frage nennt „Fakultät I", die Kandidatenliste füllt sich mit
Prüfungsanmeldungs-Seiten anderer Fakultäten — ein Fall für den
Fakultätsfilter, nicht für die Suche.

### Nach der Fehleranalyse (21.09.2026, Punkte 1–3)

Von den 19 Verlusten im besten Lauf waren drei Vertragsfehler (g29, g39,
g41 — die Antwort stand in Dokumenten auf Rang 1–3), fünf Tabellen-PDFs, die
kein Arm sieht, fünf Bedarfsfragen, deren doc2query-Fragen den Dokumentnamen
voraussetzen („Wer ist die Zielgruppe von Train the Trainer?" gegen „Welche
Weiterbildung für Leute, die Seminare geben?"), zwei fakultätsspezifische
Fragen, die fremde Fakultäten verdrängten, drei Fälle, in denen der
Fragen-Arm allein richtig lag, und n01. Umgesetzt:

1. **Vertrag:** g29/g39/g41 mit Zweitquellen → `529351a`. B0 = 50/73.
2. **Umlaute:** `crawler_utils.repariere_diakritika` (U+00A8 vor dem Vokal
   → Umlaut, NFC) läuft bei jeder PDF-Konvertierung; `repariere_diakritika_
   bestand.py` hat 113 Dateien in `var/korpus/` repariert und ihren
   `content_hash` erneuert, damit `delta.py` sie neu vektorisiert. **Der
   Index trägt den Schaden, bis der nächste Import läuft.** 19 Dateien haben
   Stellen, an denen der Vokal ganz fehlt („f ̈r") — nicht reparierbar.
3. **Fakultät aus der Frage:** `hsh_mcp_server._fakultaet_aus_frage` — nennt
   die Frage genau eine Fakultät (römisch, arabisch, „F4", voller Name) und
   hat der Aufrufer keine gewählt, gilt sie; zwei verschiedene Nennungen
   heißen keine. Das Ergebnis meldet `fakultaet_erkannt`, die Rohdaten
   tragen es, die Detailansicht zeigt es als `[f1]`.

Ergebnis `t+q_se/dbsf td12` danach: **59/73, +9 −0 gegen B0** — q06 auf
Kandidat 2, n03 auf 3. Sieben Fälle der Fallbasis nennen eine Fakultät; alle
sieben werden gefunden. Beachten: Im Fakultätsmodus verdreifacht der Server
`top_docs` (12 → 36) und mischt zwei Arme — für Runde 1 ist das lang; ob ×3
mit 12 noch nötig ist, gehört in den Volllauf.

### Volllauf-Bestätigung V0/V1 (21.09.2026, GPT-5.6 Luna, 75 Fälle, je 13 min)

| | V0 `t_se/dbsf td8` | V1 `t+q_se/dbsf td12` | paarweise |
|---|---|---|---|
| Gold in Kandidaten | 52/73 | 59/73 | +7 −0 |
| Gold in Quellen (Runde 1 behalten) | 50 | 53 | +6 −3 |
| Runde 1 behält | 50/52 (96 %) | 53/59 (90 %) | |
| Treffer, prüfbare Fälle (n = 24) | 19 | 20 | +2 −1 |
| Dauer je Fall (Median) | 8,0 s | 8,3 s | |
| Token (Median / Max) | 23,6k / 96,6k | 23,5k / 98,6k | |

Die längere Liste kostet in der Auswahl: g38 (zwei Modulhandbuch-Kopien,
Modell wählt das falsche), q17 (Fakultätsmodus → **36 Kandidaten**, Gold auf
Rang 4 nicht gewählt; in V0 mit 24 noch gewählt), g40 (Modell wählt mit
zwölf Kandidaten die Antidiskriminierungsstelle statt der Richtlinie von
2013 auf Rang 1 — vermutlich die bessere Quelle, der Vertrag kennt sie nicht).
Zeit und Token ändern sich nicht; das Maximum liegt in beiden Läufen knapp
unter `MAX_CONTEXT_TOKENS` = 100k, `_fit_to_budget` kürzt dort bereits.

**V2 = Wiederholung von V1** (Fakultätsfaktor 2 statt 3, sonst identisch):
Kandidaten 59, Quellen 53 — dieselben Summen. Auf den 64 Fällen ohne
Fakultät (identische Konfiguration) wechseln zwei: g11 verloren, g38
gewonnen. **Das Rauschen der Auswahlrunde liegt damit bei ±1–2 Fällen**;
Kandidaten sind deterministisch. Auf den 11 Fakultätsfällen ändert der
Faktor nichts (10/11 in beiden): q17 wählt das Modell auch bei 24
Kandidaten die Prüfungsordnungen statt der Prüfungsverwaltungsseite — die
beschreiben die 15-CP-Regel tatsächlich, der Vertrag verlangt aber die
Ansprechperson. Faktor 2 bleibt: gleiche Ergebnisse, kürzere Liste.

**Neue Diagnose `unbestimmt`:** Fälle ohne Pflichtbegriffe, deren Gold nicht
in den Quellen ist, zählen weder als Treffer noch als Fehler. Vorher wurden
sie als „nicht prüfbar = ok" zum `treffer_andere_quelle` — der Volllauf
meldete 70/75. In der Übersicht steht das als `51+19?`.

**Stand der Empfehlung (Retrieval-only, 5,5 min je Lauf):**
`text + questions + sparse_expanded`, DBSF, `TOP_DOCS = 12`, Keyword-Boost an,
Fakultät aus der Frage — 50 → 59 von 73 in der Kandidatenliste (paarweise
+9 −0, Vertrag 529351a). Im Volllauf: Gold in Quellen +6 −3, prüfbare
Treffer +2 −1, ohne Zeit- oder Tokenkosten; Wiederholung V2 bestätigt die
Summen bei ±1–2 Fällen Auswahlrauschen. Fakultätsfaktor seit V2 auf 2.
Damit steht der Umstellung der Konstanten (`DENSE_ARMS` um `questions`,
`TOP_DOCS = 12`) nichts mehr im Weg.

### Produktion umgestellt, Baseline P1 (22.09.2026)

`DENSE_ARMS = ("text", "questions")` in `hybrid_search.py`, `TOP_DOCS = 12`
im Server, Fakultätsfaktor 2 (Commit c9d6b85; der Server ist untracked).
Kontrolllauf **P1 `2026-09-22_0820_gpt-5.6-luna`** ohne `experiment` — der
Server meldet die Konstanten als wirksame Konfiguration, und die Suche ist
mit V2 **identisch: 59 Kandidaten, 53 Quellen, gleiche Fälle.** Was in der
Evaluation gemessen wurde, bekommt jetzt jeder Nutzer. P1 ist die Referenz
für alles Weitere (`--vergleich 2026-09-22_0820_gpt-5.6-luna <neu>`).

**Befund am Rande — Rauschen der Antwortrunde:** Drei prüfbare Fälle (n02,
q12, n06) fielen gegenüber V2 durch, obwohl die Quellen im Kontext dieselben
waren: n02 nennt den Vorlesungsbeginn nicht mehr, q12 lässt die Straßennamen
weg, n06 benennt S3 nicht, obwohl das Organigramm vorliegt. `temperature=0`
macht die Generierung über OpenRouter nicht deterministisch. Bei 24
prüfbaren Fällen sind das 12 % — die Antwortrunde streut also stärker als
die Auswahlrunde (±1–2). Konsequenz: Aussagen zur Antwortqualität brauchen
entweder mehr prüfbare Fälle (Pflichtbegriffe für g01–g50) oder
Wiederholungen; ein Unterschied von 3 prüfbaren Treffern zwischen zwei
Läufen ist keiner.

Sechs Fälle sieht kein Arm (g18, g26, g31, g33, g50, n06) — fünf davon sind
Tabellen-PDFs (Prüfungsplan, Stundenpläne, Semesterübersicht, Organigramm),
bei denen die Antwort eine Zelle ist: ein Chunking-Befund, kein Fusionsthema.
Acht Fälle
hatte ein Arm in seinen Top-10, die Kandidatenliste nicht — bei fünf davon war
es der inaktive `questions`-Arm auf Rang 1–3 (g21, g30, g43, q04, q19). Das
ist die erste Hypothese für Phase 2.

`--auswerten` und `--bericht` hängen die beiden folgenden Schritte direkt an,
damit die Lauf-ID nicht von Hand weitergereicht werden muss. Sammeln und Werten
bleiben getrennte Werkzeuge — sie werden nur nacheinander aufgerufen. Scheitert
die Auswertung, bleibt der Lauf gültig: Die Rohdaten liegen vollständig auf der
Platte, die Meldung nennt das Kommando zum Nachholen.

Der Runner wertet **nicht** aus, er sammelt nur. Je Fall landen Kandidatenliste,
Quellen, die gefundenen Chunks, Zeiten und Token-Budget roh in
`laeufe/<id>/faelle/qXX.json`. Dadurch lässt sich die Auswertung ändern, ohne
Läufe zu wiederholen.

Jeder Lauf schreibt sein eigenes `lauf.log` — alles, was Bibliotheken und
Runner melden. Vorher ging das nur auf die Konsole und landete allein dann in
einer Datei, wenn jemand von Hand umleitete.

`manifest.json` hält die vollständige Retrieval-Konfiguration fest — Arme,
Fusion, Limits, Punktzahl der Collection, Git-Commit. Ohne die ist ein
Laufvergleich wertlos: Am 17.08.2026 stieg unsere Trefferquote von 25 % auf
39 %, während sechs Dinge gleichzeitig anders waren. Das ist eine Beobachtung,
keine Messung.

### Der Vertrag ist Teil des Belegs

Die Fallbasis ist die Messlatte, und sie ändert sich — zwischen dem 18.08. und
dem 15.09.2026 dreimal, mit 12, 14, 17 oder 18 Treffern für denselben Lauf.
Deshalb trägt seit dem 15.09. jeder Lauf seinen Vertrag mit:

- `baue_fallbasis.py` schreibt nach dem Bau `fallbasis.sha256` — SHA-256 über
  die kanonische JSON (Schlüssel sortiert, Fälle nach id). Jede inhaltliche
  Änderung irgendeines Falls ändert den Hash, nichts sonst.
- `lauf.py` trägt `fallbasis.hash` (ganze Fallbasis) und `hash_auswahl` (die
  gelaufenen Fälle) ins Manifest ein und legt eine Kopie der gelaufenen Fälle
  als `fallbasis.json` in den Laufordner. `--fortsetzen` verweigert, wenn die
  Fallbasis inzwischen eine andere ist — ein Lauf misst gegen genau einen
  Vertrag.
- `auswertung.py` zeigt das Kürzel im Kopf und in der Übersicht (`Vertrag`).
  Gleiche Kürzel heißt vergleichbar. `--vergleich` warnt bei ungleichem Hash
  und nennt die Fälle, die nur in einem Lauf liegen oder einen anderen
  Fallvertrag haben. Für Läufe von vor der Versionierung wird der Hash aus
  `faelle/` rekonstruiert und mit `*` markiert; er deckt dann nur die
  gelaufenen Fälle ab.

Der erste Nutzen kam beim Einbau: Die beiden Läufe vom 17.08. teilen den
Vertrag `67aa8c5`, der vom 18.08. weicht in genau n02, n03 und q04 ab — die
präzisierten Fragen, die bis dahin nur ein Absatz weiter oben erwähnte.

## Auswertung

```bash
python auswertung/gold/auswertung.py                      # Übersicht aller Läufe
python auswertung/gold/auswertung.py --letzter            # der jüngste Lauf
python auswertung/gold/auswertung.py <lauf-id> --detail
python auswertung/gold/auswertung.py --vergleich <lauf-a> <lauf-b>
python auswertung/gold/auswertung.py --letzter --vertrag aktuell   # heutige Fallbasis statt Beleg
python auswertung/gold/bericht.py --letzter               # HTML-Bericht (Voll- oder Retrieval-Lauf)
python auswertung/gold/bericht.py --vergleich <lauf-a> <lauf-b>   # HTML-Vergleich, B gegen A
```

`--vertrag aktuell` bewertet die unveränderten Rohdaten eines alten Laufs mit
der heutigen `fallbasis.json`. Der Kopf der Ausgabe markiert das, die
`zusammenfassung.json` wird dabei nicht überschrieben. Nach der
Quellen-Nachprüfung vom 14.09.2026 stieg der Lauf vom 18.08. so von 12 auf 14
Treffer, mit dem toleranten Begriffsvergleich vom 15.09. auf 17 und mit der
Antwort-zuerst-Diagnose auf 18 von 23 — ohne dass sich am System etwas geändert
hätte. Genau deshalb steht diese Zahl nie
neben einer Messung nach altem Vertrag, ohne dass es dabeisteht.

### Die HTML-Berichte

Das Terminal beantwortet „hat sich etwas bewegt?"; die HTML-Seiten sind zum
Verstehen und Weitergeben. Drei Berichte, ein Aufruf, dieselben Zahlen aus
`auswertung.py` — der Modus steht im Manifest:

| Aufruf | Seite | Für wen |
|---|---|---|
| `bericht.py <volllauf>` | *Wo die Fälle verlorengehen* — Trichter, Vier-Felder-Diagnose, Gruppen, Lesehilfe, 75 Fallkarten mit Antwort, Vertrag und Rohdaten | Wer wissen will, warum ein Fall scheitert |
| `bericht.py <retrieval-lauf>` | *Wo das Gold steht* — Trichter bis zur Kandidatenliste, Rangverteilung, Arm-Tabelle (Top-10, Top-50, als einziger Arm), je Fall die Arme einzeln | Wer die Suche einstellt |
| `bericht.py --vergleich A B` | *Was sich geändert hat* — Konfiguration nebeneinander, Kennzahlen mit Wilson-Intervall und Δ, doppelter Trichter, die gewechselten Fälle mit beiden Antworten | Wer über eine Umstellung entscheidet |

Was die Seiten nicht verschweigen (seit 22.09.2026):

- **Nicht prüfbar** ist ein eigenes Feld unter den vieren. Die Fälle ohne
  Pflichtbegriffe, deren Gold nicht in den Quellen steht, fehlten vorher
  stillschweigend — die Ansicht zeigte „Retrievalproblem 0" neben einem
  Trichter mit elf Retrievalverlusten. Jetzt steht dabei, wo ihr Gold
  verlorenging (Suche / Kandidatenlimit / Auswahl), und die Fallkarten heißen
  nicht mehr „Korpuslücke".
- Einleitung, Lesehilfe und Fuß rechnen mit den Zahlen des Laufs (drei
  Herkünfte, Fakultätsfälle mit 120 Chunks und 24 Kandidaten), nicht mit den
  Konstanten von vor dem 15.09.
- Die Trefferliste je Fall zeigt die ersten zehn Chunks und jeden Goldtreffer
  darüber hinaus (`CHUNKS_IM_BERICHT`) — der Bericht schrumpfte damit von 4,9
  auf 2,4 MB; den Rest machen die Kandidatenlisten mit Gliederung aus, die
  zum Nachvollziehen der Auswahl gebraucht werden.
- Der Vergleich behauptet keine Signifikanz. Er nennt das Wilson-Intervall
  (bei n = 73 rund ±10 Punkte) und das gemessene Rauschen der LLM-Auswahl
  (V1 gegen V2: ±1–2 Fälle) und legt die gewechselten Fälle mit beiden
  Antworten daneben — ob ein Gewinn einer ist, entscheidet der Leser am Text.
- Jede Seite ist ein vollständiges HTML-Dokument mit `<meta charset>`; vorher
  fehlte der Kopf, und lokal geöffnet riet der Browser die Kodierung.

Ohne Argument zeigt die Auswertung alle Läufe mit Fallzahl, gefundenen
Goldquellen, Treffern und Notiz — die Frage nach einem Lauf ist fast nie „wie
heißt er", sondern „was kam raus".

Fünf Signale je Fall:

| Signal | Frage dahinter |
|---|---|
| `gold_in_kandidaten` + Rang | Hat die **Suche** es gefunden? |
| `gold_in_quellen` + Rang | Hat die **LLM-Auswahl** es behalten? |
| `politik_erfuellt` | `exact_gold_source` / `accepted_sources` beachtet? |
| `begriffe_quote` | Anteil gefundener Pflichtbegriffe (toleranter Vergleich) |
| `fragment_im_kontext` | War ein ausgewähltes Dokument unvollständig? |

Dazu drei Merkmale der Antwort: `antwort_status` (vollständig / teilweise /
keine / verweigert / nicht prüfbar), `herkunft_antwort` (gold / akzeptiert /
andere / keine) und `widerspruch` (die Antwort stellt zwei Angaben
gegeneinander — meist ein Korpusbefund: zwei Dokumente verschiedenen Alters).

Daraus die Diagnose — **die Antwort zuerst, dann die Herleitung**:

```
verbotene_aussage       verbotener Begriff steht in der Antwort
treffer                 Antwort vollständig, Goldquelle unter den Quellen
treffer_andere_quelle   Antwort vollständig, aber ohne Goldquelle — zählt als
                        Treffer; der Trichter zeigt trotzdem einen Verlust
korpus_luecke           Antwort unvollständig, Gold gar nicht im Index
retrieval_fehler        Gold nicht unter den Kandidaten
auswahl_fehler          Gold war Kandidat, fiel in Runde 1 heraus
antwort_verweigert      Gold war Quelle, Modell sagt „keine Informationen“
antwort_unvollstaendig  Gold war Quelle, aber Pflichtbegriffe fehlen
```

Ein Treffer ist eine richtige Antwort. Ob sie aus der Goldquelle stammt, ist
eine zweite Frage, die dem Retrieval-Trichter gehört, nicht dem Urteil. Bis
15.09.2026 stand die Kette andersherum — erst Retrieval, dann Antwort — und
n01 (alle fünf Fakultäten richtig, aus dem Jahresabschluss statt der
Erstsemesterseite) zählte als Fehlschlag. Der Quadrant unten links heißt
deshalb jetzt „Treffer (andere Quelle)" statt „Vertrag zu eng".

## Vergleich mit ihrem Mai-Stand

Ihr Llama-Lauf: **Goldquelle gefunden 12/20, auf Rang 1 6/20.**

Das ist über ihre sechs finalen Chunks gemessen, unsere Kandidatenliste umfasst
acht Dokumente und unsere Quellen bis zu vier. Andere Grundgesamtheit —
vergleichbar ist die Richtung, nicht die zweite Nachkommastelle. Die Auswertung
schreibt die Marke deshalb mit dem K dazu und berichtet unsere Quote zweimal:
von den möglichen und von den gestellten Fällen.

## Was hier bewusst fehlt

**Kein LLM-Judge.** Erst die deterministischen Signale ausreizen. Der Vertrag
für einen zweistufigen Judge liegt in der Fallbasis bereit; ihre eigene Grenze
dazu lautet: „Judge bleibt ein Modell, Bewertungsrauschen bleibt."

**Kein Query Assist, kein Retrieval Planner.** Das sind Features, keine
Evaluation. Bei ihnen der größte gemessene Hebel — aber sie hatten kein
doc2query. Wir lösen dasselbe Problem seit dem Index vom 14.08. von der
Indexseite und zur Laufzeit kostenlos. Ob obendrauf noch etwas zu holen ist,
soll dieses Werkzeug beantworten, nicht ein Nachbau.
