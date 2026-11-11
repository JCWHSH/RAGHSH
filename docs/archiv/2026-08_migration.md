> **Archiv.** Migrationsplan vom August 2026 zu Kollisionen und Dubletten. Phase A umgesetzt, übrige Phasen offen bzw. verworfen.

# Migrationsstrategie — URL-Kollisionen und Dublettenzusammenführung

Stand: 2026-08-07. Grundlage: `kollisionen.md`, `verifiziert.md`, `aliase.md`.

---

## 1. Bewertung des Vorschlags

> Alle Dateien mit demselben Hashwert werden zu einer Markdown-Datei zusammengefasst.
> Zeigt eine der URLs auf eine Fakultäts-Subdomain, kennen wir damit die Fakultät.

### Was daran trägt

**Die Fakultäts-Ableitung ist der stärkere Teil des Vorschlags — stärker, als er formuliert ist.**
Heute leitet `extract_faculty()` (`main.py:156`) die Fakultät allein aus dem Host ab. Jedes
Dokument, das unter `www.hs-hannover.de` liegt, bekommt `faculty: ""` — auch wenn dasselbe
Dokument zusätzlich unter `f1.hs-hannover.de` erreichbar ist und damit eindeutig einer
Fakultät zugeordnet werden könnte.

Gemessen am Korpus: **204 der 271 Dublettengruppen gewinnen dadurch eine Fakultätszuordnung,
die sie heute nicht haben.** Das ist ein echter Qualitätssprung für `frag_hsh(fakultaet=…)`.

Die Zusammenführung selbst ist bei PDFs sauber und risikoarm: 246 der 271 Gruppen sind PDFs,
byte-identisch, mehrfach im Fileadmin abgelegt (die `ATPO_2015` unter 13 URLs). Ersparnis:
**301 Dateien, 6,9 MB, ~5.800 Chunks** (4,5 % des Korpus).

### Was nicht trägt — drei Einwände

**Einwand 1: Der Vorschlag löst das schwerwiegendere Problem nicht.**
Es liegen zwei verschiedene Fehler vor, die leicht verwechselt werden:

| | Befund 1 — Kollision | Befund 2 — Dubletten |
|---|---|---|
| Richtung | verschiedene Inhalte → **eine** Datei | gleicher Inhalt → **mehrere** Dateien |
| Folge | **Datenverlust** (129 Seiten fehlen) | Redundanz (301 Dateien zu viel) |
| Ursache | `make_filename()` ohne Host | Website legt Dokumente mehrfach ab |
| Fix | Dateiname um Host erweitern | Zusammenführen |

Deduplizierung ist die Antwort auf Befund 2. Auf Befund 1 wirkt sie nicht — dort sind die
Seiten gar nicht erst im Korpus. Die Startseite `www.hs-hannover.de` fehlt vollständig; kein
Zusammenführen bringt sie zurück.

**Einwand 2: Exakte Hash-Gleichheit greift beim Personenfinder nicht — also genau dort nicht,
wo der Bug ursprünglich auffiel.**
Verifiziert am Livesystem (Prof. Begemann, `person/1000005359`):

```
f5-Variante  : 5.953 Zeichen Hauptinhalt
www-Variante : 5.906 Zeichen Hauptinhalt
identisch    : nein
Jaccard      : 0,988
Differenz    : nur in f5 → "Fakultät V", "Diakonie,", "Gesundheit", "Soziales"
```

Die Seiten unterscheiden sich um genau die Fakultätsangabe. Ihre `content_hash`-Werte sind
verschieden, die Gruppierung nach exaktem Hash erfasst sie also **nicht**. Alle 48 betroffenen
Personenfinder-Profile fallen unter diesen Fall.

Bemerkenswert: Die reichere Variante ist die mit der Fakultät. Die Intuition des Vorschlags
stimmt — nur das Kriterium „gleicher Hash" ist zu scharf, um sie zu erfassen.

**Einwand 3: `faculty` kann kein einzelner Wert bleiben.**
15 Gruppen liegen unter mehreren Fakultäten gleichzeitig (die Grundordnung unter f1 *und* f4).
Bei Zusammenführung ist die Fakultät zwangsläufig eine **Menge**. Das zieht sich durch die
gesamte Pipeline und ist der teuerste Teil der Umstellung — siehe Phase D.

### Die entscheidende Einsicht: Befund 1 maskiert Befund 2

Wo eine Kollision zugeschlagen hat, existiert nur noch **eine** Datei — die verdrängte URL
hat gar keine. Als Dublette ist sie damit unsichtbar. Die heute gemessenen 271 Gruppen sind
eine **Untergrenze**.

Daraus folgt die Reihenfolge zwingend: **erst Kollision beheben und neu crawlen, dann
deduplizieren.** Umgekehrt würde die Deduplizierung auf einem lückenhaften Korpus arbeiten
und die Lücken zementieren.

---

## 2. Zielbild

```
Korpus:   1 Dokument = 1 Datei
          Dateiname enthält den Host  → keine Kollisionen mehr
          Header führt Aliase mit     → Dubletten zusammengefasst
          faculty ist eine Liste      → Mehrfachzuordnung möglich

Header:
  ---
  source_url: "https://f5.hs-hannover.de/..."     ← kanonisch
  alias_urls:                                      ← alle weiteren
    - "https://www.hs-hannover.de/..."
  faculty: ["f5"]                                  ← Liste statt String
  content_hash: "..."
  ---
```

**Regel für die kanonische URL** (in dieser Reihenfolge):
1. URL auf einer Fakultäts-Subdomain (`f1`…`f5`) — trägt die meiste Information
2. sonst `www.hs-hannover.de`
3. sonst die kürzeste URL (stabilste, meist die TYPO3-Kurz-URL)

Bei mehreren Fakultäts-URLs entscheidet die kürzeste; alle Fakultäten wandern in `faculty`.

**Warum die Aliase im Header bleiben und nicht verworfen werden:** Der Crawler ist
inkrementell. Ruft ein späterer Lauf die Alias-URL ab, muss er *seine* Datei wiederfinden —
sonst hält er die Seite für neu und schreibt sie erneut. `matches_url()` muss deshalb
`source_url` **und** `alias_urls` prüfen. Ohne diesen Schritt bauen sich die Dubletten beim
nächsten Crawl wieder auf.

---

## 3. Phasen

Die Phasen sind so geschnitten, dass jede für sich lauffähig und einzeln rücknehmbar ist.
A und B sind Pflicht, C und D setzen darauf auf.

### Phase A — Kollision beheben (Pflicht, zuerst)

**Änderung:** `make_filename()` (`main.py:147`) nimmt den Host auf.

```python
def make_filename(url: str, today: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").replace(".hs-hannover.de", "").replace(".", "-") or "www"
    path = parsed.path.strip("/")
    slug = slugify(path) if path else "index"
    return f"{today}_{host}_{slug}.md"
```

Gleiche Änderung in `resume_crawler.py` (hat eine eigene Kopie von
`save_markdown_if_changed()`) sowie in `slug_for_url()` (`crawler_utils.py:164`), damit
Glob-Suche und Dateiname konsistent bleiben.

**Bestandsdaten:** Kein Rename der 5.510 Altdateien nötig. Begründung: Qdrant-IDs sind
UUID5 über `source_url + chunk_index`, `doc_index.documents` hat `source_url` als PRIMARY KEY
— **nichts hängt am Dateinamen**. `file_path` wird beim Rebuild ohnehin neu gefüllt.
Alte und neue Namensschemata können nebeneinander liegen; `cached_files_for_url()` findet
beide, solange die Glob-Suche auf `*_{slug}.md` **und** `*_{host}_{slug}.md` matcht.

Empfehlung trotzdem: einmalig umbenennen, um zwei Schemata zu vermeiden — als reines
`os.rename` anhand der `source_url` im Header, ohne Inhaltsänderung.

**Danach: Recrawl der 129 verlorenen URLs** über `recrawl_urls.py` mit der Liste aus
`verifiziert.md` (Abschnitt „Bestaetigt ueberschrieben"). Erwartung: ~129 neue Dateien,
darunter die Startseite der Hochschule.

**Verifikation:**
```bash
grep "Gespeichert" crawl.log | sed 's/.*: //' | sort | uniq -d   # muss leer sein
```
Ein doppelt geschriebener Pfad bedeutet, dass die Kollision noch besteht.

### Phase B — URL-Kanonisierung (billig, unabhängig)

**Änderung:** `normalize_url()` (`crawler_utils.py:144`) zusätzlich:
- Root-Slash entfernen (`https://f5…/` → `https://f5…`) — die Bedingung `path != "/"` schützt
  ihn heute ausdrücklich, dadurch werden 17 Hosts doppelt gecrawlt
- `www.` verwerfen bei `hs-hannover.de` (16 Fälle)
- `/index.php` am Pfadende entfernen (4 Fälle)

Vorsicht bei Groß-/Kleinschreibung: Der Host darf kleingeschrieben werden, der **Pfad nicht** —
`fileadmin`-Pfade sind auf dem Server case-sensitiv. Der eine gefundene Fall
(`Anerkennungsjahr.pdf` / `anerkennungsjahr.pdf`) ist damit bewusst kein Alias, sondern
werden zwei Abrufe bleiben.

Wirkung: 37 doppelte Abrufe entfallen, das visited-Set wird korrekt. Kein Bestandsdaten-Eingriff.

### Phase C — Dublettenzusammenführung (der eigentliche Vorschlag)

Neues Skript `dedupe_corpus.py`, läuft **nach** dem Crawl, **vor** der Vektorisierung.

1. Alle Dateien nach `content_hash` gruppieren
2. Gruppen mit >1 URL: kanonische URL nach der Regel oben wählen
3. Kanonische Datei behalten, Header erweitern um `alias_urls` und `faculty`-Liste
4. Übrige Dateien der Gruppe löschen (via `trash`, nicht `rm`)
5. Protokoll nach `dedupe_report.md` schreiben

**Sicherung im Skript:** Gruppen mit Body < 800 Zeichen nur mit `--include-short`
zusammenführen. Bei sehr kurzen Inhalten ist Hash-Gleichheit schwächere Evidenz für
inhaltliche Identität (5 solche Gruppen im Korpus, alle geprüft unbedenklich, aber die
Schranke kostet nichts).

**Sonderregel hochschulweite Dokumente (wichtig — sonst Regression).**
Die abgeleitete Fakultätsmenge beschreibt die *Ablage*, nicht die *Geltung*. Die `ATPO_2015`
liegt unter f1, f2 und f5 — sie gilt aber für alle fünf Fakultäten und ist nur zufällig
nicht unter f3 und f4 abgelegt.

Heute trägt die www-Kopie `faculty: ""` und wird vom permissiven Filter
(`hsh_mcp_server.py:764`) durchgelassen, eine Frage mit `fakultaet: "f3"` findet die ATPO also.
Nach naiver Zusammenführung stünde dort `["f1","f2","f5"]` und der f3-Filter würde sie
**aussortieren** — eine Verschlechterung gegenüber dem Ist-Zustand.

Regel: Liegt eine URL der Gruppe unter `Verkuendungsblatt`, `Verkuendungsblaetter` oder
`Studierendenverwaltung`, ist das Dokument hochschulweit → `faculty: []` (leer = gilt
überall), unabhängig davon, welche Fakultätskopien existieren. Eine unvollständige
Aufzählung ist schlechter als gar keine.

**Bewusst nicht enthalten: Near-Duplicate-Erkennung.** Der Personenfinder-Fall (Jaccard 0,988)
würde eine Ähnlichkeitsschwelle brauchen. Davon rate ich ab: Nach Phase A existieren beide
Varianten als eigene Dateien, die f5-Variante trägt `faculty: ["f5"]` und ist damit über den
Fakultätsfilter auffindbar — **der ursprüngliche Begemann-Bug ist damit behoben**. Der Preis
ist ein 98,8 % redundantes Dokument im Index. Eine Schwelle, die diese Paare zusammenfasst,
würde zwangsläufig auch fachlich verschiedene Seiten verschmelzen, die sich nur in wenigen
Wörtern unterscheiden — Studiengangsvarianten etwa. Das Risiko steht in keinem Verhältnis
zu 48 eingesparten Dokumenten.

### Phase D — `faculty` als Liste durch die Pipeline

Der invasivste Teil. Reihenfolge strikt einhalten, sonst bricht die Suche:

| Datei | Änderung |
|---|---|
| `main.py:156` | `extract_faculty()` → `list[str]`, Header schreibt `["f5"]` |
| `hpc_vectorizer.py:99` | Arrow-Schema `pa.string()` → `pa.list_(pa.string())` |
| `hpc_vectorizer.py:183` | `faculty` aus Header-Liste + Alias-URLs vereinigen |
| `local_importer.py:138` | Payload-Index auf `faculty` bleibt — Qdrant indiziert Keyword-Arrays elementweise, `MatchValue` trifft Arrays ohne Änderung |
| `hsh_mcp_server.py:764-768` | Filter `faculty == fakultaet` → `fakultaet in faculty` |
| `build_doc_index.py` | Spalte `faculty` als JSON-Array speichern (Schema bleibt TEXT) |

**Rückwärtskompatibilität:** Alle Lesestellen so schreiben, dass sie String *und* Liste
verkraften (`faculty if isinstance(faculty, list) else [faculty] if faculty else []`).
Damit bleibt der Korpus während der Migration gemischt lesbar und Phase D ist von A–C
zeitlich entkoppelt.

**`doc_index.db` braucht kein neues Schema:** `source_url` bleibt PRIMARY KEY. Alias-URLs
bekommen eigene Zeilen, deren `file_path` auf dieselbe Datei zeigt. Damit bleibt jede
bekannte URL auflösbar, ohne dass ein Dokument doppelt gezählt wird.

---

## 4. Reihenfolge und Aufwand

```
A  make_filename + Host, Recrawl der 129 URLs      ← Pflicht, zuerst
B  normalize_url erweitern                          ← unabhängig, billig
C  dedupe_corpus.py                                 ← braucht A
D  faculty als Liste                                ← braucht C
   ↓
   Neu-Vektorisierung (HPC) + local_importer
```

C vor D ist wesentlich: Erst nach der Zusammenführung steht fest, welche Fakultätsmengen
überhaupt entstehen — vorher wäre die Schema-Umstellung eine Wette.

Nach A **muss** neu vektorisiert werden (129 neue Dokumente). Nach C und D ebenfalls
(veränderte Payloads). Sinnvoll ist deshalb ein einziger HPC-Lauf am Ende — das setzt
voraus, dass der KISSKI-Vectorizer wieder läuft (siehe offener Punkt im Tageslog 07.08.).

## 5. Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Umbenennung verliert Dateien | Rename nur anhand `source_url` im Header, vorher `tar.xz`-Snapshot von `data/ingested` |
| Dedup löscht fälschlich | `trash` statt `rm`; `dedupe_report.md` listet jede Löschung mit Begründung |
| Alias-URL wird erneut als neu gecrawlt | `matches_url()` muss `alias_urls` mitprüfen — ohne diesen Schritt ist Phase C beim nächsten Crawl wieder aufgehoben |
| `faculty`-Liste bricht laufenden MCP-Dienst | Lesestellen tolerant gegen String und Liste; Dienst erst nach vollständigem Reimport neu starten |
| Recrawl trifft auf zwischenzeitlich geänderte Seiten | unkritisch — `content_date` hält die tatsächliche Änderung fest |
