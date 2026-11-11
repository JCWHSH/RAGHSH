# RAG-Pipeline auf KISSKI — Sitzungsbericht II

**Datum:** 13. August 2026
**Vorgänger:** `RAG-HPC-Sitzungsbericht.md` (12. August 2026)
**Cluster:** KISSKI, Partition `kisski`, NVIDIA A100-SXM4-80GB, Treiber 570.211.01

---

## 0. Kurzfassung

Drei Dinge sind passiert:

1. **vLLM läuft.** Der als „nicht gelöst" abgelegte Punkt 6.4 des Vorberichts ist
   erledigt. Die Diagnose dort war an einer Stelle falsch — nicht der torch-Pin
   war das Hindernis, sondern dass *jedes* vLLM-Wheel von PyPI gegen CUDA 13
   gebaut ist. Gelöst über CUDA Forward Compatibility ohne Root-Rechte.
2. **Der Durchsatz ist um Faktor 9,3 gestiegen** — von 3,28 auf 18,5 Chunks/s,
   bei gleichzeitig 40 % weniger zu rechnenden Chunks.
3. **Der Produktionslauf schrumpft von 16,7 auf 1,7 GPU-Stunden.**

| | Chunks | Durchsatz | Gesamtlaufzeit |
|---|---|---|---|
| Stand Vorbericht (transformers) | 196.955 | 3,28/s | 16,7 h |
| vLLM, `--batch-size 16` | 196.955 | 10,16/s | 5,4 h |
| **vLLM, `--batch-size 512`, dedupliziert** | **119.791** | **18,5/s** | **1,7 h** |

---

## 1. vLLM — die Auflösung

### 1.1 Was der Vorbericht falsch schloss

Abschnitt 6.5 leitete aus PyPI-Metadaten ab, welche vLLM-Version torch 2.11.0
verlangt, und folgerte, ein Downgrade auf 0.24.0 könnte das Treiberproblem
umgehen. Zwei Annahmen dahinter tragen nicht:

**Erstens** gibt es auf PyPI **genau ein x86_64-Wheel je vLLM-Version**, ohne
CUDA-Tag. Geprüft für 0.18.0 bis 0.27.1 — alle vorhanden. Die Vorstellung
fehlender oder 404-liefernder „cu128-Wheels" trifft auf PyPI nicht zu; sie
bezieht sich auf den separaten Nightly-Index `wheels.vllm.ai`.

**Zweitens** — und das ist der Kern — ist die CUDA-Bindung **nicht** durch den
torch-Pin bestimmt. Die Binärerweiterung `vllm._C_stable_libtorch` verlangt
`libcudart.so.13`, auch bei 0.26.0 mit `torch==2.11.0`. Der Downgrade allein
hätte also nichts gebracht, egal wie tief.

**Und die Architektur war nie das Problem:** `Qwen3_5ForConditionalGeneration`
— exakt der Wert in der lokalen `config.json` — steht seit **0.22.0** in der
Registry. Die Prüfung aus Abschnitt 6.7 des Vorberichts hätte positiv
ausfallen müssen.

| vLLM | verlangt torch | Qwen3_5 in Registry |
|---|---|---|
| 0.27.1 | 2.13.0 (CUDA 13) | ja |
| **0.26.0** | **2.11.0** | **ja** ← installiert |
| 0.25.0 | 2.11.0 | ja |
| 0.24.0 | 2.11.0 | ja |
| 0.22.0 | 2.11.0 | ja |

### 1.2 Die drei Hürden

| # | Fehler | Ursache | Behebung |
|---|---|---|---|
| 1 | `ImportError: libcudart.so.13` | vLLM-Wheel ist gegen CUDA 13 gebaut | `nvidia/cu13/lib` in `LD_LIBRARY_PATH` — die Bibliothek liegt bereits im venv |
| 2 | Treiber zu alt | Basistreiber 570.211.01 = CUDA 12.8 | **cuda-compat-13-3**, ohne Root ausgepackt |
| 3 | `FileNotFoundError: ninja`, dann `curand.h` | flashinfer kompiliert den Sampling-Kernel per JIT | Toolchain in `PATH`, `VLLM_USE_FLASHINFER_SAMPLER=0` |

### 1.3 CUDA Forward Compatibility

Der entscheidende Mechanismus. Er ist ausdrücklich für Datacenter-GPUs
vorgesehen — Consumer-Karten liefern Error 803 — und auf der A100 damit
zulässig.

Auf dem System war **nichts** davon vorhanden: kein `/usr/local/cuda*/compat`,
keine CUDA-13-Module (Lmod führt nur bis 12.9.2). Beschafft und ausgepackt:

```bash
cd $PROJEKT/cuda_compat
curl -sSLO https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/\
cuda-compat-13-3-610.57.04-1.el8.x86_64.rpm
rpm2cpio cuda-compat-13-3-610.57.04-1.el8.x86_64.rpm | cpio -idmu
# → usr/local/cuda-13.3/compat/libcuda.so.610.57.04
```

Rocky Linux 8, also `rpm2cpio` statt `dpkg -x`. 112 MB, keine Rechte nötig.

**Beleg, dass es wirkt:** `cuDriverGetVersion` meldet mit vorangestelltem
Compat-Pfad **13030** auf einem Treiber, der ohne ihn 12080 meldet.

### 1.4 Eine Fußangel in vLLMs eigener Compat-Logik

`vllm/env_override.py::_maybe_set_cuda_compatibility_path()` reagiert korrekt
auf `VLLM_ENABLE_CUDA_COMPATIBILITY` und `VLLM_CUDA_COMPATIBILITY_PATH` —
verändert aber ausschließlich `os.environ`. Der glibc-Loader liest
`LD_LIBRARY_PATH` beim Prozessstart und danach nicht mehr. Die Funktion wirkt
deshalb **nur für die gespawnten EngineCore-Worker**, nicht für den eigenen
Prozess, in dem torch importiert wird — trotz gegenteiliger Aussage im
Docstring.

Konsequenz: Der Pfad muss **zusätzlich in der Shell** gesetzt werden. Genau das
tut `env_vllm.sh`.

### 1.5 Die CUDA-13-Toolchain liegt im venv

`nvcc` (13.3.73) und `ninja` kommen als pip-Pakete mit vLLM mit:

```
$SP/nvidia/cu13/bin/nvcc
$PROJEKT/venv_vllm/bin/ninja
```

Das löst nebenbei den in Abschnitt 8 des Vorberichts gescheiterten
`causal-conv1d`-Build („`/usr/local/cuda/bin/nvcc` nicht gefunden") — der
Login-Node hat kein Toolkit, das venv jetzt schon.

`curand.h` fehlt in `cu13/include` und liegt im cu12-Paket; `CPATH` deckt das
ab. Für den Produktionsbetrieb ist es unnötig, weil der flashinfer-Sampler
abgeschaltet ist: Bei Greedy-Dekodierung wird top-k/top-p ohnehin nicht
gebraucht.

---

## 2. Umgebung — `env_vllm.sh`

Neu angelegt neben dem bestehenden `env.sh`. Sourcen, dann läuft alles.

```bash
source env_vllm.sh
```

Setzt in dieser Reihenfolge: `gcc/13.2.0` als Host-Compiler, Compat-Pfad
zuerst in `LD_LIBRARY_PATH`, dahinter CUDA-13-Runtime und die
Conda-`libstdc++` (gegen den `CXXABI_1.3.15`-Fehler aus Abschnitt 6.2 des
Vorberichts), Toolchain in `PATH`, alle Caches auf das Projekt-Dateisystem.

> **`vectorize.sbatch` fehlte `LD_LIBRARY_PATH`** — die Zeile, die Abschnitt 5
> des Vorberichts für *beide* sbatch-Dateien vorschreibt. Ergänzt.

---

## 3. Messungen

### 3.1 A/B auf identischem Bestand (508 Chunks, dieselbe A100)

| | transformers `--batch-size 16` | vLLM 0.26.0 |
|---|---|---|
| Generierung | 154,8 s | **50,0 s** |
| Durchsatz | 3,28 Chunks/s | **10,16 Chunks/s** |
| ohne Fragen | 24 (4,7 %) | **0 (0,0 %)** |
| Modell laden | 36 s | 244 s |

Die 24 transformers-Ausfälle liefern unter vLLM verwendbare Fragen. Fünf davon
gegengeprüft: inhaltlich verankert (Beta-Test, Tablet-Aktivierung, ein
Gerichtsurteil) — Guided Decoding erfindet dort nichts, es holt aus echten
Inhalten heraus, woran der freie Decoder scheiterte. Die restlichen 19 wurden
nicht angesehen.

### 3.2 `--batch-size` — der größte Einzelhebel

| batch-size | Generierung | Durchsatz |
|---|---|---|
| 16 (Vorgabe) | 50,0 s | 10,16/s |
| **512** | **27,5 s** | **18,5/s** |

**Faktor 1,8 für eine Zahl in der Kommandozeile.** Die Vorgabe 16 stammt aus
der transformers-Messung des Vorberichts — dort war sie richtig, weil
`model.generate()` im Gleichschritt arbeitet. Bei vLLM bewirkt sie das
Gegenteil: Die Hauptschleife zerlegt den Vorrat in Scheiben und ruft je Scheibe
`llm.generate()`; der Scheduler sieht nie mehr als 16 Anfragen und läuft
zwischen den Aufrufen jedes Mal leer. Derselbe Gleichschritt-Effekt, eine Ebene
höher.

Ob über 512 noch etwas geht, ist ungetestet. `max_cudagraph_capture_size=512`
im Log deutet auf eine Deckelung dort hin.

### 3.3 Wirkung der Base64-Bereinigung aus dem Vorbericht

Erstmals gemessen — offene Punkte 1 und 2 des Vorberichts:

| | vorher | nachher |
|---|---|---|
| Chunks (50 Testdateien) | 645 | 508 (−21 %) |
| ohne Fragen | 142 (22,0 %) | 0 (0,0 %) |

Die Ausfallquote ist nicht gesunken, sondern verschwunden. Die Deutung des
Vorberichts ist damit vollständig bestätigt: Die leeren Antworten waren kein
Modellfehler, sondern Base64-Müll.

---

## 4. Codeänderungen

### 4.1 `chunking.py` — Muster `bewerbung-nav`

```python
_BEWERBUNG_NAV_RE = re.compile(
    r"\[\s*(?:Jetzt\s+bewerben|Alle\s+Stellenangebote)\s*\]\(\s*https?://[^)]*\)",
    re.IGNORECASE,
)
```

Jeder Link einzeln, damit auch Reihenfolge oder Alleinstellung erfasst wird.
Der Block steht zweimal je Stellenausschreibung und besteht ausschließlich aus
Links. Gegengeprüft an 8 Fällen: trifft die drei Navigationsvarianten
vollständig, verschont „Die Bewerbung ist online über das Bewerbungsportal
möglich", „Bewerbungsschluss ist der 07.05.2026", einen echten Stellen-Link der
Fakultät IV und „Jetzt bewerben Sie sich bitte schriftlich".

Wirkung korpusweit klein: 196.955 → 196.934 Chunks. Die Testdaten waren
stellenanzeigen-lastig, der Gesamtkorpus ist es nicht.

### 4.2 `enricher.py` — Filter gegen Rückverweise

```python
_VAGUE_REF_RE = re.compile(
    r"\bdies(?:e|er|es|em|en)\s+"
    r"(?:stelle|position|ausschreibung|angebot|modul|veranstaltung|"
    r"studiengang|studiengangs|projekt|abschnitt|seite|dokument|"
    r"einrichtung|kurs|seminar|vorlesung|programm)\b", re.IGNORECASE)
```

„Wo finde ich die Bewerbung für **diese Stelle**?" ist als Suchanfrage wertlos:
Das Demonstrativpronomen zeigt auf einen Chunk zurück, den die suchende Person
noch nicht kennt.

Der Prompt verbot das bereits. Wie bei den Ja/Nein-Fragen greift die Regel erst
zuverlässig, wenn sie zusätzlich in `_clean_question()` nachgeprüft wird — der
Codekommentar dort sagt das selbst. Die Prompt-Regel wurde entsprechend
ergänzt.

**Bewusst nur Demonstrativpronomen:** „das Modul" ist durch einen nachfolgenden
Namen reparierbar („das Modul MAT-201") und darf nicht pauschal fallen;
„dieses Modul MAT-201" ist kein Deutsch. Gegengeprüft an 10 Fällen.

### 4.3 Zurückgenommen: `uniqueItems`

Der Versuch, wortgleiche Fragen innerhalb eines Chunks per JSON-Schema
auszuschließen, ist aus zwei Gründen gescheitert:

1. **xgrammar implementiert das Schlüsselwort nicht** —
   `Grammar error: Unimplemented keys: ["uniqueItems"]`. Der Lauf brach ab.
2. Es wäre **ohnehin wirkungslos** gewesen: `parse_questions()` verwirft
   wortgleiche Fragen längst über sein `seen`-Set.

Beides steht jetzt als Kommentar im Schema, damit es niemand erneut versucht.

> **Nebenbefund:** Die Fallback-Kaskade in `_sampling_params()` fängt nur
> Fehler beim *Bau* der `SamplingParams` ab, nicht bei der Generierung. Deshalb
> stürzte der Lauf ab, statt auf die nächste Variante abzustufen. Aktuell
> folgenlos, da das Schema bekannt gut ist — aber dieselbe Klasse von Lücke wie
> die aus Abschnitt 4.5 des Vorberichts.

### 4.4 Wirkung von 4.1 und 4.2

| | vorher | nachher |
|---|---|---|
| Paraphrasen ≥ 0,60 Jaccard | 3,1 % | **2,4 %** |
| wortgleiche Wiederholungen | 6,2 % | **5,5 %** |
| Ja/Nein-Fragen | 0,0 % | 0,0 % |
| ohne Fragen | 0,0 % | 0,2 % (1 Chunk) |

Alle vier anaphorischen Formulierungen sind aus der Spitzenliste verschwunden.

**Eine Kennzahl ging in die falsche Richtung** — vage Referenz 5,5 % → 6,0 %.
Das ist kein Rückschritt, sondern eine Lücke der Messung: Der Detektor prüft
„die Stelle", nicht „diese Stelle", hat also nie gemessen, was der Filter
entfernt. Der Anstieg zeigt etwas anderes: **Das Modell weicht auf „die Stelle"
aus.** Der Rückverweis ist weg, die Unbestimmtheit nur zur Hälfte.

---

## 5. Deduplizierung

### 5.1 Der Befund

40,1 % aller Chunks sind wortgleiche Wiederholungen. Ursachen: gespiegelte PDFs
(dieselbe Prüfungsordnung unter mehreren Fakultätspfaden), Navigationsreste,
Boilerplate in Stellenausschreibungen.

```
Chunks gesamt              : 196.934
eindeutige Chunktexte      : 117.921
größte Duplikatgruppe      :     234
```

Häufigste Wiederholungen: MINT-Newsletter-Listen (234×), Seminar-Kommentarblöcke
(218×), der Personenfinder-Link (197×).

### 5.2 Warum nicht auf Dateiebene

Naheliegend wäre, ganze Dateien auszusortieren. Das ist die falsche Einheit:

| Anteil duplizierter Chunks je Datei | Dateien |
|---|---|
| 0 % (völlig einzigartig) | 2.389 (42,6 %) |
| 1–49 % | 1.263 (22,5 %) |
| 50–79 % | 633 (11,3 %) |
| 80–99 % | 598 (10,7 %) |
| 100 % dupliziert | 721 (12,9 %) |

Nur 324 Dateien (5,8 %) sind als Ganzes redundant. Und ein Datei-Filter
verliert Inhalt, selbst bei der vorsichtigsten Schwelle:

```
Schwelle 100%:  721 Dateien entfernt →  3.290 Chunktexte KOMPLETT verloren
Schwelle  80%: 1319 Dateien entfernt → 24.727 Chunktexte KOMPLETT verloren
```

Der Grund ist tückisch: **Beide Partner eines Duplikatpaars erfüllen die
Bedingung** und werden beide entfernt. Der Inhalt war doppelt vorhanden —
danach ist er nullmal vorhanden. Der Fehler fällt erst auf, wenn im fertigen
Index eine Antwort fehlt.

Chunk-Deduplizierung mit **einem Vertreter je Gruppe** verliert dagegen nichts.

### 5.3 Der Schlüssel — und warum der Titel hineingehört

Der Prompt bekommt den Seitentitel mit. Zwei Chunks mit gleichem Text, aber
verschiedenem Titel erzeugen deshalb verschiedene Fragen.

```
Duplikatgruppen                   : 32.168
davon mit abweichendem Titel      : 27.437  (85,3 %)
   davon nur URL/Dateiname        : 26.545  (96,7 %)
   davon echter Titelunterschied  :    892  ( 3,3 %)
```

Bei den Stellenanzeigen **ist** der Titel die URL — solche Unterschiede sind
bedeutungslos. Die verbleibenden 892 sind es nicht:

```
Text : "### veranstaltungsort materialwerkstatt raum: 2a.0.06 expo plaza 2 …"
   Titel: Material Monday 2026: Asbest
   Titel: Material Monday 2026: Kunststoffe
   Titel: Material Monday 2026: Pflanzen und Tiere
```

Derselbe Ortsabsatz, drei verschiedene Veranstaltungen. Einmal generiert,
hinge „Wo findet Material Monday 2026: Asbest statt?" auch an den anderen
beiden Terminen.

Deshalb gruppiert `dedupe_key()` über **Text + aussagekräftigen Titel**, wobei
`informative_title()` URLs, Dateinamen und Dateiname-Stil-Titel als leer
behandelt.

| Schlüssel | behalten | entfernt |
|---|---|---|
| nur Text | 117.921 | 40,1 % |
| **Text + aussagekräftiger Titel** | **119.791** | **39,2 %** |
| Text + Titel + Überschrift | 130.007 | 34,0 % |
| Text + roher Titel + Überschrift | 171.616 | 12,9 % |

Die gewählte Variante kostet 0,9 Prozentpunkte und beseitigt das Risiko. Die
Variante mit Überschrift kostet weitere 6 Punkte und würde ausgerechnet den
NGG-Absatz *nicht* deduplizieren, weil dort die Überschrift der Stellentitel
ist.

### 5.4 Die kritische Eigenschaft: global, vor jedem Zuschnitt

`select_unique_ids()` wählt den Vertreter in Dateireihenfolge. Über eine
Teilmenge berechnet fällt die Wahl anders aus:

```
Präfix    files[:1000]  :     0 abweichende Vertreter
verstreut files[::5]    : 6.945 abweichende Vertreter   ← die Gefahr
Suffix    files[-1500:] : 2.441 abweichende Vertreter
```

22 % abweichende Vertreter bei einer verstreuten Teilmenge. Liefe der
Vektorisierer im Delta-Betrieb (`--urls`) mit einer eigenen Dedup-Rechnung,
zeigte der Join auf Chunks, die der Anreicherer nie gesehen hat.

**Deshalb rufen beide Skripte `select_unique_ids()` mit der vollständigen
Dateiliste auf und filtern erst danach** auf `--urls`, `--resume-from` und
`--shard`. Im Delta-Betrieb kostet das einen zusätzlichen Chunking-Durchlauf
(~1 min für 5.616 Dateien).

Die Regel geht über `DEDUPE_CHUNKS` und `DEDUPE_KEY_VERSION` in
`config_hash()` ein — sonst könnten zwei Läufe mit verschiedenen Regeln
zusammengejoint werden.

### 5.5 Der bewusst akzeptierte Preis

**323 von 5.604 Seiten (5,8 %) behalten keinen eigenen Chunk** und sind als
Quelle nicht mehr auffindbar. Ihr Inhalt ist andernorts wortgleich vorhanden.
Stichprobe:

```
http://itr.wp.hs-hannover.de/.../01-allgemeiner_teil_zulassung.pdf
http://itr.wp.hs-hannover.de/.../Allgemeiner_Teil_der_PO_2015.pdf
https://f1.hs-hannover.de/.../10_Ordnungen/60_Sonstige/Vorpr…
```

Gespiegelte Prüfungsordnungen. Für ein RAG-System ist gleichgültig, unter
welcher URL eine wortgleiche Information gefunden wird.

### 5.6 Verifikation

| Prüfung | Ergebnis |
|---|---|
| Determinismus (zwei Läufe) | identisch |
| Anreicherer == Vektorisierer | identisch, 119.791 IDs |
| Sharding (0 + 1 von 2) | 249 + 248 = 497, sauber |
| Anreicherer end-to-end (GPU) | `EXIT=0`, 0,2 % ohne Fragen |
| Vektorisierer end-to-end (GPU) | `EXIT=0`, 524 Punkte |
| **Anreicherungs-IDs ohne Vektor** | **0** |
| Fingerabdrücke beider Ausgaben | gleich |

---

## 6. Fingerabdrücke

| Hash | Stand |
|---|---|
| `21774915f81f9a2a` | vor der Base64-Bereinigung |
| `3757da190e218c3b` | nach Base64-Bereinigung (Vorbericht) |
| `5d4df3ab6a51ec37` | + Muster `bewerbung-nav` |
| **`8e7e7ce74f631963`** | **+ Deduplizierung — aktueller Stand** |

---

## 7. Betriebsablauf

### 7.1 Vorbedingung prüfen

```bash
cd $ARBEIT
source env_vllm.sh
$V/python -c "from chunking import config_hash; print(config_hash())"
#   erwartet: 8e7e7ce74f631963
ls enrichment.parquet enrichment.parquet.parts    # muss beides fehlen
```

Stimmt der Fingerabdruck nicht, **nicht starten** — dann läuft eine andere
Codeversion, und die Ausgabe passt später nicht zum Vektorisierer.

### 7.2 Anreicherung

```bash
sbatch enrich_vllm.sbatch          # 2 Array-Tasks, --batch-size 512
```

Nach fünf Minuten in das Log sehen (`../slurm_files/enrich_vllm_<JOBID>_0.log`,
siehe 7.2a und 7.2b). Diese Zeilen müssen kommen — die Zahlen sind die
tatsächlichen aus dem Lauf vom 13.08.2026, 16:40 Uhr:

```
Chunking-Fingerabdruck: 8e7e7ce74f631963
Gefunden: 5616 .md-Datei(en) in data/ingested
Deduplizierung: 119791 eindeutige Chunk(s) behalten.
Shard 0 von 2: 54970 von 110333 Chunk(s).
Zu verarbeiten: 54970 Chunk(s)  (zu kurz übersprungen: 19685,
                                 Duplikate übersprungen: 66916)
vLLM bereit. Denkmodus: aus, Sampling: greedy
Guided Decoding aktiv über 'structured_outputs'.
```

Die drei Zahlen gehen auf: 66.916 + 19.685 + 110.333 = 196.934. Der
`--min-chars`-Filter greift **vor** der Duplikatprüfung, deshalb erscheinen
kurze Chunks nicht in der Duplikatzahl.

Zeitrechnung: Modell laden und Deduplizierung kosten zusammen gut vier
Minuten (16:40 abgeschickt, 16:44:58 „vLLM bereit"), danach etwa 19 Chunks/s.
Ein Shard braucht also rund **50 Minuten**.

**`ohne Fragen` muss nahe null bleiben.** Im Test 0,2 %. Steigt der Wert über
etwa 5 %, abbrechen (`scancel`) — dann stimmt etwas mit Prompt oder Modell
nicht.

Vorsichtigere Variante: erst `sbatch --array=0 enrich_vllm.sbatch`, Ergebnis
ansehen, dann `--array=1`. Kostet doppelt Modellladezeit, riskiert aber nur
einen Shard.

### 7.2a Die Job-ID finden

Vier Wege, je nach Situation:

```bash
# 1. sbatch gibt sie beim Absenden aus
sbatch enrich_vllm.sbatch
#    → Submitted batch job 15267466

# 2. Was läuft gerade?
squeue -u HPC_KENNUNG
#    JOBID        NAME         STATE    TIME
#    15267466_0   enrich_vllm  RUNNING  5:04
#    15267466_1   enrich_vllm  RUNNING  5:04

# 3. Was lief heute? (auch abgeschlossene Jobs)
sacct -u HPC_KENNUNG --starttime today --format=JobID,JobName,State,Elapsed

# 4. Aus den Logdateinamen — der jüngste zuerst
ls -t $PROJEKT/slurm_files/ | head
```

**Zur Schreibweise bei Array-Jobs:** `sbatch` gibt **eine** Zahl aus, die
Array-Job-ID. Die einzelnen Tasks heißen `<ID>_<Nummer>`. In den Dateinamen
steht `%A` für die Array-Job-ID und `%a` für die Tasknummer:

```
enrich_vllm_15267466_0.log     ← Shard 0
enrich_vllm_15267466_1.log     ← Shard 1
```

Bei `vectorize.sbatch` (kein Array) heißt der Platzhalter `%j` und enthält
schlicht die Job-ID: `vectorize_15268001.log`.

### 7.2b Wo die Logs liegen

**In `$PROJEKT/slurm_files/`**, also `../slurm_files` vom Arbeitsverzeichnis
aus gesehen:

```bash
tail -f ../slurm_files/enrich_vllm_<JOBID>_0.log
```

> **Zwei Fallen an dieser Stelle.**
>
> Es gibt **zwei** Verzeichnisse namens `slurm_files`:
> `$PROJEKT/slurm_files/` (dorthin schreiben die Jobs) und
> `hsh_scraper/slurm_files/` (Reste vom März, wird nicht mehr beschrieben).
> Wer im falschen sucht, findet nur alte Dateien.
>
> `vectorize.sbatch` verwendet den **relativen** Pfad
> `../slurm_files/vectorize_%j.log`. Slurm löst `--output` gegen das
> Verzeichnis auf, aus dem *abgeschickt* wurde — nicht gegen das `cd` im
> Skript. `sbatch vectorize.sbatch` muss deshalb **aus `hsh_scraper` heraus**
> aufgerufen werden, sonst landet das Log irgendwo anders oder der Job
> scheitert am fehlenden Verzeichnis.

### 7.3 Zusammenführen — erst wenn **beide** Tasks fertig sind

```bash
# 1. Sind wirklich alle Tasks durch?
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
#    beide Zeilen müssen COMPLETED und 0:0 zeigen

# 2. Zusammenführen
source env_vllm.sh
$V/python enricher.py --merge --output enrichment.parquet
```

#### Was dabei passiert

`--merge` rechnet **nichts**. Es lädt kein Modell, braucht keine GPU und läuft
in Sekunden. Die vier Schritte:

1. `enrichment.parquet.parts/part_*.parquet` einsammeln — das Muster erfasst
   **alle** Shards (`part_s00_*` und `part_s01_*`), nicht nur einen.
2. Die Tabellen aneinanderhängen (`pa.concat_tables`). Reines Anfügen: keine
   Deduplizierung, keine Umsortierung, keine Neuberechnung. Die Zeilen stehen
   danach shardweise hintereinander — erst alle aus Shard 0, dann alle aus
   Shard 1.
3. Nach `enrichment.parquet.partial` schreiben (zstd, Stufe 3).
4. Atomar auf `enrichment.parquet` umbenennen.

Die Teildateien **bleiben liegen**. Sie sind die Grundlage des Wiederanlaufs;
nur wenn sich Prompt oder Chunking ändern, müssen sie weg (siehe 7.7).

#### Warum die Tasks nicht selbst mergen

Jeder Array-Task sieht beim Mergen die Teildateien *aller* Shards — auch die
der noch laufenden. Ein selbst mergender Task schriebe also eine Datei aus dem
Zwischenstand der anderen. Deshalb tun sie es bewusst nicht, und deshalb ist
der Merge ein eigener, manueller Schritt.

#### Die Falle: der Merge prüft nicht auf Vollständigkeit

`merge_parts()` nimmt, was es findet. Führst du zusammen, während Shard 1 noch
läuft oder gescheitert ist, entsteht eine **gültige, aber unvollständige**
Datei — ohne Fehlermeldung. Der Vektorisierer würde sie anstandslos
akzeptieren und nur eine niedrige Trefferquote melden.

Die Kontrolle ist die Zeilenzahl in der Abschlussmeldung:

```
221 Teildatei(en) → enrichment.parquet  (110333 Zeilen)
```

Diese Zahl muss der Summe aus beiden Shard-Logs entsprechen:

```bash
grep -h "Zu verarbeiten" ../slurm_files/enrich_vllm_<JOBID>_*.log
#   Shard 0: 54970 Chunk(s)
#   Shard 1: 55363 Chunk(s)   → zusammen 110.333
```

Stimmt sie nicht, fehlt ein Shard. Dann den fehlenden Task erneut abschicken
und **danach nochmals mergen** — der Merge ist wiederholbar und
verlustfrei: Er liest jedes Mal alle vorhandenen Teildateien neu und
überschreibt das Ergebnis atomar. Ein bereits vorhandenes
`enrichment.parquet` bleibt bis zur letzten Sekunde unversehrt.

#### Wie viele Teildateien zu erwarten sind

`FLUSH_ROWS = 500`: Nach je 500 fertigen Zeilen schreibt ein Shard eine
Teildatei, benannt `part_s<Shard>_<laufende Nummer>.parquet`. Bei 110.333
Chunks sind rund **221 Teildateien** zu erwarten. Der Shard steckt im Namen,
damit parallele Tasks sich nicht gegenseitig überschreiben.

### 7.4 Fragen prüfen

```bash
$V/python filter_fragen.py enrichment.parquet
$V/python qualcheck.py    enrichment.parquet
```

Vergleichswerte aus dem Testlauf: Paraphrasen 2,4 %, wortgleiche
Wiederholungen 5,5 %, Ja/Nein 0,0 %.

### 7.5 Vektorisierung

```bash
sbatch vectorize.sbatch            # erkennt enrichment.parquet selbst
```

Im Log prüfen:

```
Deduplizierung aktiv: 119791 eindeutige Chunk(s) werden vektorisiert.
Anreicherung: <N> Zeile(n) aus enrichment.parquet
avg_len gemessen: Text <X>, erweitert <Y> Tokens
```

**Die beiden `avg_len`-Werte notieren** — sie werden auf der Query-Seite
gebraucht, sonst stimmen die BM25-Termgewichte zwischen Index und Anfrage
nicht überein. Warnung zur Trefferquote unter 50 % bedeutet abweichende
Fingerabdrücke: abbrechen.

### 7.6 Übertragen und importieren

```bash
scp hsh_vectors.parquet hsh_vectors_sources.txt user@localhost:~/RAGHSH/hsh_scraper/
python local_importer.py
```

Unverändert zu beachten: Payload-Feld `questions`; zwei Sparse-Indizes
(`sparse`, `sparse_expanded`), beide `modifier=IDF`; zweiter Dense-Vektor
`questions` (1024 Dim., Cosine); Query-Seite mit `language="german"` und den
notierten `avg_len`-Werten.

### 7.7 Wenn etwas schiefgeht

**Zeitlimit oder Absturz:** denselben Shard erneut abschicken. Die Teildateien
in `enrichment.parquet.parts/` bleiben liegen, erledigte Chunks werden
übersprungen. Es kostet nur das erneute Modellladen.

**Die Falle:** Nach jeder Änderung an Prompt oder Chunking
`rm -rf enrichment.parquet.parts` — sonst überspringt der Wiederanlauf alles
und es passiert nichts.

---

## 8. Dateien

Das Arbeitsverzeichnis wurde am 13.08.2026 aufgeräumt: Es enthält nur noch,
was der Betriebsablauf aus Abschnitt 7 braucht. Alles Übrige liegt unter
`Backup/` — **verschoben, nichts gelöscht.**

### Im Arbeitsverzeichnis

| Datei | Zweck |
|---|---|
| `chunking.py` | gemeinsame Bibliothek: Chunk-Grenzen, IDs, Bereinigung, Deduplizierung |
| `enricher.py` | Fragen erzeugen (GPU) |
| `hpc_vectorizer.py` | Vektorisieren (GPU) |
| `filter_fragen.py` | Bericht über die erzeugten Fragen |
| `qualcheck.py` | **neu** — misst Paraphrasen, Wiederholungen, vage Referenz, Ja/Nein |
| `env.sh` | Umgebung für `venv_kisski` |
| `env_vllm.sh` | **neu** — vollständige vLLM-Umgebung, kommentiert |
| `enrich_vllm.sbatch` | **neu** — Array-Job, `--batch-size 512`, `SHARDS=2` |
| `enrich.sbatch` | Vorgänger auf transformers-Basis — als Rückfallebene behalten |
| `vectorize.sbatch` | Vektorisierung |
| `requirements.txt` | |
| `Sitzungsbericht_2026-08-13.md` | dieses Dokument |
| `data/`, `testdaten/`, `fastembed_cache/`, `hf_cache/`, `slurm_files/` | Daten und Caches |

Zusätzlich außerhalb: `$PROJEKT/cuda_compat/` — ausgepacktes cuda-compat-13-3.

### Geändert (Sicherungen unter `Backup/code_sicherungen/`)

| Datei | Änderung | Sicherung |
|---|---|---|
| `chunking.py` | Muster `bewerbung-nav`, Dedup-Abschnitt, `config_hash()` erweitert | `chunking.py.bak-vor-D` |
| `enricher.py` | `_VAGUE_REF_RE`, Prompt-Regel, `keep_ids` vor dem Zuschnitt | `enricher.py.bak-vor-B` |
| `hpc_vectorizer.py` | `keep_ids` vor `--urls`/`--resume-from`, `estimate_avg_len()` | `hpc_vectorizer.py.bak-vor-dedup` |
| `vectorize.sbatch` | `LD_LIBRARY_PATH` ergänzt | `vectorize.sbatch.bak` |

### `Backup/` — 52 verschobene Einträge, 6,4 MB

| Unterverzeichnis | Inhalt |
|---|---|
| `analyse_skripte/` | `dupcheck.py` (Duplikatquote), `fileanalyse.py` (Datei- gegen Chunk-Duplikation), `pruefung.py`/`pruefung2.py` (Belege zu Abschnitt 5), `keyvariants.py` (Ersparnis je Schlüssel), `konsistenz.py` (Determinismus beider Skripte), `testpatterns.py` (Gegenproben zu 4.1/4.2), `whydup.py`/`whydup2.py`, `vllm_smoke.py`, `run_hpc_vectorizer.sh` (H100-Vorgänger) |
| `code_sicherungen/` | die vier `.bak`-Stände und das frühere `alt/` |
| `testlaeufe/` | alle `enrichment_*.parquet` samt `.parts`, `vec_dedup_test.parquet` |
| `jobs/` | `smoke*.sbatch`, `bench_*.sbatch`, `vec_test.sbatch` |
| `logs/` | die zugehörigen Slurm-Protokolle |

> Die Skripte unter `analyse_skripte/` erzeugen die Zahlen dieses Berichts.
> Wer eine Aussage nachprüfen will, kopiert das betreffende Skript zurück ins
> Arbeitsverzeichnis — sie importieren `chunking.py` und laufen nur von dort.

---

## 9. Offene Punkte

| # | Punkt | Aufwand |
|---|---|---|
| 1 | **Produktionslauf** — Abschnitt 7 | ~2 h |
| 2 | Query-Seite: `language="german"` und gemessene `avg_len` übernehmen | 30 min |
| 3 | **Goldset** aus 50–100 echten Fragen mit korrektem Chunk | mehrere Std. |
| 4 | Prompt-Fingerabdruck in `PARQUET_SCHEMA` | 20 min |
| 5 | Near-Duplicate-Bericht (MinHash/LSH) für die Boilerplate-Familien | 2 h |
| 6 | Prefix Caching in vLLM einschalten | 15 min |
| 7 | Fallback-Kaskade in `_sampling_params()` auch für Generierungsfehler | 30 min |

### Zu Punkt 3 — unverändert die größte Lücke

Der Vektorisierer schreibt vier Arme nebeneinander, gemessen wird damit
weiterhin nichts. Recall@10 und MRR@10 über Baseline, nur-sparse, nur-dense und
beides wären der nächste Schritt. Alles in diesem Bericht ist
Durchsatz und Oberflächenqualität — **keine Aussage über Retrieval-Güte.**

### Zu Punkt 4 — eine bekannte Lücke

`PARQUET_SCHEMA` enthält `chunking_config_hash`, aber keinen
Prompt-Fingerabdruck. Eine Prompt-Änderung ist für den Join unsichtbar: eine
alte `enrichment.parquet` würde nach einer Überarbeitung anstandslos
akzeptiert. Der Vorbericht hat diese Falle für das Chunking sorgfältig
geschlossen und für den Prompt offen gelassen.

### Zu Punkt 5 — was die Deduplizierung nicht erreicht

Exakte Gruppierung verfehlt den NGG-Gleichstellungsabsatz, der zwischen
Stellenanzeigen variiert (Jaccard 0,63–0,95). Er erzeugt weiterhin wortgleiche
Fragen auf fünf verschiedenen Anzeigen.

Er wurde **bewusst nicht** aus `clean_body()` entfernt: Er ist beantwortbarer
Inhalt, kein Konvertierungsartefakt, und korpusweit gelöscht wäre „Werden
Schwerbehinderte bevorzugt berücksichtigt?" nirgends mehr beantwortbar. Das ist
eine Retrieval-Entscheidung, keine Bereinigung. Near-Duplicate-Behandlung wäre
der richtige Ort — mit Schwellwert ab etwa 0,95, darunter unterscheiden sich
die Texte in genau den Details, nach denen jemand sucht.

---

## 10. Was ungeprüft bleibt

Damit klar ist, wo die Aussagen dieses Berichts enden:

* **Der Durchsatz von 18,5 Chunks/s stammt aus einem Lauf über 508 Chunks der
  Testdaten** (mittlere Länge 89,7 Tokens). Der Gesamtbestand ist heterogener —
  längere Chunks aus PDFs, andere Verteilung. `--time` steht deshalb auf 2:30 h
  statt knapp kalkuliert.
* Der Vektorisierer wurde nur über die 50 Testdateien geführt, nicht über den
  Gesamtbestand.
* Von den 24 transformers-Ausfällen wurden 5 gegengeprüft, 19 nicht.
* Ob `--batch-size` über 512 hinaus noch etwas bringt, ist ungetestet.
* Die 892 Duplikatgruppen mit echtem Titelunterschied bleiben getrennt — dass
  die Trennung in allen 892 Fällen berechtigt ist, wurde an 4 Beispielen
  plausibilisiert, nicht durchgeprüft.

---

## 11. Rückblick

Der Vorbericht schloss mit dem Fazit, die Geschwindigkeitsjagd habe den
Großteil der Sitzungszeit verbraucht und nichts eingebracht. Das war zum
damaligen Zeitpunkt richtig — und ist es rückblickend nicht geblieben: Der
fehlende Baustein war ein einziges 112-MB-Paket, das ohne Root-Rechte
auspackbar ist.

Die eigentliche Lehre liegt woanders. Zwei Vermutungen dieser Sitzung haben der
Prüfung nicht standgehalten:

* Die Annahme, der Prompt sei schuld an den generischen Fragen — gemessen lag
  die Paraphrasenrate bei 3,1 %, und die wortgleichen Fragen stammten aus
  Chunktexten, die zu 0,60 bis 0,95 identisch waren. Das Modell verhielt sich
  korrekt.
* Die Annahme, `uniqueItems` würde helfen — es war nicht unterstützt *und*
  überflüssig.

Beide Male hätte ein Umbau Zeit gekostet und nichts verbessert. Der Unterschied
zum Vorbericht ist nicht mehr Sorgfalt, sondern dass die Messung vor dem Umbau
kam — was mit vLLM erstmals billig genug war: 28 Sekunden je Durchlauf statt
zweieinhalb Minuten.
