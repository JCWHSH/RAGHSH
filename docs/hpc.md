# RAGHSH auf dem HPC (KISSKI)

> **Platzhalter:** `HPC_KENNUNG` (KISSKI-Login), `PROJEKTKONTO`, `NUTZERNAME` (Projektpfad) und `HEIMSERVER` (Rechner mit `var/`) durch eigene Werte ersetzen; ebenso in `werkzeuge/hpc/env*.sh` und `*.sbatch`.

Wie der Index auf KISSKI entsteht: Zugang, Umgebung, Anreicherung (doc2query),
Vektorisierung, Delta-Fall und Kosten. Die Herkunft des heutigen Index mit allen
Zahlen steht in `docs/parquet_schema.md`, Abschnitt 0; den Ablauf vom Crawl bis
zum Import beschreibt `docs/pipeline.md`. Die Skripte liegen in `werkzeuge/hpc/`.

Stand: 11.11.2026, abgeglichen mit dem Projektverzeichnis auf dem Cluster.

---

## 1. Zugang

| | |
|---|---|
| Login | `HPC_KENNUNG@glogin9.hlrn.de`, Projekt-Account `PROJEKTKONTO` |
| Ausweichen | `glogin.hpc.gwdg.de`, dieselben Host-Keys (am 11.11.2026 brach glogin9 beim Verbindungsaufbau ab, glogin.hpc.gwdg.de landete auf glogin11) |
| Schlüssel | nur auf dem Laptop (`~/.ssh/id_ed25519`). openclaw und der Jetson haben **keinen** Zugang. |
| Projekt | `/mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME` (= `$PROJEKT`) |
| Arbeitsverzeichnis | `$PROJEKT/hsh_scraper` (= `$ARBEIT`) |

Hat glogin9 eine Störung, hilft im Laptop-`~/.ssh/config` ein Eintrag mit
`HostKeyAlias`. Dann bleibt der bekannte Host-Key gültig:

```
Host kisski
    HostName glogin.hpc.gwdg.de
    HostKeyAlias glogin9.hlrn.de
    User HPC_KENNUNG
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Die Befehle unten schreiben `kisski` für diesen Zugang.

Der Korpus liegt auf openclaw, der Schlüssel auf dem Laptop. Übertragungen
laufen deshalb über den Laptop, als Pipe ohne Zwischenkopie (siehe Abschnitt 4).

## 2. Umgebung auf dem Cluster

Alles liegt unter `$PROJEKT`. Diese Verzeichnisse und Dateien gibt es nur dort,
im Repo stehen sie nicht:

| Pfad | Inhalt |
|---|---|
| `venv_kisski/` | Conda-Umgebung, Python 3.11, onnxruntime-gpu 1.24.3 (CUDA/TensorRT), fastembed 0.7.4. Für den **Vektorisierer** und die CPU-Werkzeuge. |
| `venv_vllm/` | vLLM 0.26.0, torch 2.11.0+cu128. Nur für den **Anreicherer**. |
| `cuda_compat/` | CUDA 13.3 Forward Compatibility. Die vLLM-Wheels brauchen CUDA 13, der Cluster-Treiber hat 12.8. Läuft nur auf A100/H100. Einzelheiten stehen in `env_vllm.sh`. |
| `hf_cache/` | Modelle (63 GB): `Qwen/Qwen3.5-9B` (Anreicherung), `jinaai/jina-embeddings-v3`, Mistral (Altlast) |
| `hsh_scraper/fastembed_cache/` | jina v3 als ONNX und `Qdrant/bm25` für fastembed |
| `vllm_cache/` | JIT-Caches von vLLM, torch-inductor und flashinfer |
| `slurm_files/` | Job-Logs aller sbatch-Dateien |
| `hsh_scraper/data/ingested/` | Korpus auf dem Cluster, heute der Stand vom 11.08.2026 (`data/hsh_korpus_2026-08-11.tar.xz`, 5.616 Dateien) |
| `hsh_scraper/Backup/` | ältere Codestände, Benchmark- und Smoke-Jobs |

Zwei Umgebungsdateien (`werkzeuge/hpc/`) setzen die Variablen:

- `env.sh` für `venv_kisski`. Setzt `$PYTHON`, `FASTEMBED_CACHE_PATH`,
  `HF_HUB_CACHE` und `LD_LIBRARY_PATH`. Ohne den letzten Pfad scheitert
  der Import an `CXXABI_1.3.15`.
- `env_vllm.sh` für `venv_vllm`, mit Forward Compatibility, CUDA-13-Toolchain
  aus den pip-Paketen und den Cache-Pfaden. **Nicht** zusammen mit `env.sh` in
  derselben Shell laden: Die CUDA-13-Bibliotheken in `LD_LIBRARY_PATH` sind für
  onnxruntime im anderen venv nicht gedacht.

Compute-Nodes haben kein Internet. Modelle lädt man auf dem **Login-Node** nach,
zum Beispiel `$PYTHON hpc_vectorizer.py --download-only --cpu`. Rechenlast
gehört nicht auf den Login-Node. Die CPU-Schritte unten (Zusammenführen,
Filtern) dauern je ein bis drei Minuten, das ist dort üblich.

**Neu aufbauen** (falls das Projektverzeichnis verloren geht): `venv_kisski` als
Conda-Umgebung mit Python 3.11 aus `requirements.txt`, dann
`pip uninstall -y onnxruntime && pip install onnxruntime-gpu`. `venv_vllm` mit
`pip install vllm==0.26.0`, dazu das CUDA-13.3-Compat-Paket nach
`cuda_compat/`. Was dabei schiefging und wie es gelöst wurde, steht in den
Kommentaren von `env_vllm.sh` und in `docs/archiv/2026-08_hpc/Sitzungsbericht_2026-08-13.md`.

### Slurm

| | |
|---|---|
| Partition | `kisski`, A100 80 GB (4 je Knoten), Zeitlimit 2 Tage |
| H100 | `kisski-h100`, nicht benutzt |
| Jobs | `sbatch <datei>`, `squeue -u $USER`, `scancel <id>`, `tail -f ../slurm_files/<log>` |

## 3. Welcher Code auf den Cluster gehört

**Das ist die wichtigste Entscheidung vor jedem Lauf.** Alle HPC-Skripte
importieren `chunking.py` aus ihrem eigenen Verzeichnis, und dessen
`config_hash()` legt die Chunk-Grenzen und damit die Punkt-IDs fest.

| Stand | `chunking_config_hash` | Wo |
|---|---|---|
| Index in Produktion und Entwicklung (14.08.2026) | `5d4df3ab6a51ec37` | `werkzeuge/hpc/stand_2026-08-14/` (= `$ARBEIT/chunking.py`, `$ARBEIT/hpc_vectorizer.py`) |
| Repo heute (seit 22.09.2026: Tabellen zeilenweise, Kontextkopf) | `e73a521098f93fc3` | `src/raghsh/index/chunking.py`, `titel.py`, `hpc_vectorizer.py` |

`selection_hash` ist bei beiden `40fbfb36ee010cf8`.

Daraus folgt:

- **Delta gegen den heutigen Index:** nur mit `stand_2026-08-14/`. Mit dem
  Repo-Chunking passten die neuen Punkte nicht zum Bestand. Anreicherer und
  Vektorisierer brechen dann am Hash-Vergleich ab, und das ist gewollt.
- **Repo-Chunking in Betrieb nehmen:** Das geht nur als Volldurchlauf mit
  neuer Anreicherung (Abschnitt 5). Gemessen ist es bisher nur am Testindex
  (`auswertung/testindex/`).

Prüfen, welcher Stand auf dem Cluster liegt:

```bash
ssh kisski 'source /mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME/hsh_scraper/env.sh; \
  $PYTHON -c "import chunking as c; print(c.config_hash(), c.selection_hash())"'
```

## 4. Daten hin und zurück

Korpus von openclaw auf den Cluster (komplett, ersetzt `data/ingested`):

```bash
# auf dem Laptop
ssh HEIMSERVER 'tar cJf - -C ~/RAGHSH/var/korpus .' \
  | ssh kisski 'cd /mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME/hsh_scraper \
      && mv data/ingested data/ingested.vor_$(date +%F) && mkdir data/ingested && tar xJf - -C data/ingested'
```

Der **ganze** Korpus muss auf dem Cluster liegen, auch im Delta-Fall:
`select_unique_ids()` arbeitet global (Abschnitt 6).

Ergebnis zurück nach openclaw:

```bash
P=/mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME/hsh_scraper
ssh kisski "cat $P/hsh_vectors_delta.parquet" \
  | ssh HEIMSERVER 'cat > ~/RAGHSH/var/vektoren/hsh_vectors_delta.parquet'
```

Skripte aus dem Repo auf den Cluster:

```bash
ssh HEIMSERVER 'tar cf - -C ~/RAGHSH/werkzeuge/hpc --exclude=stand_2026-08-14 --exclude=__pycache__ . \
                                 -C stand_2026-08-14 .' \
  | ssh kisski 'tar xf - --no-overwrite-dir -C /mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME/hsh_scraper'
```

Für den Volldurchlauf mit dem neuen Chunking kommen statt `stand_2026-08-14/`
die drei Dateien aus `src/raghsh/index/` (`chunking.py`, `titel.py`,
`hpc_vectorizer.py`) hinüber. Sie fallen ohne installiertes Paket auf Importe
aus dem eigenen Verzeichnis zurück.

## 5. Volldurchlauf

Nötig bei geändertem Chunking, neuem Embedding-Modell oder neuem Prompt. Jeder
Schritt hat ein eigenes Artefakt:

```bash
cd $ARBEIT
# 1. Anreicherung, 2 Shards à ~55 min auf je einer A100
sbatch enrich_vllm.sbatch
#    wenn beide Tasks fertig sind:
source env_vllm.sh && $V/python enricher.py --merge --output enrichment.parquet

# 2. Fragenfilter (CPU, Login-Node). Erst nur berichten, dann schreiben.
source env.sh
$PYTHON filter_fragen.py enrichment.parquet
$PYTHON filter_fragen.py enrichment.parquet --ausgabe enrichment_wdh10.parquet --schreiben

# 3. Vektorisierung, ~20 min auf einer A100. Der Log nennt avg_len beider BM25-Arme.
sbatch --export=ALL,ENRICHMENT=enrichment_wdh10.parquet,AUSGABE=hsh_vectors_wdh10.parquet vectorize.sbatch
```

Danach zurückholen, `delete_qdrant.py`, Import, `build_doc_index --rebuild`
(`docs/pipeline.md`). Die neuen `avg_len`-Werte aus dem Log gehören in
`docs/parquet_schema.md` 3.5, sie sind die Vorgabe für spätere Deltas.

Zur Wahl des Anreicherers:

- `werkzeuge/hpc/enricher.py` hat den Bestand vom 13.08.2026 erzeugt.
- `auswertung/testindex/enricher_v2.py` ist die Variante mit korrigiertem
  Prompt aus den Messungen vom September (`fragen_v2_messung.tsv`). Ob sie den
  Bestand ersetzt, ist noch nicht entschieden. `--urls` hat sie nicht.

`erbe_fragen.py` war ein einmaliger Übergang. Es hat am 14.08.2026 den 1.343
Punkten der neuen Dedup-Regel die Fragen ihres wortgleichen Vertreters gegeben.
In einem Volldurchlauf reichert `enricher.py` alle Punkte der aktuellen Regel
selbst an, das Skript wird dann nicht gebraucht. Gebraucht wird es erst wieder,
wenn sich nur die Auswahlregel ändert (`selection_hash`) und eine
GPU-Stunde gespart werden soll.

`qualcheck.py <enrichment.parquet>` misst Paraphrasen je Chunk und wortgleiche
Fragen über Chunks hinweg. Nach jedem Prompt-Wechsel sinnvoll.

## 6. Delta-Durchlauf

Der Normalfall nach einem Teil-Crawl, wie in `docs/pipeline.md`. Hier steht,
was auf dem Cluster passiert. Ohne Anreicherung hätten neue Punkte nur drei
statt vier Arme und verlören in der Fusion. Deshalb läuft die Anreicherung
immer mit.

Voraussetzungen:
- Auf dem Cluster liegt `stand_2026-08-14/` (Abschnitt 3).
- `data/ingested` ist der **vollständige aktuelle** Korpus (Abschnitt 4). Ein
  Verzeichnis nur mit den Delta-Seiten genügt nicht: Anreicherer und
  Vektorisierer wählen die Gruppenvertreter der Deduplizierung über alle
  Dateien und schränken erst danach auf `--urls` ein. Über eine Teilmenge
  gerechnet fiele die Wahl anders aus.
- `delta.txt` aus `python -m raghsh.crawl.delta -o delta.txt` liegt in `$ARBEIT`.
- `enrichment_v2.parquet` (der **ungefilterte** Bestand) liegt in `$ARBEIT`.
  Eine Kopie steht in openclaw `var/vektoren/`.

```bash
cd $ARBEIT

# 1. Nur die Delta-Seiten anreichern. Ein Shard; ~4 min Modellladen + 18 Chunks/s.
sbatch --array=0 --export=ALL,SHARDS=1,URLS=delta.txt,AUSGABE=enrichment_delta.parquet enrich_vllm.sbatch
source env_vllm.sh && $V/python enricher.py --merge --output enrichment_delta.parquet
#    (neue Shell)

# 2. In den Bestand einfügen: alle Zeilen der Delta-URLs werden ersetzt.
source env.sh
$PYTHON fragen_zusammenfuehren.py --bestand enrichment_v2.parquet \
    --delta enrichment_delta.parquet --urls delta.txt --output enrichment_v3.parquet

# 3. Filtern, und zwar über den ganzen Bestand: Die Wiederholungszählung ist
#    eine Korpuseigenschaft.
$PYTHON filter_fragen.py enrichment_v3.parquet --ausgabe enrichment_v3_wdh10.parquet --schreiben

# 4. Nur die Delta-Seiten vektorisieren, mit dem avg_len des Bestands
sbatch --export=ALL,URLS=delta.txt,AVG_LEN=88.3,AVG_LEN_EXPANDED=107.1,ENRICHMENT=enrichment_v3_wdh10.parquet,AUSGABE=hsh_vectors_delta.parquet vectorize.sbatch
```

Danach `enrichment_v3.parquet` als neuen ungefilterten Bestand behalten (auch
in openclaw `var/vektoren/`). Beim nächsten Delta ist er `--bestand`.

`AVG_LEN` ist Pflicht. Ohne den Wert misst der Vektorisierer `avg_len` an den
wenigen Delta-Seiten, und deren BM25-Gewichte wären anders normalisiert als der
Rest des Index. `vectorize.sbatch` bricht deshalb ab, wenn `URLS` ohne beide
Werte gesetzt ist. Die Werte stammen aus dem Lauf vom 14.08.2026
(`docs/parquet_schema.md` 3.5). Nach einem Volldurchlauf gelten die neuen.

Grenzen des Delta-Falls:
- Eine geänderte Seite kann einen Gruppenvertreter auf einer **anderen** Seite
  ablösen. Dessen Punkt bleibt in Qdrant, bis zum nächsten Volldurchlauf
  (doppelter Text, sonst harmlos).
- Gelöschte Seiten bleiben in Qdrant („Verwaiste Chunks“ in `docs/pipeline.md`).
  In `enrichment_v3.parquet` bleiben ihre Fragen ebenfalls stehen. Das stört
  nicht, weil der Vektorisierer nur Fragen zu vorhandenen Chunks verwendet.

## 7. Kosten

Kontingent `PROJEKTKONTO`, gemessen auf A100-SXM4-80GB:

| Schritt | Umfang | Dauer |
|---|---|---|
| Anreicherung voll | ~113.000 Chunks | 1,7 GPU-h (2 × ~55 min) |
| Anreicherung Delta | einige hundert Chunks | ~5 min + Wartezeit |
| Vektorisierung voll | 121.134 Punkte, 4 Arme | ~20 min |
| Vektorisierung Delta | | wenige min, davon ~1 min Chunking über den ganzen Korpus |
| Zusammenführen, Filtern | CPU, Login-Node | je 1–3 min |

## 8. Dateien in `werkzeuge/hpc/`

| Datei | Aufgabe |
|---|---|
| `enricher.py` | doc2query mit Qwen3.5-9B. vLLM oder transformers, Shards, Wiederanlauf, `--urls` (seit 11.11.2026) |
| `fragen_zusammenfuehren.py` | Delta-Anreicherung in den ungefilterten Bestand einfügen (11.11.2026) |
| `filter_fragen.py` | zu allgemeine Fragen verwerfen (Wiederholung > 10, Dokumentfrequenz > 2 %) |
| `erbe_fragen.py` | Fragen an zusätzlich behaltene, wortgleiche Punkte vererben (Übergang 14.08.2026) |
| `qualcheck.py` | Qualitätsmaße einer Anreicherung |
| `enrich_vllm.sbatch` | Anreicherung als Slurm-Array. `SHARDS`, `URLS`, `AUSGABE` per `--export` |
| `vectorize.sbatch` | Vektorisierung. `ENRICHMENT`, `AUSGABE`, `URLS`, `AVG_LEN`, `AVG_LEN_EXPANDED` per `--export` |
| `env.sh`, `env_vllm.sh` | Umgebungen für `venv_kisski` bzw. `venv_vllm` |
| `stand_2026-08-14/` | `chunking.py` und `hpc_vectorizer.py`, mit denen der heutige Index entstand |

Berichte und Messungen der HPC-Sitzungen vom August liegen in
`docs/archiv/2026-08_hpc/`.
