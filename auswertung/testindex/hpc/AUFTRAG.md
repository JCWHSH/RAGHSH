# Auftrag: Testindex für Chunking v2 auf dem HPC vektorisieren und auswerten

Du bist ein Claude-Agent auf einem Rechner mit Zugang zum GWDG/NHR-HPC. Deine
Aufgabe ist ein klar umrissener Delta-Lauf der Vektorisierung für das Projekt
RAGHSH (RAG-System der Hochschule Hannover) und die anschließende Auswertung.
Alles, was du brauchst, liegt bereit; du musst nichts entwerfen, nur sorgfältig
ausführen, prüfen und berichten.

## Hintergrund (kurz)

RAGHSH indexiert die Website der HsH in Qdrant (Collection `hsh_knowledge`,
121.134 Chunks, vier Vektor-Arme). Das Chunking wurde am 22.09.2026 geändert
(„Chunking v2"): Markdown-Tabellen werden zeilenweise mit wiederholter
Kopfzeile geteilt, und jeder Chunk trägt einen Kontextkopf (Dokumenttitel ›
Überschrift), der mit in Dense- und BM25-Vektor eingeht. Bevor der gesamte
Korpus neu vektorisiert wird, soll ein Test zeigen, ob das im Retrieval etwas
bringt: Nur die 108 Golddokumente der Evaluations-Fallbasis werden neu
vektorisiert und in eine **Testkopie** der Collection (`hsh_knowledge_v2test`)
importiert; dann läuft die Retrieval-Evaluation dagegen.

Die Vektorisierung braucht eine GPU. Der Zielserver hat 8 GB RAM und eine CPU
von 2011 — dort darf **kein Einbettungsmodell** laufen (drei OOM-Abbrüche am
22.09.). Deshalb du und der HPC.

## Zugänge

- **Zielserver (Produktion):** `ssh HEIMSERVER` (Nutzer `christian`,
  Home `/home/christian`). Projekt: `/home/christian/RAGHSH/hsh_scraper/`,
  Python-venv dort: `.venv`. Zusätzlich ist dieser Server auf deinem Rechner
  unter **/mnt/nemo** gemountet — prüfe zuerst, welcher Pfad auf dem Mount
  `/home/christian/RAGHSH/hsh_scraper/` entspricht (`ls /mnt/nemo`), und
  benutze für Dateitransfers den Mount oder `scp`, was zuverlässiger ist.
- **HPC:** Login `HPC_KENNUNG@glogin9.hlrn.de`, Projektverzeichnis
  `/mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME/hsh_scraper/`.
  Dort existieren aus früheren Läufen ein venv mit `fastembed`,
  `onnxruntime-gpu`, `pyarrow`, `langchain-text-splitters`, der Modell-Cache
  (`FASTEMBED_CACHE_PATH`) und ein `vectorize.sbatch`. Sieh dir `vectorize.sbatch`
  an, bevor du etwas startest — es zeigt Partition, GPU-Anforderung,
  Modulbefehle und venv-Aktivierung, die auf diesem System funktionieren.
  Übernimm diese Konventionen; erfinde keine eigenen.

## Eingabedateien

Auf dem Zielserver unter `/home/christian/RAGHSH/hsh_scraper/eval_gold/testindex/hpc/`:

| Datei | Zweck | md5 (Anfang) |
|---|---|---|
| `chunking.py` | Chunking v2 (ersetzt die alte Fassung auf dem HPC) | `0454e450` |
| `titel.py` | wird von chunking.py importiert (Kontextkopf) | `efd415c8` |
| `hpc_vectorizer.py` | Vektorisierer, liest `kontext` und `einbettungstext()` | `e9faf7c2` |
| `urls.txt` | die 108 source_urls, die neu vektorisiert werden | `072e6138` |
| `enrichment_teil.parquet` | doc2query-Fragen für 1.450 unveränderte Chunks (id, questions) | `327a8259` |
| `hsh_korpus_2026-09-22.tar.xz` | `data/ingested/` — 5.619 Markdown-Dateien, Stand 22.09. (Umlaute repariert) | `a993b983` |

Prüfe die md5-Summen nach dem Kopieren (`md5sum`), bevor du weitermachst.

## Schritte auf dem HPC

1. **Arbeitsverzeichnis anlegen**, nichts Bestehendes überschreiben:
   `…/hsh_scraper/testindex_v2/`. Dorthin die sechs Dateien. Das Archiv dort
   entpacken → `testindex_v2/ingested/`. Die alte `chunking.py` im
   Projektverzeichnis **nicht** anfassen; der Lauf importiert die neue aus dem
   Arbeitsverzeichnis, weil das Skript aus diesem Verzeichnis gestartet wird.
2. **Vektorisieren** — aus `testindex_v2/`, im projekteigenen venv, auf einer
   GPU (per `sbatch` nach dem Muster von `vectorize.sbatch` oder interaktiv
   mit `srun`; auf dem Login-Knoten nicht rechnen):

   ```bash
   python hpc_vectorizer.py --require-gpu \
     --input-dir ingested \
     --urls urls.txt \
     --enrichment enrichment_teil.parquet \
     --avg-len 88.3 --avg-len-expanded 107.1 --avg-len-sample 0 \
     --output vektoren.parquet
   ```

   Erwartung: Der Lauf chunkt alle 5.619 Dateien für die korpusweite
   Deduplizierung (~1–2 min), vektorisiert dann nur die 108 URLs (~2.780
   Chunks vor Dedup, GPU-Zeit wenige Minuten) und schreibt `vektoren.parquet`
   sowie `vektoren_sources.txt`. Die `--avg-len`-Werte sind die der
   Produktion und dürfen nicht geändert werden — sonst passen die
   BM25-Gewichte der neuen Punkte nicht zu den alten.
   `--require-gpu` bricht ab, wenn keine CUDA-Session zustande kommt; das ist
   gewollt. Falls die Anreicherungsdatei eine Warnung „ohne
   chunking_config_hash" auslöst: erwartet, weiterlaufen lassen.
3. **Prüfen**, bevor etwas zurückkopiert wird (Python mit pyarrow genügt):
   - Zeilenzahl: zwischen 2.000 und 2.800.
   - Spalten enthalten `kontext`, `vector`, `vector_questions`,
     `sparse_indices`, `sparse_expanded_indices`, `chunking_config_hash`.
   - `chunking_config_hash` ist in allen Zeilen gleich und stimmt mit dem Wert
     in `/home/christian/RAGHSH/hsh_scraper/eval_gold/testindex/vorbereitung.json`
     überein (Feld `chunking_config_hash`). Weicht er ab, wurde eine andere
     `chunking.py` benutzt — dann abbrechen und berichten, nicht importieren.
   - `kontext` ist in > 95 % der Zeilen nicht leer.
   - Stichprobe: eine Zeile, deren `text` mit `|` beginnt, hat in der zweiten
     Zeile die Trennzeile `|---|…` (Kopf wiederholt).
4. **Zurückkopieren** nach
   `/home/christian/RAGHSH/hsh_scraper/eval_gold/testindex/vektoren.parquet`
   und `…/vektoren_sources.txt` (Mount oder scp). md5 vergleichen.

## Schritte auf dem Zielserver (per ssh)

Alles Folgende ist leichtgewichtig (kein Modell, kein nennenswerter Speicher).
Der MCP-Server (`systemctl --user status hsh-mcp.service`) muss laufen und
**darf nicht gestoppt oder neu gestartet werden**.

5. **Import in die Testkopie** — ausschließlich in `hsh_knowledge_v2test`,
   nie in `hsh_knowledge`:

   ```bash
   cd /home/christian/RAGHSH/hsh_scraper && source .venv/bin/activate
   HSH_COLLECTION=hsh_knowledge_v2test python local_importer.py --input eval_gold/testindex/vektoren.parquet
   ```

   Die Umgebungsvariable ist entscheidend. Prüfe vor dem Aufruf, dass sie
   gesetzt ist, und danach, dass `curl -s localhost:6333/collections/hsh_knowledge |
   python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["points_count"])'`
   weiterhin **121134** ergibt. Ist das nicht so, sofort abbrechen und berichten.
   Der Importer löscht je source_url den Altbestand in der Testkopie und
   schreibt die neuen Punkte; die Punktzahl der Testkopie ändert sich dadurch
   leicht — das ist richtig.
6. **Retrieval-Lauf gegen die Testkopie** (~6 Minuten, 75 Fälle, kein LLM):

   ```bash
   python eval_gold/lauf.py --nur-retrieval --collection hsh_knowledge_v2test \
     --notiz "Chunking v2 Testindex: 108 Golddokumente neu (Tabellen, Kontextkopf), Rest alt — Obergrenze"
   ```

   Das Manifest des Laufs muss unter `konfiguration.COLLECTION_NAME` den Wert
   `hsh_knowledge_v2test` tragen; prüfe das.
7. **Auswerten** — Vergleich gegen den Retrieval-Stand der Produktion,
   Lauf `2026-09-21_1922_retrieval_t+q_se_dbsf_td12` (59 von 73 in der
   Kandidatenliste):

   ```bash
   python eval_gold/auswertung.py <neue-lauf-id>
   python eval_gold/auswertung.py --vergleich 2026-09-21_1922_retrieval_t+q_se_dbsf_td12 <neue-lauf-id>
   python eval_gold/bericht.py <neue-lauf-id>
   python eval_gold/bericht.py --vergleich 2026-09-21_1922_retrieval_t+q_se_dbsf_td12 <neue-lauf-id>
   ```

## Bericht

Am Ende ein kurzer Bericht mit: Zeilenzahl und Hash des Parquets, GPU-Zeit,
Punktzahl der Testkopie vor/nach dem Import, Punktzahl der Produktion
(unverändert 121134), Lauf-ID, die Zahlen „Gold in Kandidaten" alt/neu, die
Liste „Neu in Kandidaten" und „Verloren" aus dem Vergleich, und die Pfade der
beiden HTML-Seiten. Besonders interessant sind die Fälle g31, g33, g34, g50
(Tabellen), g44 und g23–g28 (Kontextkopf). Bewerte nicht, ob sich der volle
Re-Import lohnt — das entscheidet Christian anhand der Zahlen; nenne aber
ausdrücklich, dass der Test eine Obergrenze misst (nur die Golddokumente haben
den neuen Index, ihre Konkurrenten nicht).

## Regeln

- Nichts committen, nichts pushen. Keine Datei außerhalb von `testindex_v2/`
  (HPC) bzw. `eval_gold/testindex/` und `eval_gold/laeufe/` (Zielserver)
  verändern.
- Auf dem Zielserver nichts löschen; falls doch nötig, `trash` statt `rm`.
- Auf dem Zielserver **keine** Einbettungsmodelle laden, keinen
  `hpc_vectorizer.py`, keinen zweiten MCP-Server. Nur Importer, `lauf.py`
  (ein MCP-Client) und die Auswertung.
- Wenn ein Schritt anders ausgeht als hier beschrieben, nicht improvisieren:
  anhalten, den Zustand beschreiben, berichten. Ein halb importierter
  Testindex ist unkritisch (Testkopie), ein Eingriff in `hsh_knowledge` wäre es
  nicht.
- Alle Ausgaben, Zeiten und Abweichungen mitschreiben; sie gehören in den
  Bericht.
