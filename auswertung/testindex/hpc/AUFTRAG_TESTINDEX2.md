# Auftrag: Testindex mit den neuen doc2query-Fragen

Du bist derselbe Claude-Agent, der den Testindex für Chunking v2 (AUFTRAG.md)
und den Fragenlauf (AUFTRAG_FRAGEN.md) ausgeführt hat. Beides ist erledigt und
ausgewertet. Dieser Auftrag ist der Abschluss: Er bringt beide Änderungen im
selben Index zusammen und misst sie da, wo es zählt — im Retrieval, nicht in
der Kosinus-Ähnlichkeit.

## Warum noch ein Lauf

Der Chunking-Test (`2026-09-22_1117_retrieval_v2test`) sieht in der Summe
schlechter aus als die Produktion (57 statt 59 Fälle in der Kandidatenliste),
ist es aber nicht:

- Gold auf Rang 1 stieg von **20 auf 26**, 19 Fälle verbesserten ihren Rang,
  7 verschlechterten ihn.
- Die drei verlorenen Fälle (g11, g34, g38) verloren **ausschließlich den
  Fragen-Arm**: `enrichment_teil.parquet` übernahm Fragen nur dort, wo der
  Chunktext unverändert blieb — umchunkte Tabellen fielen durch. Bei DBSF
  fällt ein Dokument, das nur noch aus einem Arm kommt, aus den Top 12.

Dazu die Zahl, die der Fragenlauf geliefert hat und die im Bericht unterging:
Von den 211 Tabellenchunks des Testindex tragen mit den **alten** Fragen nur
**18** einen Fragen-Arm, mit den **neuen** sind es **208**. Der Fragen-Arm
existiert für Tabellen heute praktisch nicht.

Beide Befunde führen auf denselben Lauf: dieselben Chunks, aber vollständige
Fragen. Gegen den 57er-Lauf isoliert er den Frageneffekt bei identischem
Chunking; gegen die Produktion zeigt er, was Chunking und Fragen zusammen
bringen.

## Ein Lauf, keine neue Anreicherung

Die Fragen sind fertig. `enrichment_gold_v2.parquet` (1952 Zeilen) liegt auf
dem Zielserver; doc2query läuft **nicht** noch einmal. Was auf der GPU läuft,
ist allein der Vektorisierer — die Fragen müssen zu Vektoren werden, damit sie
im Fragen-Arm wirken, und das sind wenige Minuten.

Ein Vorbehalt gehört in den Bericht, nicht in einen zweiten Lauf: Drei
Beispiele im damaligen Prompt waren wörtlich Nutzerfragen aus der Fallbasis
(g23 Open Access, g27 TOEIC, g28 Train-the-Trainer). Deren Fragen sind zu gut,
weil sie abgeschrieben sind. Weise diese drei Fälle gesondert aus und zähle
sie nicht mit; die übrigen 70 sind sauber. Der Prompt ist inzwischen
korrigiert — das wirkt sich erst auf den Korpuslauf aus, nicht auf diesen Test.

## Zugänge

Unverändert zu AUFTRAG.md:

- **Zielserver, zwei Sichten auf dasselbe Verzeichnis** — halte sie
  auseinander:
  - **Dateien lesen und schreiben** über den Mount:
    `/mnt/nemo/RAGHSH/hsh_scraper/`
  - **Befehle ausführen** über `ssh HEIMSERVER`; dort heißt dasselbe
    Verzeichnis `/home/christian/RAGHSH/hsh_scraper/`, venv `.venv`. Ein
    Mount-Pfad in einem ssh-Befehl schlägt fehl — den Mount kennt nur dein
    eigener Rechner.
- **HPC:** `HPC_KENNUNG@glogin9.hlrn.de`,
  `/mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME/hsh_scraper/`.
  Deine Arbeitsverzeichnisse `testindex_v2/` (Vektorisierung) und `fragen_v2/`
  (doc2query) stehen noch — benutze sie weiter, lege nichts neu an.

## Eingabedateien

Über den Mount unter `/mnt/nemo/RAGHSH/hsh_scraper/eval_gold/testindex/`:

| Datei | Zweck | md5 (Anfang) |
|---|---|---|
| `enrichment_gold_v2.parquet` | Fragen aus deinem Lauf, 1952 Zeilen | `17153d43` |
| `enricher_v2.py` | **korrigierte** Fassung, ersetzt deine auf dem HPC | `72c75586` |
| `hpc/AUFTRAG_FRAGEN.md` | der Prompt in seiner korrigierten Fassung | `d6f47355` |

Die übrigen Eingaben (`urls.txt`, `ingested/`, `chunking.py`, `titel.py`,
`hpc_vectorizer.py`) liegen unverändert in `testindex_v2/` auf dem HPC.

Der Chunking-Stand muss über alle Läufe derselbe sein:
`chunking_config_hash = e73a521098f93fc3`. Er steht in
`/mnt/nemo/RAGHSH/hsh_scraper/eval_gold/testindex/vorbereitung.json`, in
`vektoren.parquet` und in `enrichment_gold_v2.parquet` im selben Verzeichnis. Weicht irgendwo etwas ab: abbrechen, berichten.

## Schritte

1. `/mnt/nemo/RAGHSH/hsh_scraper/eval_gold/testindex/enrichment_gold_v2.parquet`
   nach `testindex_v2/` auf dem HPC kopieren, md5 prüfen (`17153d43…`).
2. Vektorisieren, aus `testindex_v2/`, auf einer GPU:

   ```bash
   python hpc_vectorizer.py --require-gpu \
     --input-dir ingested \
     --urls urls.txt \
     --enrichment enrichment_gold_v2.parquet \
     --avg-len 88.3 --avg-len-expanded 107.1 --avg-len-sample 0 \
     --output vektoren_fragen.parquet
   ```

   Die `--avg-len`-Werte bleiben auf den Produktionswerten, auch wenn die
   neuen Fragen die Dokumentlängen ändern — sonst passen die BM25-Gewichte
   der neuen Punkte nicht zu den alten. Setze das Slurm-TimeLimit realistisch
   (15–20 min); dein Backfilling-Befund aus dem letzten Lauf gilt weiter.
3. Prüfen: ~2.021 Zeilen, `chunking_config_hash` durchgehend `e73a521098f93fc3`,
   und — der eigentliche Punkt dieses Laufs — **mindestens 1.900 Zeilen mit
   nicht-leerem `questions`** und **mindestens 200 der Zeilen, deren `text`
   mit `|` beginnt, mit Fragen**. Wird das nicht erreicht, hat die
   Anreicherung nicht gegriffen: abbrechen, berichten.
4. Zurückkopieren nach
   `/mnt/nemo/RAGHSH/hsh_scraper/eval_gold/testindex/vektoren_fragen.parquet`,
   md5 vergleichen.
5. Import **nur** in die Testkopie — dieser Block läuft per ssh **auf** dem
   Zielserver, also mit dessen eigenen Pfaden:

   ```bash
   cd /home/christian/RAGHSH/hsh_scraper && source .venv/bin/activate
   HSH_COLLECTION=hsh_knowledge_v2test python local_importer.py \
     --input eval_gold/testindex/vektoren_fragen.parquet
   ```

   Danach muss `hsh_knowledge` weiterhin **121134** Punkte haben (die
   Testkopie hatte vor diesem Auftrag 121214). Ist das nicht so: sofort
   anhalten und berichten.
6. Retrieval-Lauf (~6 min, kein LLM), ebenfalls per ssh:

   ```bash
   python eval_gold/lauf.py --nur-retrieval --collection hsh_knowledge_v2test \
     --notiz "Chunking v2 + Fragen v2 (alter Prompt) auf 108 Golddokumenten — Obergrenze"
   ```

7. Auswerten, gegen **beide** Vorläufer, per ssh:

   ```bash
   python eval_gold/auswertung.py <lauf-id>
   python eval_gold/auswertung.py --vergleich 2026-09-22_1117_retrieval_v2test <lauf-id>
   python eval_gold/auswertung.py --vergleich 2026-09-21_1922_retrieval_t+q_se_dbsf_td12 <lauf-id>
   python eval_gold/bericht.py <lauf-id>
   python eval_gold/bericht.py --vergleich 2026-09-22_1117_retrieval_v2test <lauf-id>
   ```

### Was danach kommt (nicht Teil dieses Auftrags)

Der Korpuslauf würde die Fragen mit dem korrigierten Prompt neu erzeugen
(`enricher_v2.py`, md5 `72c75586`). Fang damit nicht an — darüber wird anhand
der Zahlen aus diesem Lauf entschieden.

## Bericht

Eine Tabelle über drei Zustände — Produktion,
`2026-09-22_1117_retrieval_v2test` und dieser Lauf — mit je: Gold in
Kandidaten (von 73), Gold auf Rang 1, Median-Rang, Diagnosen. Dazu:

- „Neu in Kandidaten" und „Verloren" gegen den 1117er-Lauf und gegen die
  Produktion.
- Was aus **g11, g34, g38** geworden ist (die drei Kunstverluste) und aus
  **g31, g33, g50, g26** (Tabellenfälle, bisher gar nicht in der
  Kandidatenliste).
- Was aus **n05, q07, g02** geworden ist — die Fälle, in denen die
  Ähnlichkeitsmessung den größten Verlust zeigte.
- **g23, g27, g28 gesondert ausweisen und nicht mitzählen** (siehe oben).
- Zahl der Chunks mit Fragen und der Tabellenchunks mit Fragen.
- GPU-Zeit, Wartezeit, md5 und Zeilenzahl des Parquets, Punktzahl der
  Testkopie vor/nach dem Import, Punktzahl der Produktion (unverändert
  121134), die Lauf-ID und die Pfade der HTML-Seiten.

Nenne wieder ausdrücklich, dass der Test eine **Obergrenze** misst: Nur die
108 Golddokumente haben neues Chunking und neue Fragen, ihre Konkurrenten im
Korpus nicht. Bewerte nicht, ob der volle Re-Import sich lohnt — das
entscheidet Christian anhand der Zahlen.

## Regeln

- Nichts committen, nichts pushen. Auf dem Zielserver nur unterhalb von
  `eval_gold/testindex/` und `eval_gold/laeufe/` schreiben — egal ob über den
  Mount oder über ssh.
- Nichts löschen; falls doch nötig, `trash` statt `rm`.
- Der MCP-Server (`systemctl --user status hsh-mcp.service`) läuft und **darf
  nicht gestoppt oder neu gestartet werden**. Auf dem Zielserver **kein**
  Einbettungsmodell laden, kein `hpc_vectorizer.py`, kein zweiter MCP-Server —
  die Maschine hat 8 GB RAM und eine CPU von 2011.
- Import ausschließlich in `hsh_knowledge_v2test`, nie in `hsh_knowledge`.
- Wenn ein Schritt anders ausgeht als beschrieben, nicht improvisieren:
  anhalten, Zustand beschreiben, berichten.
