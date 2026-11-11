#!/bin/bash
#SBATCH --job-name=hsh-vectorize
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=../slurm_files/vectorize_%j.log
#SBATCH --error=../slurm_files/vectorize_%j.log

# Vektorisierung: Dense (jina-embeddings-v3, GPU) + Sparse (BM25, CPU)
# über data/ingested nach hsh_vectors.parquet.
#
# Deutlich kürzer als die Anreicherung — Embedding ist ein Forward-Pass, keine
# autoregressive Generierung. Die 6 h sind großzügig bemessen.
#
# REIHENFOLGE: Dieser Job läuft NACH der Anreicherung. enrichment_v2.parquet muss
# vollständig zusammengefügt vorliegen (enricher.py --merge), und beide Läufe
# müssen dieselbe chunking.py gesehen haben. Letzteres prüft der Vektorisierer
# über den chunking_config_hash und bricht bei Abweichung ab — sonst hingen die
# Fragen an fremden Textstellen, ohne dass es auffiele.
#
# ACHTUNG, keine Wiederanlaufmöglichkeit: Parquet lässt sich nicht
# fortschreiben. Ein abgebrochener Lauf muss vollständig wiederholt werden.
# Die vorhandene Ausgabe bleibt dabei unversehrt (es wird in eine
# .partial-Datei geschrieben und erst am Ende umbenannt).
#
# Voraussetzung: Die Modelle müssen im Cache liegen. Einmalig auf dem
# LOGIN-Node, Compute-Nodes haben kein Internet:
#     python hpc_vectorizer.py --download-only --cpu
#
# Aufruf:
#     cd hsh_scraper && sbatch vectorize.sbatch

set -euo pipefail

PROJEKT=/mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME
ARBEIT=$PROJEKT/hsh_scraper
PYTHON=$PROJEKT/venv_kisski/bin/python

# Es werden ALLE Varianten nebeneinander geschrieben: BM25 über den Text,
# BM25 über Text+Fragen, Dense über den Text, Dense über die Fragen. Welche
# davon das Retrieval nutzt, entscheidet später die Query — ein A/B-Vergleich
# braucht also KEINEN zweiten Vektorisierungslauf mehr.
# enrichment_v2*, nicht enrichment.parquet: Seit dem 14.08.2026 behält die
# Deduplizierung je Gruppe zusätzlich einen Vertreter pro Fakultät und
# Organisation. Die 1.343 dadurch hinzugekommenen Punkte haben ihre Fragen von
# ihrem wortgleichen Gruppenvertreter geerbt (erbe_fragen.py, ohne GPU). Die
# alte Datei trägt noch den alten Fingerabdruck und würde abgewiesen.
#
# Beide Werte sind von außen überschreibbar, weil zum Fragenfilter mehrere
# Varianten gegeneinander zu messen sind (Messung_Fragenfilter_2026-08-14.md).
# Der Name der Ausgabe nennt die Variante — sonst ist nach dem dritten Lauf
# nicht mehr feststellbar, welche Datei welche Einstellung enthält:
#
#   sbatch vectorize.sbatch
#       Voreinstellung: Wiederholung 10, reparierte Wortregel und Korpus
#   sbatch --export=ALL,ENRICHMENT=enrichment_v2.parquet,AUSGABE=hsh_vectors_ungefiltert.parquet vectorize.sbatch
#       ungefiltert, alle 329.506 Fragen
#
# Der Text- und der Dense-Arm sind in allen Varianten identisch; sie
# unterscheiden sich ausschließlich in questions, sparse_expanded_* und
# vector_questions. Damit misst ein Vergleich wirklich den Filter.
ENRICHMENT=${ENRICHMENT:-enrichment_v2_wdh10.parquet}
AUSGABE=${AUSGABE:-hsh_vectors_wdh10.parquet}
# Abschaltbar, falls Rechenzeit oder Speicher knapp werden:
#   --no-expanded-sparse    zweiten BM25-Arm weglassen
#   --no-question-vector    Dense-Vektor der Fragen weglassen (spart GPU-Zeit)
EXTRA_ARGS=()

# Delta-Betrieb (seit 11.11.2026), alles per --export abschaltbar/setzbar:
#   URLS=delta.txt          nur diese Seiten (Dedup trotzdem über alles)
#   AVG_LEN / AVG_LEN_EXPANDED
#       Im Delta UNBEDINGT die Werte des Bestands übergeben (Index vom
#       14.08.2026: 88.3 und 107.1, docs/parquet_schema.md 3.5). Sonst misst der
#       Vektorisierer avg_len an den wenigen Delta-Seiten, und deren BM25-Gewichte
#       passen nicht zum Rest des Index.
if [[ -n "${URLS:-}" ]]; then
    EXTRA_ARGS+=(--urls "$URLS")
    if [[ -z "${AVG_LEN:-}" || -z "${AVG_LEN_EXPANDED:-}" ]]; then
        echo "FEHLER: Delta-Lauf (URLS=$URLS) ohne AVG_LEN/AVG_LEN_EXPANDED." >&2
        exit 1
    fi
fi
if [[ -n "${AVG_LEN:-}" ]]; then EXTRA_ARGS+=(--avg-len "$AVG_LEN"); fi
if [[ -n "${AVG_LEN_EXPANDED:-}" ]]; then EXTRA_ARGS+=(--avg-len-expanded "$AVG_LEN_EXPANDED"); fi

cd "$ARBEIT"
mkdir -p ../slurm_files

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# fastembed liest NICHT HF_HOME, sondern FASTEMBED_CACHE_PATH. Ohne diese
# Angabe landet der Cache in $TMPDIR — node-lokal und auf einem Compute-Node
# ohne Internet leer.
export FASTEMBED_CACHE_PATH=$ARBEIT/fastembed_cache
# venv_kisski ist eine Conda-Umgebung und bringt eigene Systembibliotheken
# mit. Ohne diesen Pfad scheitert der Import mit
#   ImportError: /lib64/libstdc++.so.6: version CXXABI_1.3.15 not found
export LD_LIBRARY_PATH=$PROJEKT/venv_kisski/lib:${LD_LIBRARY_PATH:-}

echo "=== Start $(date '+%F %T') ==="
echo "Knoten     : $(hostname)"
echo "Job        : ${SLURM_JOB_ID:-interaktiv}"
echo "Anreicherung: $ENRICHMENT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# Ohne Anreicherungsdatei ganz ohne --enrichment laufen: Die Fragen-Spalten
# bleiben dann leer, alles andere ist identisch. Sonst stürbe der Job an einer
# fehlenden Datei, obwohl der Vektorisierer auch allein sinnvoll ist.
ENRICHMENT_ARGS=()
if [[ -e "$ENRICHMENT" ]]; then
    ENRICHMENT_ARGS=(--enrichment "$ENRICHMENT")
else
    echo "HINWEIS: $ENRICHMENT nicht gefunden — es wird OHNE Fragen vektorisiert."
    echo "         Läuft die Anreicherung noch? (squeue -u $USER)"
fi

# --require-gpu bricht ab, wenn die Inferenz-Session nicht auf CUDA läuft.
# Ohne das Flag rechnet der Job im Fehlerfall still auf der CPU weiter — Tage
# statt Minuten, ohne dass eine Meldung erscheint.
# Die '+'-Form der Expansion: Ein leeres Array gilt unter 'set -u' in bash vor
# 4.4 als ungebundene Variable, der Job stürbe dann ausgerechnet im Fall ohne
# Anreicherung. Welche bash auf dem Compute-Node läuft, ist nicht garantiert.
# Die Sprache des BM25-Modells steht in hpc_vectorizer.py auf 'german'
# (fastembed selbst nähme 'english'). avg_len wird an einer Stichprobe
# gemessen, je einmal für den Text- und den erweiterten Arm — die gemessenen
# Werte stehen im Log und gehören in die Query-Seite übernommen.
"$PYTHON" hpc_vectorizer.py \
    --require-gpu \
    --output "$AUSGABE" \
    ${ENRICHMENT_ARGS[@]+"${ENRICHMENT_ARGS[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "=== Ende $(date '+%F %T') ==="
echo "Nächster Schritt: $AUSGABE und ${AUSGABE%.parquet}_sources.txt"
echo "auf den lokalen Rechner übertragen, dann local_importer.py."
echo "Im Importer beachten:"
echo "  - Payload-Feld 'questions'"
echo "  - zwei Sparse-Indizes ('sparse', 'sparse_expanded'), beide modifier=IDF"
echo "  - zweiter Dense-Vektor 'questions' (1024 Dim., Cosine)"
echo "  - BM25 der Query-Seite mit language='german' und dem avg_len aus dem Log"
