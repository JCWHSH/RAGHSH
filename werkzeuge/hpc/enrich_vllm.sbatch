#!/bin/bash -l
# enrich_vllm.sbatch — Anreicherung des Gesamtbestands mit vLLM.
#
# Stand 13.08.2026, alle Zahlen gemessen auf A100-SXM4-80GB:
#
#   Chunks im Bestand        196.934
#   nach Deduplizierung      119.791   (39,2 % entfernt)
#   abzüglich --min-chars    ~113.000  (5,2 % im Test)
#   Durchsatz                18,5 Chunks/s   (vLLM 0.26.0, --batch-size 512)
#   → Gesamt                 ~1,7 GPU-Stunden
#   → je Shard bei SHARDS=2  ~51 min + 4 min Modellladen + 1 min Dedup
#
# --time ist mit 2:30 h großzügig gesetzt: Der Durchsatz wurde an 508 Chunks
# gemessen, der Gesamtbestand ist heterogener. Lieber Luft als ein Abbruch
# nach 90 % — der Wiederanlauf über die Teildateien funktioniert zwar, kostet
# aber erneut das Modellladen.

#SBATCH --job-name=enrich_vllm
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME/slurm_files/enrich_vllm_%A_%a.log
#SBATCH --error=/mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME/slurm_files/enrich_vllm_%A_%a.log

# MUSS zu --array=0-(SHARDS-1) passen. Stimmen die beiden nicht überein,
# bleiben Chunks unbearbeitet oder werden doppelt gerechnet.
SHARDS=${SHARDS:-2}

# Delta-Betrieb (seit 11.11.2026): nur die Seiten aus einer URL-Liste,
# ein Shard reicht. Aufruf siehe docs/hpc.md:
#   sbatch --array=0 --export=ALL,SHARDS=1,URLS=delta.txt,AUSGABE=enrichment_delta.parquet enrich_vllm.sbatch
AUSGABE=${AUSGABE:-enrichment.parquet}
URL_ARGS=()
if [[ -n "${URLS:-}" ]]; then
    URL_ARGS=(--urls "$URLS")
fi

set -uo pipefail
source /mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME/hsh_scraper/env_vllm.sh
export HF_HUB_OFFLINE=1
mkdir -p "$PROJEKT/slurm_files"

echo "=== Shard ${SLURM_ARRAY_TASK_ID:-0} von ${SHARDS} auf $(hostname), $(date '+%F %T') ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$V/python" enricher.py \
    --input-dir data/ingested \
    --output "$AUSGABE" \
    ${URL_ARGS[@]+"${URL_ARGS[@]}"} \
    --backend vllm \
    --batch-size 512 \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SHARDS}"

status=$?
echo "EXIT=$status  $(date '+%F %T')"
if [[ $status -eq 0 ]]; then
    echo
    echo "Dieser Shard ist fertig. Erst wenn ALLE ${SHARDS} Tasks durch sind:"
    echo "    source env_vllm.sh"
    echo "    \$V/python enricher.py --merge --output $AUSGABE"
fi
exit $status
