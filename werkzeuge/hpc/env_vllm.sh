# env_vllm.sh — Umgebung fuer den vLLM-Pfad auf KISSKI
# Aufruf:  source env_vllm.sh
#
# Verifiziert am 13.08.2026 auf ggpu171 (A100-SXM4-80GB, Treiber 570.211.01)
# mit vLLM 0.26.0 + torch 2.11.0+cu128.
#
# Warum das noetig ist: Die vLLM-Wheels von PyPI sind gegen CUDA 13 gebaut
# (vllm._C_stable_libtorch braucht libcudart.so.13) — unabhaengig davon,
# welches torch sie pinnen. Der Cluster-Treiber ist CUDA 12.8. Die Luecke
# schliesst CUDA Forward Compatibility, die auf Datacenter-GPUs wie der A100
# unterstuetzt ist (Consumer-Karten nicht: die liefern Error 803).

module load gcc/13.2.0 2>/dev/null   # Host-Compiler fuer den flashinfer-JIT

export PROJEKT=/mnt/vast-kisski/projects/PROJEKTKONTO/users/NUTZERNAME
export ARBEIT=$PROJEKT/hsh_scraper
export V=$PROJEKT/venv_vllm/bin
export SP=$PROJEKT/venv_vllm/lib/python3.11/site-packages
export COMPAT=$PROJEKT/cuda_compat/usr/local/cuda-13.3/compat

# --- 1. CUDA Forward Compatibility -----------------------------------------
# Die Compat-libcuda (610.57.04) meldet cuDriverGetVersion=13030 auf dem
# 12.8-Treiber. vLLM setzt VLLM_CUDA_COMPATIBILITY_PATH nur in os.environ —
# das wirkt erst fuer die gespawnten Worker, nicht fuer den eigenen Prozess.
# Deshalb muss der Pfad zusaetzlich hier in LD_LIBRARY_PATH stehen.
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH=$COMPAT

# --- 2. CUDA-13-Runtime + Conda-libstdc++ (ICU/sqlite3-Pfad) ---------------
export LD_LIBRARY_PATH=$COMPAT:$SP/nvidia/cu13/lib:$PROJEKT/venv_kisski/lib:${LD_LIBRARY_PATH:-}

# --- 3. CUDA-13-Toolchain aus den pip-Paketen ------------------------------
# Enthaelt nvcc und ninja — loest nebenbei auch den causal-conv1d-Build,
# fuer den der Login-Node kein Toolkit hatte.
export CUDA_HOME=$SP/nvidia/cu13
export PATH=$V:$SP/nvidia/cu13/bin:$PATH
export CPATH=$SP/nvidia/curand/include:$SP/nvidia/cu13/include:${CPATH:-}

# flashinfer JIT-baut seinen Sampling-Kernel und braucht curand.h. Bei
# Greedy-Dekodierung wird top-k/top-p ohnehin nicht gebraucht — abschalten
# spart die Kompilierung und eine Fehlerquelle.
export VLLM_USE_FLASHINFER_SAMPLER=0

# --- 4. Caches auf das Projekt-Filesystem ----------------------------------
export HF_HUB_CACHE=$PROJEKT/hf_cache
export FASTEMBED_CACHE_PATH=$ARBEIT/fastembed_cache
export VLLM_CACHE_ROOT=$PROJEKT/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=$PROJEKT/vllm_cache/inductor
export FLASHINFER_WORKSPACE_BASE=$PROJEKT/vllm_cache/flashinfer
export FLASHINFER_JIT_DIR=$PROJEKT/vllm_cache/flashinfer/jit
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_JIT_DIR"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

cd "$ARBEIT"
