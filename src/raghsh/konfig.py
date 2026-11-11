"""Orte des Systems an einer Stelle.

Code liegt im Repository, Laufzeitdaten (Korpus, Index, Vektoren, Logs,
Caches) unter ``var/``, Geheimnisse in ``.env`` im Wurzelverzeichnis. Jeder Ort
lässt sich per Umgebungsvariable umlenken — etwa auf dem Jetson
``RAGHSH_VAR=/srv/raghsh``, ohne den Code anzufassen.

``WURZEL`` setzt eine editierbare Installation voraus (``pip install -e .``):
Dann liegt dieses Modul unter ``<wurzel>/src/raghsh/``. Bei einer normalen
Installation ins site-packages ``RAGHSH_WURZEL`` setzen.
"""
from __future__ import annotations

import os
from pathlib import Path


def _ort(variable: str, standard: Path) -> Path:
    wert = os.environ.get(variable)
    return Path(wert).expanduser() if wert else standard


WURZEL = _ort("RAGHSH_WURZEL", Path(__file__).resolve().parents[2])

# Geheimnisse (nie versioniert)
ENV_DATEI = _ort("RAGHSH_ENV", WURZEL / ".env")
OPENROUTER_KEY_DATEI = _ort("RAGHSH_OPENROUTER_KEY", WURZEL / "openrouter_key.txt")

# Laufzeitdaten (gitignored)
VAR = _ort("RAGHSH_VAR", WURZEL / "var")
KORPUS = _ort("RAGHSH_KORPUS", VAR / "korpus")          # gecrawlte Seiten als Markdown
DATEN = VAR / "daten"                                    # Delta-Archive und Listen des Crawlers
DOC_INDEX = _ort("RAGHSH_DOC_INDEX", VAR / "doc_index.db")
QA_LOG = _ort("RAGHSH_QA_LOG", VAR / "qa_log.jsonl")     # Fragen mit Einwilligung (DSGVO, 12 Monate)
VEKTOREN = VAR / "vektoren"                              # Parquet-Dateien von HPC und Import
CACHE = VAR / "cache"
LOGS = VAR / "logs"
DASHBOARD = VAR / "agenten_dashboard"

# Versionierte Daten
STAMMDATEN = WURZEL / "stammdaten"
PO_DATEN = _ort("RAGHSH_PO_DATEN", STAMMDATEN / "po")
AUSWERTUNG = WURZEL / "auswertung"
LAEUFE = AUSWERTUNG / "laeufe"
