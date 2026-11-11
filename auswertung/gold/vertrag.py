"""Fingerabdruck der Fallbasis — damit jeder Lauf weiß, wogegen er gemessen wurde.

Die Fallbasis ist die Messlatte. Sie hat sich zwischen dem 18.08. und dem
15.09.2026 dreimal geändert, und derselbe Lauf lieferte je nach Fassung 12, 14,
17 oder 18 Treffer. Ohne Fingerabdruck erkennt die Auswertung nicht, ob zwei
Läufe überhaupt dieselbe Frage beantworten.

Ein Hash, drei Stellen:
- `baue_fallbasis.py` schreibt ihn nach dem Bau in `fallbasis.sha256`.
- `lauf.py` legt eine Kopie der benutzten Fallbasis in den Laufordner und
  trägt den Hash ins Manifest ein; `--fortsetzen` verweigert, wenn die
  Fallbasis inzwischen eine andere ist.
- `auswertung.py` zeigt ihn im Kopf und in der Übersicht und warnt bei
  `--vergleich`, wenn er sich unterscheidet.

Der Hash läuft über die kanonische JSON-Form (Schlüssel sortiert, Fälle nach
id sortiert, keine Leerzeichen). Er ändert sich bei jeder inhaltlichen
Änderung irgendeines Falls — einem Pflichtbegriff, einer akzeptierten URL,
einer Referenzantwort — und bei nichts sonst.

Für Läufe von vor dem 15.09.2026 fehlt der Eintrag im Manifest. Ihr Vertrag
steht aber in jeder faelle/qXX.json unter `fall`; daraus lässt er sich
rekonstruieren (`hash_aus_faellen`). Die Auswertung kennzeichnet das.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_BASE = Path(__file__).resolve().parent
FALLBASIS = _BASE / "fallbasis.json"
HASHDATEI = _BASE / "fallbasis.sha256"


def kanonisch(faelle: list[dict]) -> str:
    return json.dumps(sorted(faelle, key=lambda f: f["id"]),
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def hash_fallbasis(faelle: list[dict]) -> str:
    return hashlib.sha256(kanonisch(faelle).encode("utf-8")).hexdigest()


def hash_fall(fall: dict) -> str:
    """Fingerabdruck eines einzelnen Falls — für den fallweisen Vergleich."""
    return hashlib.sha256(kanonisch([fall]).encode("utf-8")).hexdigest()


def kurz(h: str | None) -> str:
    return (h or "")[:7] or "—"


def lade_fallbasis(pfad: Path = FALLBASIS) -> list[dict]:
    return json.loads(pfad.read_text(encoding="utf-8"))


def hash_aus_faellen(datensaetze: list[dict]) -> str:
    """Rekonstruiert den Vertrag eines alten Laufs aus den Rohdaten.

    Deckt nur die Fälle ab, die gelaufen sind — ein Probelauf mit drei Fällen
    hat damit einen anderen Hash als der Volllauf derselben Fallbasis. Das ist
    kein Fehler, sondern die Wahrheit: Gemessen wurde gegen diese drei.
    """
    return hash_fallbasis([d["fall"] for d in datensaetze])
