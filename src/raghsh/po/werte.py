#!/usr/bin/env python3
"""
Tabellenurteil über PO-Werte je Fassung (S087, F6).

Liest zwei Tabellen aus stammdaten/po/:
- po_werte.tsv: je (Studiengang, Fassung, Punkt) ein oder mehrere Werte mit
  wörtlichem Auszug und Fundstelle (geprüft von stammdaten/po/po_werte_pruefen.py);
- po_punkte.tsv: je Punkt die Begriffe, an denen er in einer Frage erkannt
  wird, und Begriffe, die ihn ausschließen.

Begriffe (Spalte `begriffe`): Alternativen getrennt durch „ | “; eine
Alternative besteht aus Teilen, verbunden durch „ + “, die alle in der Frage
stehen müssen; ein Teil nennt Varianten getrennt durch „/“. Eine Variante ist
ein ganzes Wort oder eine Wortfolge; ein „*“ am Ende erlaubt beliebige
Wortenden („Leistungspunkt*“). Groß-/Kleinschreibung zählt nicht. Steht eine
Variante aus `ohne` in der Frage, gilt der Punkt als nicht erkannt.

Modulbezogene Punkte tragen im Namen „:<modul>“. Sie zählen nur, wenn die
Frage genau ein Modul des Studiengangs nennt — per Kürzel oder Modulname aus
stammdaten/po/module.tsv (ganzes Wort, auch als Kompositum mit Prüfung/Modul/…,
längster Name zuerst wie im Server). Der Punkt heißt dann z. B.
„zulassung_modulpruefung:BRS-212“.

Urteil für (Frage, Studiengang, geltende Fassungen):
- genau ein Punkt erkannt, sonst `keins`;
- Werte je Fassung als Menge der Zeilen vergleichen;
- `verschieden`, sobald zwei geltende Fassungen mit Zeilen sich unterscheiden;
- `gleich` nur, wenn alle geltenden Fassungen Zeilen haben und übereinstimmen;
- sonst `keins` (bisheriger Weg, Fassungsvergleich im Server).

Im Antwortpfad (S088): hsh_mcp_server.frag_hsh fragt das Urteil vor dem
LLM-Fassungsvergleich ab. Kein LLM, kein Qdrant.
"""
import csv
import re
from functools import lru_cache
from pathlib import Path

from raghsh import konfig

PO_DATEN = konfig.PO_DATEN
WERTE_TSV = PO_DATEN / "po_werte.tsv"
PUNKTE_TSV = PO_DATEN / "po_punkte.tsv"
MODULE_TSV = PO_DATEN / "module.tsv"
MODUL_PLATZHALTER = "<modul>"
# wie _MODUL_GRUNDWORT in server/mcp_server.py: „Projektprüfung“ nennt das Modul „Projekt“
_MODUL_GRUNDWORT = r"(?:s?(?:prüfung|modul|bericht|note|leistung)\w*)?"


def _variante(v: str) -> str:
    """Variante → Regex: ganzes Wort bzw. Wortfolge, „*“ am Ende = beliebiges Wortende."""
    v = v.strip()
    offen = v.endswith("*")
    teile = [re.escape(w) for w in v.rstrip("*").split()]
    return r"(?<!\w)" + r"\s+".join(teile) + (r"\w*" if offen else r"(?!\w)")


def _teil(t: str) -> re.Pattern:
    return re.compile("|".join(_variante(v) for v in t.split("/") if v.strip()), re.I)


@lru_cache(maxsize=1)
def punkte() -> list[tuple[str, list[list[re.Pattern]], re.Pattern | None]]:
    """[(Punkt, Alternativen [[Teil-Muster]], Ausschluss-Muster)] aus po_punkte.tsv."""
    aus = []
    with open(PUNKTE_TSV, encoding="utf-8", newline="") as fh:
        for z in csv.DictReader(fh, delimiter="\t"):
            alternativen = [[_teil(t) for t in alt.split(" + ")]
                            for alt in z["begriffe"].split(" | ") if alt.strip()]
            ohne = _teil(z["ohne"]) if (z.get("ohne") or "").strip() else None
            aus.append((z["punkt"], alternativen, ohne))
    return aus


@lru_cache(maxsize=1)
def werte() -> dict[tuple[str, str], dict[int, frozenset[str]]]:
    """(Studiengang, Punkt) → {Fassung: Menge der Werte} aus po_werte.tsv."""
    tab: dict = {}
    with open(WERTE_TSV, encoding="utf-8", newline="") as fh:
        for z in csv.DictReader(fh, delimiter="\t"):
            je = tab.setdefault((z["studiengang"], z["punkt"]), {})
            je[int(z["fassung"])] = je.get(int(z["fassung"]), frozenset()) | {z["wert"]}
    return tab


@lru_cache(maxsize=1)
def zeilen() -> list[dict]:
    """Alle Zeilen von po_werte.tsv (für Kontext und Beleg)."""
    with open(WERTE_TSV, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def fundstellen(studiengang: str, punkt: str) -> list[str]:
    """URLs der Fundstellen (ohne „#Chunk“) aller Zeilen zu (Studiengang, Punkt),
    in Tabellenreihenfolge, ohne Dubletten."""
    urls: list[str] = []
    for z in zeilen():
        if z["studiengang"] == studiengang and z["punkt"] == punkt:
            url = z["fundstelle"].rsplit("#", 1)[0]
            if url not in urls:
                urls.append(url)
    return urls


@lru_cache(maxsize=1)
def _module() -> dict[str, list[tuple[re.Pattern, frozenset[str]]]]:
    """Studiengang → [(Muster aus Kürzel oder Modulname, Kürzel)], längster Name zuerst.
    Zeilen ohne Fassung bleiben draußen (wie im Server: Kürzel je Fassung)."""
    namen: dict = {}
    with open(MODULE_TSV, encoding="utf-8", newline="") as fh:
        for z in csv.DictReader(fh, delimiter="\t"):
            if z["studiengang"] and z["fassung"]:
                je = namen.setdefault(z["studiengang"], {})
                je.setdefault(z["modulname"], set()).add(z["modulkuerzel"])
                je.setdefault(z["modulkuerzel"], set()).add(z["modulkuerzel"])
    return {sg: [(re.compile(r"(?<!\w)" + re.escape(n) + _MODUL_GRUNDWORT + r"(?![\w-])", re.I), frozenset(ks))
                 for n, ks in sorted(nk.items(), key=lambda t: (-len(t[0]), t[0]))]
            for sg, nk in namen.items()}


def module_in_frage(frage: str, studiengang: str) -> set[str]:
    """Kürzel der Module des Studiengangs, die die Frage nennt (Name oder Kürzel);
    ein gefundener Name verdeckt die kürzeren darin."""
    kuerzel: set[str] = set()
    for muster, ks in _module().get(studiengang, []):
        if muster.search(frage):
            frage = muster.sub(" ", frage)
            kuerzel |= ks
    return kuerzel


def punkte_in_frage(frage: str, studiengang: str) -> list[str]:
    """Alle Punkte, die die Frage nach po_punkte.tsv nennt (modulbezogene mit Kürzel)."""
    gefunden = []
    for punkt, alternativen, ohne in punkte():
        if ohne and ohne.search(frage):
            continue
        if not any(all(t.search(frage) for t in alt) for alt in alternativen):
            continue
        if MODUL_PLATZHALTER in punkt:
            ks = module_in_frage(frage, studiengang)
            if len(ks) != 1:
                continue
            punkt = punkt.replace(MODUL_PLATZHALTER, next(iter(ks)))
        gefunden.append(punkt)
    return gefunden


def urteil(frage: str, studiengang: str, geltend: list[int]) -> dict:
    """→ {"urteil": gleich|verschieden|keins, "punkt", "punkte", "werte", "ohne_zeile"}."""
    erkannt = punkte_in_frage(frage, studiengang)
    aus = {"urteil": "keins", "punkt": None, "punkte": erkannt, "werte": {}, "ohne_zeile": []}
    if len(erkannt) != 1 or not geltend:
        return aus
    punkt = erkannt[0]
    je = werte().get((studiengang, punkt), {})
    mit = {f: je[f] for f in geltend if f in je}
    aus.update(punkt=punkt, werte={f: sorted(w) for f, w in mit.items()},
               ohne_zeile=[f for f in geltend if f not in je])
    if len(set(mit.values())) >= 2:
        aus["urteil"] = "verschieden"
    elif mit and not aus["ohne_zeile"]:
        aus["urteil"] = "gleich"
    return aus
