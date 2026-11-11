"""
Neue Fälle von Beitragenden — Vorlage, Prüfbericht, Übernahme.

    python auswertung/gold/beitraege.py --vorlage [PFAD]         leere beitraege.xlsx anlegen
    python auswertung/gold/beitraege.py PFAD.xlsx                 Prüfbericht, nichts wird geschrieben
    python auswertung/gold/beitraege.py PFAD.xlsx --uebernehmen   Fälle an fallbasis.json anhängen

Die Vorlage hat fünf Spalten, drei davon Pflicht: Frage, Antwort, Quelle —
dazu „Wörtlich erwartet" und „Anmerkung". Das ist alles, was ein Mensch wissen
kann. Die übrigen Felder des Vertrags (siehe README) entstehen hier:

abgeleitet — das System weiß es besser als der Mensch
    gold_im_korpus         gegen doc_index.db, wie in baue_fallbasis.py
    accepted_source_urls   zweite und weitere Quellen, plus Vertreter-URLs
    source_policy          exact_gold_source bei einer Quelle, sonst accepted_sources
    id, herkunft           bNN fortlaufend, beitrag_<kürzel>
    pflicht_begriffe       aus „Wörtlich erwartet", durch den Selbsttest geprüft

voreingestellt — Kuratierungsentscheidungen mit vernünftigem Default
    case_type=direct, expected_behavior=direct_answer, case_difficulty=mittel,
    fairness_risk=niedrig, fakultaet="", required_facts=[Antwort],
    optional_facts=[], forbidden_claims=[], evaluation_notes=Anmerkung

`quellen_geprueft` bleibt leer: Beitragen ist keine Prüfung. Die Sichtung
geschieht am Prüfbericht; erst --uebernehmen ändert den Vertrag und damit den
Hash in fallbasis.sha256.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))

from baue_fallbasis import (  # noqa: E402
    _korpus_texte, ergaenze_vertreter, gold_vorhanden, lade_korpus_urls, lade_vertretung)
from raghsh.crawl.crawler_utils import normalize_url  # noqa: E402
from vergleich import anzeige, enthaelt_begriff  # noqa: E402
from vertrag import FALLBASIS, HASHDATEI, hash_fallbasis, kurz  # noqa: E402

VORLAGE = _BASE / "beitraege.xlsx"
PRAEFIX = "b"

SPALTEN = [
    ("Frage", 55, "Pflicht. So, wie ein Studierender sie stellen würde."),
    ("Antwort", 65, "Pflicht. Die richtige Antwort in ganzen Sätzen — so, wie sie auf der HsH-Seite steht, "
                    "nicht aus dem Gedächtnis."),
    ("Quelle", 55, "Pflicht. Die URL der Seite oder des PDFs, auf der die Antwort steht. "
                   "Mehrere Quellen: eine je Zeile (Alt+Enter); die erste ist die Hauptquelle."),
    ("Wörtlich erwartet", 30, "Optional. 1–3 Wörter oder Zahlen, die in JEDER richtigen Antwort vorkommen müssen "
                              "(Name, Raum, Frist, Fachbegriff) — eines je Zeile. Alternativen, von denen eine genügt, "
                              "in einer Zeile mit ' | ' trennen. Leer heißt: nur die Quelle wird geprüft, nicht der Text."),
    ("Anmerkung", 45, "Optional. Alles, was beim Prüfen hilft: Mehrdeutigkeiten, typische Falschantworten, "
                      "warum diese Quelle die richtige ist."),
    ("Von", 12, "Kürzel der beitragenden Person."),
]
KOPF = [s[0] for s in SPALTEN]

# Ein bestehender, geprüfter Fall (n01) als Beispielzeile — der Import erkennt
# ihn am Wortlaut der Frage und übergeht ihn.
BEISPIEL = {
    "Frage": "[Nutzerfrage – nicht veröffentlicht]",
    "Antwort": "Fünf: I Elektro- und Informationstechnik, II Maschinenbau und Bioverfahrenstechnik, "
               "III Medien, Information und Design, IV Wirtschaft und Informatik, "
               "V Diakonie, Gesundheit und Soziales.",
    "Quelle": "https://www.hs-hannover.de/ueber-uns/organisation/akademische-angelegenheiten/"
              "studierende/erstsemesterinformationen",
    "Wörtlich erwartet": "Elektro- und Informationstechnik\nDiakonie",
    "Anmerkung": "Beispielzeile — wird beim Import übergangen.",
    "Von": "Beispiel",
}

# Wortüberlappung wie in eval/pruefen.py (dort nicht importierbar: zieht das
# Crawler-Chunking nach). Wörter ab 5 Zeichen, die nichts über den Inhalt sagen.
STOPP = {
    "welche", "welcher", "welches", "welchem", "welchen", "einen", "einem", "eine",
    "einer", "eines", "nicht", "muss", "kann", "wenn", "wird", "werden", "haben",
    "meine", "meiner", "meinem", "meinen", "sich", "auch", "noch", "schon", "aber",
    "oder", "dass", "diese", "dieser", "diesem", "diesen", "beim", "nach", "über",
    "unter", "gibt", "viele", "lange", "wann", "wohin", "womit", "damit", "immer",
    "sowie", "sowohl", "etwas", "wieder", "sonst", "meines", "ihrer", "ihren",
    "hochschule", "hannover",
}
WORT = re.compile(r"[a-zäöüßA-ZÄÖÜ0-9]+")


def inhaltswoerter(text: str) -> set[str]:
    return {w for w in (m.group().lower() for m in WORT.finditer(text))
            if len(w) >= 5 and w not in STOPP}


UEBERLAPPUNG_FRAGE_ANTWORT = 0.6   # Anteil der Inhaltswörter der Frage, die in der Antwort stehen
AEHNLICHKEIT_FRAGEN = 0.5          # Jaccard der Inhaltswörter zweier Fragen


# --- Vorlage ------------------------------------------------------------------

def schreibe_vorlage(ziel: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Beiträge"
    kopf = Font(bold=True)
    umbruch = Alignment(wrap_text=True, vertical="top")
    for i, (name, breite, _) in enumerate(SPALTEN, start=1):
        z = ws.cell(row=1, column=i, value=name)
        z.font, z.alignment = kopf, umbruch
        z.fill = PatternFill("solid", fgColor="DDEBF7")
        ws.column_dimensions[get_column_letter(i)].width = breite
    grau = Font(italic=True, color="808080")
    for i, name in enumerate(KOPF, start=1):
        z = ws.cell(row=2, column=i, value=BEISPIEL[name])
        z.font, z.alignment = grau, umbruch
    for r in range(3, 60):
        for i in range(1, len(KOPF) + 1):
            ws.cell(row=r, column=i).alignment = umbruch
    ws.freeze_panes = "A2"

    an = wb.create_sheet("Anleitung")
    an.column_dimensions["A"].width = 20
    an.column_dimensions["B"].width = 100
    an["A1"], an["B1"] = "Spalte", "Was hinein soll"
    an["A1"].font = an["B1"].font = kopf
    for r, (name, _, erkl) in enumerate(SPALTEN, start=2):
        an.cell(row=r, column=1, value=name).alignment = umbruch
        an.cell(row=r, column=2, value=erkl).alignment = umbruch
    r = len(SPALTEN) + 3
    for text in [
        "Eine Zeile je Frage. Die graue Zeile 2 ist ein Beispiel und wird nicht übernommen.",
        "Gute Fragen: so formuliert, wie jemand sie wirklich stellt — nicht mit den Wörtern der Quelle. "
        "Wenn die Frage den Seitentext wiederholt, findet ihn schon eine Stichwortsuche, und wir messen nichts.",
        "Gute Antworten: nur, was auf der Quelle steht. Was Sie dazuwissen, gehört in die Anmerkung.",
        "Alles Weitere (Schwierigkeit, Prüfstatus, ob die Seite im Index liegt) wird beim Import ermittelt — "
        "Sie müssen es nicht wissen.",
    ]:
        an.cell(row=r, column=2, value=text).alignment = umbruch
        r += 1
    wb.save(ziel)


# --- Lesen --------------------------------------------------------------------

def _zeilen(text) -> list[str]:
    return [z.strip() for z in str(text or "").splitlines() if z.strip()]


def lies_beitraege(pfad: Path) -> list[dict]:
    wb = load_workbook(pfad, data_only=True)
    ws = wb["Beiträge"] if "Beiträge" in wb.sheetnames else wb.active
    zeilen = ws.iter_rows(values_only=True)
    kopf = [str(z).strip() if z is not None else "" for z in next(zeilen)]
    fehlend = [k for k in KOPF[:3] if k not in kopf]
    if fehlend:
        raise SystemExit(f"Spalten fehlen: {fehlend} — ist das eine Datei aus --vorlage?")
    beitraege = []
    for nr, zeile in enumerate(zeilen, start=2):
        roh = {k: (v if v is not None else "") for k, v in zip(kopf, zeile)}
        if not any(str(v).strip() for v in roh.values()):
            continue
        if str(roh.get("Frage", "")).strip() == BEISPIEL["Frage"]:
            continue
        roh["_zeile"] = nr
        beitraege.append(roh)
    return beitraege


# --- Fall bauen ---------------------------------------------------------------

def _kuerzel(von) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(von or "").strip().lower()).strip("_")
    return f"beitrag_{s}" if s else "beitrag"


def _begriffe(text) -> list:
    eintraege = []
    for z in _zeilen(text):
        alt = [a.strip() for a in z.split("|") if a.strip()]
        eintraege.append(alt if len(alt) > 1 else alt[0])
    return eintraege


def baue_fall(roh: dict, fall_id: str, korpus: set[str], vertretung: dict) -> dict:
    quellen = _zeilen(roh.get("Quelle"))
    antwort = str(roh.get("Antwort", "")).strip()
    fall = {
        "id": fall_id,
        "case_type": "direct",
        "question": str(roh.get("Frage", "")).strip(),
        "reference_answer": antwort,
        "source_url": quellen[0] if quellen else "",
        "expected_behavior": "direct_answer",
        "source_policy": "accepted_sources" if len(quellen) > 1 else "exact_gold_source",
        "accepted_source_urls": quellen[1:],
        "required_facts": [antwort] if antwort else [],
        "optional_facts": [],
        "forbidden_claims": [],
        "case_difficulty": "mittel",
        "fairness_risk": "niedrig",
        "evaluation_notes": str(roh.get("Anmerkung", "")).strip(),
        "herkunft": _kuerzel(roh.get("Von")),
        "fakultaet": "",
        "pflicht_begriffe": _begriffe(roh.get("Wörtlich erwartet")),
    }
    if fall["source_url"]:
        fall = ergaenze_vertreter(fall, vertretung)
        fall.pop("source_policy_original", None)   # ohne Historie: es gab keinen Vorwert
    fall["gold_im_korpus"] = gold_vorhanden(fall, korpus) if fall["source_url"] else False
    return fall


def naechste_ids(bestehend: list[dict], anzahl: int) -> list[str]:
    nummern = [int(f["id"][len(PRAEFIX):]) for f in bestehend
               if f["id"].startswith(PRAEFIX) and f["id"][len(PRAEFIX):].isdigit()]
    start = max(nummern, default=0) + 1
    return [f"{PRAEFIX}{n:02d}" for n in range(start, start + anzahl)]


# --- Prüfen -------------------------------------------------------------------

def _norm_frage(text: str) -> str:
    return re.sub(r"[^a-zäöüß0-9 ]", "", text.casefold()).strip()


def pruefe(fall: dict, bestehend: list[dict], korpus: set[str], vertretung: dict,
           texte: dict[str, str]) -> list[tuple[str, str]]:
    """(stufe, meldung) — FEHLER blockiert die Übernahme, warn und hinweis nicht."""
    m: list[tuple[str, str]] = []
    for feld, name in (("question", "Frage"), ("reference_answer", "Antwort"), ("source_url", "Quelle")):
        if not fall[feld]:
            m.append(("FEHLER", f"{name} fehlt"))
    if any(m):
        return m

    for url in [fall["source_url"], *fall["accepted_source_urls"]]:
        if not re.match(r"https?://", url):
            m.append(("FEHLER", f"keine URL: {url!r}"))
        elif "hs-hannover.de" not in url:
            m.append(("warn", f"Quelle liegt nicht auf hs-hannover.de: {url}"))

    frage_n = _norm_frage(fall["question"])
    for b in bestehend:
        if _norm_frage(b["question"]) == frage_n:
            m.append(("FEHLER", f"Frage gibt es schon: {b['id']}"))

    fehlt = [anzeige(b) for b in fall["pflicht_begriffe"] if not enthaelt_begriff(fall["reference_answer"], b)]
    if fehlt:
        m.append(("FEHLER", f"„Wörtlich erwartet“ steht nicht in der Antwort: {fehlt}"))

    if not fall["gold_im_korpus"]:
        m.append(("warn", "keine der Quellen im Korpus → Fall misst „Gold fehlt“, nicht das Retrieval"))
    for url in [fall["source_url"], *fall["accepted_source_urls"]]:
        n = normalize_url(url)
        if n not in korpus:
            if fall["gold_im_korpus"]:
                m.append(("hinweis", f"nicht im Korpus (andere Quelle trägt): {url}"))
            continue
        text = texte.get(n)
        if text is not None and fall["pflicht_begriffe"]:
            fehlt = [anzeige(b) for b in fall["pflicht_begriffe"] if not enthaelt_begriff(text, b)]
            if fehlt:
                m.append(("warn", f"Quelle trägt nicht {fehlt} — {url}"))

    if not fall["pflicht_begriffe"]:
        m.append(("hinweis", "kein „Wörtlich erwartet“ → Antworttext nicht prüfbar, nur die Quelle"))

    f = inhaltswoerter(fall["question"])
    a = inhaltswoerter(fall["reference_answer"])
    if f:
        anteil = len(f & a) / len(f)
        if anteil >= UEBERLAPPUNG_FRAGE_ANTWORT and len(f) >= 3:   # bei 1–2 Wörtern sagt der Anteil nichts
            m.append(("warn", f"Frage wiederholt die Antwort ({anteil:.0%} gemeinsame Inhaltswörter: "
                              f"{sorted(f & a)}) — Stichwortsuche genügt"))
        for b in bestehend:
            g = inhaltswoerter(b["question"])
            if g and len(f & g) / len(f | g) >= AEHNLICHKEIT_FRAGEN:
                m.append(("hinweis", f"ähnelt {b['id']}: {b['question']!r}"))
    if fall.get("vertreter_von"):
        m.append(("hinweis", f"Vertreter-URL ergänzt: {list(fall['vertreter_von'])}"))
    return m


# --- Hauptprogramm ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datei", nargs="?", type=Path, help="ausgefüllte beitraege.xlsx")
    ap.add_argument("--vorlage", action="store_true", help="leere Vorlage schreiben (Ziel: DATEI oder beitraege.xlsx)")
    ap.add_argument("--uebernehmen", action="store_true", help="geprüfte Fälle an fallbasis.json anhängen")
    args = ap.parse_args()

    if args.vorlage:
        ziel = args.datei or VORLAGE
        if ziel.exists():
            raise SystemExit(f"{ziel} gibt es schon — nicht überschrieben.")
        schreibe_vorlage(ziel)
        print(f"Vorlage → {ziel}")
        return
    if not args.datei:
        ap.error("Datei angeben oder --vorlage")

    bestehend = json.loads(FALLBASIS.read_text(encoding="utf-8"))
    korpus = lade_korpus_urls()
    vertretung = lade_vertretung()
    texte = _korpus_texte()

    roh = lies_beitraege(args.datei)
    if not roh:
        print("Keine Beiträge in der Datei (nur Kopf und Beispiel).")
        return
    ids = naechste_ids(bestehend, len(roh))
    faelle = [baue_fall(r, i, korpus, vertretung) for r, i in zip(roh, ids)]

    # Neue Fälle untereinander auch auf Dopplung prüfen: jeder gegen die vorherigen
    fehler = 0
    print(f"{len(faelle)} Beiträge aus {args.datei.name}\n")
    for r, fall, k in zip(roh, faelle, range(len(faelle))):
        meldungen = pruefe(fall, bestehend + faelle[:k], korpus, vertretung, texte)
        fehler += sum(1 for s, _ in meldungen if s == "FEHLER")
        gold = "Gold im Korpus" if fall["gold_im_korpus"] else "Gold FEHLT im Korpus"
        print(f"{fall['id']}  (Zeile {r['_zeile']}, {fall['herkunft']})  {gold}")
        print(f"      {fall['question']}")
        for stufe, text in meldungen:
            print(f"      {stufe:<7} {text}")
        if not meldungen:
            print("      ok")
        print()

    ohne_gold = sum(1 for f in faelle if not f["gold_im_korpus"])
    print(f"Fehler: {fehler}   ·   ohne Gold im Korpus: {ohne_gold}/{len(faelle)}")

    if not args.uebernehmen:
        print("Nichts geschrieben. Übernahme mit --uebernehmen.")
        return
    if fehler:
        raise SystemExit("Übernahme verweigert — erst die FEHLER beheben.")
    alle = bestehend + faelle
    FALLBASIS.write_text(json.dumps(alle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    h = hash_fallbasis(alle)
    HASHDATEI.write_text(h + "\n", encoding="utf-8")
    print(f"{len(faelle)} Fälle angehängt → {FALLBASIS.name} ({len(alle)} gesamt), Vertrag {kurz(h)}")


if __name__ == "__main__":
    main()
