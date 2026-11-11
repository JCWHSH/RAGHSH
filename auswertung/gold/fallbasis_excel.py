"""
Fallbasis als Excel-Liste — zum Ergänzen weiterer Fragen und Goldantworten.

    python auswertung/gold/fallbasis_excel.py                 fallbasis.json → fallbasis.xlsx
    python auswertung/gold/fallbasis_excel.py --nach-json     fallbasis.xlsx → fallbasis.json

Ein Fall ist eine Zeile, ein Feld eine Spalte. Listen (required_facts,
accepted_source_urls, …) stehen mit einem Eintrag je Zeile in der Zelle
(Alt+Enter in Excel). Alternativen in pflicht_begriffe, von denen eine genügt,
stehen in EINER Zeile durch " | " getrennt.

Die seltenen verschachtelten Felder (answer_variants, vertreter_von) bleiben
als JSON-Text in ihrer Spalte — lesbar, aber nicht für die Handbearbeitung
gedacht. Eine leere Zelle heißt „Feld nicht gesetzt".

Der Rückweg schreibt fallbasis.json neu und aktualisiert fallbasis.sha256,
damit lauf.py weiß, dass die Messlatte eine andere ist. gold_im_korpus wird
dabei NICHT geprüft — das tut baue_fallbasis.py gegen doc_index.db; hier gilt,
was in der Tabelle steht.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

_BASE = Path(__file__).resolve().parent
FALLBASIS = _BASE / "fallbasis.json"
EXCEL = _BASE / "fallbasis.xlsx"

# Spalte, Breite, Erklärung für die Legende. Reihenfolge = Reihenfolge im Blatt.
SPALTEN = [
    ("id", 6, "Eindeutig. qNN = Chargy/Beidawi, nNN = Nutzerfragen; neue Fälle z. B. eNN."),
    ("herkunft", 16, "chargy_beidawi | nutzer_gold — oder ein neuer Wert für neue Fälle."),
    ("question", 50, "Die Frage, wie sie dem System gestellt wird."),
    ("reference_answer", 60, "Die Goldantwort — was eine richtige Antwort sagen muss."),
    ("pflicht_begriffe", 30, "Kurze WÖRTLICHE Zeichenketten für die maschinelle Prüfung; einer je Zeile. "
                             "Alternativen, von denen eine genügt, in einer Zeile mit ' | '. "
                             "Leer = nicht prüfbar (nicht: erfüllt)."),
    ("required_facts", 45, "Ganze Sätze, die eine richtige Antwort enthalten muss (für einen späteren Judge); einer je Zeile."),
    ("optional_facts", 40, "Sätze, die enthalten sein dürfen, aber nicht müssen; einer je Zeile."),
    ("forbidden_claims", 40, "Aussagen, die eine Antwort NICHT machen darf; eine je Zeile."),
    ("source_url", 50, "Die Goldquelle (Seite oder PDF), aus der die Antwort stammt."),
    ("accepted_source_urls", 50, "Weitere Quellen, die als Beleg gelten; eine je Zeile."),
    ("source_policy", 20, "exact_gold_source = nur source_url zählt | accepted_sources = auch accepted_source_urls."),
    ("gold_im_korpus", 10, "ja | nein — liegt die Goldquelle im Index? Beim Bau über baue_fallbasis.py wird das geprüft."),
    ("fakultaet", 10, "Filter, den der Lauf setzt (f1…f5); leer = keine Einschränkung."),
    ("case_type", 12, "direct (bisher einziger Wert)."),
    ("expected_behavior", 22, "direct_answer | missing_detail_answer."),
    ("case_difficulty", 12, "leicht | mittel | schwer."),
    ("fairness_risk", 12, "niedrig | mittel | hoch."),
    ("evaluation_notes", 50, "Hinweise für die Bewertung — worauf es bei diesem Fall ankommt."),
    ("tags", 16, "Freie Schlagworte, eines je Zeile (z. B. negativfall)."),
    ("quellen_geprueft", 14, "Datum der letzten Prüfung der Quellen (JJJJ-MM-TT)."),
    ("quellen_grund", 45, "Warum diese Quelle die richtige ist."),
    ("question_original", 40, "Nur bei präzisierten Fragen: der ursprüngliche Wortlaut."),
    ("anpassung", 45, "Nur bei präzisierten Fragen: warum und wie geändert."),
    ("source_url_original", 40, "Nur wenn die Goldquelle ersetzt wurde: die ursprüngliche."),
    ("source_policy_original", 20, "Nur wenn source_policy geändert wurde: der ursprüngliche Wert."),
    ("vertreter_von", 40, "JSON — welche Vertreter-URL für welche nicht indexierte URL steht. Nicht per Hand pflegen."),
    ("answer_variants", 40, "JSON — mehrere gleichermaßen gültige Antwortvarianten. Nicht per Hand pflegen."),
]
FELDER = [s[0] for s in SPALTEN]

LISTENFELDER = {"pflicht_begriffe", "required_facts", "optional_facts",
                "forbidden_claims", "accepted_source_urls", "tags"}
JSONFELDER = {"vertreter_von", "answer_variants"}
AUSWAHL = {
    "herkunft": ["chargy_beidawi", "nutzer_gold"],
    "source_policy": ["exact_gold_source", "accepted_sources"],
    "gold_im_korpus": ["ja", "nein"],
    "fakultaet": ["", "f1", "f2", "f3", "f4", "f5"],
    "case_type": ["direct"],
    "expected_behavior": ["direct_answer", "missing_detail_answer"],
    "case_difficulty": ["leicht", "mittel", "schwer"],
    "fairness_risk": ["niedrig", "mittel", "hoch"],
}
# Felder, die jeder Fall trägt — der Rückweg schreibt sie auch dann, wenn die
# Zelle leer ist. Alle anderen Felder erscheinen im JSON nur, wenn gesetzt.
PFLICHTFELDER = ["id", "case_type", "question", "reference_answer", "source_url",
                 "expected_behavior", "source_policy", "accepted_source_urls",
                 "required_facts", "optional_facts", "forbidden_claims",
                 "case_difficulty", "fairness_risk", "evaluation_notes",
                 "herkunft", "fakultaet", "pflicht_begriffe", "gold_im_korpus"]
# Schlüsselreihenfolge im JSON — wie baue_fallbasis.py sie schreibt, damit ein
# Rundlauf ohne Änderung keinen Diff erzeugt.
JSON_REIHENFOLGE = PFLICHTFELDER[:-1] + [
    "question_original", "anpassung", "tags", "answer_variants",
    "source_url_original", "source_policy_original", "vertreter_von",
    "quellen_geprueft", "quellen_grund", "gold_im_korpus"]
assert set(JSON_REIHENFOLGE) == set(FELDER)
MAX_ZEILEN = 500  # Reichweite der Dropdowns


# --- JSON → Zelle -----------------------------------------------------------

def zelle_aus(feld: str, wert) -> str:
    if wert is None:
        return ""
    if feld == "gold_im_korpus":
        return "ja" if wert else "nein"
    if feld in JSONFELDER:
        return json.dumps(wert, ensure_ascii=False, indent=1)
    if feld in LISTENFELDER:
        return "\n".join(" | ".join(e) if isinstance(e, list) else e for e in wert)
    return str(wert)


def wert_aus(feld: str, zelle) -> object:
    text = "" if zelle is None else str(zelle).strip()
    if feld == "gold_im_korpus":
        return text.lower() in ("ja", "true", "1", "x")
    if feld in JSONFELDER:
        return json.loads(text) if text else None
    if feld in LISTENFELDER:
        eintraege = []
        for zeile in text.splitlines():
            zeile = zeile.strip()
            if not zeile:
                continue
            if feld == "pflicht_begriffe" and "|" in zeile:
                zeile = [a.strip() for a in zeile.split("|") if a.strip()]
            eintraege.append(zeile)
        return eintraege
    return text if text else None


# --- Export -----------------------------------------------------------------

def nach_excel(faelle: list[dict], ziel: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Fälle"

    kopf = Font(bold=True)
    fuellung = PatternFill("solid", fgColor="DDEBF7")
    umbruch = Alignment(wrap_text=True, vertical="top")

    for spalte, (feld, breite, _) in enumerate(SPALTEN, start=1):
        z = ws.cell(row=1, column=spalte, value=feld)
        z.font, z.fill, z.alignment = kopf, fuellung, umbruch
        ws.column_dimensions[get_column_letter(spalte)].width = breite

    for zeile, fall in enumerate(faelle, start=2):
        unbekannt = set(fall) - set(FELDER)
        if unbekannt:
            raise SystemExit(f"{fall['id']}: Felder ohne Spalte: {sorted(unbekannt)}")
        for spalte, feld in enumerate(FELDER, start=1):
            z = ws.cell(row=zeile, column=spalte, value=zelle_aus(feld, fall.get(feld)))
            z.alignment = umbruch

    for feld, werte in AUSWAHL.items():
        spalte = get_column_letter(FELDER.index(feld) + 1)
        dv = DataValidation(type="list", formula1='"' + ",".join(werte) + '"',
                            allow_blank=True, showErrorMessage=False)
        ws.add_data_validation(dv)
        dv.add(f"{spalte}2:{spalte}{MAX_ZEILEN}")

    ws.freeze_panes = "D2"  # id, herkunft, question bleiben beim Scrollen stehen
    ws.auto_filter.ref = f"A1:{get_column_letter(len(FELDER))}{len(faelle) + 1}"

    lg = wb.create_sheet("Legende")
    lg.column_dimensions["A"].width = 24
    lg.column_dimensions["B"].width = 100
    lg["A1"], lg["B1"] = "Spalte", "Bedeutung"
    lg["A1"].font = lg["B1"].font = kopf
    for zeile, (feld, _, erklaerung) in enumerate(SPALTEN, start=2):
        lg.cell(row=zeile, column=1, value=feld).alignment = umbruch
        lg.cell(row=zeile, column=2, value=erklaerung).alignment = umbruch
    zeile = len(SPALTEN) + 3
    for text in [
        "Listen: ein Eintrag je Zeile in der Zelle (Alt+Enter).",
        "pflicht_begriffe: Alternativen, von denen eine genügt, in einer Zeile mit ' | ' trennen.",
        "Leere Zelle = Feld nicht gesetzt. Für neue Fälle genügen die Pflichtspalten "
        "(id, herkunft, question, reference_answer, pflicht_begriffe, required_facts, source_url, "
        "source_policy, gold_im_korpus, case_type, expected_behavior, case_difficulty, fairness_risk).",
        "Zurück ins JSON: python auswertung/gold/fallbasis_excel.py --nach-json",
    ]:
        lg.cell(row=zeile, column=2, value=text).alignment = umbruch
        zeile += 1

    wb.save(ziel)


# --- Import -----------------------------------------------------------------

def aus_excel(quelle: Path) -> list[dict]:
    ws = load_workbook(quelle)["Fälle"]
    zeilen = ws.iter_rows(values_only=True)
    kopf = [str(z).strip() if z is not None else "" for z in next(zeilen)]
    fehlend = set(PFLICHTFELDER) - set(kopf)
    if fehlend:
        raise SystemExit(f"Spalten fehlen im Blatt: {sorted(fehlend)}")

    faelle, ids = [], set()
    for nr, zeile in enumerate(zeilen, start=2):
        roh = dict(zip(kopf, zeile))
        if not (roh.get("id") or "").strip():
            continue  # Leerzeile
        fall = {}
        for feld in JSON_REIHENFOLGE:
            if feld not in roh:
                continue
            wert = wert_aus(feld, roh[feld])
            if feld in PFLICHTFELDER:
                fall[feld] = "" if wert is None and feld not in LISTENFELDER else wert
            elif wert not in (None, [], ""):
                fall[feld] = wert
        if fall["id"] in ids:
            raise SystemExit(f"Zeile {nr}: id {fall['id']} doppelt")
        ids.add(fall["id"])
        for feld in ("question", "reference_answer", "source_url"):
            if not fall.get(feld):
                raise SystemExit(f"Zeile {nr} ({fall['id']}): {feld} ist leer")
        for feld, werte in AUSWAHL.items():
            if feld == "gold_im_korpus":
                continue
            if fall.get(feld, "") not in werte and feld != "herkunft":
                raise SystemExit(f"Zeile {nr} ({fall['id']}): {feld}={fall.get(feld)!r} "
                                 f"nicht in {werte}")
        faelle.append(fall)
    return faelle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nach-json", action="store_true",
                    help="Excel zurück nach fallbasis.json schreiben (statt Export)")
    ap.add_argument("--excel", type=Path, default=EXCEL)
    args = ap.parse_args()

    if args.nach_json:
        faelle = aus_excel(args.excel)
        FALLBASIS.write_text(json.dumps(faelle, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        from vertrag import HASHDATEI, hash_fallbasis, kurz  # noqa: E402
        h = hash_fallbasis(faelle)
        HASHDATEI.write_text(h + "\n", encoding="utf-8")
        print(f"{len(faelle)} Fälle → {FALLBASIS.name}, Vertrag {kurz(h)}")
    else:
        faelle = json.loads(FALLBASIS.read_text(encoding="utf-8"))
        nach_excel(faelle, args.excel)
        print(f"{len(faelle)} Fälle → {args.excel}")


if __name__ == "__main__":
    main()
