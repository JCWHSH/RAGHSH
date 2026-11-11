"""Vorschläge für Pflichtbegriffe der 50 Fälle aus eval/fragen.csv (g01–g50).

Die 50 Fälle haben eine Goldquelle, aber keine Pflichtbegriffe — deshalb kann
die Auswertung bei ihnen nur sagen, ob das richtige Dokument im Kontext lag,
nicht ob die Antwort stimmt. 16 von ihnen landen als „nicht prüfbar" außerhalb
der Diagnose, und die Antwortrunde streut messbar (P1 gegen V2: 3 von 24
prüfbaren Fällen bei gleichen Quellen). Mehr prüfbare Fälle sind der einzige
Weg, Rauschen von Wirkung zu trennen.

Was das Skript tut
------------------
Je Frage Kandidaten sammeln und nach der Regel des Builder-Selbsttests prüfen:
ein Begriff taugt nur, wenn er (1) in der Referenzantwort steht und (2) im
Korpustext der Goldquelle — beides mit dem toleranten Vergleich aus
vergleich.py, also genau so, wie die Auswertung später prüft.

Kandidaten kommen aus zwei Quellen:
- der Spalte `belegstelle` (die exakte Passage, „**Färbeküche** // 2A.1.11"),
  zerlegt an „//" und von Markdown befreit — das ist die stärkste Quelle,
  weil ein Mensch sie beim Anlegen der Frage aus dem Dokument kopiert hat;
- der Referenzantwort per Muster: Raumnummern, Daten, Uhrzeiten, Zahlen mit
  Einheit, E-Mail, Straßen, Eigennamen mit Titel, Anführungen.

Was es nicht tut: entscheiden. Die Ausgabe ist ein Vorschlag je Fall mit
Beleg, den ein Mensch durchsieht. Erst `begriffe_g.json` (kuratiert) liest der
Builder; diese Datei hier ist Arbeitsmaterial.

    python auswertung/gold/begriffe_vorschlag.py            # schreibt begriffe_g_vorschlag.json + .md
    python auswertung/gold/begriffe_vorschlag.py --pruefe   # prüft die kuratierte begriffe_g.json
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))

from baue_fallbasis import FRAGEN50_CSV, _korpus_texte, lade_vertretung  # noqa: E402
from raghsh.crawl.crawler_utils import normalize_url  # noqa: E402
from vergleich import enthaelt  # noqa: E402

ZIEL_JSON = _BASE / "begriffe_g_vorschlag.json"
ZIEL_MD = _BASE / "begriffe_g_vorschlag.md"
MAX_BEGRIFFE = 3

# Muster in der Referenzantwort, grob nach Aussagekraft geordnet.
_MUSTER = [
    ("raum", re.compile(r"\b\d[A-Z]\.\d{1,2}\.\d{2,3}\b")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("datum", re.compile(r"\b\d{1,2}\.\s?(?:\d{1,2}\.|Januar|Februar|März|April|Mai|Juni|Juli|August|"
                         r"September|Oktober|November|Dezember)(?:\s?\d{2,4})?\b")),
    ("uhrzeit", re.compile(r"\b\d{1,2}(?:[:.]\d{2})?\s?(?:bis|–|-)\s?\d{1,2}(?:[:.]\d{2})?\s?Uhr\b|\b\d{1,2}[:.]\d{2}\s?Uhr\b")),
    ("menge", re.compile(r"(?<![\d.,])\d{1,3}(?:\.\d{3})*(?:,\d+)?\s?(?:CP|Credits?|ECTS|Leistungspunkte|LP|Wochen?|Semester|"
                         r"Monate?|Tage?|Stunden?|SWS|Euro|€|Prozent|%|Seiten|Zeichen|Plätze)\b")),
    ("strasse", re.compile(r"\b[A-ZÄÖÜ][\wäöüß-]+(?:straße|strasse|str\.|weg|allee|platz|ring)\s\d+[a-z]?\b")),
    ("name", re.compile(r"\b(?:Prof\.|Dr\.|Herr|Frau)\s(?:Dr\.\s)?(?:[A-ZÄÖÜ][\wäöüß-]+\s?){1,3}")),
    ("zitat", re.compile(r"[„\"]([^“\"”]{3,60})[“\"”]")),
    ("kuerzel", re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]{1,4})+\b|\b[A-Z][A-Z0-9]{2,7}\b")),
]
# Zahlen allein sind zu unspezifisch — „2" trifft alles. Als Begriff nur mit Einheit.
_ZU_ALLGEMEIN = {"hochschule", "hannover", "hs hannover", "studium", "studierende", "fakultät",
                 "bachelor", "master", "raum", "informationen", "seite", "website", "hsh"}


def _saeubere(s: str) -> str:
    s = re.sub(r"[*_`#>|]+", " ", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)     # Markdown-Link → Text
    s = " ".join(s.split())
    return s.strip(" .,;:–-—„“\"'()")


_STOPP = {"der", "die", "das", "des", "dem", "den", "ein", "eine", "einer", "eines", "einem", "einen",
          "und", "oder", "in", "im", "an", "am", "auf", "bei", "mit", "von", "vom", "zu", "zum", "zur",
          "fuer", "ist", "sind", "wird", "werden", "kann", "koennen", "sich", "nicht", "nur", "auch",
          "als", "bis", "nach", "ueber", "unter", "durch", "aus", "es", "sie", "er", "ihre", "ihr",
          "ihren", "ihrer", "ihrem", "sein", "seine", "seiner", "seinem", "dieser", "diese", "dieses",
          "je", "pro", "per", "bitte", "sowie", "bzw", "etwa", "rund", "ca", "z.b", "the", "of", "and"}


def gemeinsame_folgen(belegstelle: str, antwort: str) -> list[str]:
    """Wortfolgen (1–4 Tokens), die in Belegstelle UND Antwort stehen.

    Die Belegstelle ist oft ein ganzer Satz und die Antwort eine Paraphrase —
    dann liefert weder Muster noch Belegstelle einen kurzen Begriff, aber die
    Schnittmenge der Wörter tut es („Justiziariat", „amtsärztliches Attest",
    „Zweitveröffentlichung"). Gesucht wird auf den normalisierten Tokens, die
    Rückgabe ist der Wortlaut der Antwort, damit der Begriff lesbar bleibt.
    """
    from vergleich import tokens
    a_tok = tokens(antwort)
    b_tok = set(tokens(belegstelle))
    b_pref = {t[:5] for t in b_tok if len(t) >= 5}      # Flexion: „lehrgaenge"/„lehrgang"
    # Wortlaut je Antwort-Token zurückgewinnen: die Antwort erneut grob zerlegen
    roh = [w.strip(" .,;:!?()[]{}<>|…„“\"'*") for w in re.split(r"[\s/–—-]+", antwort)]
    roh = [w for w in roh if w]

    def passt(t: str) -> bool:
        if t in _STOPP or len(t) < 4 or t.isdigit():
            return False
        return t in b_tok or (len(t) >= 5 and t[:5] in b_pref)

    treffer: list[str] = []
    i = 0
    while i < len(a_tok):
        if not passt(a_tok[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(a_tok) and j - i < 3 and (passt(a_tok[j + 1]) or
                                                     (a_tok[j + 1] in _STOPP and j + 2 < len(a_tok) and passt(a_tok[j + 2]))):
            j += 1
        while a_tok[j] in _STOPP:
            j -= 1
        # Wortlaut: passende Rohwörter in Reihenfolge einsammeln
        folge = a_tok[i:j + 1]
        woerter = [w for w in roh if tokens(w) and tokens(w)[0] in folge]
        text = " ".join(dict.fromkeys(woerter))[:60] if woerter else " ".join(folge)
        treffer.append(text)
        i = j + 1
    # längste zuerst — „amtsärztliches Attest" vor „Attest"
    return sorted(dict.fromkeys(treffer), key=lambda t: -len(t))


def kandidaten(belegstelle: str, antwort: str) -> list[tuple[str, str]]:
    """(Begriff, Herkunft) — dedupliziert, Reihenfolge nach Aussagekraft."""
    gesehen: set[str] = set()
    ergebnis: list[tuple[str, str]] = []

    def nimm(begriff: str, woher: str) -> None:
        b = _saeubere(begriff)
        if not (2 <= len(b) <= 70) or b.casefold() in _ZU_ALLGEMEIN or b.casefold() in gesehen:
            return
        if re.fullmatch(r"\d+", b):
            return
        gesehen.add(b.casefold())
        ergebnis.append((b, woher))

    for teil in re.split(r"\s*//\s*|\s*\|\s*", belegstelle or ""):
        teil = _saeubere(teil)
        if not teil:
            continue
        # Eine Belegstelle kann ein ganzer Satz sein — dann zusätzlich die
        # Muster darin, der Satz selbst nur, wenn er kurz ist.
        if len(teil) <= 45:
            nimm(teil, "belegstelle")
        for art, muster in _MUSTER:
            for m in muster.finditer(teil):
                nimm(m.group(1) if m.groups() else m.group(0), f"belegstelle/{art}")
    for art, muster in _MUSTER:
        for m in muster.finditer(antwort or ""):
            nimm(m.group(1) if m.groups() else m.group(0), f"antwort/{art}")
    for folge in gemeinsame_folgen(belegstelle, antwort):
        nimm(folge, "gemeinsam")
    return ergebnis


KURATIERT = _BASE / "begriffe_g.json"


def _goldtext(texte: dict, vertretung: dict, url: str) -> str | None:
    n = normalize_url(url)
    if n in texte:
        return texte[n]
    for v in vertretung.get(n, []):
        if v in texte:
            return texte[v]
    return None


def pruefe_kuratiert() -> None:
    """begriffe_g.json gegen Referenzantwort und Goldtext prüfen — dieselben
    Regeln wie der Builder-Selbsttest, aber vor dem Bau und je Alternative."""
    from vergleich import anzeige, enthaelt_begriff
    kur = json.loads(KURATIERT.read_text(encoding="utf-8"))
    texte, vertretung = _korpus_texte(), lade_vertretung()
    fehler = 0
    with FRAGEN50_CSV.open(encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            fid = f"g{i:02d}"
            begriffe = kur.get(fid)
            if not begriffe:
                print(f"  {fid}  — keine Begriffe")
                continue
            gold = _goldtext(texte, vertretung, row.get("source_url", ""))
            teile = []
            for b in begriffe:
                in_a = enthaelt_begriff(row.get("antwort", ""), b)
                in_g = None if gold is None else enthaelt_begriff(gold, b)
                alts = b if isinstance(b, list) else [b]
                alt_g = "" if gold is None or not isinstance(b, list) else \
                    " [" + "/".join("✓" if enthaelt(gold, a) else "·" for a in alts) + "]"
                mark = ("✓" if in_a else "✗A") + ("" if in_g else ("?" if in_g is None else "✗G"))
                teile.append(f"{anzeige(b)} {mark}{alt_g}")
                fehler += not in_a
            print(f"  {fid}  {' | '.join(teile)}")
    print(f"\n✗A = nicht in der Referenzantwort (Fehler): {fehler}. ✗G = nicht im Goldtext (Warnung). "
          f"[✓/·] = welche Alternative im Goldtext steht.")


def main() -> None:
    if "--pruefe" in sys.argv:
        pruefe_kuratiert()
        return
    texte = _korpus_texte()
    vertretung = lade_vertretung()
    faelle = []
    with FRAGEN50_CSV.open(encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            fid = f"g{i:02d}"
            antwort, beleg, url = row.get("antwort", ""), row.get("belegstelle", ""), row.get("source_url", "")
            n = normalize_url(url)
            goldtext = texte.get(n)
            if goldtext is None:
                # ohne eigene Datei: Vertreter-URLs versuchen
                for v in vertretung.get(n, []):
                    if v in texte:
                        goldtext = texte[v]
                        break
            geprueft = []
            for begriff, woher in kandidaten(beleg, antwort):
                in_antwort = enthaelt(antwort, begriff)
                im_gold = enthaelt(goldtext, begriff) if goldtext is not None else None
                geprueft.append({"begriff": begriff, "woher": woher,
                                 "in_antwort": in_antwort, "im_gold": im_gold})
            tauglich = [g for g in geprueft if g["in_antwort"] and g["im_gold"] is not False]
            # Aussagekraft: Belegstelle vor Mustern, spezifische Werte vor Namen/Zitaten
            rang = {"belegstelle": 0, "raum": 1, "email": 1, "datum": 2, "uhrzeit": 2, "menge": 2,
                    "strasse": 3, "name": 4, "kuerzel": 5, "zitat": 6, "gemeinsam": 7}
            tauglich.sort(key=lambda g: rang.get(g["woher"].split("/")[-1], 9))
            vorschlag = [g["begriff"] for g in tauglich[:MAX_BEGRIFFE]]
            faelle.append({
                "id": fid, "frage": row["frage"], "antwort": antwort, "belegstelle": beleg,
                "source_url": url, "goldtext_vorhanden": goldtext is not None,
                "vorschlag": vorschlag, "kandidaten": geprueft,
            })

    ZIEL_JSON.write_text(json.dumps(faelle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zeilen = ["# Vorschläge Pflichtbegriffe g01–g50", "",
              "Regel: Begriff steht in der Referenzantwort **und** im Goldtext (toleranter Vergleich).",
              "✓ = beide Prüfungen bestanden, (A) = nur in der Antwort, (G?) = Goldtext nicht im Korpus.", "",
              "| Fall | Frage | Belegstelle | Referenzantwort | Vorschlag | verworfen |", "|---|---|---|---|---|---|"]
    for f in faelle:
        verworfen = [f'{g["begriff"]} ({"A" if not g["im_gold"] else "–"})'
                     for g in f["kandidaten"] if g["begriff"] not in f["vorschlag"]][:4]
        zeilen.append(f'| {f["id"]} | {f["frage"]} | {_saeubere(f["belegstelle"])[:80]} | {f["antwort"][:90]} | '
                      f'**{" · ".join(f["vorschlag"]) or "—"}** | {"; ".join(verworfen)} |')
    ohne = [f["id"] for f in faelle if not f["vorschlag"]]
    zeilen += ["", f"Ohne tauglichen Vorschlag: {', '.join(ohne) or '—'}"]
    ZIEL_MD.write_text("\n".join(zeilen) + "\n", encoding="utf-8")
    print(f"{len(faelle)} Fälle, {sum(1 for f in faelle if f['vorschlag'])} mit Vorschlag, "
          f"ohne: {', '.join(ohne) or '—'}")
    print(f"→ {ZIEL_JSON.name}, {ZIEL_MD.name}")


if __name__ == "__main__":
    main()
