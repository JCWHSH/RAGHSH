"""Toleranter Begriffsvergleich für die Auswertung.

Die einzige Stelle im Projekt, die entscheidet, ob ein Pflichtbegriff in einem
Text steht. Sie wird von zwei Seiten benutzt — `auswertung.py` prüft damit die
Antwort des RAG, `baue_fallbasis.py` prüft damit den Vertrag gegen sich selbst
und gegen den Korpus. Beide müssen dasselbe meinen, sonst besteht ein Vertrag
den Selbsttest und fällt trotzdem in der Auswertung durch.

Warum nicht `begriff in text`: Der wörtliche Vergleich hat im Lauf vom
18.08.2026 vier von vier „Antwortproblemen" erzeugt, die keine waren —
`01.01.` gegen „01.01 - 31.01.", `11 Bachelor` gegen „11 praxisnahe
Bachelor-Studiengänge", `support-it@` gegen „Support-it(at)", `4 Master`
gegen „vier Masterstudiengänge". Das RAG hatte recht, die Messlatte lag schief.

Was der Vergleich toleriert
---------------------------
- Groß-/Kleinschreibung, Unicode-Form, ß/Umlaute (ae/oe/ue), HTML-Tags („<br>")
- typografische Anführungen, Binde- und Gedankenstriche, Interpunktion am
  Tokenrand („01.01." ≡ „01.01", „Bachelor-Studiengänge" → bachelor
  studiengaenge)
- „(at)" für „@" in E-Mail-Adressen
- Zahlwörter bis 99 als Ziffern („vier" ≡ „4", „zwölf" ≡ „12")
- Tausenderpunkte („9.000" ≡ „9000"), Prozentzeichen („40%" ≡ „40 Prozent")
- Tabellenstriche „|" als Worttrenner (Markdown-Tabellen aus PDFs)
- Datumsformen: „1. September 2026", „1 Sept. 2026", „01.09.2026", „1.9.2026"
  werden alle zu 01.09.2026; ohne Jahr entsprechend 01.09. Ein Begriff ohne
  Jahr („31. Mai") trifft den Text mit Jahr („31. Mai 2026"), nicht umgekehrt
- eingeschobene Wörter: die Tokens des Begriffs müssen in Reihenfolge stehen,
  zwischen zwei Tokens dürfen bis zu MAX_LUECKE fremde Tokens liegen —
  außer zwischen Zahl und Einheit (V005, 30.09.2026): beginnt ein Begriff mit
  einer Zahl und folgt eine Einheit aus _EINHEITEN („10 CP", „12 Wochen"),
  müssen beide direkt nebeneinander stehen. Sonst träfe „§ 10 sind 12 CP"
  den Begriff „10 CP". „11 Bachelor" darf weiter „11 praxisnahe Bachelor-…"
  treffen, weil „Bachelor" keine Einheit ist
- Flexion und Komposita per Präfix: ein Begriffstoken ab MIN_PRAEFIX Zeichen
  trifft ein Texttoken, das mit ihm beginnt („zentrum" → „zentrums",
  „master" → „masterstudiengaenge"). Kurze Tokens („cp", „4") müssen exakt
  stimmen — sonst träfe „4" auch „40".

Was er nicht toleriert, mit Absicht
------------------------------------
- andere Zahlen („4 Master" trifft nicht „14 Master")
- vertauschte Reihenfolge („Lehre und Beratung" trifft nicht „Beratung und
  Lehre")
- große Abstände: „Stephan … sieben Wörter … Rittmüller" ist kein Treffer
- Wörter zwischen Zahl und Einheit: „10 weitere CP" trifft „10 CP" nicht (V005)

Alternativen
------------
Ein Eintrag in `pflicht_begriffe` ist entweder ein String oder eine Liste von
Strings; bei einer Liste genügt einer davon („Prüfungsverwaltung" ODER
„Anmeldung Studienprojekt"). `enthaelt_begriff` nimmt beides.

Tests: `pytest auswertung/gold/test_vergleich.py`
"""
from __future__ import annotations

import re
import unicodedata

MAX_LUECKE = 3      # fremde Tokens zwischen zwei Begriffstokens
MIN_PRAEFIX = 4     # ab dieser Länge genügt Präfixgleichheit

# Einheiten, die direkt auf ihre Zahl folgen müssen (V005): steht das zweite
# Token eines Begriffs hier und ist das erste eine Zahl, gilt keine Lücke.
# Das Texttoken wird danach wie sonst per _passt verglichen.
_EINHEITEN = {
    "c", "cr", "cp", "credit", "credits", "ects", "leistungspunkte", "lp", "sws",
    "semester", "woche", "wochen", "monat", "monate", "monaten", "jahr", "jahre",
    "jahren", "tag", "tage", "tagen", "stunde", "stunden", "minute", "minuten",
    "euro", "eur", "€", "prozent",
}
_ZAHL = re.compile(r"\d+(?:[.,]\d+)?")

_ZAHLWOERTER = {
    "null": 0, "ein": 1, "eine": 1, "einen": 1, "einem": 1, "einer": 1, "eins": 1,
    "zwei": 2, "drei": 3, "vier": 4, "fuenf": 5, "sechs": 6, "sieben": 7,
    "acht": 8, "neun": 9, "zehn": 10, "elf": 11, "zwoelf": 12, "dreizehn": 13,
    "vierzehn": 14, "fuenfzehn": 15, "sechzehn": 16, "siebzehn": 17,
    "achtzehn": 18, "neunzehn": 19, "zwanzig": 20, "dreissig": 30,
    "vierzig": 40, "fuenfzig": 50, "sechzig": 60, "siebzig": 70,
    "achtzig": 80, "neunzig": 90,
}
_EINER = {"ein": 1, "zwei": 2, "drei": 3, "vier": 4, "fuenf": 5, "sechs": 6,
          "sieben": 7, "acht": 8, "neun": 9}
_ZEHNER = {"zwanzig": 20, "dreissig": 30, "vierzig": 40, "fuenfzig": 50,
           "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90}

_MONATE = {
    "januar": 1, "jan": 1, "februar": 2, "feb": 2, "maerz": 3, "mrz": 3, "mar": 3,
    "april": 4, "apr": 4, "mai": 5, "juni": 6, "jun": 6, "juli": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "oktober": 10,
    "okt": 10, "november": 11, "nov": 11, "dezember": 12, "dez": 12,
}

_UMLAUTE = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_STRICHE = re.compile(r"[‐-―−\-–—/|]")         # Binde-/Gedankenstriche, Schrägstrich, Tabellenstrich
_ANFUEHRUNG = re.compile(r"[\"'„“”‚‘’«»‹›*_`]")
_TAUSENDER = re.compile(r"\b(\d{1,3})\.(\d{3})\b(?!\.\d)")
# (?<![\w.]) statt \b: In „Raum 2A.1.11." darf „1.11." kein Datum werden —
# sonst heißt der Token 2a.01.11 und trifft den Begriff 2A.1.11 nicht mehr.
_DATUM_NUM = re.compile(r"(?<![\w.])(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})?(?!\d)")
# Nur echte Tags (<br>, </td>, <span class=…>), nicht „<==**" oder „a < b > c":
# eine gierige Fassung hatte in einer Datei 4 KB zwischen zwei Winkeln gelöscht.
_HTML_TAG = re.compile(r"</?[a-z][a-z0-9]*(?:\s[^<>\n]{0,80})?/?>")
_DATUM_WORT = re.compile(
    r"\b(\d{1,2})\.?\s+(" + "|".join(sorted(_MONATE, key=len, reverse=True)) + r")\.?(?:\s+(\d{4}))?\b")
_INTERPUNKTION = re.compile(r"(?<!\d)[,;:!?()\[\]{}<>]|[,;:!?()\[\]{}<>](?!\d)")
_ZAHLWORT_ZUSAMMEN = re.compile(
    r"\b(" + "|".join(_EINER) + r")und(" + "|".join(_ZEHNER) + r")\b")


def _grundform(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = _HTML_TAG.sub(" ", text)          # <br> in Tabellen-PDFs klebt sonst am Token
    text = text.replace("(at)", "@").replace("[at]", "@").replace(" at ", " @ ")
    text = re.sub(r"(?<=\d)\s?%", " prozent", text)     # „40%" ≡ „40 Prozent"
    text = text.translate(_UMLAUTE)
    text = _ANFUEHRUNG.sub(" ", text)
    return text


def _datum_num(m: re.Match) -> str:
    tag, monat, jahr = m.group(1), m.group(2), m.group(3)
    if not (1 <= int(tag) <= 31 and 1 <= int(monat) <= 12):
        return m.group(0)
    kern = f"{int(tag):02d}.{int(monat):02d}"
    if jahr:
        jahr = jahr if len(jahr) == 4 else "20" + jahr
        return f"{kern}.{jahr}"
    return kern


def _datum_wort(m: re.Match) -> str:
    tag, monat, jahr = m.group(1), m.group(2), m.group(3)
    if not 1 <= int(tag) <= 31:
        return m.group(0)
    kern = f"{int(tag):02d}.{_MONATE[monat]:02d}"
    return f"{kern}.{jahr}" if jahr else kern


def _zahlwort(tok: str) -> str:
    if tok in _ZAHLWOERTER:
        return str(_ZAHLWOERTER[tok])
    m = _ZAHLWORT_ZUSAMMEN.fullmatch(tok)
    if m:
        return str(_EINER[m.group(1)] + _ZEHNER[m.group(2)])
    return tok


def tokens(text: str) -> list[str]:
    """Normalisierte Tokenfolge eines Textes (siehe Modulkopf)."""
    t = _grundform(text)
    t = _DATUM_WORT.sub(_datum_wort, t)
    t = _DATUM_NUM.sub(_datum_num, t)
    t = _TAUSENDER.sub(r"\1\2", t)
    t = _STRICHE.sub(" ", t)
    # Interpunktion trennt Wörter — auch mitten im „Wort" („Semester(Sommersemester),"
    # aus PDF-Tabellen). Zwischen Ziffern bleibt sie („7.029,40", „1b.0.43").
    t = _INTERPUNKTION.sub(" ", t)
    ergebnis = []
    for roh in t.split():
        # Interpunktion am Rand weg, innen bleibt sie (1b.0.43, hs-hannover.de, 01.09.2026)
        tok = roh.strip(".,;:!?()[]{}<>|…")
        if not tok:
            continue
        ergebnis.append(_zahlwort(tok))
    return ergebnis


_DATUM_OHNE_JAHR = re.compile(r"\d{2}\.\d{2}")


def _passt(begriff_tok: str, text_tok: str) -> bool:
    if begriff_tok == text_tok:
        return True
    if len(begriff_tok) >= MIN_PRAEFIX and text_tok.startswith(begriff_tok):
        if not begriff_tok[0].isdigit():
            return True                     # Flexion/Kompositum: zentrum → zentrums
        # Zahlen und Kennungen nur exakt (4 ≠ 40, 1b.0.43 ≠ 1b.0.431) — mit
        # einer Ausnahme: ein Datum ohne Jahr trifft dasselbe Datum mit Jahr.
        return bool(_DATUM_OHNE_JAHR.fullmatch(begriff_tok)) and text_tok[5:6] == "."
    return False


def enthaelt(text: str, begriff: str) -> bool:
    """Steht `begriff` — tolerant, siehe Modulkopf — in `text`?"""
    muster = tokens(begriff)
    if not muster:
        return False
    folge = tokens(text)
    starts = [i for i, tok in enumerate(folge) if _passt(muster[0], tok)]
    for start in starts:
        pos = start
        ok = True
        for k, m in enumerate(muster[1:]):
            luecke = MAX_LUECKE
            if k == 0 and _ZAHL.fullmatch(muster[0]) and m in _EINHEITEN:
                luecke = 0                  # Zahl und Einheit direkt nebeneinander (V005)
            grenze = min(len(folge), pos + 1 + luecke + 1)
            naechste = next((j for j in range(pos + 1, grenze) if _passt(m, folge[j])), None)
            if naechste is None:
                ok = False
                break
            pos = naechste
        if ok:
            return True
    return False


def enthaelt_begriff(text: str, eintrag: str | list[str]) -> bool:
    """Ein Eintrag aus `pflicht_begriffe`: String oder Liste von Alternativen."""
    if isinstance(eintrag, str):
        return enthaelt(text, eintrag)
    return any(enthaelt(text, alt) for alt in eintrag)


def anzeige(eintrag: str | list[str]) -> str:
    """Lesbare Form eines Eintrags für Berichte („A | B")."""
    return eintrag if isinstance(eintrag, str) else " | ".join(eintrag)
