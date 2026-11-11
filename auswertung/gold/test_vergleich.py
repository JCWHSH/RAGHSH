"""Tests für den toleranten Begriffsvergleich.

Die Positivfälle sind die Fehlbewertungen des Laufs vom 18.08.2026, jeweils
mit dem echten Wortlaut aus Antwort oder Quelle. Die Negativfälle sichern,
dass die Toleranz nicht in Beliebigkeit kippt.

    pytest auswertung/gold/test_vergleich.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vergleich import enthaelt, enthaelt_begriff, tokens  # noqa: E402


# ── Positiv: die Fehlbewertungen vom 18.08.2026 ───────────────────────────
@pytest.mark.parametrize("text, begriff", [
    # q08 — Punkt vor dem Strich fehlt im Original
    ("„für das Sommersemester 01.01 - 31.01. eines jeden Jahres“", "01.01."),
    ("„für das Sommersemester 01.01 - 31.01. eines jeden Jahres“", "31.01."),
    # q14 — eingeschobenes Wort, Bindestrich-Kompositum
    ("Dokument 1 nennt „11 praxisnahe Bachelor-Studiengänge“ [1]", "11 Bachelor"),
    # q16 — Zahlwort und Kompositum
    ("15 Bachelor- und vier Masterstudiengänge", "4 Master"),
    # q09 — (at)-Schreibweise und Großschreibung
    ("per E-Mail über „Support-it(at)hs-hannover.de“ [1]", "support-it@hs-hannover.de"),
    # q20 — Genitiv
    ("Tutor*innen des Zentrums für Lehre und Beratung", "Zentrum für Lehre und Beratung"),
    # n02 — Datum in Wortform
    ("Das Wintersemester beginnt am „1 September“ [1].", "01.09."),
    ("Semesterbeginn ist der 1. September 2026.", "01.09.2026"),
    ("Semesterbeginn ist der 1.9.2026.", "01.09.2026"),
    # q07 — Begriff ohne Jahr, Text mit Jahr
    ("Bewerbungsfrist: 1. April bis 31. Mai 2026", "31. Mai"),
    ("Bewerbungsfrist: 1. April bis 31. Mai 2026", "1. April"),
    # n04 — Zahlwort
    ("beträgt die Aufbewahrungsfrist „drei Jahre“", "3 Jahre"),
    ("beträgt die Aufbewahrungsfrist 3 Jahre", "drei Jahre"),
    # q18 — Tausenderpunkt, Näherungszeichen
    ("studieren ungefähr „≈ 9.000“ Studierende", "9000 Studierende"),
    ("rund 9000 Studierende", "9.000 Studierende"),
    # n03 — Kompositum mit Bindestrich
    ("Es besteht aus einer 10-wöchigen Praxisphase.", "10-wöchig"),
    # Umlaut-Schreibweisen
    ("Stephan Rittmueller ist zuständig", "Stephan Rittmüller"),
    ("Anfahrt über Stammestraße oder Lodemannweg", "Stammestrasse"),
    # Raum- und Kennungen bleiben ganz — auch vor dem Satzpunkt (g01, 22.09.2026)
    ("im „Gebäude 1B“ [1], „Raum 1B.0.43“ [1]", "1B.0.43"),
    ("In der Färbeküche der Kostümwerkstatt, Raum 2A.1.11.", "2A.1.11"),
    # HTML-Reste aus Tabellen-PDFs (g34)
    ("VIF-101<br>Staatsrecht<br>Obermöller       1H.2.62", "1H.2.62"),
    ("MDT-341<br>Cyber-Sicherheit<br>Hußmann 1H.1.27", "Cyber-Sicherheit"),
    ("Obermöller       1H.2.62|||||VIF-131", "1H.2.62"),
    ("|Sozialer Arbeit|Christian Bremer|2|1|", "Christian Bremer"),
    ("**2. Semester(Sommersemester), RGU**", "Sommersemester"),
    ("GroupPresentation(= results of workshops),20 min", "20 min"),
    ("die Note des Einzelfachs Mathematik (40% Gewicht)", "40 Prozent"),
    # kein Tag: Pfeile und Vergleiche bleiben Text
    ("**<==** start of picture text ... Düwelstraße 16-18", "Düwelstraße 16"),
    # Zusammengesetzte Zahlwörter
    ("es gibt einundzwanzig Wahlmodule", "21 Wahlmodule"),
])
def test_positiv(text, begriff):
    assert enthaelt(text, begriff), f"{begriff!r} nicht gefunden in {text!r}\n{tokens(text)}"


# ── Negativ: was nicht treffen darf ────────────────────────────────────────
@pytest.mark.parametrize("text, begriff", [
    ("Dokument 1 nennt „14 Bachelorstudiengänge“", "4 Bachelor"),       # andere Zahl
    ("Dokument 1 nennt „14 Bachelorstudiengänge“", "15 Bachelor"),
    ("ein 16-wöchiges Vorpraktikum", "6 Wochen"),                       # 16 ≠ 6, und wöchig ≠ wochen
    ("das Semester dauert 40 Wochen", "4 Wochen"),                      # kein Präfix bei Zahlen
    ("Beratung und Lehre", "Lehre und Beratung"),                        # Reihenfolge
    ("Stephan ist für Fragen, Anmeldungen, iCMS, Umbuchungen und Rittmüller",
     "Stephan Rittmüller"),                                              # Lücke > 3
    ("Raum 1B.0.44", "1B.0.43"),                                         # Kennung exakt
    ("15 Credit Points", "15 CP"),                                       # kurze Tokens exakt
    ("am 01.10.2026", "01.09.2026"),                                     # anderes Datum
    ("Frist bis 31. Mai", "31. Mai 2026"),                                # Jahr gefordert, fehlt
    ("am 31.052026", "31.05"),                                            # kein Datum
    ("", "irgendwas"),
    ("irgendwas", ""),
])
def test_negativ(text, begriff):
    assert not enthaelt(text, begriff), f"{begriff!r} fälschlich gefunden in {text!r}\n{tokens(text)}"


# ── Alternativen ───────────────────────────────────────────────────────────
def test_alternativen():
    antwort = "Die Anmeldung finden Sie unter „Anmeldung Studienprojekt“ [2]."
    assert enthaelt_begriff(antwort, ["Prüfungsverwaltung", "Anmeldung Studienprojekt"])
    assert not enthaelt_begriff(antwort, ["Prüfungsverwaltung", "Prüfungsamt"])
    assert enthaelt_begriff(antwort, "Studienprojekt")


# ── Tokenisierung im Detail ────────────────────────────────────────────────
@pytest.mark.parametrize("text, erwartet", [
    ("01.01 - 31.01.", ["01.01", "31.01"]),
    ("1. September 2026", ["01.09.2026"]),
    ("15.06. - 15.07. eines jeden Jahres", ["15.06", "15.07", "eines", "jeden", "jahres"]),
    ("≈ 9.000 Studierende", ["≈", "9000", "studierende"]),
    ("Support-it(at)hs-hannover.de", ["support", "it@hs", "hannover.de"]),
    ("vier Master-Studiengänge", ["4", "master", "studiengaenge"]),
    ("Version 2.5 vom 13.12.2012", ["version", "2.5", "vom", "13.12.2012"]),
    ("Raum 2A.1.11.", ["raum", "2a.1.11"]),
    ("am 1.11. beginnt es", ["am", "01.11", "beginnt", "es"]),
])
def test_tokens(text, erwartet):
    assert tokens(text) == erwartet


# ── V005: Zahl und Einheit direkt nebeneinander ───────────────────────────
@pytest.mark.parametrize("text, begriff", [
    ("Nach § 10 sind es 12 CP.", "10 CP"),
    ("In § 10 Abs. 2 stehen 12 Credits.", "10 Credits"),
    ("Die 10 weiteren CP entfallen.", "10 CP"),
    ("21 von 30 Wochen sind Präsenz.", "21 Wochen"),
])
def test_v005_zahl_einheit_ohne_luecke(text, begriff):
    assert not enthaelt(text, begriff)


@pytest.mark.parametrize("text, begriff", [
    ("maximal 10 CP, einmalig", "10 CP"),
    ("bis zu zehn Credits", "10 Credits"),
    ("Die Bearbeitungszeit beträgt 16 Wochen.", "16 Wochen"),
    ("Kosten: 500 € pro Semester", "500 €"),
    ("Dokument 1 nennt „11 praxisnahe Bachelor-Studiengänge“ [1]", "11 Bachelor"),
    ("Anmeldung für das Campusportal", "Anmeldung Campusportal"),
])
def test_v005_unveraendert(text, begriff):
    assert enthaelt(text, begriff)
