"""
Baut auswertung/gold/fallbasis.json aus zwei Quellen.

Quelle 1 — Chargy/Beidawi, 20 Fälle
    https://github.com/MohamedSalemCharghy/RAGHSH
    src/hsh_scraper/evals/validation_cases.json

    Ihre dreizehn Vertragsfelder werden unter ihren Originalnamen übernommen
    und bis auf eine dokumentierte Ausnahme (q04, siehe PRAEZISIERT) inhaltlich
    unverändert gelassen (question, reference_answer, source_url,
    accepted_source_urls, source_policy, required_facts, optional_facts,
    forbidden_claims, case_type, expected_behavior, case_difficulty,
    fairness_risk, evaluation_notes). Nur so bleibt vergleichbar, was
    verglichen werden soll, und ein späterer LLM-Judge kann direkt auf ihrem
    Vertrag arbeiten statt auf einer Übersetzung.

Quelle 2 — unsere sechs Nutzerfragen aus eval/fragen_nutzer.csv
    Werden auf dasselbe Schema abgebildet: antwort → reference_answer,
    belegstelle → Ausgangspunkt für pflicht_begriffe.

Quelle 3 — die 50 geschichteten Fragen aus eval/fragen.csv (seit 21.09.2026)
    Gleiche Abbildung, aber OHNE pflicht_begriffe: Für Retrieval-Experimente
    genügen Frage und Goldquelle, und dafür wurden sie gebraucht. Im Volllauf
    zählen sie damit als „nicht prüfbar" (Antwort ok, wenn Gold in den
    Quellen). Begriffe kommen nach, sobald jemand sie kuratiert hat — dann
    ändert sich der Vertrag, und das Manifest sagt es.

Vier eigene Felder kommen hinzu
------------------------------
herkunft         chargy_beidawi | nutzer_gold | eval_gold_50
gold_im_korpus   wird hier gegen doc_index.db geprüft, nicht geraten
pflicht_begriffe kurze WÖRTLICHE Zeichenketten für die deterministische Prüfung
fakultaet        Filter, den der Lauf setzt (vorerst überall leer)

Zu pflicht_begriffe
-------------------
`required_facts` sind ganze Sätze („Stephan Rittmüller ist die genannte
Ansprechperson.") und maschinell nicht prüfbar — eine korrekte Antwort
formuliert anders. `pflicht_begriffe` ist die prüfbare Kurzform desselben
Fakts („Stephan Rittmüller"). Beide bleiben nebeneinander stehen: die Sätze für
einen späteren Judge, die Begriffe für die Auswertung heute.

Die Auswahl ist bewusst konservativ. Aufgenommen wird nur, was eine richtige
Antwort mit hoher Wahrscheinlichkeit WÖRTLICH enthält — Eigennamen, Raum- und
Aktenzeichen, Fristen, feststehende Bezeichnungen. Zahlen ohne Kontext ("11")
sind ausgeschlossen, weil sie in jedem Datum mitzählen würden.

Wo sich kein solcher Begriff finden lässt, bleibt die Liste LEER. Das ist kein
Versäumnis, sondern eine Aussage: Die Auswertung wertet eine leere Liste als
„nicht prüfbar" und nicht als „erfüllt" (siehe auswertung.py). Betrifft q18,
wo mehrere offizielle Zahlen gleichermaßen gültig sind.

Aufruf:  python auswertung/gold/baue_fallbasis.py [--quelle PFAD]
"""

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent
from raghsh import konfig  # noqa: E402

from raghsh.crawl.crawler_utils import normalize_url  # noqa: E402

sys.path.insert(0, str(_BASE))
from vergleich import anzeige, enthaelt_begriff  # noqa: E402
from vertrag import HASHDATEI, hash_fallbasis, kurz  # noqa: E402

DOC_INDEX   = konfig.DOC_INDEX
NUTZER_CSV  = konfig.AUSWERTUNG / "fragen" / "fragen_nutzer.csv"
FRAGEN50_CSV = konfig.AUSWERTUNG / "fragen" / "fragen.csv"
ZIEL        = _BASE / "fallbasis.json"

# ---------------------------------------------------------------------------
# pflicht_begriffe je Fall — aus required_facts abgeleitet, siehe Modulkopf
# ---------------------------------------------------------------------------

# Ein Eintrag ist ein String oder eine Liste von Alternativen (einer genügt).
# Der Vergleich ist tolerant (auswertung/gold/vergleich.py): Zahlwörter, Datumsformen,
# Interpunktion, Flexion, eingeschobene Wörter — die Begriffe müssen also nicht
# mehr jede Schreibweise vorwegnehmen, wohl aber den Kern der Aussage treffen.
BEGRIFFE: dict[str, list] = {
    "q01": ["Sensor- und Automatisierungstechnik", "Elektrische Energiesysteme"],
    "q02": ["6 Wochen"],   # „zulassungsfrei“ steht in der Frage (30.09.2026)
    "q03": ["Bachelor of Engineering", "7 Semester"],
    "q04": ["Bewerbungsportal"],
    # Die Antwort nennt den Link oder die Seite — beides belegt den Fundort.
    # „Studienprojekt“ / „Anmeldung Studienprojekt“ stehen in der Frage (30.09.2026)
    "q05": [["Prüfungsverwaltung", "Prüfungsamt", "Formular"]],
    "q06": ["Stephan Rittmüller", "F1-Pruefungsverwaltung"],
    "q07": ["1. April", "31. Mai"],
    "q08": ["01.01.", "31.01.", "15.06.", "15.07."],
    "q09": ["1B.0.43", "support-it@hs-hannover.de", "4444"],
    "q10": ["Ricklinger Stadtweg 120", "Studentenwerk"],
    "q11": ["iCMS"],   # „Notenspiegel“ steht in der Frage (30.09.2026)
    # Gefragt ist nur, ob es Parkplätze gibt; die Anfahrtsstraßen sind eine Alternative (30.09.2026).
    "q12": [["unmittelbarer Nähe", "Stammestraße", "Lodemannweg"]],
    "q13": ["ohne Voranmeldung"],
    "q14": ["11 Bachelor"],
    "q15": ["Elektrische Antriebe und Automatisierungstechnik"],
    "q16": ["15 Bachelor", "4 Master"],   # gestrichen, siehe AUSGESCHLOSSEN
    "q17": ["15 CP", "Stephan Rittmüller"],
    # Leer mit Absicht: die Quelle nennt 9.000 und 10.000 nebeneinander, beide
    # Angaben sind laut Fallvertrag gültig. Ein Begriffstest wäre hier Willkür.
    "q18": [],
    "q19": ["Allgemeiner Maschinenbau", "Intelligente Produktionssysteme"],
    "q20": ["Zentrum für Lehre und Beratung"],
    "n01": ["Elektro- und Informationstechnik", "Diakonie"],
    "n02": ["01.09.2026", "21.09.2026"],
    "n03": [["10 Wochen", "10-wöchig", "zehnwöchig"]],   # Fassung 2024, siehe WEITERE_QUELLEN
    "n04": ["drei Jahre"],
    # „Erfindungsmeldung“ steht in der Frage; zuständig ist seit der Drittmittel-
    # richtlinie 2024 D1 – Personal und Recht (vorher Justiziariat), beides gilt.
    "n05": [["Justiziariat", "Dezernat 1", "D1", "Team Recht", "Personal und Recht"]],
    # S-Kürzel mit Namen (01.10.2026, Christian): nur so fällt eine vertauschte
    # Zuordnung auf — der Organigramm-Text im HEP-PDF ist verrutscht und legt
    # „S2 Forschung …“ nahe; richtig ist das Organigramm der Zentralen Verwaltung.
    "n06": ["S1 Strategische Hochschulentwicklung", "S2 Kommunikation und Marketing",
            "S3 Forschung, Entwicklung und Transfer"],
    # Ersatz für q10/q20 (30.09.2026), Belege in eval/fragen_nutzer.csv
    "n07": ["3E.1.19"],
    "n08": ["B2"],
}

# Die Nutzerfragen: Reihenfolge der CSV bestimmt n01…n08 (n07/n08 seit 30.09.2026 als Ersatz für q10/q20).
# Zwei Fälle bekommen eine gelockerte Quellenpolitik, weil die Frage von
# mehreren offiziellen Seiten gleichwertig beantwortet wird — genau das
# Fairness-Problem, das Chargy/Beidawi in ihrem Bericht benennen und das unsere
# bisherige Messung als Fehltreffer gewertet hat.
NUTZER_POLITIK = {
    "n01": ("accepted_sources", "hoch"),
    "n02": ("exact_gold_source", "mittel"),
    "n03": ("exact_gold_source", "niedrig"),
    "n04": ("exact_gold_source", "niedrig"),
    "n05": ("exact_gold_source", "niedrig"),
    "n06": ("exact_gold_source", "niedrig"),
}


# ---------------------------------------------------------------------------
# Präzisierte Fragen
# ---------------------------------------------------------------------------
# Drei Fragen waren so gestellt, dass keine Quelle sie eindeutig beantworten
# kann: „[Nutzerfrage – nicht veröffentlicht]“ und „Wie lange dauert eine
# Praxisphase?“ haben je Fakultät andere Antworten, „Wie stelle ich einen
# Online-Antrag?“ je nach Antragsart. Gemessen wurde damit nicht das System,
# sondern das Raten der wahrscheinlichsten Lesart.
#
# Die Originalfrage bleibt als `question_original` erhalten, der Grund als
# `anpassung`. Beides steht im Bericht, damit ein Vergleich mit Läufen vor
# dieser Änderung — und mit dem Lauf von Charghy und El Beidawi bei q04 —
# nicht stillschweigend falsch wird.
#
# `patch` zieht den Vertrag nach, wo die neue Frage ihn sonst widerlegen
# würde: Bei q04 war die Mehrdeutigkeit der Prüfgegenstand; ist sie in der
# Frage aufgelöst, kann der Fall keine Rückfrage mehr verlangen.
PRAEZISIERT: dict[str, dict] = {
    # Durchsicht aller Fälle 30.09.2026: fehlender Kontext
    "q02": {
        "question": "Ist der Bachelorstudiengang Mechatronik an der Fakultät I zulassungsfrei und "
                    "welches Vorpraktikum wird verlangt?",
        "grund": "Es gibt Mechatronik (MEC, Fakultät I) und Mechatronik dual (MTD, Fakultät II).",
    },
    "q03": {
        "question": "Welchen Abschluss hat Mechatronik an der Fakultät I der HsH, wann beginnt das "
                    "Studium und wie lange ist die Regelstudienzeit?",
        "grund": "Es gibt Mechatronik (MEC, Fakultät I) und Mechatronik dual (MTD, Fakultät II); "
                 "Luna zählte beide auf.",
    },
    "g02": {
        "question": "In welchem Raum sitzt die Sachbearbeiterin des Praktikumsamts der Fakultät V?",
        "grund": "Ohne Fakultät war offen, welches Praktikumsamt gemeint ist.",
    },
    "g12": {
        "question": "Im Master Business Development (MBD) mit Doppelabschluss an der Robert Gordon "
                    "University (RGU): In welchem Semester bin ich dort?",
        "grund": "Nur Abkürzungen, kein Studiengang.",
    },
    "g12": {
        "question": "Wann gehe ich im Doppelmaster Business Development an die Robert Gordon University?",
        "grund": "Original („Wann bin ich im Doppelmaster an der RGU?“) nur mit Abkürzungen. "
                 "Realitätsnah formuliert (Christian, 30.09.2026). Die Quellen widersprechen sich "
                 "(RGU-Flyer 2022: 2. und 3. Semester; Studienplan 2023 und Application Notes: nur "
                 "2. Semester) — eine gute Antwort folgt dem neueren Studienplan (Ziel 1: aktuell).",
        "patch": {
            "reference_answer": "Im 2. Semester (Sommersemester; Beginn an der RGU um den 20. Januar), "
                                "das 3. Semester ist wieder an der HsH (Studienplan 2023).",
            # „Wann“ lässt sich auch mit dem Monat beantworten; das englische Dokument sagt „Summer Term“.
            # „nach dem ersten Semester“ ist dieselbe Aussage (Christian, 30.09.2026).
            "pflicht_begriffe": [["zweiten Semester", "zweites Semester", "2. Semester",
                                  "nach dem ersten Semester"],
                                 ["Sommersemester", "Summer Term", "Januar"]],
        },
    },
    "g49": {
        "question": "Ich bekomme ein Stipendium nach der Stipendien-Rahmenordnung der HsH. Wie wirkt "
                    "sich eine Geburt auf die Zahlung aus?",
        "grund": "Offen war, welches Stipendium gemeint ist (Rahmenordnung der HsH oder "
                 "Deutschlandstipendium).",
    },
    "g22": {
        "question": "Mein Studium dauert wegen einer schweren Erkrankung deutlich länger. Kann ich mir "
                    "die Langzeitstudiengebühr erlassen lassen, und wo stelle ich den Antrag?",
        "grund": "Gold war ein Covid-Formular; Frage auf die aktuelle Seite Langzeitstudiengebühren "
                 "ausgerichtet.",
        "patch": {
            "reference_answer": "Ja, bei unbilliger Härte, die u. a. bei einer schweren Erkrankung vorliegt, "
                                "kann die Langzeitstudiengebühr auf Antrag erlassen werden. Der Online-Antrag "
                                "steht im Campusmanagement unter „Studienservice“.",
            # wörtlich aus der Goldseite, damit die Belegpunkte gefunden werden
            "required_facts": ["kann die Langzeitstudiengebühr auf Antrag erlassen werden",
                               "Den Online-Antrag finden Sie in Ihrem Campusmanagement"],
        },
    },
    "g29": {
        "question": "In welchem Bachelor der Fakultät IV lerne ich sowohl Programmieren als auch Gestalten?",
        "grund": "Ohne Fakultät kamen mehrere Studiengänge in Frage.",
    },
    "g30": {
        "question": "Welches Wahlpflichtmodul der Abteilung Maschinenbau (Fakultät II) beschäftigt sich "
                    "im Wintersemester 2026/27 mit Drohnen?",
        "grund": "Studiengang und Semester fehlten; Goldquelle ist die Wahlpflichtliste WS26 der Abt. M.",
    },
    "g32": {
        "question": "Warum haben die Studierenden im Projekt QpLuS IM die interaktiven Lernbausteine "
                    "kaum angenommen?",
        "grund": "Ohne Projektbezug war die Frage nicht beantwortbar.",
    },
    "g34": {
        "question": "In welchem Raum wird im Wintersemester 2026/27 im ersten Semester der "
                    "Verwaltungsinformatik Staatsrecht gelesen?",
        "grund": "Semester fehlte; der Raum wechselt (SoSe 2026 1H.2.62, WiSe 2026/27 1H.1.19).",
        "patch": {"reference_answer": "In Raum 1H.1.19 (VIF-101 Staatsrecht, Obermöller).",
                  "required_facts": ["Staatsrecht Obermöller 1H.1.19"]},
    },
    "g38": {
        "question": "Zwischen welchen beiden Modulen kann ich mich im dritten Semester des Masters "
                    "Management für Pflege- und Gesundheitsberufe entscheiden?",
        "grund": "Es gibt mehrere Pflege-Master; Luna zählte zwei auf.",
    },
    # Durchsicht 30.09.2026 (siehe AUSGESCHLOSSEN): Fragen, die ohne Kontext
    # auch ein Mensch nicht eindeutig beantworten könnte.
    "g18": {
        "question": "Ich studiere Informatik an der Fakultät IV und mein Prüfungsfach taucht "
                    "bei der Prüfungsanmeldung nicht auf. An wen wende ich mich?",
        "grund": "Ohne Fakultät und Studiengang war offen, welche Prüfungsverwaltung gemeint "
                 "ist (Luna antwortete korrekt für Fakultät II). Die Goldquelle ist der "
                 "Prüfungsplan Informatik der Fakultät IV.",
    },
    "g23": {
        "question": "Ich forsche an der HsH und mein Zeitschriftenaufsatz ist nur kostenpflichtig "
                    "zugänglich. Was erwartet die Open-Access-Policy der HsH von mir, "
                    "damit er trotzdem frei lesbar wird?",
        "grund": "Original („Mein Aufsatz ist nur gegen Gebühr lesbar – was erwartet die "
                 "Hochschule von mir?“) war ohne Bezug auf Forschende und Open Access nicht "
                 "verständlich.",
    },
    "g26": {
        "question": "Mir fehlen Grundkenntnisse in Automatisierungstechnik, ich möchte aber die "
                    "Weiterbildung „IT-Sicherheit in Produktionsanlagen“ der HsH-Akademie "
                    "besuchen. Was wird empfohlen?",
        "grund": "Original fragte nach „Vorwissen zu Anlagensteuerung“ ohne Bezug zur "
                 "Weiterbildung; das RAG antwortete plausibel zum EIT-Studium.",
    },
    "g31": {
        "question": "Soziale Arbeit berufsbegleitend (SBW), Wintersemester 2026/27: Welche "
                    "Lehrveranstaltung im 2. Semester behandelt Sozialpsychiatrie, und wer lehrt sie?",
        "grund": "Studiengang und Semester fehlten; die Goldquelle ist der SBW-Stundenplan "
                 "WiSe 2026/27.",
    },
    "g33": {
        "question": "Wo sitze ich im Wintersemester 2026/27 in der Veranstaltung zur Sicherheit "
                    "im Master Digitale Transformation?",
        "grund": "Ohne Semester war offen, welcher Stundenplan gilt (SoSe 2026 nennt einen "
                 "anderen Raum). Goldquelle: Stundenplan MDT WiSe 2026/27.",
    },
    "g43": {
        "question": "Ich studiere Religionspädagogik und Soziale Arbeit nach der Prüfungsordnung "
                    "2014 und möchte in Teilzeit studieren. Was muss ich zusammen mit dem Antrag "
                    "vorlegen?",
        "grund": "Teilzeit ist je Studiengang und Fassung geregelt; die Nachweispflicht steht "
                 "in den BT-PO BRS 2011–2015, ab 2018 nicht mehr. Die Goldquelle ist die BT-PO "
                 "BRS 2014.",
    },
    "q17": {
        "question": "Wer ist in der Prüfungsverwaltung der Fakultät I Ansprechpartner für die "
                    "15-CP-Überprüfung?",
        "grund": "Original („Gibt es Informationen zur 15-CP-Überprüfung in der Fakultät I und "
                 "ist dort ein freier Versuch beschrieben?“) fragte nicht nach der Person, die "
                 "der Vertrag verlangt; die verschachtelte Doppelfrage mass Raten statt Retrieval.",
        "patch": {
            "expected_behavior": "direct_answer",
            "reference_answer": "Stephan Rittmüller; die Seite der Prüfungsverwaltung der "
                                "Fakultät I nennt ihn als Ansprechpartner u. a. für die "
                                "15 CP Überprüfung.",
            "required_facts": ["Stephan Rittmüller ist in der Prüfungsverwaltung der Fakultät I "
                               "Ansprechpartner für die 15 CP Überprüfung."],
            "optional_facts": ["Er ist auch für allgemeine Fragen, Prüfungsanmeldungen, iCMS "
                               "und Umbuchungen zuständig."],
            "forbidden_claims": ["Eine Freiversuchsregel aus der Quelle ableiten oder erfinden."],
            "evaluation_notes": "Präzisiert 30.09.2026: gefragt ist die Ansprechperson.",
            # „15 CP“ steht jetzt in der Frage und prüft nichts mehr.
            "pflicht_begriffe": ["Stephan Rittmüller"],
        },
    },
    "n02": {
        "question": "[Nutzerfrage – nicht veröffentlicht]",
        "grund": "Ohne Fakultät blieb offen, welche Vorlesungszeiten gemeint sind. "
                 "Die Goldquelle gilt hochschulweit und nimmt nur Fakultät V aus, "
                 "trägt die Antwort für Fakultät I also unverändert.",
    },
    "n03": {
        "question": "[Nutzerfrage – nicht veröffentlicht]",
        "grund": "Jede Fakultät hat eine eigene Praxisphasenordnung mit eigener "
                 "Dauer. Die Goldquelle ist die Ordnung der Fakultät I; ohne diese "
                 "Angabe war die Frage nicht eindeutig beantwortbar.",
    },
    "q04": {
        "question": "Wie stelle ich einen Online-Antrag für die Einschreibung?",
        "grund": "Original von Charghy und El Beidawi: „Wie stelle ich einen "
                 "Online-Antrag?“ — bewusst mehrdeutig zwischen Bewerbung "
                 "und Antrag eingeschriebener Studierender. Mit der Präzisierung "
                 "ist der Bewerbungsweg gemeint; der Vertrag verlangt deshalb keine "
                 "Rückfrage mehr.",
        "patch": {
            "case_type": "direct",
            "expected_behavior": "direct_answer",
            "clarification": None,
            "required_facts": [
                "Das Bewerbungsverfahren läuft ausschließlich digital über das "
                "Bewerbungsportal der Hochschule Hannover.",
                "Erforderliche Unterlagen werden online eingereicht; postalischer "
                "Versand ist nicht mehr nötig.",
                "Der Bearbeitungsstatus ist im Portal einsehbar.",
            ],
            "optional_facts": [
                "Die Umstellung gilt ab dem Sommersemester 2026.",
                "Zulassung oder Ablehnung kommt über das Portal.",
                "Eingeschriebene Studierende finden ausgewählte Anträge nach dem "
                "Login unter Service → Anträge.",
            ],
            "forbidden_claims": [
                "Ein postalisches Verfahren als notwendig darstellen.",
            ],
            "case_difficulty": "mittel",
            "fairness_risk": "mittel",
            "evaluation_notes": "Gefragt ist der Weg für Bewerber:innen. Das "
                                "Wort „Einschreibung“ steht so nicht auf der "
                                "Seite; gemeint ist der Abschnitt „Für "
                                "Studieninteressierte und Bewerber:innen“.",
        },
    },
}


# ---------------------------------------------------------------------------
# Quellen-Nachprüfung
# ---------------------------------------------------------------------------
# Am 14.09.2026 wurden alle 26 Referenzantworten gegen die Live-Seiten der HsH
# geprüft (Seiten per curl, PDFs per PyMuPDF): Kein Fakt hat sich geändert.
# Zugleich wurde je Fall gesucht, welche WEITEREN Dokumente dieselbe Antwort
# tragen — im Web (site:hs-hannover.de) und im eigenen Korpus (var/korpus,
# Volltext). Aufgenommen ist nur, was beides besteht: live verifiziert UND
# alle Pflichtbegriffe im Text. Ausnahmen sind markiert.
#
# Warum das nötig war: Der Vertrag von Charghy/Beidawi kennt fast überall nur
# EINE Goldquelle (`exact_gold_source`). Dieselben Fakten stehen an der HsH
# aber regelmäßig auf drei bis fünf offiziellen Seiten. Eine richtige Antwort
# aus einer davon zählte bisher als Fehlschlag — q09 etwa wurde vollständig
# richtig beantwortet (Raum, Durchwahl, Adresse) und trotzdem als
# „Korpuslücke“ aus der Quote gestrichen, weil nur die Startseite des
# IT-Portals als Gold eingetragen war und die gecrawlte Unterseite nicht.
#
# Wo Quellen ergänzt sind, wechselt `source_policy` auf `accepted_sources`,
# sonst ignoriert die Auswertung sie. Die ursprüngliche Politik bleibt als
# `source_policy_original` erhalten.
#
# Zwei inhaltliche Korrekturen stecken mit drin:
#   n03  Die Goldquelle (Praxisphasenordnung 2012) ist durch die 2. Änderung
#        vom 01.09.2024 abgelöst. Dauer weiterhin 10 Wochen, aber die neue
#        Fassung nennt keine „18 Credits“ mehr — der Begriff fliegt raus.
#   q16  gestrichen, siehe AUSGESCHLOSSEN.
QUELLEN_GEPRUEFT = "2026-09-14"   # g29/g39/g41: 2026-09-21

# Fälle mit Verfallsdatum (30.09.2026): Die Antwort hängt an einem Semester
# (Stundenplan, Wahlpflichtliste, Semestertermine). Nach dem Datum warnt der
# Bau; der Fall muss dann auf das neue Semester umgestellt oder gestrichen werden.
GUELTIG_BIS: dict[str, str] = {
    "g30": "2027-02-28", "g31": "2027-02-28", "g33": "2027-02-28", "g34": "2027-02-28",
    "g35": "2027-08-31", "n02": "2027-08-31",
}

# Fälle, die nicht in die Fallbasis kommen — mit Grund. Sie bleiben in der
# Quelldatei von Charghy/Beidawi unverändert; alte Läufe tragen sie noch in
# ihren Rohdaten, `auswertung.py --vertrag aktuell` lässt sie dann weg.
AUSGESCHLOSSEN: dict[str, str] = {
    "q16": "Entscheidung 15.09.2026: f3.hs-hannover.de/ nennt live 15 Bachelor-"
           "studiengänge, f3.hs-hannover.de/studium nennt live 14. Die Website "
           "widerspricht sich selbst — der Fall misst dann nicht das RAG, sondern "
           "welche Seite es zuerst liest.",
    # Durchsicht 30.09.2026 (Christian/Nemo) der 17 im Luna-Lauf 2026-09-30_1306
    # nicht beantworteten Fälle: gestrichen, weil die Frage mehrdeutig oder das
    # Gold veraltet ist; ersetzt durch n07/n08 (q10/q20).
    "g09": "Entscheidung 30.09.2026: Mehrere Caritas-Ausschreibungen im Korpus, die "
           "Frage nennt nicht welche; Stellenanzeigen veralten zudem schnell.",
    "g11": "Entscheidung 30.09.2026: Gold ist eine archivierte Kursbeschreibung der "
           "Summer School 2019; die Frage ist mit dem Master Digitale Transformation "
           "verwechselbar.",
    "g40": "Entscheidung 30.09.2026: Gold („Justiziariat“, Richtlinie 2013) ist durch "
           "die Richtlinie vom 29.03.2024 (Beschwerdestelle Dezernat 1 – Team Recht) und "
           "die Antidiskriminierungsstelle überholt.",
    "g42": "Entscheidung 30.09.2026: „In welchem Teil“ lässt Studienabschnitt und "
           "Semester gleichermaßen zu; beide Antworten sind richtig.",
    "g08": "Entscheidung 30.09.2026: Stellenanzeige (Albemarle, 01/2026) — veraltet "
           "schnell, derselbe Grund wie bei g09.",
    "q10": "Entscheidung 30.09.2026: Goldseite nie gecrawlt (Korpuslücke), der Fall "
           "misst nichts. Ersetzt durch n07.",
    "q20": "Entscheidung 30.09.2026: Goldseite (Meldung von 2019) nie gecrawlt "
           "(Korpuslücke). Ersetzt durch n08.",
}

WEITERE_QUELLEN: dict[str, dict] = {
    # ── Durchsicht aller Fälle 30.09.2026 (Christian/Nemo)
    "g18": {
        "accepted": [
            "https://f4.hs-hannover.de/fileadmin/HsH/Forms/Fakultaet_IV/StudServ/Pruefungsplan/Informatik/2026-06-23-Pruefungsplan_SS26.pdf",
            "https://f4.hs-hannover.de/fileadmin/HsH/Forms/Fakultaet_IV/StudServ/OFI/Informatik/Broschueren/2025-08-25-InfobroschuereErstsemesterInformatik_2025-2026_0825.pdf",
            "https://f4.hs-hannover.de/fileadmin/HsH/Forms/Fakultaet_IV/StudServ/OFI/Informatik/Broschueren/25-09-10_Studienhandbuch_2025_AbtlgInf_StudyGuide_CSDept_080925.pdf",
        ],
        "begriffe": [["Dzinblewski", "f4-i-dekanat"]],
        "grund": "Gold war nur der archivierte Prüfungsplan. Der aktuelle Prüfungsplan SS26 enthält "
                 "dieselbe Anweisung; laut Studienhandbuch 2025 ist Frau Dzinblewski das Sekretariat "
                 "der Abteilung Informatik mit der Adresse F4-I-Dekanat@hs-hannover.de — Name oder "
                 "Adresse sind gleichwertig richtig (30.09.2026).",
    },
    "g10": {
        "accepted": [],
        "begriffe": [["Courses offered in English", "exchange students", "Schedule"]],
        "grund": "„Semesterplan“ stand in der Frage und prüfte nichts (30.09.2026).",
    },
    "g16": {
        "accepted": [
            "https://www.hs-hannover.de/hsh-akademie/weiterbildungen-a-z/projektmanagement-level-c/b-1",
            "https://www.hs-hannover.de/hsh-akademie/weiterbildungen-a-z/projektmanagement-level-d-1",
            "https://www.hs-hannover.de/hsh-akademie/weiterbildungen-a-z/projektmanagement-level-d-1-2",
            "https://www.hs-hannover.de/hsh-akademie/weiterbildungen-a-z/projektmanagement-level-d-winter",
            "https://www.hs-hannover.de/hsh-akademie/weiterbildungsangebot/projektmanagement",
        ],
        "grund": "Gold war ein einzelner Informationsabend; die dauerhaften Kursseiten enthalten "
                 "denselben Satz („bereitet Sie optimal auf diese Führungsaufgaben vor“).",
    },
    "g22": {
        "gold": "https://www.hs-hannover.de/ueber-uns/organisation/akademische-angelegenheiten/studierende/langzeitstudiengebuehren",
        "accepted": [],
        "begriffe": [["unbillige Härte", "Härtefall"], "Campusmanagement"],
        "grund": "Gold war ein Covid-Antragsformular. Die Seite Langzeitstudiengebühren nennt den "
                 "Erlass bei unbilliger Härte (u. a. schwere Erkrankung) und den Online-Antrag im "
                 "Campusmanagement unter „Studienservice“.",
    },
    "g34": {
        "gold": "https://f4.hs-hannover.de/fileadmin/HsH/Forms/Fakultaet_IV/StudServ/Stundenplan/WI/26-07-27-WiSe26-VIF-1.Studienabschnitt.pdf",
        "accepted": [],
        "begriffe": ["1H.1.19"],
        "grund": "Auf den aktuellen Stundenplan WiSe 2026/27 umgestellt (SoSe 2026: 1H.2.62, "
                 "WiSe 2026/27: 1H.1.19, 1. Semester, VIF-101 Staatsrecht, Obermöller).",
    },
    "g44": {
        "accepted": [
            "https://www.hs-hannover.de/fileadmin/HsH/Forms/Hochschule_Hannover/UEber_uns/Organe_und_Gremien/Verkuendungsblatt/2021/01-2021/02_PO_MDT_2021.pdf",
            "https://f4.hs-hannover.de/fileadmin/HsH/Forms/Fakultaet_IV/StudServ/OFI/WI/Ordnungen/01-MDT-PO2021-210318.pdf",
        ],
        "grund": "PO MDT 2021 nennt dieselben vier Monate (höchstens sechs) wie PO 2018 (30.09.2026).",
    },
    "n08": {
        "accepted": [
            "https://www.hs-hannover.de/hsh-akademie/wissenswelt-sprachen-und-studienkompetenzen/schreibwerkstatt/kurse",
        ],
        "grund": "Dieselbe Kursliste der Schreibwerkstatt steht unter Servicezentrum und HsH-Akademie (30.09.2026).",
    },
    "q01": {
        "accepted": [
            "https://f1.hs-hannover.de/fileadmin/HsH/Fakultaet_I/20_Dokumente/30_Zeitplaene/Erstsemester/Erstsemester_SoSe26_Entwurf.pdf",
        ],
        "grund": "Erstsemesterinformation der Fakultät I nennt beide konsekutiven Master (ESA, EEE, je 3 Semester).",
    },
    "q02": {
        "accepted": [
            "https://www.hs-hannover.de/studium/studienangebot/bachelor-studiengaenge",
        ],
        "grund": "Zentrale Studienangebotsseite: Mechatronik zulassungsfrei, Vorpraktikum.",
    },
    "q03": {
        "accepted": [
            "https://www.hs-hannover.de/studium/studienangebot/bachelor-studiengaenge",
        ],
        "grund": "Zentrale Studienangebotsseite: B.Eng., Sommer- und Wintersemester, 7 Semester.",
    },
    "q04": {
        "accepted": [
            "https://www.hs-hannover.de/ueber-uns/organisation/akademische-angelegenheiten/studieninteressierte/bewerbung",
        ],
        "grund": "Bewerbungsseite: „vollständig papierlos“, „ausschließlich digital im Bewerbungsportal“.",
    },
    "q05": {
        "accepted": [
            "https://f1.hs-hannover.de/fileadmin/HsH/Fakultaet_I/20_Dokumente/60_Pruefungsamt/Formular_Studentenprojekt_Anmeldung.pdf",
            "https://f1.hs-hannover.de/fileadmin/HsH/Fakultaet_I/20_Dokumente/60_Pruefungsamt/Wichtige_Hinweise_fuer_Projekte_bzw._Studienprojekte.pdf",
        ],
        "grund": "Die beiden PDFs, auf die die Goldseite verlinkt; wer sie zitiert, hat die Anmeldung gefunden.",
    },
    "q07": {
        "accepted": [
            "https://www.hs-hannover.de/studium/studieren-an-der-hsh/rund-ums-studium/stipendien/wir-stipendium-studierende/timetable",
            "https://www.hs-hannover.de/studium/studieren-an-der-hsh/rund-ums-studium/stipendien/landesstipendium-niedersachsen",
        ],
        "grund": "Timetable und Landesstipendium nennen dieselbe Frist 1. April – 31. Mai.",
    },
    "q09": {
        "accepted": [
            "https://service.it.hs-hannover.de/services/support",
            "https://www.hs-hannover.de/ueber-uns/organisation/hochschul-it",
        ],
        "grund": "Beide Seiten sind gecrawlt und nennen Raum 1B.0.43, Durchwahl 4444 und support-it@. "
                 "Die Startseite des IT-Portals (Gold) ist es nicht — der Fall war nie eine Korpuslücke.",
    },
    "q12": {
        "accepted": [
            "https://f1.hs-hannover.de/service/standorte-oeffnungszeiten",
        ],
        "grund": "Fakultät I wiederholt den Anfahrtstext wörtlich (Stammestraße, Lodemannweg).",
    },
    "q14": {
        "accepted": [
            "https://www.hs-hannover.de/fileadmin/HsH/Hochschule_Hannover/Organisation/Kommunikation_und_Marketing/Broschueren/2022-26_Hochschulentwicklungsplan_Web.pdf",
        ],
        "grund": "Hochschulentwicklungsplan 2022–26, Basisdaten Fakultät II: 11 Bachelor-Studiengänge.",
    },
    "q15": {
        "accepted": [
            "https://f1.hs-hannover.de/ueber-uns/fachgebiete/elektrische-antriebe-und-automatisierungstechnik",
        ],
        "grund": "Fachgebietsseite nennt Kreim als Fachgebietsleiter.",
    },
    "q18": {
        "accepted": [
            "https://www.hs-hannover.de/ueber-uns/profil/daten-und-fakten/geschichte-der-hsh",
            "https://www.hs-hannover.de/fileadmin/HsH/Hochschule_Hannover/Organisation/Kommunikation_und_Marketing/PDF__Sonstige/IMAGE_Praesentation_27.06.2025_web.pdf",
        ],
        "grund": "Geschichtsseite („rund 9.000“) und Imagepräsentation 2025 („≈ 9.000“) sind offizielle "
                 "gerundete Angaben im Sinne des Vertrags. /ueber-uns (10.000) ist nicht im Korpus.",
    },
    "q19": {
        "accepted": [
            "https://f2.hs-hannover.de/fileadmin/HsH/Fakultaet_II/Abt_M/Dokumente/Ordnungen/MAB/F2_BA_MAB_BTPO_2025-07-15.pdf",
            "https://www.hs-hannover.de/fileadmin/HsH/Hochschule_Hannover/UEber_uns/Organe_und_Gremien/Verkuendungsblaetter/09-2025/06_F2_BA_MAB_VB_EBU.pdf",
            "https://www.hs-hannover.de/studium/studienangebot/bachelor-studiengaenge",
            "https://f2.hs-hannover.de/studium/studierende/semesteruebersichten",
        ],
        "grund": "BTPO 2025 (Verkündungsblatt 09/2025) und die zentrale Studienangebotsseite nennen alle drei "
                 "Schwerpunkte. Die PO 2019 kennt nur zwei und ist für Einschreibungen ab WS 2025 abgelöst.",
        # PO-abhängig: PO 2019 gilt noch bis 31.08.2028 (Christian, 01.10.2026)
        "patch": {"rueckfrage_zulaessig": True,
                  "evaluation_notes": "Die Antwort muss die drei Studienschwerpunkte korrekt berücksichtigen: Allgemeiner Maschinenbau mit vier Wahlpflichtmodulen aus über 20 Wahlmöglichkeiten; Intelligente Produktionssysteme mit zwei Wahlpflichtmodulen und einem Projekt; Nachhaltige Werkstoffe, Produkte und Prozesse ebenfalls mit zwei Wahlpflichtmodulen und einem Projekt. Keine konkreten Modulnamen erfinden, weil die Quelle an dieser Stelle keine konkreten Modulnamen nennt. PO-abhängig (01.10.2026, Christian): Die PO 2019 gilt für Einschreibungen vor WS 2025 noch bis 31.08.2028. Treffer ist die Antwort mit den Schwerpunkten der PO 2025 (oder eine Antwort je Fassung); eine Rückfrage nach der PO ist zulässig (kein Treffer, aber kein Fehler)."},
    },
    "n01": {
        "accepted": [
            "https://www.hs-hannover.de/fileadmin/HsH/Hochschule_Hannover/Organisation/Servicezentrum_Beratung/Studienberatung/250630_Z3_Broschuere_Bachelor_Master_HsH.pdf",
            "https://www.hs-hannover.de/ueber-uns/organisation/servicezentrum-beratung/studienberatung/studieninteressierte-1",
            # Live vorhanden, aber nicht gecrawlt — Crawl-Kandidat, kein Retrieval-Ziel.
            "https://www.hs-hannover.de/studium/studieren-an-der-hsh/warum-an-der-hsh-studieren/fakultaeten-der-hochschule",
        ],
        "grund": "Studienberatungsbroschüre 2025 und Studienberatungsseite listen alle fünf Fakultäten mit Namen.",
    },
    "n02": {
        "accepted": [
            "https://www.hs-hannover.de/ueber-uns/organisation/akademische-angelegenheiten/studierende/wichtige-termine-fristen",
        ],
        "grund": "Zentrale Terminseite: Semesterbeginn 01.09.2026, Vorlesungsbeginn 21.09.2026, "
                 "abweichend nur Fakultät V — gilt also für Fakultät I.",
    },
    "n03": {
        "gold": "https://f1.hs-hannover.de/fileadmin/HsH/Fakultaet_I/20_Dokumente/10_Ordnungen/20_Praxisphasenordnung/01_Praxisphasenordnung_F1_2024_v06.pdf",
        "accepted": [
            "https://f1.hs-hannover.de/fileadmin/HsH/Fakultaet_I/20_Dokumente/60_Pruefungsamt/Verkuendet_01_Praxisphasenordnung_F1_2024_v06.pdf",
            "https://www.hs-hannover.de/fileadmin/HsH/Forms/Hochschule_Hannover/UEber_uns/Organe_und_Gremien/Verkuendungsblatt/07-2024/01_Praxisphasenordnung_F1_2024_v06.pdf",
            "https://f1.hs-hannover.de/studium/studiengaenge-und-programme/bachelor-studiengaenge/elektrotechnik-und-informationstechnik-eit-2-1",
        ],
        "patch": {
            "reference_answer": "An der Fakultät I in der Regel 10 Wochen Vollzeittätigkeit (mindestens "
                                "35 h/Woche) im 7. Regelstudiensemester, ohne Urlaub; aufteilbar in zwei "
                                "Teile (nicht vor dem 4. Semester), der erste Teil höchstens 6 Wochen; in "
                                "Teilzeit bei mindestens 350 h und höchstens doppeltem Zeitraum.",
            "required_facts": [
                "Die Praxisphase umfasst in der Regel ohne Inanspruchnahme von Urlaub einen "
                "Zeitraum von 10 Wochen Vollzeittätigkeit bei einer wöchentlichen Arbeitszeit "
                "von mindestens 35 h und wird als eine zusammenhängende Praxisphase im "
                "7. Regelstudiensemester abgeleistet.",
            ],
            "evaluation_notes": "Goldquelle am 14.09.2026 von der Fassung 2012 auf die 2. Änderung "
                                "(Verkündungsblatt 07/2024 vom 01.09.2024) umgestellt; die Studiengangsseite "
                                "EIT verlinkt genau diese. Die alte Fassung nannte „18 Credits“, die neue "
                                "nicht mehr. Die drei Kopien der 2024er Ordnung sind inhaltsgleich.",
        },
        "grund": "Praxisphasenordnung 2012 durch Fassung 2024 abgelöst.",
    },
    "n04": {
        "accepted": [
            "https://www.hs-hannover.de/fileadmin/HsH/Forms/Hochschule_Hannover/UEber_uns/Organe_und_Gremien/Verkuendungsblatt/2019/06-2019/HsH_Verkuendungsblatt_06_2019.pdf",
        ],
        "grund": "Gesamtausgabe des Verkündungsblatts 06/2019 enthält die Richtlinie wörtlich.",
    },
    "n05": {
        "accepted": [
            "https://www.hs-hannover.de/fileadmin/HsH/Forms/Hochschule_Hannover/UEber_uns/Organe_und_Gremien/Verkuendungsblatt/2015/13-2015/HsH_Verkuendungsblatt_13_2015.pdf",
            "https://www.hs-hannover.de/fileadmin/HsH/Forms/Hochschule_Hannover/UEber_uns/Organe_und_Gremien/Verkuendungsblatt/08-2024/Drittmittelrichtlinie.pdf",
        ],
        "grund": "Gesamtausgabe des Verkündungsblatts 13/2015 enthält die IP-Strategie wörtlich. "
                 "30.09.2026: Die Drittmittelrichtlinie 2024 regelt die Meldung neu (an D1 – "
                 "Personal und Recht) und ist als Quelle aufgenommen.",
        "patch": {
            "reference_answer": "Mit dem Formblatt „Erfindungsmeldung“, schriftlich und von allen "
                                "Erfindern unterschrieben, so früh wie möglich. Zuständig ist nach der "
                                "Drittmittelrichtlinie 2024 D1 – Personal und Recht (Team Recht); die "
                                "IP-Strategie 2015 nennt noch das Justiziariat.",
        },
    },
    # ── eval_gold_50, Nachprüfung 21.09.2026 nach dem Phase-3-Lauf: die Antwort
    #    stand in Dokumenten auf Rang 1–3, die der Vertrag nicht kannte.
    "g29": {
        "accepted": [
            "https://mint.hs-hannover.de/studieren/mint-studiengaenge/mediendesigninformatik",
        ],
        "grund": "MINT-Seite Mediendesigninformatik: „2/3 Informatik + 1/3 Mediendesign … viel "
                 "Programmieren … Concept Design“ — beantwortet „Programmieren und Gestalten“ "
                 "ebenso wie das Qualifikationsziele-PDF von 2016.",
    },
    "g39": {
        "accepted": [
            "https://f3.hs-hannover.de/fileadmin/HsH/Forms/Fakultaet_III/Ordnungen_DM/2022_PO_MDM.pdf",
            "https://www.hs-hannover.de/fileadmin/HsH/Forms/Hochschule_Hannover/UEber_uns/Organe_und_Gremien/Verkuendungsblatt/06-2022/13_MDM.pdf",
        ],
        "grund": "Die Prüfungsordnung MDM 2022 (und ihre Verkündungsblatt-Kopie) enthält dieselbe "
                 "Tabelle „MAA (Master-Arbeit) 30 bis 90 Seiten“ wie das Modulhandbuch.",
        # 30.09.2026: „30“ allein traf jede Zahl 30 in der Antwort.
        "begriffe": ["30 bis 90 Seiten"],
    },
    "g41": {
        "accepted": [
            "https://f5.hs-hannover.de/studium/bachelor-studiengaenge/heilpaedagogik-inklusive-bildung-und-begleitung-berufsintegrierend-hbi",
        ],
        "grund": "Studiengangsseite HBI, Abschnitt Zulassungsvoraussetzungen: „min. 2-jährigen "
                 "Berufstätigkeit in einem heilpädagogischen Tätigkeitsfeld“ — wörtlich die ZZO.",
    },
}
# Nicht ergänzt, mit Grund:
#   q06, q08, q11, q13, q17  — nur die Goldseite trägt den Fakt (FAQ verlinkt bei q08 lediglich).
#   q10, q20                 — Gold live vorhanden, aber nicht gecrawlt; keine zweite HsH-Quelle.
#                              Öffnungszeiten der Mensa nennt nur das Studentenwerk (extern).
#   n06                      — S1–S3 mit Kürzeln stehen nur im Organigramm; die Einzelseiten
#                              (S1, KOM, FuE) nennen sich „Stabsabteilung“, aber ohne Kürzel.


def ergaenze_quellen(fall: dict) -> dict:
    """Trägt die nachgeprüften Quellen ein und zieht Politik und Vertrag nach."""
    eintrag = WEITERE_QUELLEN.get(fall["id"])
    if not eintrag:
        return fall
    if eintrag.get("gold") and eintrag["gold"] != fall["source_url"]:
        fall["source_url_original"] = fall["source_url"]
        fall["source_url"] = eintrag["gold"]
    vorhanden = {normalize_url(u) for u in [fall["source_url"], *fall.get("accepted_source_urls", [])]}
    neu = [u for u in eintrag.get("accepted", []) if normalize_url(u) not in vorhanden]
    fall["accepted_source_urls"] = [*fall.get("accepted_source_urls", []), *neu]
    if fall["accepted_source_urls"] and fall.get("source_policy") != "accepted_sources":
        fall["source_policy_original"] = fall.get("source_policy")
        fall["source_policy"] = "accepted_sources"
    for schluessel, wert in (eintrag.get("patch") or {}).items():
        fall[schluessel] = wert
    if "begriffe" in eintrag:
        fall["pflicht_begriffe"] = eintrag["begriffe"]
    fall["quellen_geprueft"] = QUELLEN_GEPRUEFT
    fall["quellen_grund"] = eintrag["grund"]
    return fall


def praezisiere(fall: dict) -> dict:
    """Ersetzt die Frage und zieht den Vertrag nach, wo hinterlegt."""
    eintrag = PRAEZISIERT.get(fall["id"])
    if not eintrag:
        return fall
    fall["question_original"] = fall["question"]
    fall["question"] = eintrag["question"]
    fall["anpassung"] = eintrag["grund"]
    for schluessel, wert in (eintrag.get("patch") or {}).items():
        if wert is None:
            fall.pop(schluessel, None)
        else:
            fall[schluessel] = wert
    return fall


def lade_korpus_urls() -> set[str]:
    """Normalisierte source_urls aller Dokumente mit Chunks im Index."""
    if not DOC_INDEX.exists():
        raise SystemExit(f"doc_index.db fehlt: {DOC_INDEX} — erst build_doc_index.py laufen lassen.")
    conn = sqlite3.connect(f"file:{DOC_INDEX}?mode=ro", uri=True)
    try:
        return {
            normalize_url(r[0])
            for r in conn.execute("SELECT source_url FROM documents WHERE chunk_count > 0")
        }
    finally:
        conn.close()


def lade_vertretung() -> dict[str, list[str]]:
    """source_url → Hauptvertreter-URLs (normalisiert).

    Die Deduplizierung beim Import behält je Gruppe wortgleicher Chunks einen
    Vertreter; 310 Dokumente haben deshalb keinen einzigen eigenen Punkt im
    Index und tauchen im Retrieval NUR unter der URL ihres Vertreters auf
    (Details: `_load_document_text` in server/mcp_server.py). Für die Fallbasis
    heißt das: Wer eine solche Kopie als Quelle einträgt, muss den Vertreter
    mit eintragen, sonst kann der Fall nicht getroffen werden.

    Als Vertreter zählt nur, wer mindestens die Hälfte der Positionen eines
    Dokuments stellt. Einzelne Boilerplate-Absätze (Inkrafttreten,
    Rechtsbehelf) zeigen sonst auf beliebige fremde Ordnungen — das wären
    keine Quellen für die Antwort, sondern Zufallstreffer.
    """
    conn = sqlite3.connect(f"file:{DOC_INDEX}?mode=ro", uri=True)
    try:
        zeilen = conn.execute(
            "SELECT source_url, vertreter_url, COUNT(*) FROM chunk_vertretung "
            "WHERE selbst = 0 AND vertreter_url != '' AND vertreter_url != source_url "
            "GROUP BY source_url, vertreter_url").fetchall()
        gesamt = dict(conn.execute(
            "SELECT source_url, COUNT(*) FROM chunk_vertretung GROUP BY source_url").fetchall())
    finally:
        conn.close()
    ergebnis: dict[str, list[str]] = {}
    for url, vurl, n in zeilen:
        if n * 2 >= gesamt.get(url, 0):
            ergebnis.setdefault(normalize_url(url), []).append(normalize_url(vurl))
    return ergebnis


def ergaenze_vertreter(fall: dict, vertretung: dict[str, list[str]]) -> dict:
    """Trägt Hauptvertreter von Gold- und akzeptierten Quellen als akzeptiert ein."""
    urls = [fall["source_url"], *fall.get("accepted_source_urls", [])]
    vorhanden = {normalize_url(u) for u in urls if u}
    neu: dict[str, str] = {}
    for u in urls:
        for v in vertretung.get(normalize_url(u), []):
            if v not in vorhanden and v not in neu:
                neu[v] = u
    if neu:
        fall["accepted_source_urls"] = [*fall.get("accepted_source_urls", []), *neu]
        fall["vertreter_von"] = neu            # Vertreter → vertretenes Original
        if fall.get("source_policy") != "accepted_sources":
            fall["source_policy_original"] = fall.get("source_policy")
            fall["source_policy"] = "accepted_sources"
    return fall


def lade_positionen() -> dict[str, list[tuple[int, str, str]]]:
    """source_url → [(chunk_index, punkt_id, vertreter_url)] aus chunk_vertretung."""
    conn = sqlite3.connect(f"file:{DOC_INDEX}?mode=ro", uri=True)
    try:
        zeilen = conn.execute(
            "SELECT source_url, chunk_index, vertreten_durch, vertreter_url "
            "FROM chunk_vertretung ORDER BY source_url, chunk_index").fetchall()
    finally:
        conn.close()
    ergebnis: dict[str, list] = {}
    for url, idx, pid, vurl in zeilen:
        ergebnis.setdefault(normalize_url(url), []).append((idx, pid, normalize_url(vurl or url)))
    return ergebnis


def ergaenze_belegpunkte(fall: dict, positionen: dict, texte: dict[str, str]) -> dict:
    """Punktgenaue Ziele: Unter welchen Punkt-IDs liegt der Text der Goldquelle,
    und welche davon tragen die Belegstelle?

    Hintergrund (g46, 21.09.2026): Die Richtlinie zu Lehraufträgen 2024 hat 17
    von 18 Chunks an vier andere Dokumente abgegeben — Verkündungsblatt,
    Revision 2025, zwei Infopakete. Kein Vertreter erreichte die Hälfte, also
    bekam der Fall keine Zweitquelle; die Suche fand den Text auf Rang 1 und
    3, unter fremden URLs, und die Auswertung zählte „nicht gefunden".

    gold_punkte   alle Punkt-IDs, die eine Position der Goldquelle vertreten
    beleg_punkte  davon die, deren Text einen required_fact enthält
    beleg_urls    die URLs, unter denen diese Belegpunkte im Index stehen

    Die Auswertung zählt einen Chunk-Treffer über die ID und eine Kandidaten-
    URL, wenn sie einen Belegpunkt trägt — nicht, wenn sie nur eine Kopfzeile
    vertritt. Fehlt eine Belegstelle im Vertrag oder wird sie in keinem
    Chunk gefunden, bleibt beleg_* leer und es gilt die URL-Regel wie bisher.
    """
    ziele = [fall["source_url"], *fall.get("accepted_source_urls", [])]
    gold_punkte: list[str] = []
    beleg_punkte: list[str] = []
    beleg_urls: list[str] = []
    fakten = [t for t in (fall.get("required_facts") or []) if t]
    for url in ziele:
        for _idx, pid, vurl in positionen.get(normalize_url(url), []):
            if pid not in gold_punkte:
                gold_punkte.append(pid)
            text = texte.get(pid, "")
            if fakten and text and any(enthaelt_begriff(text, fakt) for fakt in fakten):
                if pid not in beleg_punkte:
                    beleg_punkte.append(pid)
                if vurl not in beleg_urls:
                    beleg_urls.append(vurl)
    fall["gold_punkte"] = gold_punkte
    fall["beleg_punkte"] = beleg_punkte
    fall["beleg_urls"] = beleg_urls
    return fall


def lade_punkttexte(punkt_ids: list[str]) -> dict[str, str]:
    """Texte der Punkte aus Qdrant — nur die, die eine Goldposition vertreten."""
    from qdrant_client import QdrantClient
    client = QdrantClient(url="http://localhost:6333", timeout=30)
    texte: dict[str, str] = {}
    for i in range(0, len(punkt_ids), 500):
        for p_ in client.retrieve("hsh_knowledge", ids=punkt_ids[i:i + 500],
                                  with_payload=["text"], with_vectors=False):
            texte[str(p_.id)] = (p_.payload or {}).get("text", "")
    return texte


def gold_vorhanden(fall: dict, korpus: set[str]) -> bool:
    """Liegt die Gold- oder eine akzeptierte Quelle überhaupt in unserem Index?

    Ohne diese Prüfung erscheint ein Fall als Retrieval-Fehler, obwohl das
    Dokument gar nicht gecrawlt wurde. Bei q09, q10 und q20 ist genau das der
    Fall — dieselben drei Fälle stehen in ihrem Bericht unter „Retrieval
    debuggen", was die falsche Ebene ist.
    """
    kandidaten = [fall["source_url"], *fall.get("accepted_source_urls", [])]
    return any(normalize_url(u) in korpus for u in kandidaten if u)


def aus_chargy_beidawi(pfad: Path, korpus: set[str], vertretung: dict) -> list[dict]:
    faelle = json.loads(pfad.read_text(encoding="utf-8"))
    ergebnis = []
    for c in faelle:
        if c["id"] in AUSGESCHLOSSEN:
            continue
        fall = dict(c)                       # ihre Felder unverändert
        fall["herkunft"] = "chargy_beidawi"
        fall["fakultaet"] = ""
        fall["pflicht_begriffe"] = BEGRIFFE.get(c["id"], [])
        fall = ergaenze_vertreter(ergaenze_quellen(praezisiere(fall)), vertretung)
        # Erst nach der Quellenergänzung prüfen: q09 ist nur mit ihnen im Korpus.
        fall["gold_im_korpus"] = gold_vorhanden(fall, korpus)
        ergebnis.append(fall)
    return ergebnis


def aus_nutzerfragen(korpus: set[str], vertretung: dict) -> list[dict]:
    if not NUTZER_CSV.exists():
        raise SystemExit(f"Fehlt: {NUTZER_CSV}")
    ergebnis = []
    with NUTZER_CSV.open(encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            fid = f"n{i:02d}"
            politik, fairness = NUTZER_POLITIK.get(fid, ("exact_gold_source", "niedrig"))
            fall = {
                "id": fid,
                "case_type": "direct",
                "question": row["frage"],
                "reference_answer": row.get("antwort", ""),
                "source_url": row.get("source_url", ""),
                "expected_behavior": "direct_answer",
                "source_policy": politik,
                "accepted_source_urls": [],
                # Die Belegstelle ist der wörtliche Fundort im Dokument und
                # damit die beste vorhandene Entsprechung eines Pflichtfakts.
                "required_facts": [row.get("belegstelle", "")] if row.get("belegstelle") else [],
                "optional_facts": [],
                "forbidden_claims": [],
                "case_difficulty": "mittel",
                "fairness_risk": fairness,
                "evaluation_notes": f"Aus unserem Goldstandard, Typ {row.get('typ','')}/"
                                    f"{row.get('frageform','')}.",
                "herkunft": "nutzer_gold",
                "fakultaet": "",
                "pflicht_begriffe": BEGRIFFE.get(fid, []),
            }
            fall = ergaenze_vertreter(ergaenze_quellen(praezisiere(fall)), vertretung)
            fall["gold_im_korpus"] = gold_vorhanden(fall, korpus)
            ergebnis.append(fall)
    return ergebnis


INGESTED = konfig.KORPUS


def _korpus_texte() -> dict[str, str]:
    """source_url → normalisierter Volltext aller gecrawlten Markdown-Dateien."""
    import re
    texte: dict[str, str] = {}
    if not INGESTED.exists():
        return texte
    for pfad in INGESTED.glob("*.md"):
        text = pfad.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'^source_url:\s*"?([^"\n]+)"?', text, re.M)
        if m:
            texte[normalize_url(m.group(1).strip())] = " ".join(text.split()).casefold()
    return texte


def selbsttest(faelle: list[dict], korpus: set[str], vertretung: dict[str, list[str]]) -> list[str]:
    """Prüft den Vertrag gegen sich selbst und gegen den Korpus.

    Drei Dinge müssen stimmen, sonst misst der Lauf später den Vertrag statt
    das RAG: (1) Die Referenzantwort enthält ihre eigenen Pflichtbegriffe
    (derselbe tolerante Vergleich wie in der Auswertung),
    (2) jede Gold- und akzeptierte Quelle liegt im Korpus, (3) der Korpustext
    dieser Quelle enthält die Pflichtbegriffe. (2) und (3) sind Warnungen —
    eine Quelle darf bewusst „nur live“ eingetragen sein —, (1) ist ein Fehler.
    """
    texte = _korpus_texte()
    meldungen: list[str] = []
    for f in faelle:
        ref = f.get("reference_answer", "")
        fehlt = [anzeige(b) for b in f["pflicht_begriffe"] if not enthaelt_begriff(ref, b)]
        if fehlt:
            meldungen.append(f"FEHLER {f['id']}: Referenzantwort enthält nicht {fehlt}")
        for url in [f["source_url"], *f.get("accepted_source_urls", [])]:
            n = normalize_url(url)
            if n not in korpus:
                if any(v in korpus for v in vertretung.get(n, [])):
                    continue                # ohne eigene Chunks, aber vertreten
                meldungen.append(f"warn   {f['id']}: nicht im Korpus — {url}")
                continue
            text = texte.get(n)
            if text is None:
                continue
            fehlt = [anzeige(b) for b in f["pflicht_begriffe"] if not enthaelt_begriff(text, b)]
            if fehlt:
                meldungen.append(f"warn   {f['id']}: Quelle trägt nicht {fehlt} — {url}")
    return meldungen


BEGRIFFE_G = _BASE / "begriffe_g.json"


def aus_fragen50(korpus: set[str], vertretung: dict) -> list[dict]:
    """Die 50 Fragen aus eval/fragen.csv — Gold aus der CSV, Pflichtbegriffe
    aus `begriffe_g.json`.

    Die Begriffe sind kuratiert (22.09.2026, Vorschläge aus
    begriffe_vorschlag.py, von Hand durchgesehen): konservativ, Listen als
    Alternativen. Fehlt die Datei oder ein Eintrag, bleibt der Fall ohne
    Begriffe und damit nur über die Quellenlage bewertbar — wie bis zum 22.09.

    Schwierigkeit aus der Frageform: Ort und Fakt sind Nachschlagefragen
    (leicht), Vorgehen und Auswahl verlangen Zusammenhang (mittel). Das ist
    eine Heuristik für die Gruppenansicht, kein Urteil.
    """
    if not FRAGEN50_CSV.exists():
        raise SystemExit(f"Fehlt: {FRAGEN50_CSV}")
    begriffe_g = {}
    if BEGRIFFE_G.exists():
        begriffe_g = {k: v for k, v in json.loads(BEGRIFFE_G.read_text(encoding="utf-8")).items()
                      if not k.startswith("_")}
    ergebnis = []
    with FRAGEN50_CSV.open(encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            fid = f"g{i:02d}"
            fall = {
                "id": fid,
                "case_type": "direct",
                "question": row["frage"],
                "reference_answer": row.get("antwort", ""),
                "source_url": row.get("source_url", ""),
                "expected_behavior": "direct_answer",
                "source_policy": "exact_gold_source",
                "accepted_source_urls": [],
                "required_facts": [row.get("belegstelle", "")] if row.get("belegstelle") else [],
                "optional_facts": [],
                "forbidden_claims": [],
                "case_difficulty": "leicht" if row.get("frageform") in ("ort", "fakt") else "mittel",
                "fairness_risk": "niedrig",
                "evaluation_notes": f"Aus eval/fragen.csv, Typ {row.get('typ','')}/"
                                    f"{row.get('frageform','')}. Aufgenommen 21.09.2026; "
                                    f"Pflichtbegriffe seit 22.09.2026 aus begriffe_g.json."
                                    + (" Belegstelle: " + row["belegstelle"][:160] if row.get("belegstelle") else ""),
                "herkunft": "eval_gold_50",
                "fakultaet": "",
                "pflicht_begriffe": begriffe_g.get(fid, []),
            }
            if fid in AUSGESCHLOSSEN:        # Nummer bleibt belegt, damit g-IDs stabil sind
                continue
            fall = ergaenze_vertreter(ergaenze_quellen(praezisiere(fall)), vertretung)
            fall["gold_im_korpus"] = gold_vorhanden(fall, korpus)
            ergebnis.append(fall)
    return ergebnis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quelle", type=Path, required=True,
                    help="validation_cases.json von Chargy/Beidawi")
    args = ap.parse_args()

    korpus = lade_korpus_urls()
    vertretung = lade_vertretung()
    faelle = (aus_chargy_beidawi(args.quelle, korpus, vertretung)
              + aus_nutzerfragen(korpus, vertretung)
              + aus_fragen50(korpus, vertretung))

    # Punktgenaue Ziele (siehe ergaenze_belegpunkte): erst alle Punkt-IDs der
    # Goldpositionen sammeln, deren Texte in einem Rutsch aus Qdrant holen,
    # dann je Fall die Belegpunkte bestimmen.
    positionen = lade_positionen()
    alle_ids = sorted({pid for f in faelle
                       for url in [f["source_url"], *f.get("accepted_source_urls", [])]
                       for _i, pid, _v in positionen.get(normalize_url(url), [])})
    try:
        texte = lade_punkttexte(alle_ids)
    except Exception as exc:
        print(f"WARNUNG: Qdrant nicht erreichbar ({exc}) — beleg_punkte bleiben leer.")
        texte = {}
    for f in faelle:
        ergaenze_belegpunkte(f, positionen, texte)
        if f["id"] in GUELTIG_BIS:
            f["gueltig_bis"] = GUELTIG_BIS[f["id"]]

    ZIEL.write_text(json.dumps(faelle, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    fingerabdruck = hash_fallbasis(faelle)
    HASHDATEI.write_text(fingerabdruck + "\n", encoding="utf-8")

    ohne_gold = [f["id"] for f in faelle if not f["gold_im_korpus"]]
    ohne_begriffe = [f["id"] for f in faelle if not f["pflicht_begriffe"]]
    print(f"{ZIEL.name}: {len(faelle)} Fälle geschrieben"
          f" (ausgeschlossen: {', '.join(AUSGESCHLOSSEN) or '—'})  ·  Vertrag {kurz(fingerabdruck)}")
    print(f"  Chargy/Beidawi : {sum(1 for f in faelle if f['herkunft']=='chargy_beidawi')}")
    print(f"  Nutzer-Gold    : {sum(1 for f in faelle if f['herkunft']=='nutzer_gold')}")
    print(f"  eval_gold_50   : {sum(1 for f in faelle if f['herkunft']=='eval_gold_50')}")
    print(f"  Gold NICHT im Korpus ({len(ohne_gold)}): {', '.join(ohne_gold) or '—'}")
    print(f"  ohne pflicht_begriffe ({len(ohne_begriffe)}): {', '.join(ohne_begriffe) or '—'}")
    geaendert = [f["id"] for f in faelle if f.get("question_original")]
    print(f"  präzisierte Fragen ({len(geaendert)}): {', '.join(geaendert) or '—'}")
    ergaenzt = [f["id"] for f in faelle if f.get("quellen_geprueft")]
    print(f"  Quellen ergänzt ({len(ergaenzt)}, Stand {QUELLEN_GEPRUEFT}): {', '.join(ergaenzt) or '—'}")
    mit_vertreter = [f["id"] for f in faelle if f.get("vertreter_von")]
    print(f"  Vertreter-URLs ergänzt ({len(mit_vertreter)}): {', '.join(mit_vertreter) or '—'}")
    ohne_beleg = [f["id"] for f in faelle if f["gold_im_korpus"] and not f.get("beleg_punkte")]
    fremd = [f["id"] for f in faelle if f.get("beleg_urls")
             and any(normalize_url(u) not in {normalize_url(x) for x in
                     [f["source_url"], *f.get("accepted_source_urls", [])]} for u in f["beleg_urls"])]
    print(f"  Belegstelle in keinem Chunk gefunden ({len(ohne_beleg)}): {', '.join(ohne_beleg) or '—'}")
    print(f"  Belegstelle liegt (auch) unter fremder URL ({len(fremd)}): {', '.join(fremd) or '—'}")

    from datetime import date
    abgelaufen = [f["id"] for f in faelle if f.get("gueltig_bis", "9999") < date.today().isoformat()]
    print(f"  mit Verfallsdatum ({len(GUELTIG_BIS)}): {', '.join(sorted(GUELTIG_BIS))}"
          + (f"  ·  ABGELAUFEN: {', '.join(abgelaufen)}" if abgelaufen else ""))

    meldungen = selbsttest(faelle, korpus, vertretung)
    if meldungen:
        print("\nSelbsttest:")
        for m in meldungen:
            print("  " + m)
    if any(m.startswith("FEHLER") for m in meldungen):
        raise SystemExit("Vertrag widerspricht sich — siehe FEHLER oben.")


if __name__ == "__main__":
    main()
