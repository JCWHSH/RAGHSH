"""Modultabelle stammdaten/po/module.tsv: Modulkürzel → Modulname aus den Modulhandbüchern
(S073 Entwurf, S077 übernommen nach F5; kein LLM).

Quelle: alle Dokumente mit po_dokumenttyp = Modulhandbuch in
stammdaten/po/po_zuordnung.tsv, Text aus var/korpus/ (Zuordnung über
die source_url im Kopf der Markdown-Datei, nie über den Dateinamen).

Ein Modulkürzel ist <Präfix>-<Nummer>[-<Teilnummer>], Präfix ein Kürzel aus
stammdaten/po/studiengaenge.tsv (oder BSKE/BUE-artige Präfixe, die im Dokument
mindestens MIN_PRAEFIX-mal mit verschiedenen Nummern stehen). Ein Name wird nur
aus festen Stellen gelesen:
  zelle      |KÜRZEL|Name|          (Übersichtstabellen, Inhaltsverzeichnisse)
  inline     KÜRZEL[:/ ]Name ....   am Zeilen-/Zellen-/Überschriftsanfang oder
             nach „Modul“, „Teilmodul“, „Lehrveranstaltung“, „<Seitenzahl>“
  kuerzel    „Kürzel: KÜRZEL“ → Name aus der Modulüberschrift „Modul N – Name“
             in den MAX_RUECK Zeilen davor (Modulhandbücher ohne Kürzel im Titel)
Erwähnungen im Fließtext („Teilnahme an Modul BRS-212 ist …“) zählen nicht.
Je (Dokument, Kürzel) gewinnt der häufigste Name; Fundstelle = Zeile in der
Markdown-Datei (1-basiert) des ersten Belegs dieses Namens.

Zweizeilige Titel im Feld „Modulbezeichnung / Titel“ (`A<br>B`) werden zu
„A B“ verbunden (S077). Steht in einer Zelle oder Zeile nach dem Namen ein
`<br>` mit kleingeschriebener Fortsetzung, gilt der Name als „endet offen“.

Ausgabe: stammdaten/po/module.tsv — nur Zeilen ohne Markierung (F5: „Wortende nicht
belegt“, „endet offen“, „Zeichensalat“ bleiben draußen; die Zahl je Grund steht
in der Ausgabe des Skripts). Gleiche Eingabe → gleiche Ausgabe.
Aufruf (aus dem Wurzelverzeichnis): python werkzeuge/po/module_ziehen.py
"""
from __future__ import annotations

import collections
import csv
import glob
import re
from pathlib import Path

from raghsh import konfig

PO_DATEN = konfig.PO_DATEN
AUS = PO_DATEN / "module.tsv"
MIN_PRAEFIX = 5
MAX_RUECK = 6
NAME_MIN, NAME_MAX = 3, 120

KUERZEL = r"(?P<k>(?P<p>[A-Z][A-Za-z]{1,5})-(?P<n>\d{2,4})(?:-(?P<t>\d{1,2}))?)(?![\d-])"
# Davor erlaubt: Zeilenanfang, Zelle, Umbruch, Überschrift, Modul-Wörter, TOC-Seitenzahl, „:“
VOR = r"(?:^|\||<br>|#+\s|\bModul\s|\bTeilmodul\s|\bLehrveranstaltung\s|\bModulbezeichnung\s|:\s|\s\d{1,3}\s)"
ZELLE = re.compile(r"\|\s*(?:\*\*)?\s*" + KUERZEL + r"\s*(?:\*\*)?\s*\|\s*(?:\*\*)?(?P<name>[^|]*)")
INLINE = re.compile(VOR + r"\s*(?:\*\*)?\s*" + KUERZEL + r"\s*(?::|/|–|_)?\s+(?P<name>[A-ZÄÖÜ][^|]*)")
KUERZEL_FELD = re.compile(r"Kürzel\s*:?\s*(?:\*\*)?\s*(?:\|\s*)*(?:\*\*)?\s*" + KUERZEL)
# Modulüberschrift „Modul N – Name“ oder Feld „(Teil)Modulbezeichnung / Titel | Name“ vor dem Kürzel-Feld
MODUL_TITEL = re.compile(r"Modul\s+\d{1,2}\s*[–-]\s*(?:Modulbeschreibung:\s*)?(?P<name>[A-ZÄÖÜ][^|*]{2,})"
                         r"|(?:Teilmodul|Modul)(?:bezeichnung|titel|name)(?:\s*/\s*Titel)?\s*(?:\*\*)?\s*(?:\|\s*)+(?:\*\*)?(?P<name2>[A-ZÄÖÜ][^|]{2,})", re.I)

# Namensende: Zelle, Umbruch, Fettdruck-Ende, Punktreihe, Seitenangabe, Pflicht-Angabe
ENDE = re.compile(r"\||<br>|\*\*|\s?\.{3,}|\s?(?:\. ){3,}|\sS\.\s?\d|,\s*(?:Pflicht|Wahlpflicht|Wahl)\b|\s{3,}|\s\d{1,3}\s*$")
# Was nach dem Kürzel steht, aber kein Name ist
KEIN_NAME = re.compile(r"^(?:V/Ü|V\b|VL\b|Ü\b|Pflicht|Wahlpflicht|PF|WP|Seite|Page|ist|sind|und|oder|bzw|im|in|der|die|das|K\d|M\b|H\b|Vorlesung|Seminar|Übung|Labor|Projekt\s*$)", re.I)


def _name_saeubern(roh: str, umbruch_anhaengen: bool = False) -> str:
    """Name bis zum ersten Endezeichen. Im Titelfeld (umbruch_anhaengen) ist
    `<br>` nur ein Zeilenumbruch im selben Titel: die Fortsetzung wird angehängt."""
    if umbruch_anhaengen:
        roh = re.sub(r"\*\*", "", roh).replace("<br>", " ")
    m = ENDE.search(roh)
    name = roh[: m.start()] if m else roh
    name = re.sub(r"\*\*|<br>|\\", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" :–-/.,;")
    # „A<br>b…“: Fortsetzung in Kleinschrift → Name abgeschnitten
    if m and m.group(0) == "<br>" and re.match(r"(?:\*\*)?\s*[a-zäöü]", roh[m.end():]):
        name += " …"
    return name


def _gueltig(name: str) -> bool:
    if not (NAME_MIN <= len(name) <= NAME_MAX):
        return False
    if not re.match(r"[A-ZÄÖÜ0-9]", name):
        return False
    if re.fullmatch(r"[\d\s.,/()-]+", name):
        return False
    # „Projekt“ allein ist ein gültiger Modulname, als Lehrform aber nur mit Zusatz;
    # deshalb nur Lehrformen ohne Zusatz ausschließen
    if KEIN_NAME.match(name) and not re.match(r"Projekt\b", name):
        return False
    return True


STOPP_ENDE = {"und", "der", "die", "das", "des", "für", "in", "im", "mit", "von", "zur", "zum", "auf", "&"}


def _unsicher(name: str, woerter: collections.Counter) -> str:
    """Grund, warum der Name abgeschnitten sein könnte, sonst ''."""
    # PDF-Zeichensalat („Unternehmens lanun p g“): mehrere einzelne Kleinbuchstaben
    if len(re.findall(r"(?:^|\s)[a-zäöü](?=\s|$)", name)) >= 2:
        return "Zeichensalat"
    letztes = name.split()[-1]
    if name.endswith(("-", "…")) or letztes.lower() in STOPP_ENDE:
        return "endet offen"
    w = re.findall(r"\w+", letztes)
    # Silbentrennung im PDF: letztes Wort kommt sonst nirgends als ganzes Wort vor
    if w and len(w[-1]) >= 4 and woerter[w[-1]] < 2:
        return "Wortende nicht belegt"
    return ""


def ingested_index() -> dict[str, str]:
    idx = {}
    for f in glob.glob(str(konfig.KORPUS / "*.md")):
        with open(f, errors="ignore") as fh:
            kopf = fh.read(1500)
        m = re.search(r'^source_url:\s*"?([^"\n]+)"?', kopf, re.M)
        if m:
            idx[m.group(1).strip()] = f
    return idx


def praefixe_bekannt() -> set[str]:
    with open(PO_DATEN / "studiengaenge.tsv", newline="") as fh:
        return {r["kuerzel"] for r in csv.DictReader(fh, delimiter="\t")}


def kandidaten(zeilen: list[str], erlaubt: set[str]):
    """liefert (kürzel, name, zeile, regel)"""
    for i, z in enumerate(zeilen):
        for m in ZELLE.finditer(z):
            if m.group("p") in erlaubt:
                yield m.group("k"), _name_saeubern(m.group("name")), i + 1, "zelle"
        for m in INLINE.finditer(z):
            if m.group("p") in erlaubt:
                yield m.group("k"), _name_saeubern(m.group("name")), i + 1, "inline"
        for m in KUERZEL_FELD.finditer(z):
            if m.group("p") not in erlaubt:
                continue
            for j in range(i, max(-1, i - MAX_RUECK), -1):
                t = list(MODUL_TITEL.finditer(zeilen[j]))
                if t:
                    name = t[-1].group("name") or t[-1].group("name2")
                    yield m.group("k"), _name_saeubern(name, umbruch_anhaengen=True), i + 1, "kuerzel"
                    break


def main() -> None:
    idx = ingested_index()
    bekannt = praefixe_bekannt()
    with open(PO_DATEN / "po_zuordnung.tsv", newline="") as fh:
        docs = [r for r in csv.DictReader(fh, delimiter="\t") if r["po_dokumenttyp"] == "Modulhandbuch"]

    zeilen_aus, abdeckung = [], []
    for d in docs:
        url = d["source_url"]
        datei = idx.get(url)
        if not datei:
            abdeckung.append((d["po_studiengang"], d["po_fassung"], url, 0, "keine Datei in var/korpus"))
            continue
        # PDF-Auszüge schreiben Bindestriche teils als U+2010/U+2011
        zeilen = open(datei, errors="ignore").read().replace("\u2010", "-").replace("\u2011", "-").split("\n")
        text = "\n".join(zeilen)
        # Präfixe: bekannte Studiengänge + häufige eigene (z. B. BSKE, BUE)
        nummern = collections.defaultdict(set)
        for m in re.finditer(KUERZEL, text):
            nummern[m.group("p")].add(m.group("n"))
        erlaubt = {p for p, n in nummern.items() if p in bekannt or len(n) >= MIN_PRAEFIX}

        stimmen = collections.defaultdict(collections.Counter)
        erste = {}
        for k, name, zeile, regel in kandidaten(zeilen, erlaubt):
            if not _gueltig(name):
                continue
            stimmen[k][name] += 1
            erste.setdefault((k, name), (zeile, regel))
        sg = d["po_studiengang"]
        if not sg and nummern:
            # Dokument ohne Studiengang in po_zuordnung: häufigster bekannter Präfix, markiert
            haeufig = collections.Counter(m.group("p") for m in re.finditer(KUERZEL, text) if m.group("p") in bekannt)
            sg = (haeufig.most_common(1)[0][0] + "?") if haeufig else ""
        woerter = collections.Counter(re.findall(r"\w+", text))
        for k in sorted(stimmen):
            # abgeschnittene Fassungen (Zeilenumbruch in der Zelle) dem längeren Namen zuschlagen
            for kurz in list(stimmen[k]):
                laenger = [l for l in stimmen[k] if l != kurz and l.startswith(kurz.removesuffix(" …"))]
                if laenger:
                    stimmen[k][max(laenger, key=len)] += stimmen[k].pop(kurz)
            name, n = stimmen[k].most_common(1)[0]
            zeile, regel = erste[(k, name)]
            andere = len(stimmen[k]) - 1
            zeilen_aus.append((sg, d["po_fassung"], k, name, url, f"md-Zeile {zeile}", regel, n, andere,
                               _unsicher(name, woerter)))
        if stimmen:
            abdeckung.append((sg, d["po_fassung"], url, len(stimmen), ""))
        else:
            grund = "kein Modulkürzel im Text" if not nummern else "Kürzel nur im Fließtext / ohne lesbaren Namen"
            abdeckung.append((sg, d["po_fassung"], url, 0, grund))

    unsicher = collections.Counter(z[-1] for z in zeilen_aus if z[-1])
    sicher = [z[:6] for z in zeilen_aus if not z[-1]]
    with open(AUS, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["studiengang", "fassung", "modulkuerzel", "modulname", "source_url", "fundstelle"])
        w.writerows(sicher)
    print(f"{len(docs)} Dokumente, {sum(1 for a in abdeckung if a[3])} mit Treffern, {len(zeilen_aus)} Zeilen, "
          f"davon {len(sicher)} nach {AUS.name}; markiert: {dict(sorted(unsicher.items()))}")
    for a in abdeckung:
        if not a[3]:
            print(f"ohne Treffer: {a[0] or '-'} {a[1] or '-'} {a[2]} ({a[4]})")

if __name__ == "__main__":
    main()
