#!/usr/bin/env python3
"""
Geltungsfristen der PO-Fassungen offline extrahieren (S038).

Liest die Prüfungsordnungen aus stammdaten/po/po_zuordnung.tsv (Text aus
var/korpus/, Zuordnung über `source_url` im Kopf, nie über den Slug) und
sucht allgemeine Formulierungen:

- uebergang_bis: „… längstens (jedoch) bis zum <Datum>“,
  „… längstens bis zum Ende des Sommersemesters JJJJ“,
  „… letztmalig im/zum/für das …semester JJJJ“ — nur in Sätzen, die von einer
  Prüfungsordnung/Fassung sprechen.
- ausser_kraft: „Sie/Diese Prüfungsordnung tritt zum/mit Ablauf des Ende des
  …semesters JJJJ außer Kraft“ bzw. „… am <Datum> außer Kraft“.
- in_kraft: „… tritt zum …semester JJJJ / am <Datum> / ab <Datum> in Kraft“
  (nur datierte; „am Tage nach der Veröffentlichung“ hat kein Datum).

Betroffene Fassung (Regel, im Bericht S038 begründet):
- in_kraft betrifft die eigene Fassung des Dokuments.
- ausser_kraft mit „Sie/Diese …“ betrifft die eigene Fassung; „der bisherige/
  bestehende … tritt außer Kraft“ betrifft die Vorgängerfassung (siehe unten).
- uebergang_bis nennt die Vorgängerfassung selbst („in der 5. Änderung
  (veröffentlicht im Verkündungsblatt Nr. 9/2016 …)“, „der im
  Verkündungsblatt 2/2006 verkündete …“): betroffen ist die jüngste ältere
  Fassung desselben Studiengangs mit Jahr ≤ dem genannten Jahr.
- sonst („in den bisherigen/jeweils geltenden Fassungen“, „nach der bisher
  gültigen Prüfungsordnung“): betroffen sind alle älteren Fassungen desselben
  Studiengangs, **aber je ältere Fassung zählt nur die Bestimmung der
  nächstjüngeren Fassung, die eine solche Bestimmung enthält** — die Fassung,
  die sie tatsächlich abgelöst hat. Beispiel IMC: PO 2019 lässt die
  bisherigen Fassungen bis 31.08.2026 gelten, PO 2021 die ihren bis
  31.08.2028; für IMC 2014 gilt 2026-08-31 (von 2019), für 2019 2028-08-31.
  Abweichende Daten späterer Fassungen landen in `po_geltung_unklar.tsv`.
- Dokumente ohne Studiengang oder ohne Fassung (Allgemeine Teile,
  Änderungsordnungen ohne Stammjahr) liefern keine Zeile; ihre Funde stehen
  in `po_geltung_unklar.tsv`.

Deterministisch, kein LLM. Aufruf:  python werkzeuge/po/extrahiere_geltung.py
Ausgabe: stammdaten/po/po_geltung.tsv, stammdaten/po/po_geltung_unklar.tsv
"""
import calendar
import csv
import re
from collections import defaultdict
from pathlib import Path

from raghsh import konfig

HIER = konfig.PO_DATEN
ZUORDNUNG = HIER / "po_zuordnung.tsv"
KORPUS = konfig.KORPUS
AUSGABE = HIER / "po_geltung.tsv"
UNKLAR = HIER / "po_geltung_unklar.tsv"
SPALTEN = ["studiengang", "fassung", "art", "datum", "beleg_url", "wortlaut"]
MAX_WORTLAUT = 300

MONATE = {"januar": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4, "mai": 5,
          "juni": 6, "juli": 7, "august": 8, "september": 9, "oktober": 10,
          "november": 11, "dezember": 12}
_MON = "|".join(MONATE)
# „31. August 2026“, „28.Februar 2015“, „28.02.2023“, „1.9.2009“
def DATUM(n):
    return (rf"(?P<tag{n}>\d{{1,2}})\.\s?(?:(?P<mon{n}>{_MON})|(?P<monz{n}>\d{{1,2}})\.)\s?"
            rf"(?P<jahr{n}>(?:19|20)\d\d)")
# „Sommersemesters 2027“, „Wintersemesters 2029/2030“, „Wintersemester 2018/19“
def SEMESTER(n):
    return (rf"(?P<sem{n}>Sommer|Winter)\s?-?\s?se\w*ters?\s(?P<sjahr{n}>(?:19|20)\d\d)"
            rf"(?:\s?/\s?\d{{2,4}})?")

UEBERGANG = re.compile(
    rf"längstens\s(?:jedoch\s)?bis\s(?:zum\s)?(?:(?:{DATUM(1)})|(?:Ende\sdes\s{SEMESTER(1)}))"
    rf"|letztmal\w*\s(?:im|zum|für\sdas|in\sdem)\s{SEMESTER(2)}", re.I)
AUSSER = re.compile(
    rf"(?:(?:zum|mit)\s(?:Ende|Ablauf)\sdes\s{SEMESTER(1)}|am\s{DATUM(1)})\s(?:\w+\s){{0,2}}außer\sKraft", re.I)
INKRAFT = re.compile(
    rf"(?:(?:zum|ab\sdem|mit\sdem|am|ab)\s(?:(?:{SEMESTER(1)})|(?:{DATUM(1)})))\s(?:\w+\s){{0,2}}in\sKraft", re.I)
# Satz spricht von einer Ordnung/Fassung (sonst z. B. Abgabefristen)
ORDNUNGSBEZUG = re.compile(r"Prüfungsordnung|Fassung|Ordnung|bisher", re.I)
BISHERIG = re.compile(r"bisherig|bisher\sgelt|bisher\sgült|bestehend|jeweils\sgeltend|vorherig", re.I)
EIGENE = re.compile(r"\b(?:Sie|Diese\s(?:\w+\s){0,2}(?:Prüfungs)?[Oo]rdnung)\s", re.I)
# Vorgängerfassung ausdrücklich genannt: „Verkündungsblatt Nr. 9/2016“, „Verkündungsblatt 2/2006“
GENANNT = re.compile(r"Verkündungsblatt\s(?:der\s\w+\s\w+\s)?(?:Nr\.\s?)?\d{1,2}\s?/\s?((?:19|20)\d\d)")

# Stichtag der neuen Fassung: „… die ihr Studium vor dem 1. September 2019 aufgenommen/begonnen …“
STICHTAG = re.compile(r"vor\sdem\s\d{1,2}\.\s?(?:\w+|\d{1,2}\.)\s?((?:19|20)\d\d)\s(?:\w+\s){0,3}?(?:aufgenommen|begonnen|immatrikuliert)", re.I)
EINZAHL = re.compile(r"in\sder\sbisherigen\sFassung|nach\sder\sbisher\s(?:gültigen|geltenden)\sPrüfungsordnung", re.I)


def lies_rumpf(pfad):
    text = pfad.read_text(encoding="utf-8", errors="replace")
    if text.startswith("---"):
        text = text[text.find("\n---", 3) + 4:]
    text = text.replace("­", "")
    text = re.sub(r"[*#_`>|]", " ", text)
    return re.sub(r"\s+", " ", text)


def url_zu_datei():
    m = {}
    for p in sorted(KORPUS.glob("*.md")):
        with open(p, encoding="utf-8", errors="replace") as f:
            kopf = f.read(2000)
        r = re.search(r"^source_url:\s*\"?([^\"\n]+)", kopf, re.M)
        if r:
            m.setdefault(r.group(1).strip(), p)
    return m


def semesterende(sem, jahr):
    """Letzter Tag: SoSe JJJJ → 31.08.JJJJ, WiSe JJJJ(/JJ) → Februarende JJJJ+1."""
    jahr = int(jahr)
    if sem.lower().startswith("sommer"):
        return f"{jahr}-08-31"
    j = jahr + 1
    return f"{j}-02-{calendar.monthrange(j, 2)[1]:02d}"


def semesterbeginn(sem, jahr):
    jahr = int(jahr)
    return f"{jahr}-09-01" if sem.lower().startswith("winter") else f"{jahr}-03-01"


def datum_aus(m, beginn=False):
    g = {k: v for k, v in m.groupdict().items() if v}
    for n in "12":
        if g.get("jahr" + n):
            mon = MONATE[g["mon" + n].lower()] if g.get("mon" + n) else int(g["monz" + n])
            tag = int(g["tag" + n])
            if not (1 <= mon <= 12 and 1 <= tag <= 31):
                return None
            return f"{g['jahr' + n]}-{mon:02d}-{tag:02d}"
        if g.get("sjahr" + n):
            if beginn:
                return semesterbeginn(g["sem" + n], g["sjahr" + n])
            return semesterende(g["sem" + n], g["sjahr" + n])
    return None


def satz_um(text, a, e):
    """Satz um den Treffer: ab dem letzten Absatz-/Satzanfang vor a, bis Satzende nach e."""
    vor = text[max(0, a - 600):a]
    anf = max(vor.rfind(" - ("), vor.rfind(". Für"), vor.rfind(". Die"), vor.rfind(". Der"),
              vor.rfind(". Sie"), vor.rfind(". Gleichzeitig"), vor.rfind("§"))
    start = a - len(vor) + (anf + 2 if anf >= 0 else 0)
    ende_m = re.search(r"(?<!\d)\.(?:\s|$)", text[e:e + 200])
    ende = e + (ende_m.end() if ende_m else 0)
    return text[start:ende].strip(" -")


def kuerze(s):
    return s if len(s) <= MAX_WORTLAUT else "… " + s[-(MAX_WORTLAUT - 2):]


def funde(text):
    """[(art, datum, bezug, genannt_jahr, satz)] — bezug: 'eigene' | 'vorgaenger'."""
    out = []
    for m in UEBERGANG.finditer(text):
        satz = satz_um(text, m.start(), m.end())
        if not ORDNUNGSBEZUG.search(satz):
            continue
        d = datum_aus(m)
        if not d:
            continue
        g = GENANNT.search(satz[:satz.find("längstens") if "längstens" in satz else len(satz)])
        out.append(("uebergang_bis", d, "vorgaenger", int(g.group(1)) if g else None, satz))
    for m in AUSSER.finditer(text):
        satz = satz_um(text, m.start(), m.end())
        d = datum_aus(m)
        if not d:
            continue
        bezug = "vorgaenger" if BISHERIG.search(satz) and not EIGENE.search(satz) else "eigene"
        g = GENANNT.search(satz) if bezug == "vorgaenger" else None
        out.append(("ausser_kraft", d, bezug, int(g.group(1)) if g else None, satz))
    for m in INKRAFT.finditer(text):
        satz = satz_um(text, m.start(), m.end())
        d = datum_aus(m, beginn=True)
        if not d:
            continue
        out.append(("in_kraft", d, "eigene", None, satz))
    return out


def main():
    zeilen = list(csv.DictReader(open(ZUORDNUNG, encoding="utf-8"), delimiter="\t"))
    dateien = url_zu_datei()
    po = [z for z in zeilen if z["po_dokumenttyp"] == "Pruefungsordnung"]
    fassungen = defaultdict(set)                 # sg -> {Jahr}
    for z in po:
        if z["po_studiengang"] and z["po_fassung"]:
            fassungen[z["po_studiengang"]].add(int(z["po_fassung"]))

    ergebnis = set()                             # (sg, f, art, datum, url, wortlaut)
    unklar = []                                  # [grund, sg, fassung_quelle, art, datum, url, wortlaut]
    # Vorgänger-Bestimmungen „bisherige Fassungen“ je sg: {quellfassung: [(art, datum, url, satz)]}
    bisherige = defaultdict(lambda: defaultdict(list))
    wortgleich = defaultdict(set)                # (sg, art, datum, wortlaut) -> {Quellfassung}
    docs_mit_frist = set()
    for z in po:
        pfad = dateien.get(z["source_url"])
        if not pfad:
            continue
        fs = funde(lies_rumpf(pfad))
        if any(a in ("uebergang_bis", "ausser_kraft") for a, *_ in fs):
            docs_mit_frist.add(z["source_url"])
        sg, f = z["po_studiengang"], z["po_fassung"]
        for art, datum, bezug, genannt, satz in fs:
            w = kuerze(satz)
            if not sg or not f:
                unklar.append(["ohne_studiengang_oder_fassung", sg, f, art, datum, z["source_url"], w])
                continue
            f = int(f)
            if bezug == "eigene":
                ergebnis.add((sg, f, art, datum, z["source_url"], w))
            elif genannt:
                # Nennt der Satz den Studiengang „… (KÜRZEL)“ ausdrücklich, gilt dieser
                # (Sammelbekanntmachungen regeln mehrere Studiengänge in einem Dokument).
                ks = [k for k in re.findall(r"\(([A-Z]{2,6})\)", satz) if k in fassungen]
                sg_b = ks[0] if ks else sg
                kand = [x for x in fassungen[sg_b] if x < f and x <= genannt]
                if kand:
                    ergebnis.add((sg_b, max(kand), art, datum, z["source_url"], w))
                else:
                    unklar.append(["genannte_fassung_fehlt", sg_b, f, art, datum, z["source_url"], w])
            else:
                # Quellfassung: die Fassung, die der Stichtag „vor dem 1. September JJJJ
                # begonnen“ einführt — Änderungsordnungen und Neubekanntmachungen
                # wiederholen die Bestimmung ihrer Stammfassung.
                q = f
                st = STICHTAG.search(satz)
                if st:
                    ab = [x for x in fassungen[sg] if x >= int(st.group(1))]
                    q = min(ab) if ab else f
                nur_vorige = bool(EINZAHL.search(satz))
                wortgleich[(sg, art, datum, w)].add(q)
                bisherige[sg][q].append((art, datum, z["source_url"], w, nur_vorige))

    # Wortgleiche Bestimmung in mehreren Fassungen (ohne Stichtag): sie gehört der
    # ältesten, die sie enthält; jüngere Fassungen haben sie nur übernommen.
    for sg, quellen in bisherige.items():
        for q in list(quellen):
            quellen[q] = [e for e in quellen[q] if min(wortgleich[(sg, e[0], e[1], e[3])]) == q]
            if not quellen[q]:
                del quellen[q]

    # „bisherige Fassungen“: je ältere Fassung die Bestimmung der nächstjüngeren
    # Quellfassung; Einzahl („in der bisherigen Fassung“, „nach der bisher gültigen
    # Prüfungsordnung“) betrifft nur die unmittelbar vorige Fassung.
    for sg, quellen in bisherige.items():
        for alt in sorted(fassungen[sg]):
            juengere = sorted(q for q in quellen if q > alt)
            if not juengere:
                continue
            naechste = juengere[0]
            vorige = max(x for x in fassungen[sg] if x < naechste)
            for art, datum, url, w, nur_vorige in quellen[naechste]:
                if nur_vorige and alt != vorige:
                    continue
                ergebnis.add((sg, alt, art, datum, url, w))
            eigene_daten = {(a, d) for a, d, *_ in quellen[naechste]}
            for q in juengere[1:]:
                for art, datum, url, w, _ in quellen[q]:
                    if (art, datum) not in eigene_daten:
                        unklar.append([f"spaetere_fassung_{q}_nennt_anderes_datum_fuer_{alt}",
                                       sg, q, art, datum, url, w])
        for q in quellen:
            if not any(x < q for x in fassungen[sg]):
                for art, datum, url, w, _ in quellen[q]:
                    unklar.append(["keine_aeltere_fassung_im_korpus", sg, q, art, datum, url, w])

    # Je (sg, fassung, art, datum) eine Belegzeile (erste URL in Sortierung) — Kopien
    # derselben Ordnung (Fakultätsseite und Verkündungsblatt) ergeben sonst Dubletten.
    beste = {}
    for r in sorted(ergebnis):
        beste.setdefault(r[:4], r)
    # in_kraft: nur das früheste Datum je Fassung — spätere stammen aus
    # Änderungsordnungen, die unter ihrer Stammfassung geführt werden.
    frueh = {}
    for k in sorted(k for k in beste if k[2] == "in_kraft"):
        frueh.setdefault(k[:2], k)
    beste = {k: r for k, r in beste.items() if k[2] != "in_kraft" or frueh[k[:2]] == k}
    with open(AUSGABE, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(SPALTEN)
        for r in sorted(beste.values()):
            w.writerow(r)
    with open(UNKLAR, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["grund", "studiengang", "fassung_quelle", "art", "datum", "beleg_url", "wortlaut"])
        for r in sorted(set(map(tuple, unklar)), key=lambda r: tuple(map(str, r))):
            w.writerow(r)

    zeilen_aus = list(beste.values())
    print(f"PO-Dokumente: {len(po)}, davon mit datierter Frist (uebergang_bis/ausser_kraft): {len(docs_mit_frist)}")
    for art in ("uebergang_bis", "ausser_kraft", "in_kraft"):
        fs = {(r[0], r[1]) for r in zeilen_aus if r[2] == art}
        print(f"  {art}: {sum(1 for r in zeilen_aus if r[2] == art)} Zeilen, {len(fs)} Fassungen")
    print(f"  Zeilen gesamt {len(zeilen_aus)}, unklar {len(set(map(tuple, unklar)))}")


if __name__ == "__main__":
    main()
