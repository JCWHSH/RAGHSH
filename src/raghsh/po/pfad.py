"""PO-Pfad von frag_hsh: Studiengang und Fakultät aus der Frage, Fassung der
Prüfungsordnung, Fassungsvielfalt, Geltung, Schranke, Wertetabelle,
LLM-Fassungsvergleich, Rückfrage und Kopfzeilen.

Herausgelöst aus dem MCP-Server (vorher dort Zeilen 1553–2699); die Begründungen
stehen weiter an den Konstanten. Ablauf und Datenquellen: docs/po_logik.md.

Laufzeitobjekte (Qdrant-Client, Embedder, LLM-Aufruf, Collection) entstehen
erst in der Lifespan des Servers. Der Server trägt sich deshalb über
``verbinden()`` ein; die Funktionen hier lesen sie zur Aufrufzeit über
``_server``.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import lru_cache
import numpy as np
from qdrant_client import models
from raghsh.po import filter as po_filter
from raghsh.po import werte as po_werte
from raghsh.suche.hybrid_search import GLEICHSTAND_RESERVE, embed_query_dense, embed_query_sparse, nach_score

_server = None   # das Servermodul, gesetzt durch verbinden()


def verbinden(server_modul) -> None:
    """Vom MCP-Server beim Import aufgerufen (siehe dort)."""
    global _server
    _server = server_modul


# Fakultät aus der Frage lesen. Anlass (eval_gold, 21.09.2026): „An wen wende
# ich mich in der Fakultät I bei Problemen mit der Prüfungsanmeldung?" — die
# Kandidatenliste füllte sich mit Prüfungsanmeldungs-Anleitungen von F2, F3 und
# F4, das Gold (Prüfungsverwaltung F1) lag auf Chunk 19. Der Fakultätsfilter
# existierte, aber nur die Oberfläche setzte ihn. Nennt die Frage die Fakultät
# ausdrücklich — Ziffer, römische Zahl, Kürzel oder voller Name —, gilt sie
# als gesetzt, sofern der Aufrufer nichts anderes gewählt hat. Fachbegriffe
# allein („Informatik") reichen nicht: Die kommen in jeder Fakultät vor.
_FAK_ROEMISCH = {"I": "f1", "II": "f2", "III": "f3", "IV": "f4", "V": "f5"}
_FAK_MUSTER = [
    (re.compile(r"\bFakult[äa]t\s*(IV|V|III|II|I)(?![IVX\w])"), lambda m: _FAK_ROEMISCH[m.group(1)]),
    (re.compile(r"\bFakult[äa]t\s*([1-5])(?!\d)"),                lambda m: f"f{m.group(1)}"),
    (re.compile(r"\bF\s?([1-5])\b"),                              lambda m: f"f{m.group(1)}"),
    (re.compile(r"Elektro-\s*und Informationstechnik", re.I),      lambda m: "f1"),
    (re.compile(r"Maschinenbau und Bioverfahrenstechnik", re.I),     lambda m: "f2"),
    (re.compile(r"Medien,\s*Information und Design", re.I),         lambda m: "f3"),
    (re.compile(r"Wirtschaft und Informatik", re.I),                 lambda m: "f4"),
    (re.compile(r"Diakonie,\s*Gesundheit und Soziales", re.I),      lambda m: "f5"),
]


def _fakultaet_aus_frage(frage: str) -> str:
    """f1…f5, wenn die Frage genau eine Fakultät nennt; sonst ''.

    Zwei verschiedene Nennungen („Wechsel von Fakultät I nach IV") heißen:
    keine — dann wäre jeder Filter falsch.
    """
    gefunden = {wert(m) for muster, wert in _FAK_MUSTER for m in muster.finditer(frage)}
    return gefunden.pop() if len(gefunden) == 1 else ""


# PO aus der Frage lesen (S022, Ziel 3). Analog zur Fakultät: Nennt die Frage
# einen Studiengang (Kürzel oder Langform aus studiengaenge.tsv) und dazu eine
# Fassung („PO 2018“, „Prüfungsordnung von 2016“) oder einen Neubeginn („ich
# fange jetzt an“ → jüngste aktuelle Fassung, F1 Punkt 5), gilt diese PO, sofern
# der Aufrufer keine mitgibt. Ohne erkennbaren Studiengang, bei mehreren
# Studiengängen oder mehreren Jahren: keine PO. Die Erkennung wird nicht
# geloggt (Fragetext), nur im Ergebnis unter `po_erkannt` gemeldet.
_PO_JAHR = re.compile(r"\b(?:PO|Prüfungsordnung)\b[^\w.?!]{0,3}(?:(?:von|vom|aus)\s+)?(20\d\d)\b", re.I)
_PO_NEUBEGINN = re.compile(
    r"\b(?:fange|fangen)\b[^.?!]{0,80}\ban\b"
    r"|\b(?:beginne|starte)\b[^.?!]{0,80}\b(?:Studium|neu|jetzt)\b"
    r"|\bErstsemester|\bStudienanf[äa]nger", re.I)
STUDIENGAENGE_TSV = po_filter.PO_DATEN / "studiengaenge.tsv"


_ADJ_ENDUNG = re.compile(r"(?:es|en|er|em|e)$")


def _wort_muster(w: str) -> str:
    """Ein Wort einer Langform mit deutscher Flexion: Adjektivendung weg,
    bis zu zwei Zeichen zulassen („Medizinisches“ trifft „Medizinischem“,
    „Angewandte“ trifft „Angewandter“)."""
    if len(w) <= 3:
        return re.escape(w)
    stamm = _ADJ_ENDUNG.sub("", w) if len(w) > 5 else w
    return re.escape(stamm) + r"\w{0,2}"


def _langform_muster(name: str) -> re.Pattern:
    """Langform → Muster, das deutsche Flexionsendungen zulässt."""
    woerter = re.findall(r"[\wÄÖÜäöüß]+|&", name)
    # Trenner auch „/“, Klammern und „.“ (S054): sonst trafen „Szenografie/Kostüm/…“
    # und „Wirtschaftsingenieur/in (Technischer Vertrieb)“ ihre eigene Langform nie (S053).
    return re.compile(r"\b" + r"[\s\-–,/().]+".join(map(_wort_muster, woerter)) + r"\b", re.I)


def _kurzform_muster(name: str) -> re.Pattern | None:
    """Kurzform einer Langform aus Adjektiven + Kernwort („Angewandte
    Informatik“ → „Informatik“) — nur zusammen mit dem Abschluss
    („Informatik-Bachelor“, „Master Informatik“). Das Kernwort allein kommt
    in zu vielen anderen Zusammenhängen vor (Fakultätsname, Abteilung)."""
    woerter = re.findall(r"[\wÄÖÜäöüß]+", name)
    if len(woerter) < 2 or not all(_ADJ_ENDUNG.search(w) for w in woerter[:-1]):
        return None
    kern = re.escape(woerter[-1])
    return re.compile(rf"\b{kern}[\s\-]*(?:Bachelor|Master)"
                      rf"|\b(?:Bachelor|Master)\w*\s+(?:in\s+|im\s+)?{kern}\b", re.I)


@lru_cache(maxsize=1)
def _studiengang_daten() -> dict[str, dict]:
    """Kürzel → {abschluss, fakultaet, dual, namen} für jeden Studiengang mit
    PO im PoIndex. `dual`: ein Name (Langform oder Variante) sagt „dual“."""
    mit_po = {s for s, _ in po_filter.lade().urls}
    daten = {}
    with open(STUDIENGAENGE_TSV, encoding="utf-8") as f:
        for z in csv.DictReader(f, delimiter="\t"):
            if z["kuerzel"] not in mit_po:
                continue
            namen = [z["langform"]]
            v = z.get("vermerk") or ""
            if v.startswith("Langform-Varianten:"):
                namen += [n.strip() for n in v.split("|")[0].split(":", 1)[1].split(";")]
            namen = [n for n in namen if n]
            daten[z["kuerzel"]] = {
                "abschluss": z.get("abschluss") or "", "fakultaet": z.get("fakultaet") or "",
                "dual": any(re.search(r"\bdual\b", n, re.I) for n in namen), "namen": namen}
    return daten


def _studiengang_langform(sg: str) -> str:
    """Langform des Studiengangs aus studiengaenge.tsv, sonst das Kürzel
    (auch bei leerer Namensliste)."""
    namen = (_studiengang_daten().get(sg) or {}).get("namen") or [sg]
    return namen[0]


KURZFORMEN_TSV = po_filter.PO_DATEN / "kurzformen.tsv"


@lru_cache(maxsize=1)
def _kurzformen() -> list[tuple[str, re.Pattern, list[str], bool]]:
    """(Form, Muster, Kürzel, mehrdeutig?) je Zeile aus kurzformen.tsv (S080, F4).

    Ganzes Wort; Formen bis zwei Buchstaben („PR“, „ET“) nur case-sensitiv,
    sonst träfe „pr“/„et“ beliebigen Text. Leerzeichen in der Form lassen
    auch einen Bindestrich zu („Data-Science“).
    Mehrere Kürzel bedeuten je nach Spalte `mehrdeutig` zweierlei (S081):
    „nein“ = eine Familie mit demselben Namen (Medizininformatik → BMI,MMI),
    die _studiengang_familie_aus_frage wie bei einer Langform über den
    Abschluss einengt; „ja“ = verschiedene Studiengänge, nie raten."""
    eintraege = []
    with open(KURZFORMEN_TSV, encoding="utf-8") as f:
        for z in csv.DictReader(f, delimiter="\t"):
            form = z["kurzform"]
            m = r"[\s\-]".join(map(re.escape, form.split()))
            m = re.compile(rf"\b{m}\b", 0 if len(form) <= 2 else re.I)
            eintraege.append((form, m, [k.strip() for k in z["kuerzel"].split(",") if k.strip()],
                              z["mehrdeutig"] == "ja"))
    return eintraege


@lru_cache(maxsize=1)
def _studiengang_muster() -> list[tuple[str, re.Pattern, int, bool]]:
    """(Kürzel, Muster, Länge, Kurzform?) für jeden Studiengang mit PO im PoIndex.
    Eindeutige Formen aus kurzformen.tsv zählen als Kurzform (S080), aber nur
    im PO-Kontext (S081, siehe _studiengang_familie_aus_frage)."""
    muster = []
    daten = _studiengang_daten()
    for sg, d in daten.items():
        muster.append((sg, re.compile(rf"\b{re.escape(sg)}\b"), len(sg), False))
        for n in d["namen"]:
            muster.append((sg, _langform_muster(n), len(n), False))
            k = _kurzform_muster(n)
            if k:
                muster.append((sg, k, len(n), True))
    for form, m, kuerzel, mehrdeutig in _kurzformen():
        if not mehrdeutig:
            muster += [(sg, m, len(form), True) for sg in kuerzel if sg in daten]
    return muster


# Wortanfang-Regel (S080): ein Wort ab 5 Buchstaben direkt vor oder nach dem
# Abschluss („Mathe-Bachelor“, „Master Wirtschaftsinfo“), das ein längeres Wort
# einer Langform anfängt („Mathe“ → „Mathematik“). Nur mit Abschluss, aus
# demselben Grund wie bei _kurzform_muster.
_WORTANFANG = re.compile(
    r"\b([A-Za-zÄÖÜäöüß]{5,})[\s\-]*(?:Bachelor|Master)"
    r"|\b(?:Bachelor|Master)\w*\s+(?:in\s+|im\s+)?([A-Za-zÄÖÜäöüß]{5,})\b", re.I)


@lru_cache(maxsize=1)
def _langform_woerter() -> list[tuple[str, str]]:
    """(Wort klein, Kürzel) aller Wörter der Langformen und Varianten."""
    return sorted({(w.lower(), sg) for sg, d in _studiengang_daten().items()
                   for n in d["namen"] for w in re.findall(r"[\wÄÖÜäöüß]+", n)})


def _wortanfang_treffer(frage: str) -> list[tuple[int, int, str]]:
    """(Anfang, Ende, Kürzel) je Wortanfang-Treffer, siehe _WORTANFANG."""
    treffer = []
    for t in _WORTANFANG.finditer(frage):
        g = 1 if t.group(1) else 2
        wort = t.group(g).lower()
        if _ABSCHLUSS.fullmatch(wort):
            continue
        treffer += [(t.start(g), t.end(g), sg) for w, sg in _langform_woerter()
                    if len(w) > len(wort) and w.startswith(wort)]
    return treffer


# PO-Kontext für alle Formen aus kurzformen.tsv (S080 mehrdeutige, S081 auch
# eindeutige; F4): Die Frage nennt einen Abschluss oder die PO. Ohne das
# bezeichnen „Informatik“, „IM“, „MBA“, „PR“ meist Abteilungen, Projekte,
# Module oder das Fach (eval_gold g18/g32/g36/g37 als Wache). „PO“ nur als
# ganzes Wort (S081), sonst träfe es „Politik“, „Podcast“, „Position“.
_PO_KONTEXT = re.compile(r"\b(?:Bachelor|Master|Prüfungsordnung)|\bPOs?\b", re.I)


def _studiengang_mehrdeutig(frage: str, familie: list[str]) -> list[str]:
    """Kürzel, auf die eine mehrdeutige Kurzform der Frage passt — nur ohne
    erkannten Studiengang und nur im PO-Kontext; sonst []. Nie geraten (F4):
    frag_hsh fragt dann nach dem Studiengang zurück (S085)."""
    if familie or not _PO_KONTEXT.search(frage):
        return []
    daten = _studiengang_daten()
    return sorted({sg for _f, m, kuerzel, mehrdeutig in _kurzformen() if mehrdeutig
                   and m.search(frage) for sg in kuerzel if sg in daten})


def _studiengang_rueckfrage(kuerzel: list[str]) -> dict:
    """Rückfrage nach dem Studiengang bei mehrdeutiger Kurzform (S085).

    Optionen tragen Kürzel und Langform; die Wahl kommt als
    `po: {"studiengang": …}` (ohne Fassung) zurück."""
    optionen = [{"studiengang": sg, "name": _studiengang_langform(sg)} for sg in kuerzel]
    liste = ", ".join(f"{o['name']} ({o['studiengang']})" if o["name"] != o["studiengang"]
                      else o["studiengang"] for o in optionen)
    return {"frage": (f"Die Antwort hängt vom Studiengang und seiner Prüfungsordnung ab. "
                      f"Welchen Studiengang meinen Sie? Zur Auswahl: {liste}."),
            "optionen": optionen}


def _studiengang_gewaehlt(sg: str) -> str:
    """Kürzel aus `po` ohne Fassung (Wahl aus der Studiengangs-Rückfrage, S085);
    Groß/klein egal. Unbekannt → ValueError mit den vorhandenen Kürzeln."""
    bekannt = sorted(_studiengang_daten())
    treffer = next((s for s in bekannt if s.lower() == str(sg).lower()), None)
    if not treffer:
        raise ValueError(f"Studiengang {sg!r} hat keine bekannte PO; vorhanden: "
                         + ", ".join(bekannt))
    return treffer


_ABSCHLUSS = re.compile(r"\b(Bachelor|Master)", re.I)
_DUAL = re.compile(r"\bdual", re.I)


def _studiengang_familie_aus_frage(frage: str) -> list[str]:
    """Kürzel des genannten Studiengangs — oder seiner Familie (S032).

    Überlappen sich Treffer („Informatik“ in „Angewandte Informatik“),
    zählt der längste. Eine Stelle kann mehrere Studiengänge treffen, wenn
    sie denselben Namen tragen (IID/IIM, BMI/MMI, MEC/MTD): Das ist eine
    Familie, keine Mehrdeutigkeit. Nennt die Frage an verschiedenen Stellen
    verschiedene Studiengänge, bleibt nur, was alle Stellen gemeinsam haben
    (sonst []). Kurzformen („Informatik-Bachelor“) zählen nur ohne
    Volltreffer. Die Familie wird mit dem eingeengt, was die Frage sonst
    sagt: Abschluss, Fakultät, „dual“; ausgelaufene Studiengänge fallen weg,
    wenn ein anderer bleibt. Formen aus kurzformen.tsv zählen nur im
    PO-Kontext (S081, F4-Nachtrag): „PR und Öffentlichkeitsarbeit“ ist eine
    Sachfrage, kein Studiengang.
    """
    treffer = {False: [], True: []}
    tabelle = set() if _PO_KONTEXT.search(frage) else {m for _f, m, _k, _md in _kurzformen()}
    for sg, m, laenge, kurz in _studiengang_muster():
        if m in tabelle:
            continue
        for t in m.finditer(frage):
            treffer[kurz].append((t.start(), t.end(), sg))
    treffer[True] += _wortanfang_treffer(frage)
    treffer = treffer[False] or treffer[True]
    # Treffer, die in einem längeren Treffer enthalten sind, fallen weg
    stellen: dict[tuple[int, int], set[str]] = {}
    for a, e, sg in treffer:
        if not any(a2 <= a and e <= e2 and (e2 - a2) > (e - a) for a2, e2, _ in treffer):
            stellen.setdefault((a, e), set()).add(sg)
    if not stellen:
        return []
    familie = set.intersection(*stellen.values())
    daten = _studiengang_daten()

    def einengen(bedingung):
        nonlocal familie
        rest = {sg for sg in familie if bedingung(daten.get(sg, {}))}
        if rest:
            familie = rest

    if len(familie) > 1:
        abschluesse = {m.group(1).capitalize() for m in _ABSCHLUSS.finditer(frage)}
        if len(abschluesse) == 1:
            a = abschluesse.pop()
            einengen(lambda d: d.get("abschluss") == a)
    if len(familie) > 1:
        fak = _fakultaet_aus_frage(frage)
        if fak:
            einengen(lambda d: d.get("fakultaet") == fak)
    if len(familie) > 1 and _DUAL.search(frage):
        einengen(lambda d: d.get("dual"))
    if len(familie) > 1:
        ausgelaufen = po_filter.lade().ausgelaufen
        familie = {sg for sg in familie if sg not in ausgelaufen} or familie
    return sorted(familie)


def _fakultaet_des_studiengangs(sg: str | None, familie: list[str]) -> str:
    """Fakultätsschlüssel (f1…f5) des erkannten Studiengangs, sonst "" (S043).

    Ohne PO-Angabe zählt die Familie aus der Frage, wenn alle Mitglieder zur
    selben Fakultät gehören. Fehlt die Fakultät in studiengaenge.tsv, wird das
    mit dem Kürzel geloggt (nie der Fragetext) und ungefiltert gesucht."""
    daten = _studiengang_daten()
    familie = [sg] if sg else familie
    fak = {daten.get(s, {}).get("fakultaet", "") for s in familie}
    if len(fak) == 1 and (f := fak.pop()) in {k for k, _ in _server.FACULTIES}:
        return f
    if familie:
        print(f"  ⚠ Studiengang {','.join(sorted(familie))} ohne eindeutige Fakultät "
              "— Suche ungefiltert")
    return ""


def _urls_des_studiengangs(sg: str | None, familie: list[str]) -> set[str]:
    """Qdrant-URLs aller Fassungen des erkannten Studiengangs bzw. seiner
    Familie (PoIndex.urls), leer ohne Erkennung (S043)."""
    idx = po_filter.lade()
    familie = [sg] if sg else familie
    return set().union(*(idx.urls[(s, f)] for s in familie for f in idx.fassungen(s)))


def _po_aus_frage(frage: str, familie: list[str],
                  nur: set[str] | None = None) -> dict | None:
    """{"studiengang", "fassung"} oder None (siehe Kommentar oben).

    `familie`: _studiengang_familie_aus_frage(frage), einmal in frag_hsh berechnet.
    `nur`: Familie auf diese Studiengänge einschränken (S032: die, die in den
    Kandidaten stehen — siehe frag_hsh)."""
    jahre = {int(m.group(1)) for m in _PO_JAHR.finditer(frage)}
    neubeginn = bool(_PO_NEUBEGINN.search(frage))
    if len(jahre) > 1 or not (jahre or neubeginn):
        return None
    if nur is not None:
        familie = [sg for sg in familie if sg in nur]
    idx = po_filter.lade()
    if jahre:
        # Familie (S032): gilt, wenn genau ein Mitglied diese Fassung hat
        jahr = jahre.pop()
        mit = [sg for sg in familie if jahr in idx.fassungen(sg)]
        return {"studiengang": mit[0], "fassung": jahr} if len(mit) == 1 else None
    if len(familie) != 1:
        return None
    sg = familie[0]
    if sg in idx.ausgelaufen:                   # F1 Punkt 5: nicht anbieten
        return None
    aktuell = [f for f in idx.fassungen(sg) if idx.aktuell.get((sg, f))]
    return {"studiengang": sg, "fassung": max(aktuell)} if aktuell else None


def _po_plan(po: dict | None) -> dict | None:
    """PO-Angabe → Filter und auszuschließende URLs (S011, E004/E006).

    Der Qdrant-Filter schließt die anderen Fassungen desselben Studiengangs
    aus (Nicht-PO-Seiten und Allgemeine Teile bleiben); `andere` sind deren
    URLs, für den Keyword-Boost, der ungefiltert sucht. URLs, die zugleich
    Chunks der gewünschten Fassung tragen (Textgleichheit, E004), gelten
    nicht als „andere Fassung“. Die frühere Variante B (Nachrang statt
    Ausschluss) war gemessen schlechter (E006) und ist seit S075 entfernt.
    """
    if not po:
        return None
    try:
        sg, fassung = str(po["studiengang"]), int(po["fassung"])
    except (KeyError, TypeError, ValueError):
        raise ValueError('po erwartet {"studiengang": "<Kürzel>", "fassung": <Jahr>}')
    idx = po_filter.lade()
    bekannt = sorted({s for s, _ in idx.urls})
    if sg not in bekannt:                       # Kürzel ohne Rücksicht auf Groß/klein
        sg = next((s for s in bekannt if s.lower() == sg.lower()), sg)
    if sg not in bekannt:
        raise ValueError(f"Studiengang {sg!r} hat keine bekannte PO; vorhanden: "
                         + ", ".join(bekannt))
    vorhanden = idx.fassungen(sg)
    if fassung not in vorhanden:
        raise ValueError(f"Keine PO {fassung} für {sg!r} bekannt; vorhanden: "
                         + ", ".join(map(str, vorhanden)))
    return {"studiengang": sg, "fassung": fassung,
            "filter": idx.filter(sg, fassung),
            "andere": idx.ausschluss(sg, fassung) - idx.urls[(sg, fassung)]}


# Fassungsvielfalt in den Kandidaten (S026, Ziel 3 „Rückfrage, wenn nötig“,
# Teil a). Umkehrung von PoIndex.urls: Qdrant-URL → {(Studiengang, Fassung)}.
# Eine Konflikt-URL (mehrere Fassungen desselben Studiengangs, E004) zählt
# für keine Fassung allein. Nur Kürzel, Jahre und Ränge; kein Fragetext.
@lru_cache(maxsize=1)
def _po_url_fassungen() -> dict[str, dict[str, int]]:
    """URL → {Studiengang: Fassung}, nur eindeutige (URL, Studiengang)-Paare."""
    idx = po_filter.lade()
    je_url: dict[str, dict[str, set]] = {}
    for (sg, f), urls in idx.urls.items():
        for u in urls:
            je_url.setdefault(u, {}).setdefault(sg, set()).add(f)
    return {u: {sg: next(iter(fs)) for sg, fs in m.items() if len(fs) == 1}
            for u, m in je_url.items()}


@lru_cache(maxsize=1)
def _po_punkt_fassungen() -> dict[str, dict[str, int]]:
    """Punkt-ID → {Studiengang: Fassung}, nur Punkte genau einer Fassung (S027).

    Auf einer Konflikt-URL (E004) tragen meist nicht alle Chunks alle
    Fassungen; ein Chunk, der nur zu einer Fassung gehört, zählt für diese.
    """
    idx = po_filter.lade()
    je_pid: dict[str, dict[str, set]] = {}
    for (sg, f), pids in idx.punkte.items():
        for pid in pids:
            je_pid.setdefault(str(pid), {}).setdefault(sg, set()).add(f)
    return {pid: {sg: next(iter(fs)) for sg, fs in m.items() if len(fs) == 1}
            for pid, m in je_pid.items()}


def _po_vielfalt(familie: list[str], kandidaten: list[dict],
                 results: list[dict] | None = None) -> dict | None:
    """{"studiengang", "fassungen", "je_fassung": {Fassung: [Rang, …]},
    "chunks": {Fassung: [Punkt-ID, …]}} oder None.

    Studiengang: der in der Frage genannte bzw. das einzige Mitglied seiner
    Familie in den Kandidaten (S032); sonst keine Vielfalt. Vielfalt heißt: mindestens zwei
    Fassungen dieses Studiengangs stehen in den Kandidaten oder gelten laut
    PoIndex (S039, `aus_index`). Zugeordnet wird
    über die URL und — für Konflikt-URLs — über die gefundenen Chunks (S027);
    `chunks` nennt je Fassung die Treffer-Chunks in Trefferreihenfolge.
    """
    zuordnung = _po_url_fassungen()
    punkt_zuordnung = _po_punkt_fassungen()
    rang_je_url = {k["url"]: k["nr"] for k in kandidaten}
    je_sg: dict[str, dict[int, list[int]]] = {}
    chunks: dict[str, dict[int, list[str]]] = {}
    for k in kandidaten:
        for sg, f in zuordnung.get(k["url"], {}).items():
            je_sg.setdefault(sg, {}).setdefault(f, []).append(k["nr"])
    for r in results or []:
        url = (r.get("payload") or {}).get("source_url", "")
        if url not in rang_je_url:
            continue
        pid = str(r.get("id", ""))
        fs = dict(zuordnung.get(url, {}))
        fs.update(punkt_zuordnung.get(pid, {}))
        for sg, f in fs.items():
            raenge = je_sg.setdefault(sg, {}).setdefault(f, [])
            if rang_je_url[url] not in raenge:
                raenge.append(rang_je_url[url])
                raenge.sort()
            chunks.setdefault(sg, {}).setdefault(f, []).append(pid)
    # S032: nur der Studiengang aus der Frage zählt. Bei einer Familie
    # (gleicher Name) muss genau ein Mitglied in den Kandidaten stehen. Ohne
    # erkannten Studiengang keine Vielfalt — die frühere Mehrheitsregel über
    # die Kandidaten fragte zu Studiengängen zurück, nach denen niemand fragte.
    # S039: Nennt die Frage genau einen Studiengang, zählt er auch dann, wenn
    # keine seiner Fassungen in den Kandidaten steht.
    vorhanden = [s for s in familie if s in je_sg]
    if len(vorhanden) == 1:
        sg = vorhanden[0]
    elif len(familie) == 1 and not vorhanden:
        sg = familie[0]
    else:
        return None
    je_fassung = je_sg.get(sg, {})
    # S039: Die Kandidaten zeigen oft nur einen Teil der Fassungen (eine,
    # zwei von vier). Verglichen werden deshalb alle geltenden PO-Fassungen
    # des erkannten Studiengangs aus dem PoIndex; die Nachsuche je Fassung
    # braucht keine Kandidaten. `aus_index` nennt die so hinzugekommenen.
    idx = po_filter.lade()
    stichtag = _po_stichtag()
    aus_index = [f for f in idx.fassungen(sg, nur_po=True)
                 if f not in je_fassung and not idx.abgelaufen(sg, f, stichtag)]
    fassungen = sorted(set(je_fassung) | set(aus_index))
    if len(fassungen) < 2:
        return None
    return {"studiengang": sg, "fassungen": fassungen,
            "je_fassung": {f: je_fassung.get(f, []) for f in fassungen},
            "chunks": {f: chunks.get(sg, {}).get(f, []) for f in fassungen},
            "aus_index": aus_index}


# Schranke vor dem Fassungsvergleich (S064, S057 Ursache A): Nennt die Frage
# nur einen Studiengang, betrifft aber nicht die PO (Stundenplan, Raum,
# Zulassungsordnung), lief der Vergleich trotzdem — GWDG-Anfragen, ~8 s je
# Frage und das Risiko einer unnötigen Rückfrage. Verglichen wird nur, wenn
# ein PO-Dokument (`po_dokumenttyp` „Pruefungsordnung“) des Studiengangs unter
# den ersten PO_SCHRANKE_RANG Kandidaten steht; die Vertreter-URL eines
# PO-Dokuments zählt mit. Modulhandbücher und Studienpläne zählen nicht.
PO_SCHRANKE_RANG = 3


@lru_cache(maxsize=1)
def _po_dokument_urls() -> dict[str, set[str]]:
    """Studiengang → Qdrant-URLs seiner PO-Dokumente (eigene und Vertreter-URLs)."""
    idx = po_filter.lade()
    aus: dict[str, set[str]] = {}
    for z in idx.zeilen:
        if z["po_dokumenttyp"] != "Pruefungsordnung" or not z["po_studiengang"]:
            continue
        urls = aus.setdefault(z["po_studiengang"], set())
        urls.add(z["source_url"])
        urls.update(v for _i, _pid, v, _s in idx.chunks.get(z["source_url"], []))
    return aus


def _po_schranke_rang(studiengang: str, kandidaten: list[dict]) -> int | None:
    """Kandidatenrang des ersten PO-Dokuments des Studiengangs oder None."""
    urls = _po_dokument_urls().get(studiengang, set())
    return min((k["nr"] for k in kandidaten if k["url"] in urls), default=None)


# Rückfrage entscheiden (S027, Ziel 3): Ergeben die Auszüge der Fassungen
# verschiedene Antworten auf genau diese Frage? Nur „verschieden“ führt zur
# Rückfrage; „zusatzbedingung“, „gleich“ und „unklar“ antworten normal —
# unnötige Rückfragen sind ein Fehler, keine Vorsicht. „zusatzbedingung“
# (S039, Christian 01.10.2026): der gefragte Wert ist gleich, eine Fassung
# verlangt zusätzlich etwas; die Antwort nennt den Wert und weist auf die
# Bedingung hin (`_po_hinweis`).
PO_VERGLEICH_SYSTEM = """Du vergleichst Auszüge aus mehreren Fassungen der Prüfungsordnung (PO) \
eines Studiengangs der Hochschule Hannover.

1. Bestimme den gefragten Punkt (z. B. Zahl der Credits, Frist, Dauer, Zahl der \
Schwerpunkte, Voraussetzung).
2. Nenne je Fassung den Wert zu genau diesem Punkt, so kurz wie möglich (Zahl oder \
wenige Wörter, mit §). Behandeln die Auszüge einer Fassung den Punkt nicht: "".
3. Nenne je Fassung unter "zusatz" eine Bedingung, die nur diese Fassung zusätzlich \
zum gleichen Wert verlangt (z. B. "bestandene Vorprüfung"); sonst weglassen.
4. Urteil, nur über die Fassungen mit Wert:
- "verschieden": der gefragte Wert selbst unterscheidet sich (andere Zahl, Frist, \
Dauer, andere Voraussetzung statt der bisherigen) — auch nach den Gleich-Regeln \
unten. Prüfe vorher: Weist eine Fassung einen Bestandteil getrennt aus und eine \
andere die Summe einschließlich dieses Bestandteils, ist das kein Unterschied.
- "zusatzbedingung": der Wert ist überall gleich, aber mindestens eine Fassung \
verlangt zusätzlich eine Bedingung, die andere nicht nennen.
- "gleich": alle Werte stimmen überein.
- "unklar": höchstens eine Fassung nennt einen Wert, oder die Frage betrifft nicht \
diese Prüfungsordnung.
Gleich sind: dieselbe Zahl in anderer Schreibweise oder Einheit (12 = zwölf = 12 CP \
= 12 Credits = 12 Leistungspunkte); eine Summe und ihr Gesamtwert (12 + 3 Kolloquium \
= 15 inkl. Kolloquium); eine andere Paragraphennummer bei gleichem Inhalt. Fehlt der \
Wert in einer Fassung nur, weil ihr Auszug den Punkt nicht behandelt, ist das kein \
Unterschied.

Antworte knapp und ausschließlich mit JSON:
{"werte": {"<Fassung>": "<Wert (§)>", ...}, "zusatz": {"<Fassung>": "<Bedingung>"}, \
"urteil": "verschieden" | "zusatzbedingung" | "gleich" | "unklar"}"""

PO_VERGLEICH_ZEICHEN = 1200     # je Auszug (S039: 1500 → 1200, schnellerer Vergleich)
PO_VERGLEICH_CHUNKS = 3         # je Fassung (S033: 2 → 3, der Punkt stand oft im dritten)
PO_NACHSUCHE_ARM = 10           # Treffer je Arm (Dense/Sparse) vor der RRF-Fusion der Nachsuche
# Obergrenze für Nachsuche + Vergleichsaufruf zusammen (S027). Überschritten →
# Urteil „unklar“, normale Antwort. Die Nachsuche selbst braucht < 0,1 s
# (HasId-Filter, gemessen gegen source_url-Filter); die Zeit geht an das LLM.
# Für Messläufe mit langsamen Modellen (GLM) per Umgebung anhebbar, z. B.
# HSH_PO_VERGLEICH_GRENZE_S=120 (Christian, 30.09.2026: GLM-Nachtlauf).
# Christian, 03.10.2026: 60 s statt 30 s, weil GLM über GWDG (Standard seit 03.10.) für
# den Vergleich meist 25–40 s braucht und bei 30 s fast immer „unklar“ endete.
PO_VERGLEICH_GRENZE_S = float(os.getenv("HSH_PO_VERGLEICH_GRENZE_S", "60"))
# Der Vergleichsaufruf dauert meist 3–10 s, einzelne Aufrufe hängen aber
# (pr05/pr07 einmal 4 s, einmal > 30 s, S039; GWDG-Hänger unabhängig von der
# Eingabegröße, S053). Deshalb startet nach PO_VERGLEICH_ABSTAND_S ohne Antwort
# ein weiterer, paralleler Versuch (der hängende läuft weiter); die erste
# Antwort zählt, alle zusammen enden an der Grenze (S054). Bis S053: erster
# Versuch bis zur halben Grenze abgebrochen, dann eine Wiederholung — ein
# Versuch, der kurz nach dem Abbruch fertig geworden wäre, war verloren.
PO_VERGLEICH_VERSUCHE = 3
PO_VERGLEICH_ABSTAND_S = 15.0
# Eigener kleiner Executor für die Vergleichsaufrufe (S056): Verworfene Versuche
# belegen so nicht den Standard-Threadpool von asyncio.to_thread, und jeder
# Aufruf endet spätestens an der Grenze (Client-Timeout = Restzeit, keine Retries).
_po_vergleich_pool = ThreadPoolExecutor(max_workers=2 * PO_VERGLEICH_VERSUCHE,
                                        thread_name_prefix="po-vergleich")

# Wörter, die in der Nachsuche nichts unterscheiden: Innerhalb der Punkte einer
# Fassung gehört jeder Chunk zu diesem Studiengang und dieser PO (S033).
_PO_RAHMENWORTE = re.compile(
    r"\b(?:Zwei-Fächer-|Zwei-Fach-)?(?:Bachelor|Master)(?:s|studium|studiengang\w*)?\b(?!-?[Aa]rbeit)"
    r"|\b(?:Studiengang|Studiengangs|Studium|Prüfungsordnung|PO)\b"
    r"|\b(?:im|in|am|an der|der)\s+(?=[,.?!:]|$)"
    r"|\b20\d\d\b", re.I)


def _po_sachbegriff(frage: str) -> str:
    """Suchanfrage für die Nachsuche: die Frage ohne Studiengangsnamen,
    Kürzel, Fakultät, Abschluss und PO-Jahr (S033). Diese Wörter trafen in
    der Sparse-Suche Titel und Kopfzeilen statt der Stelle, die den gefragten
    Punkt regelt. Bleibt fast nichts übrig, gilt die ganze Frage."""
    rest = frage
    for _sg, m, _l, _k in sorted(_studiengang_muster(), key=lambda t: -t[2]):
        rest = m.sub(" ", rest)
    for m, _w in _FAK_MUSTER:
        rest = m.sub(" ", rest)
    rest = re.sub(r"\bFakult[äa]t\b", " ", rest)
    rest = _PO_RAHMENWORTE.sub(" ", rest)
    rest = re.sub(r"\(\s*\)|(?<!\w)-(?!\w)", " ", rest)
    rest = re.sub(r"\s+([,.?!:])", r"\1", re.sub(r"\s+", " ", rest)).strip(" ,:-")
    return rest if len(re.findall(r"\w{3,}", rest)) >= 2 else frage


# Nachsuche nach Dokumenttyp (S065, S057 Ursache 2): Die Punkte einer Fassung
# stammen oft überwiegend aus Modulhandbüchern (MAB 2019: 642 von 731; in 52
# von 264 Fassungen über die Hälfte), die 3-Chunk-Nachsuche landete dort statt
# in der PO. Gesucht wird deshalb zuerst unter den Punkten der PO-Dokumente
# (`po_dokumenttyp` „Pruefungsordnung“, Vertretungspunkte zählen mit), die
# übrigen Punkte (Modulhandbuch, Studienplan) füllen nur auf. Dazu je Treffer
# der Vorgänger-Chunk desselben Dokuments: die Nachsuche traf oft den Chunk
# hinter der Stelle (die Überschrift steht vorn, die Regel am Ende des
# vorigen Chunks). Vom Vorgänger kommt nur das Ende, zusammen höchstens
# (PO_NACHSUCHE_ZUSCHLAG − 1) × Länge der Trefferauszüge — die Eingabe wächst
# um höchstens 30 % (S054: die Eingabegröße darf den Vergleich nicht verlangsamen).
# Die Brücke zwischen dem Modulnamen der Frage und dem Modulkürzel im
# PO-Paragrafen (die PO nennt nur „BRS-212“) schlägt seit S077 die
# Modultabelle (_po_modulkuerzel); der Begleit-Treffer aus dem Modulhandbuch
# (S065) ist damit entfallen.
PO_NACHSUCHE_ZUSCHLAG = 1.3
# Obergrenze der Vergleichseingabe (alle Fassungen, Zeichen) für den Vorgänger-
# Zuschlag: Fälle mit vielen Fassungen (8 bei BRS) wuchsen sonst über den Stand
# vor S065 (max. ~20 500 Zeichen); die Treffer selbst kürzt sie nicht.
PO_NACHSUCHE_EINGABE_MAX = 20000


# Modultabelle (S077, F5): stammdaten/po/module.tsv (aus den Modulhandbüchern,
# stammdaten/po/module_ziehen.py) verbindet je Studiengang und Fassung Modulname
# und Modulkürzel. Ein Name zählt nur als ganzes Wort; ein Kompositum aus Name
# und einem der Grundwörter unten („Projektprüfung“, „Praktikumsbericht“)
# nennt das Modul ebenfalls. Längster Name zuerst, ein gefundener Name
# verdeckt die kürzeren darin („Projektorientiertes Lernen“ ≠ „Projekt“).
# Zeilen ohne Fassung bleiben draußen: dasselbe Kürzel kann in einer anderen
# Fassung ein anderes Modul sein.
MODULE_TSV = po_filter.PO_DATEN / "module.tsv"
_MODUL_GRUNDWORT = r"(?:s?(?:prüfung|modul|bericht|note|leistung)\w*)?"
PO_MODUL_KUERZEL_MAX = 3        # je Fassung an den Suchtext angehängte Kürzel


@lru_cache(maxsize=1)
def _po_module() -> dict[tuple[str, int], list[tuple[re.Pattern, list[str]]]]:
    """(Studiengang, Fassung) → [(Muster des Modulnamens, Kürzel)], längster Name zuerst."""
    namen: dict = {}
    with open(MODULE_TSV, encoding="utf-8", newline="") as fh:
        for z in csv.DictReader(fh, delimiter="\t"):
            if z["studiengang"] and z["fassung"]:
                k = (z["studiengang"], int(z["fassung"]))
                namen.setdefault(k, {}).setdefault(z["modulname"], set()).add(z["modulkuerzel"])
    return {k: [(re.compile(r"(?<!\w)" + re.escape(n) + _MODUL_GRUNDWORT + r"(?!\w)", re.I), sorted(ks))
                for n, ks in sorted(nk.items(), key=lambda t: (-len(t[0]), t[0]))]
            for k, nk in namen.items()}


def _po_modulkuerzel(text: str, studiengang: str, fassung: int) -> list[str]:
    """Kürzel der Module dieser Fassung, deren Name im Text steht (höchstens
    PO_MODUL_KUERZEL_MAX, in der Reihenfolge der Namen)."""
    kuerzel: list[str] = []
    for muster, ks in _po_module().get((studiengang, fassung), []):
        if muster.search(text):
            text = muster.sub(" ", text)
            kuerzel += [k for k in ks if k not in kuerzel]
    return kuerzel[:PO_MODUL_KUERZEL_MAX]


@lru_cache(maxsize=1)
def _po_nachsuche_karte() -> tuple[dict, dict]:
    """→ ({(Studiengang, Fassung): Punkt-IDs der PO-Dokumente},
    {(Studiengang, Fassung): {Punkt-ID: Punkt-ID des Vorgänger-Chunks}}).
    Trägt ein Punkt mehrere Chunks der Fassung (Vertretung), zählt der Chunk
    aus einem PO-Dokument."""
    idx = po_filter.lade()
    typ = {z["source_url"]: z["po_dokumenttyp"] for z in idx.zeilen}
    po_punkte: dict = {}
    vorgaenger: dict = {}
    for u, k in idx.schluessel.items():
        if k is None:
            continue
        chunks = idx.chunks.get(u, [])
        pid_an = {i: pid for i, pid, _v, _s in chunks}
        ist_po = typ.get(u) == "Pruefungsordnung"
        vg = vorgaenger.setdefault(k, {})
        for i, pid, _v, _s in chunks:
            if ist_po:
                po_punkte.setdefault(k, set()).add(pid)
            davor = pid_an.get(i - 1)
            if davor and davor != pid and (ist_po or pid not in vg):
                vg[pid] = davor
    return po_punkte, vorgaenger


def _po_nachsuche(frage: str, studiengang: str, fassungen: list[int]) -> dict[int, list[str]]:
    """Je Fassung die Auszüge für den Vergleich → {Fassung: [Text, …]}.

    Die Treffer-Chunks einer Fassung behandeln oft nicht den gefragten Punkt
    (sie kamen über eine andere Stelle in die Kandidaten, S026/S027). Deshalb
    eine eigene kleine Suche je Fassung, beschränkt auf die Punkte, die Chunks
    dieser Fassung tragen (PoIndex.punkte): Dense „text“ und Sparse
    „sparse_expanded“, RRF — mit dem Sachbegriff der Frage (S033), je Fassung
    ergänzt um die Kürzel der genannten Module (S077). Erst über die
    PO-Dokumente, aus dem Rest (Modulhandbuch, Studienplan) nur, was zum
    Auffüllen fehlt; je Treffer der Vorgänger-Chunk davor (S065).
    """
    idx = po_filter.lade()
    po_punkte, vorgaenger = _po_nachsuche_karte()
    anfrage = _po_sachbegriff(frage)

    @lru_cache(maxsize=None)
    def vektoren(text: str):
        return embed_query_dense(_server._embedder, text), embed_query_sparse(_server._sparse_embedder, text)

    def suche(text: str, pids: list[str], limit: int) -> list[str]:
        if not pids or limit <= 0:
            return []
        dense, sparse = vektoren(text)
        flt = models.Filter(must=[models.HasIdCondition(has_id=pids)])
        # Arme getrennt (eine Batch-Anfrage), je Arm nach Tie-Break (S079) geschnitten, RRF hier
        # statt in Qdrant: dort schnitt der Prefetch bei Gleichstand an der
        # Armgrenze zufällig (S082). Formel wie Qdrant 1.17 (Fusion.RRF, k=2):
        # Σ 1/(Rang ab 0 + 2) in float32 — S082 nachgerechnet, ohne Gleichstand
        # im Arm bitgleich mit Qdrants Scores.
        arme = _server._qdrant.query_batch_points(_server.COLLECTION_NAME, requests=[
            models.QueryRequest(query=vektor, using=using, filter=flt,
                                limit=PO_NACHSUCHE_ARM + GLEICHSTAND_RESERVE,
                                with_payload=["source_url"])
            for vektor, using in ((dense, "text"), (sparse, "sparse_expanded"))])
        rrf: dict[str, dict] = {}
        for antwort in arme:
            for rang, t in enumerate(nach_score(antwort.points, PO_NACHSUCHE_ARM)):
                eintrag = rrf.setdefault(str(t.id), {"id": str(t.id), "score": np.float32(0),
                                                     "payload": t.payload})
                eintrag["score"] += np.float32(1) / np.float32(rang + 2)
        return [t["id"] for t in nach_score(list(rrf.values()), limit)]

    treffer_je: dict[int, list[str]] = {}
    for f in fassungen:
        alle = set(idx.punkte.get((studiengang, f), {}))
        if not alle:
            continue
        po = alle & po_punkte.get((studiengang, f), set())
        text = " ".join([anfrage] + _po_modulkuerzel(anfrage, studiengang, f))
        treffer = suche(text, sorted(po), PO_VERGLEICH_CHUNKS)
        treffer += suche(text, sorted(alle - po), PO_VERGLEICH_CHUNKS - len(treffer))
        treffer_je[f] = treffer

    # Vorgänger nur, wenn nicht selbst Treffer; Texte in einem Aufruf holen
    vor_je = {f: vorgaenger.get((studiengang, f), {}) for f in treffer_je}
    ids = sorted({p for f, tr in treffer_je.items()
                  for p in tr + [vor_je[f].get(t) for t in tr] if p})
    text = {str(p.id): (p.payload or {}).get("text", "")
            for p in (_server._qdrant.retrieve(_server.COLLECTION_NAME, ids=ids, with_payload=["text"])
                      if ids else [])}

    # Vorgänger: je Fassung höchstens (ZUSCHLAG − 1) × Länge der Trefferauszüge,
    # über alle Fassungen zusammen nur bis PO_NACHSUCHE_EINGABE_MAX (anteilig gekürzt)
    treffer_je = {f: [t for t in tr if text.get(t)] for f, tr in treffer_je.items()}
    laenge = {f: sum(min(len(text[t]), PO_VERGLEICH_ZEICHEN) for t in tr)
              for f, tr in treffer_je.items()}
    budget = {f: int(n * (PO_NACHSUCHE_ZUSCHLAG - 1)) for f, n in laenge.items()}
    frei = max(PO_NACHSUCHE_EINGABE_MAX - sum(laenge.values()), 0)
    if sum(budget.values()) > frei:
        budget = {f: int(b * frei / sum(budget.values())) for f, b in budget.items()}

    texte: dict[int, list[str]] = {}
    for f, tr in treffer_je.items():
        # bessere Treffer zuerst; vom Vorgänger zählt das Ende (nahe am Treffer)
        rest = budget[f]
        gesehen = set(tr)
        for t in tr:
            auszug = text[t][:PO_VERGLEICH_ZEICHEN]
            v = vor_je[f].get(t)
            if v and v not in gesehen and text.get(v) and rest > 0:
                gesehen.add(v)
                davor = text[v][-min(rest, PO_VERGLEICH_ZEICHEN):]
                rest -= len(davor)
                auszug = davor + "\n" + auszug
            texte.setdefault(f, []).append(auszug)
    return texte


def _po_jahreswerte(roh, verglichen: list[int]) -> dict[int, str]:
    """{"2021": "…", …} aus der LLM-Antwort → {Fassung: Wert}, nur verglichene, nicht leer."""
    werte: dict[int, str] = {}
    for k, w in (roh.items() if isinstance(roh, dict) else []):
        jahr = re.search(r"20\d\d", str(k))
        if jahr and int(jahr.group(0)) in verglichen and str(w or "").strip():
            werte[int(jahr.group(0))] = str(w).strip()[:120]
    return werte


def _po_urteil_lesen(antwort: str, verglichen: list[int]) -> tuple[str, dict[int, str], dict[int, str]]:
    """LLM-Antwort → (Urteil, {Fassung: Wert}, {Fassung: Zusatzbedingung}).
    Der Code vergleicht keine Werte (S030 scheiterte an „12 3“ ≠ „12 credits“);
    er prüft nur, ob das Urteil zu den genannten Werten passen kann:
    „verschieden“, „zusatzbedingung“ und „gleich“ brauchen Werte aus
    mindestens zwei Fassungen, sonst „unklar“; „zusatzbedingung“ ohne
    genannte Bedingung ist „gleich“."""
    daten = {}
    m = re.search(r"\{.*\}", antwort or "", re.S)
    if m:
        try:
            daten = json.loads(m.group(0))
        except ValueError:
            daten = {}
    if not isinstance(daten, dict):
        daten = {}
    werte = _po_jahreswerte(daten.get("werte"), verglichen)
    zusatz = _po_jahreswerte(daten.get("zusatz"), verglichen)
    urteil = daten.get("urteil")
    if urteil not in PO_URTEILE:
        u = re.search(r'"urteil"\s*:\s*"(' + "|".join(PO_URTEILE) + r')"', antwort or "")
        urteil = u.group(1) if u else "unklar"
    if urteil != "unklar" and len(werte) < 2:
        urteil = "unklar"
    if urteil == "zusatzbedingung" and not zusatz:
        urteil = "gleich"
    if urteil != "zusatzbedingung":
        zusatz = {}
    return urteil, werte, zusatz


PO_URTEILE = ("verschieden", "zusatzbedingung", "gleich", "unklar")


def _po_vergleich_nachrichten(frage: str, vielfalt: dict) -> tuple[list[dict] | None, list[int]]:
    """Nachsuche je Fassung → (Nachrichten für den Vergleichsaufruf, verglichene
    Fassungen). Findet sich für eine Fassung nichts, wird sie nicht verglichen;
    bleiben weniger als zwei, gibt es keine Nachrichten."""
    texte = _po_nachsuche(frage, vielfalt["studiengang"], vielfalt["fassungen"])
    bloecke, verglichen = [], []
    for f in vielfalt["fassungen"]:
        if not texte.get(f):
            continue
        verglichen.append(f)
        auszuege = "\n\n".join(texte[f])   # schon gekürzt (_po_nachsuche)
        bloecke.append(f"[PO {vielfalt['studiengang']} Fassung {f}]\n{auszuege}")
    if len(verglichen) < 2:
        return None, verglichen
    return [
        {"role": "system", "content": PO_VERGLEICH_SYSTEM},
        {"role": "user", "content": f"Frage: {frage}\n\n" + "\n\n---\n\n".join(bloecke)},
    ], verglichen


async def _po_entscheiden(frage: str, vielfalt: dict, modell: str) -> dict:
    """Fassungsvergleich mit Zeitgrenze → {urteil, verglichen, werte, zusatz,
    nicht_verglichen, dauer_s, versuche[, grund]}.

    Nachsuche einmal, dann LLM-Aufrufe (JSON, temperature 0): ohne Antwort nach
    PO_VERGLEICH_ABSTAND_S startet ein weiterer paralleler Versuch (höchstens
    PO_VERGLEICH_VERSUCHE), die erste Antwort zählt (S054). Scheitert ein
    Versuch mit Fehler, startet der nächste sofort. Zeitüberschreitung und
    Fehler ergeben „unklar“ mit Grund."""
    t_vergleich = time.perf_counter()
    grund, versuche = None, 0
    urteil, verglichen, werte, zusatz = "unklar", [], {}, {}
    laufend: set[asyncio.Future] = set()
    nummer: dict[asyncio.Future, int] = {}
    loop = asyncio.get_running_loop()
    try:
        nachrichten, verglichen = await asyncio.wait_for(
            asyncio.to_thread(_po_vergleich_nachrichten, frage, vielfalt),
            timeout=PO_VERGLEICH_GRENZE_S / 2)
        ende = t_vergleich + PO_VERGLEICH_GRENZE_S
        naechster = time.perf_counter()
        antwort, fehler = None, None
        while nachrichten:
            jetzt = time.perf_counter()
            if (versuche < PO_VERGLEICH_VERSUCHE and jetzt < ende
                    and (jetzt >= naechster or not laufend)):
                versuche += 1
                t = loop.run_in_executor(_po_vergleich_pool, _server._ask_llm, modell, nachrichten,
                                         ende - jetzt)
                nummer[t] = versuche
                laufend.add(t)
                print(f"  ⏱ PO-Vergleich: Versuch {versuche} gestartet nach "
                      f"{jetzt - t_vergleich:.1f}s", flush=True)
                naechster = jetzt + PO_VERGLEICH_ABSTAND_S
            if not laufend or jetzt >= ende:
                grund = grund or "zeit"
                break
            frist = ende - jetzt
            if versuche < PO_VERGLEICH_VERSUCHE:
                frist = min(frist, max(naechster - jetzt, 0.0))
            fertig, laufend = await asyncio.wait(laufend, timeout=frist,
                                                 return_when=asyncio.FIRST_COMPLETED)
            for t in fertig:
                print(f"  ⏱ PO-Vergleich: Antwort nach {time.perf_counter() - t_vergleich:.1f}s "
                      f"(Versuch {nummer[t]})"
                      + (f" ({type(t.exception()).__name__})" if t.exception() else ""), flush=True)
                if t.exception() is None:
                    antwort = t.result()[0]
                    break
                fehler = t.exception()
            if antwort is not None:
                break
            if fehler is not None and not laufend:
                grund = f"fehler:{type(fehler).__name__}"
                if versuche >= PO_VERGLEICH_VERSUCHE:
                    break
                grund = None
        if antwort is not None:
            urteil, werte, zusatz = _po_urteil_lesen(antwort, verglichen)
            grund = None
        elif nachrichten:
            grund = grund or "zeit"
        if grund:
            verglichen = []
    except asyncio.TimeoutError:
        verglichen, grund = [], "zeit"
    except Exception as e:  # Qdrant, Embedding, LLM — Art vermerken, nicht still schlucken
        urteil, verglichen, werte, zusatz, grund = "unklar", [], {}, {}, f"fehler:{type(e).__name__}"
    finally:
        # Hängende Versuche: Ihr Ergebnis wird verworfen; die Threads enden spätestens
        # mit dem Client-Timeout an der Grenze (S056).
        for t in laufend:
            t.cancel()
    entscheidung = {
        "urteil": urteil, "verglichen": verglichen, "werte": werte, "zusatz": zusatz,
        "nicht_verglichen": [f for f in vielfalt["fassungen"] if f not in verglichen],
        "dauer_s": round(time.perf_counter() - t_vergleich, 2), "versuche": versuche}
    if grund:
        entscheidung["grund"] = grund
    return entscheidung


def _po_hinweis(studiengang: str, entscheidung: dict | None) -> str:
    """Einzeiliger Kontext-Hinweis für das Antwort-LLM bei Urteil
    „zusatzbedingung“ (S039): gleicher Wert, Bedingung samt Fassung. Sonst ""."""
    if not entscheidung or entscheidung.get("urteil") != "zusatzbedingung":
        return ""
    werte = entscheidung.get("werte") or {}
    # §-Angaben weglassen: Sie unterscheiden sich zwischen den Fassungen.
    wert = re.sub(r"\s*\(?§[^)]*\)?", "", werte[min(werte)]).strip() if werte else ""
    bedingungen = "; ".join(f"nach PO {f} zusätzlich: {b}"
                            for f, b in sorted(entscheidung["zusatz"].items()))
    jahre = ", ".join(str(f) for f in sorted(werte))
    # Langform statt Kürzel: interne Kürzel gehören nicht in die Antwort.
    name = _studiengang_langform(studiengang)
    return (f"Hinweis aus dem Vergleich der Prüfungsordnungen im Studiengang {name} "
            f"(Fassungen {jahre}): Der gefragte Wert ist in allen gleich ({wert}); "
            f"{bedingungen}.")


# Gewählte bzw. aus der Frage erkannte PO (S048). Nennt die Frage die Fassung
# nicht selbst (Folgefrage nach einer Rückfrage), wussten Auswahl und Antwort
# nichts von ihr: bei BVM 2016 stand PO_BVM_2016.pdf auf Kandidatenrang 8/9,
# die Auswahl nahm das aktuelle Curriculum. Deshalb (1) eine Zeile zur Fassung
# für beide LLM-Runden und (2) wenigstens ein Dokument der Fassung in der
# Auswahl, sofern eines in den Kandidaten steht.
def _po_gewaehlt_zeile(po: dict | None) -> str:
    """„Gewählte Prüfungsordnung: <Langform>, Fassung <Jahr>.“ oder ""."""
    if not po:
        return ""
    name = _studiengang_langform(po["studiengang"])
    return (f"Gewählte Prüfungsordnung: {name}, Fassung {po['fassung']} — "
            f"die Antwort muss aus dieser Fassung stammen.")


def _po_auswahl_ergaenzen(selected: list[int], candidate_urls: list[str],
                          po: dict | None) -> list[int]:
    """Fehlt in der Auswahl ein Dokument der gewählten Fassung, kommt der
    bestplatzierte Kandidat dieser Fassung an den Anfang. Bevorzugt URLs, die nur zu
    dieser Fassung gehören; eine Konflikt-URL (E004) trägt oft nur geteilte
    Chunks und den Text einer anderen Fassung."""
    if not po:
        return selected
    sg, fassung = po["studiengang"], int(po["fassung"])
    menge = po_filter.lade().urls.get((sg, fassung), set())
    eindeutig = _po_url_fassungen()
    for nur_eindeutig in (True, False):
        passt = [i for i, u in enumerate(candidate_urls, 1) if u in menge
                 and (not nur_eindeutig or eindeutig.get(u, {}).get(sg) == fassung)]
        if passt:
            if any(i in passt for i in selected):
                return selected
            # vorn statt hinten: _fit_to_budget nimmt das erste Dokument immer mit
            return [passt[0]] + selected
    return selected


# Fassung im Antwortkontext (S049, Hinweis Christian 16:46). Bisher stand je
# Dokument nur Titel, URL, Fakultät und Stand; zu welcher PO-Fassung ein
# Verkündungsblatt oder eine Änderungssatzung gehört, musste das Modell raten.
# Jedes Dokument, das der PoIndex einer Fassung zuordnet, bekommt deshalb eine
# Kopfzeile mit Langform, Kürzel und Fassung und, wenn po_geltung.tsv es
# belegt, dem Geltungsende. Metadatum: geht nicht in die Belegprüfung (die
# sieht nur Titel und Text).
# Zuordnung: die eigene Zeile der URL in der PO-Zuordnung; dazu jede Fassung,
# deren Dokument die URL überwiegend vertritt (wortgleiche Dublette). Vertritt
# sie von einer anderen Fassung desselben Studiengangs nur einzelne Abschnitte
# (Konflikt-URL, E004: PO_BA_BVM_2026 trägt 1–3 Chunks von 2018/2022), wird
# diese Fassung als „wortgleiche Abschnitte“ genannt, nicht als eigene —
# sonst stünde über der PO 2026 „Fassungen 2018, 2022, 2026“. Einzelne geteilte
# Abschnitte fremder Studiengänge (Allgemeiner Teil) bleiben ungenannt.
@lru_cache(maxsize=1)
def _po_url_zuordnung() -> dict[str, dict[str, tuple[list[int], list[int]]]]:
    """URL → {Studiengang: (Fassungen des Dokuments, Fassungen mit wortgleichen
    Abschnitten)}."""
    idx = po_filter.lade()
    eigen: dict[str, dict[str, set]] = {}
    teil: dict[str, dict[str, set]] = {}
    for u, k in idx.schluessel.items():
        if k is None:
            continue
        eigen.setdefault(u, {}).setdefault(k[0], set()).add(k[1])
        chunks = idx.chunks.get(u, [])
        je_vertreter: dict[str, int] = {}
        for _, _, v, _ in chunks:
            je_vertreter[v] = je_vertreter.get(v, 0) + 1
        for v, n in je_vertreter.items():
            ziel = eigen if 2 * n >= len(chunks) else teil
            ziel.setdefault(v, {}).setdefault(k[0], set()).add(k[1])
    zuordnung = {}
    for u, m in eigen.items():
        zuordnung[u] = {sg: (sorted(fs), sorted(teil.get(u, {}).get(sg, set()) - fs))
                        for sg, fs in m.items()}
    return zuordnung


@lru_cache(maxsize=1)
def _po_geltung_eintraege() -> dict[tuple[str, int], tuple[str, date]]:
    """(Studiengang, Fassung) → (Art, Datum) des Geltungsendes aus po_geltung.tsv
    (uebergang_bis/ausser_kraft; mehrere Fristen: die späteste, wie PoIndex)."""
    eintraege: dict[tuple[str, int], tuple[str, date]] = {}
    if not po_filter.GELTUNG.exists():
        return eintraege
    with open(po_filter.GELTUNG, encoding="utf-8") as f:
        for z in csv.DictReader(f, delimiter="\t"):
            if z["art"] not in ("uebergang_bis", "ausser_kraft"):
                continue
            k, d = (z["studiengang"], int(z["fassung"])), date.fromisoformat(z["datum"])
            if k not in eintraege or d > eintraege[k][1]:
                eintraege[k] = (z["art"], d)
    return eintraege


def _po_geltung_text(sg: str, fassung: int, stichtag: date | None = None) -> str:
    """„gilt übergangsweise bis …“ / „nicht mehr gültig seit …“ oder ""."""
    eintrag = _po_geltung_eintraege().get((sg, fassung))
    if not eintrag:
        # ohne eigene Zeile: abgelaufen, wenn eine jüngere Fassung es ist (S033, monoton)
        return ("nicht mehr gültig" if po_filter.lade().abgelaufen(sg, fassung, stichtag or _po_stichtag())
                else "")
    art, d = eintrag
    if d < (stichtag or _po_stichtag()):
        return f"nicht mehr gültig seit {d:%d.%m.%Y}"
    return (f"gilt übergangsweise bis {d:%d.%m.%Y}" if art == "uebergang_bis"
            else f"gilt bis {d:%d.%m.%Y}")


def _po_kopfzeile(url: str, stichtag: date | None = None) -> str:
    """„Prüfungsordnung: <Langform> (<Kürzel>), Fassung <Jahr> (<Geltung>)“ oder ""."""
    def mit_geltung(sg: str, fassungen: list[int]) -> str:
        jahre = []
        for f in fassungen:
            geltung = _po_geltung_text(sg, f, stichtag)
            jahre.append(f"{f} ({geltung})" if geltung else str(f))
        return ("Fassung " if len(jahre) == 1 else "Fassungen ") + ", ".join(jahre)

    teile = []
    for sg, (fassungen, abschnitte) in sorted(_po_url_zuordnung().get(url, {}).items()):
        name = _studiengang_langform(sg)
        text = f"{name} ({sg}), {mit_geltung(sg, fassungen)}"
        if abschnitte:
            text += f"; enthält auch wortgleiche Abschnitte der {mit_geltung(sg, abschnitte)}"
        teile.append(text)
    return ("Prüfungsordnung: " + "; ".join(teile)) if teile else ""


# Geltung (S033, V007): Eine Fassung, deren Übergangsfrist oder Geltung laut
# stammdaten/po/po_geltung.tsv (S038) vor dem Stichtag endet, geht nicht in den
# Fassungsvergleich und nicht in die Rückfrage-Optionen; ebenso alle älteren
# Fassungen desselben Studiengangs (monoton, PoIndex.abgelaufen). Fassungen
# ohne Eintrag und ohne abgelaufene jüngere gelten als nicht abgelaufen.
# Aufgefüllt und angeboten werden nur Fassungen mit eigentlichem PO-Dokument
# (nicht reine Modulhandbuch-Fassungen). Stichtag für Tests per Umgebung,
# z. B. HSH_PO_STICHTAG=2026-08-01; sonst das heutige Datum.
def _po_stichtag() -> date:
    wert = os.getenv("HSH_PO_STICHTAG", "").strip()
    return date.fromisoformat(wert) if wert else date.today()


def _po_geltende(vielfalt: dict, stichtag: date | None = None) -> tuple[list[int], list[int]]:
    """→ (zu vergleichende, abgelaufene) Fassungen der Vielfalt zum Stichtag.

    Bleiben aus den Kandidaten weniger als zwei geltende Fassungen, kommen die
    übrigen geltenden Fassungen des Studiengangs aus dem PoIndex dazu (die
    Nachsuche braucht keine Kandidaten): Nur wenn der Studiengang insgesamt
    höchstens eine geltende Fassung hat, wird ohne Vergleich geantwortet.
    """
    idx = po_filter.lade()
    stichtag = stichtag or _po_stichtag()
    sg = vielfalt["studiengang"]
    abgelaufen = [f for f in vielfalt["fassungen"] if idx.abgelaufen(sg, f, stichtag)]
    geltend = [f for f in vielfalt["fassungen"] if f not in abgelaufen]
    if abgelaufen and len(geltend) < 2:
        geltend = [f for f in idx.fassungen(sg, nur_po=True) if not idx.abgelaufen(sg, f, stichtag)]
    return geltend, abgelaufen


def _po_rueckfrage(vielfalt: dict, stichtag: date | None = None) -> dict:
    """Rückfrage mit den vorhandenen, noch geltenden Fassungen als Auswahl (ZIEL.md).

    Nicht mehr aktuelle, aber noch geltende Fassungen und ausgelaufene
    Studiengänge bleiben in der Liste (F1 Punkt 5); Fassungen, deren Geltung vor
    dem Stichtag endet, fallen weg (S033). Im Text heißt nur die neueste Fassung
    „aktuell“, die übrigen werden nicht als „älter“ markiert — sie gelten je nach
    Studienbeginn weiter (S096); bei ausgelaufenen Studiengängen sagt der Text das.
    """
    idx = po_filter.lade()
    stichtag = stichtag or _po_stichtag()
    sg = vielfalt["studiengang"]
    name = _studiengang_langform(sg)
    optionen = [{"studiengang": sg, "name": name, "fassung": f,
                 "aktuell": bool(idx.aktuell.get((sg, f))) and sg not in idx.ausgelaufen}
                for f in idx.fassungen(sg, nur_po=True) if not idx.abgelaufen(sg, f, stichtag)]
    liste = ", ".join(f"PO {o['fassung']}" + (" (aktuelle Fassung)" if o["aktuell"] else "")
                      for o in reversed(optionen))
    liste += (" — der Studiengang läuft aus, alle gelten noch je nach Studienbeginn"
              if sg in idx.ausgelaufen else " — alle noch gültig, je nach Studienbeginn")
    # Langform mit Kürzel (S084): das Kürzel allein ist intern (ZIEL.md, Ziel 1)
    name = f"{name} ({sg})" if name != sg else sg
    return {"frage": (f"Die Antwort hängt davon ab, nach welcher Prüfungsordnung Sie "
                      f"im Studiengang {name} studieren. Welche Fassung gilt für Sie? "
                      f"Vorhanden: {liste}."),
            "optionen": optionen}


# Tabellenurteil (S088, F6): Hat der Studiengang Zeilen in stammdaten/po/po_werte.tsv,
# entscheidet die Wertetabelle vor dem LLM-Fassungsvergleich (`_po_entscheiden`):
# „verschieden“ → Rückfrage, „gleich“ → direkte Antwort, „keins“ → bisheriger Weg.
# Verglichen werden alle geltenden PO-Fassungen des Studiengangs (PoIndex), nicht
# nur die der Kandidaten — sonst könnte „gleich“ auf einer Teilmenge entstehen.
# „gleich“ zählt nur, wenn eine Fundstelle der Tabelle unter den Kandidaten steht
# (die Antwort braucht den Beleg); sonst „keins“.
# S091: Nennt die Frage einen anderen Studiengang (`familie` nicht leer und ohne
# `studiengang`), gilt die Tabelle nicht — die Vielfalt kommt dann nur aus den
# Kandidaten. Fehlt eine Tabelle oder ist sie defekt, ebenfalls „keins“
# (bisheriger Weg) statt einer Ausnahme im Antwortpfad.
def _po_tabellenurteil(frage: str, studiengang: str, kandidaten: list[dict],
                       familie: list[str] | None = None,
                       stichtag: date | None = None) -> dict:
    """→ po_werte.urteil(...) über die geltenden Fassungen, dazu beleg_url
    (bestplatzierte Fundstelle unter den Kandidaten) bei „gleich“."""
    keins = {"urteil": "keins", "punkt": None, "punkte": [], "werte": {}, "ohne_zeile": [],
             "beleg_url": None}
    if familie and studiengang not in familie:
        return keins
    idx = po_filter.lade()
    geltend, _abgelaufen = _po_geltende(
        {"studiengang": studiengang, "fassungen": idx.fassungen(studiengang, nur_po=True)}, stichtag)
    try:
        tabelle = po_werte.urteil(frage, studiengang, geltend)
        urls = (set(po_werte.fundstellen(studiengang, tabelle["punkt"]))
                if tabelle["urteil"] == "gleich" else set())
    except (OSError, KeyError, ValueError, csv.Error) as exc:
        print(f"  ⚠ PO-Wertetabelle nicht lesbar ({type(exc).__name__}) → keins", flush=True)
        return keins
    tabelle["beleg_url"] = None
    if tabelle["urteil"] == "gleich":
        beleg = min((k for k in kandidaten if k["url"] in urls), key=lambda k: k["nr"], default=None)
        if beleg:
            tabelle["beleg_url"] = beleg["url"]
        else:
            tabelle["urteil"] = "keins"
    return tabelle
