#!/usr/bin/env python3
"""
PO-Filter über URL-Mengen statt Payload (S016, E004).

Lädt stammdaten/po/po_zuordnung.tsv und bildet je (Studiengang, Fassung) die
Qdrant-`source_url`s, unter denen die Suche die Dokumente dieser Fassung
tatsächlich findet: eigene Punkte, sonst die Vertreter-URL aus
doc_index.db.chunk_vertretung (chunkgenau). Qdrant wird nicht verändert; der
Filter läuft über den vorhandenen `source_url`-Index.

Regeln (E004):
- Ein PO-Dokument gehört zu genau einem Schlüssel (Studiengang, Fassung) aus
  der TSV. Dokumente ohne Studiengang (Allgemeine Teile) gehören zu keinem
  Studiengang und werden nie ausgeschlossen.
- Die Menge einer Fassung enthält die eigenen URLs ihrer Dokumente und jede
  URL, die Chunks ihrer Dokumente vertritt.
- Konflikt: eine URL, die zu mehreren Fassungen desselben Studiengangs gehört
  (eigene Fassung + vertretene Chunks einer anderen, oder Vertreter für
  mehrere Fassungen). Sie wird keiner Fassung allein zugeschlagen: Der
  Filter einer Fassung schließt sie aus, lässt aber die Punkte stehen, die
  Chunks der gewünschten Fassung vertreten (`HasId`-Ausnahme).

Aufruf:  python po_filter.py --pruefen [--fehlend datei] [--pruefung faelle.json]

Laufzeitdaten liegen in stammdaten/po/ (S036). Die Prüfung (4) liest die
po_gegeben-Fälle nur, wenn `--pruefung` sie angibt (PO-Testmenge,
auswertung/po/pruefung.json) — der Server liest nichts aus auswertung/gold/.
"""
import argparse
import csv
import json
import os
import sqlite3
from collections import defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path

from qdrant_client import models

from raghsh import konfig

PO_DATEN = konfig.PO_DATEN
ZUORDNUNG = PO_DATEN / "po_zuordnung.tsv"
DOC_INDEX = konfig.DOC_INDEX
GELTUNG = PO_DATEN / "po_geltung.tsv"
QDRANT_URL = os.environ.get("HSH_QDRANT_URL", "http://localhost:6333")
COLLECTION = "hsh_knowledge"

# F1 Punkt 5: Kürzel, deren jüngste Fassung vor diesem Jahr liegt, gelten als
# ausgelaufen (nie aktuell, bei „Neubeginn“ nicht anbieten). Wie extrahiere_po.py.
AUSGELAUFEN_VOR = 2016


class PoIndex:
    """URL-Mengen und Vertretungspunkte je (Studiengang, Fassung)."""

    def __init__(self, zuordnung=ZUORDNUNG, doc_index=DOC_INDEX, geltung=GELTUNG):
        self.zeilen = list(csv.DictReader(open(zuordnung, encoding="utf-8"), delimiter="\t"))
        self.schluessel = {}                  # TSV-URL -> (sg, fassung) oder None
        self.aktuell = {}                     # (sg, fassung) -> bool
        self.mit_po = set()                   # (sg, fassung) mit eigentlichem PO-Dokument
        juengste = defaultdict(int)
        for z in self.zeilen:
            if z["po_studiengang"] and z["po_fassung"]:
                k = (z["po_studiengang"], int(z["po_fassung"]))
                self.schluessel[z["source_url"]] = k
                if z["po_dokumenttyp"] == "Pruefungsordnung":
                    self.mit_po.add(k)
                self.aktuell[k] = self.aktuell.get(k, False) or z["po_aktuell"] == "true"
                juengste[k[0]] = max(juengste[k[0]], k[1])
            else:
                self.schluessel[z["source_url"]] = None
        self.ausgelaufen = {sg for sg, j in juengste.items() if j < AUSGELAUFEN_VOR}

        # chunk_vertretung nur für die TSV-Dokumente laden
        con = sqlite3.connect(doc_index)
        self.chunks = defaultdict(list)       # TSV-URL -> [(chunk_index, punkt_id, vertreter_url, selbst)]
        tsv = set(self.schluessel)
        for u, i, pid, v, s in con.execute(
                "SELECT source_url, chunk_index, vertreten_durch, vertreter_url, selbst FROM chunk_vertretung"):
            if u in tsv:
                self.chunks[u].append((i, pid, v or u, s))
        con.close()

        self.urls = defaultdict(set)          # (sg, f) -> Qdrant-URLs
        self.punkte = defaultdict(dict)       # (sg, f) -> {Punkt-ID: URL} der Punkte, die Chunks dieser Fassung tragen
        for u, k in self.schluessel.items():
            if k is None:
                continue
            for i, pid, v, s in self.chunks.get(u, []):
                self.urls[k].add(v)
                self.punkte[k][pid] = v

        # Konflikte: URL in mehreren Fassungen desselben Studiengangs
        je_url = defaultdict(set)
        for (sg, f), us in self.urls.items():
            for v in us:
                je_url[(sg, v)].add(f)
        self.konflikte = {(sg, v): fs for (sg, v), fs in je_url.items() if len(fs) > 1}

        # Geltungsende je Fassung (S038/S033): nur uebergang_bis und
        # ausser_kraft — in_kraft ist als Beginn unzuverlässig. Fassungen ohne
        # Zeile sind „unbekannt“ und gelten nicht als abgelaufen.
        self.geltung_bis = {}                 # (sg, f) -> date
        if Path(geltung).exists():
            for z in csv.DictReader(open(geltung, encoding="utf-8"), delimiter="\t"):
                if z["art"] in ("uebergang_bis", "ausser_kraft"):
                    k = (z["studiengang"], int(z["fassung"]))
                    d = date.fromisoformat(z["datum"])
                    # mehrere Fristen: die späteste zählt (eher zu wenig weglassen)
                    self.geltung_bis[k] = max(d, self.geltung_bis.get(k, d))

    # ── Abfragen ──────────────────────────────────────────────────────────
    def fassungen(self, studiengang, nur_po=False):
        """Fassungen des Studiengangs; `nur_po`: nur solche mit eigentlichem
        PO-Dokument (Dokumenttyp „Pruefungsordnung“, Besonderer Teil oder
        Gesamtfassung) — Fassungen, die nur aus Modulhandbüchern oder
        Studienplänen bestehen (z. B. MAB 2010), sind keine eigene PO (S033)."""
        return sorted(f for sg, f in self.urls
                      if sg == studiengang and (not nur_po or (sg, f) in self.mit_po))

    def abgelaufen(self, studiengang, fassung, stichtag=None):
        """True, wenn die Geltung der Fassung oder einer jüngeren Fassung
        desselben Studiengangs vor dem Stichtag (Standard: heute) endet.

        Übergangsfristen sind monoton (S033): Ist eine Fassung abgelaufen,
        sind es alle älteren auch, auch ohne eigene Zeile in po_geltung.tsv.
        """
        stichtag = stichtag or date.today()
        return any(bis < stichtag for (sg, f), bis in self.geltung_bis.items()
                   if sg == studiengang and f >= fassung)

    def ausschluss(self, studiengang, fassung):
        """URLs der anderen Fassungen desselben Studiengangs (inkl. Konflikt-URLs)."""
        andere = set()
        for f in self.fassungen(studiengang):
            if f != fassung:
                andere |= self.urls[(studiengang, f)]
        return andere

    def filter(self, studiengang, fassung):
        """Qdrant-Filter: alles außer den anderen Fassungen; Punkte, die Chunks
        der gewünschten Fassung tragen, bleiben auch auf geteilten URLs erhalten."""
        k = (studiengang, fassung)
        if k not in self.urls:
            raise KeyError(f"Keine Fassung {fassung} für {studiengang}")
        aus = self.ausschluss(studiengang, fassung)
        if not aus:
            return None
        innen = [models.FieldCondition(key="source_url", match=models.MatchAny(any=sorted(aus)))]
        # Punkte mit Chunks der gewünschten Fassung, die auf ausgeschlossenen URLs liegen
        geteilt = [pid for pid, v in self.punkte[k].items() if v in aus]
        return models.Filter(must_not=[models.Filter(
            must=innen,
            must_not=[models.HasIdCondition(has_id=sorted(geteilt))] if geteilt else None)])

    def abdeckung(self):
        """Je TSV-Dokument: eigen / nur vertreten / fehlend, dazu Konfliktart."""
        tab = {}
        for u, k in self.schluessel.items():
            ch = self.chunks.get(u, [])
            eigen = any(s for _, _, _, s in ch)
            fremd = {v for _, _, v, s in ch if not s and v != u}
            status = "eigen" if eigen else ("vertreten" if fremd else "fehlt")
            arten = set()
            if k:
                for v in fremd:
                    kv = self.schluessel.get(v)
                    if kv and kv[0] == k[0] and kv[1] != k[1]:
                        arten.add("andere_fassung")
                    elif kv and kv[0] != k[0]:
                        arten.add("anderer_studiengang")
                    elif kv is None and v in self.schluessel:
                        arten.add("allgemeiner_teil")
            # Rangfolge: die für den Filter schwerste Art zählt
            konflikt = next((a for a in ("andere_fassung", "anderer_studiengang", "allgemeiner_teil")
                             if a in arten), "")
            tab[u] = (status, konflikt)
        return tab


@lru_cache(maxsize=1)
def lade():
    return PoIndex()


# ── CLI ───────────────────────────────────────────────────────────────────
def _pruefen(fehlend_datei, pruefung=None):
    from collections import Counter
    from qdrant_client import QdrantClient
    idx = lade()
    tab = idx.abdeckung()
    mit_sg = [u for u, k in idx.schluessel.items() if k]
    print(f"Dokumente in TSV: {len(tab)} (mit Studiengang/Fassung: {len(mit_sg)})")
    for teil, urls in (("alle", list(tab)), ("mit Studiengang", mit_sg)):
        c = Counter(tab[u][0] for u in urls)
        print(f"  {teil:16s} eigen {c['eigen']} · nur vertreten {c['vertreten']} · fehlend {c['fehlt']}")
    c = Counter((tab[u][0], tab[u][1]) for u in mit_sg if tab[u][1])
    print("Dokumente mit Vertretung aus fremdem Schlüssel (Status, Art):", dict(c))
    print(f"Konflikt-URLs (mehrere Fassungen desselben Studiengangs): {len(idx.konflikte)} "
          f"in {len({sg for sg, _ in idx.konflikte})} Studiengängen")
    fehlend = sorted(u for u, (s, _) in tab.items() if s == "fehlt")
    if fehlend_datei:
        Path(fehlend_datei).write_text("".join(u + "\n" for u in fehlend), encoding="utf-8")
    print(f"Wirklich fehlend: {len(fehlend)}", *fehlend, sep="\n  ")

    client = QdrantClient(url=QDRANT_URL, timeout=60)
    info = client.get_collection(COLLECTION)
    print(f"Qdrant: {info.points_count} Punkte, Status {info.status}, Indizes {sorted(info.payload_schema)}")

    # (4) po_gegeben-Fälle: Gold-Punkte unter dem Filter der genannten Fassung
    # vorhanden, unter dem Filter jeder anderen Fassung nicht (außer geteilt)
    # (nur mit --pruefung; Testfälle sind keine Laufzeitdaten)
    faelle = [f for f in json.load(open(pruefung, encoding="utf-8"))["faelle"]
              if f["kategorie"] == "po_gegeben"] if pruefung else []
    ok = 0
    for fall in faelle:
        sg, fa = fall["po"]["studiengang"], fall["po"]["fassung"]
        gold = [u for x in fall["fassungen"] if x["fassung"] == fa for u in x["urls"]]
        # eigene Punkte der Gold-URLs; geteilte = textgleiche Chunks auf fremden URLs
        eig_pids = sorted({pid for u in gold for _, pid, v, s in idx.chunks.get(u, []) if s and v == u})
        get_pids = sorted({pid for u in gold for _, pid, v, s in idx.chunks.get(u, []) if not s} - set(eig_pids))
        in_eig = all(idx.schluessel.get(u) == (sg, fa) for u in gold)
        drin = _zaehle(client, idx.filter(sg, fa), eig_pids + get_pids)
        # ausgenommen: Gold-Punkte, die zugleich Chunks der anderen Fassung tragen (textgleich)
        fremd = {f: _zaehle(client, idx.filter(sg, f), sorted(set(eig_pids) - set(idx.punkte[(sg, f)])))
                 for f in idx.fassungen(sg) if f != fa}
        falsch = {f: n for f, n in fremd.items() if n}
        gut = in_eig and drin == len(eig_pids) + len(get_pids) and not falsch
        ok += gut
        print(f"  {fall['id']} {sg} {fa}: TSV-Schlüssel {'ok' if in_eig else 'ABW'}, Gold-Punkte "
              f"(eigen+geteilt) {drin}/{len(eig_pids) + len(get_pids)} unter eigenem Filter; "
              f"eigene Gold-Punkte unter anderen Fassungen {falsch or 0}{'' if gut else '  ← ABW'}")
    print(f"(4) po_gegeben: {ok}/{len(faelle)}")

    # (5) je Studiengang mit ≥ 2 Fassungen: count mit Filter > 0
    mehr = sorted({sg for sg, _ in idx.urls if len(idx.fassungen(sg)) >= 2})
    leer = []
    for sg in mehr:
        for f in idx.fassungen(sg):
            flt = idx.filter(sg, f)
            eigene = models.Filter(must=[models.FieldCondition(
                key="source_url", match=models.MatchAny(any=sorted(idx.urls[(sg, f)])))],
                must_not=flt.must_not if flt else None)
            if client.count(COLLECTION, count_filter=eigene, exact=True).count == 0:
                leer.append((sg, f))
    print(f"(5) Studiengänge mit ≥ 2 Fassungen: {len(mehr)}, Fassungen gesamt "
          f"{sum(len(idx.fassungen(s)) for s in mehr)}; Fassungen ohne Punkte unter ihrem Filter: "
          f"{len(leer)} {leer}")
    print(f"Ausgelaufen (< {AUSGELAUFEN_VOR}): {sorted(idx.ausgelaufen)}")


def _zaehle(client, flt, pids):
    if not pids:
        return 0
    bed = models.Filter(must=[models.HasIdCondition(has_id=pids)],
                        must_not=flt.must_not if flt else None)
    return client.count(COLLECTION, count_filter=bed, exact=True).count


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pruefen", action="store_true", help="Abdeckung und Filter gegen Qdrant prüfen")
    ap.add_argument("--fehlend", help="Datei für wirklich fehlende Dokumente")
    ap.add_argument("--pruefung", help="PO-Testfälle für Prüfung (4), z. B. auswertung/po/pruefung.json")
    a = ap.parse_args()
    if a.pruefen:
        _pruefen(a.fehlend, a.pruefung)
    else:
        ap.print_help()
