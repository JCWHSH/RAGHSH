#!/usr/bin/env python3
"""
Prüft po_werte.tsv gegen Qdrant (S087, F6): Jede Zeile muss mit ihrem
wörtlichen Textauszug im Chunk stehen, auf den `punkt_id` zeigt, und dieser
Chunk muss zur Fundstelle (`source_url#chunk_index`) gehören. Leerraum wird
für den Vergleich zusammengefasst.

Nach jedem Neucrawl/Re-Import laufen lassen: Zeilen, deren Auszug nicht mehr
im Chunk steht, werden gemeldet (Wert prüfen, Punkt-ID/Fundstelle erneuern).
Dazu formale Prüfungen: Pflichtspalten gefüllt, Fassung eine Zahl, Punkt in
po_punkte.tsv (modulbezogene Punkte „name:<Kürzel>“ gegen „name:<modul>“).

Aufruf (aus dem Wurzelverzeichnis, venv aktiv):  python werkzeuge/po/po_werte_pruefen.py [--qdrant URL]
Rückgabewert 0, wenn alle Zeilen bestehen, sonst 1.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from raghsh import konfig

HIER = konfig.PO_DATEN
WERTE = HIER / "po_werte.tsv"
PUNKTE = HIER / "po_punkte.tsv"
SPALTEN = ["studiengang", "fassung", "punkt", "wert", "fundstelle", "paragraph", "punkt_id", "textauszug"]
COLLECTION = "hsh_knowledge"


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--qdrant", default="http://localhost:6333")
    args = ap.parse_args()

    zeilen = list(csv.DictReader(open(WERTE, encoding="utf-8", newline=""), delimiter="\t"))
    punkte = {z["punkt"] for z in csv.DictReader(open(PUNKTE, encoding="utf-8", newline=""), delimiter="\t")}
    qc = QdrantClient(url=args.qdrant, timeout=60)
    ids = sorted({z["punkt_id"] for z in zeilen if z.get("punkt_id")})
    chunks = {str(p.id): p.payload for p in
              qc.retrieve(COLLECTION, ids=ids, with_payload=["text", "source_url", "chunk_index"])}

    fehler, kaputt = [], set()

    def melde(nr, text):
        fehler.append(text)
        kaputt.add(nr)

    for nr, z in enumerate(zeilen, start=2):          # Zeile 1 = Kopf
        ort = f"Z. {nr} {z.get('studiengang')} {z.get('fassung')} {z.get('punkt')}"
        leer = [s for s in SPALTEN if not (z.get(s) or "").strip() and s != "paragraph"]
        if leer:
            melde(nr, f"{ort}: leer: {', '.join(leer)}")
            continue
        if not z["fassung"].isdigit():
            melde(nr, f"{ort}: Fassung keine Zahl")
        name = z["punkt"].split(":")[0]
        if z["punkt"] not in punkte and f"{name}:<modul>" not in punkte:
            melde(nr, f"{ort}: Punkt fehlt in po_punkte.tsv")
        p = chunks.get(z["punkt_id"])
        if p is None:
            melde(nr, f"{ort}: Punkt-ID {z['punkt_id']} nicht in Qdrant")
            continue
        url, _, ci = z["fundstelle"].rpartition("#")
        if p.get("source_url") != url or str(p.get("chunk_index")) != ci:
            melde(nr, f"{ort}: Fundstelle passt nicht zum Chunk "
                       f"({p.get('source_url')}#{p.get('chunk_index')})")
        if norm(z["textauszug"]) not in norm(p.get("text")):
            melde(nr, f"{ort}: Auszug nicht mehr im Chunk")

    gut = len(zeilen) - len(kaputt)
    print(f"Auszug im Chunk und Zeile in Ordnung: {gut}/{len(zeilen)}")
    for f in fehler:
        print("  ✗", f)
    return 0 if not fehler else 1


if __name__ == "__main__":
    sys.exit(main())
