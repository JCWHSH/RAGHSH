"""Fragen an die neu behaltenen Chunks vererben — ohne die GPU anzufassen.

Warum es das gibt
-----------------
Die Deduplizierung behält seit dem 14.08.2026 je Gruppe wortgleicher Chunks
nicht mehr nur EINEN Vertreter, sondern zusätzlich einen je Fakultät und
Organisation, die in der Gruppe vorkommt (siehe select_unique_ids()). Dadurch
kommen 1.343 Punkte hinzu, für die enrichment.parquet keine Zeile hat.

Diese Chunks sind mit ihrem Gruppenvertreter **wortgleich** — das ist die
Definition der Gruppe. Die Fragen, die das Modell zum Vertreter erzeugt hat,
passen deshalb ohne Abstriche auch zu ihnen. Sie neu erzeugen zu lassen hieße,
eine GPU-Stunde für dieselbe Antwort auszugeben.

Was das Skript tut
------------------
1. Prüft, dass sich die Chunk-GRENZEN nicht verschoben haben. Beweis statt
   Behauptung: enrichment.parquet speichert je Zeile das section_heading, das
   beim Anreichern galt; es muss zeichengenau zu dem passen, was das heutige
   Chunking für dieselbe ID liefert. Weicht eine einzige Zeile ab, bricht das
   Skript ab — dann sind die Fragen tatsächlich verrutscht und das hier wäre
   der falsche Weg.
2. Prüft, dass die neue Behaltemenge eine Obermenge der alten ist. Nur dann
   ist jede vorhandene Zeile weiterhin gültig.
3. Schreibt enrichment_v2.parquet: alle alten Zeilen mit den neuen
   Fingerabdrücken, plus je neuem Chunk eine geerbte Zeile.

Die geerbten Zeilen tragen die Spalte 'geerbt_von' mit der ID des Vertreters,
aus dem die Fragen stammen. Bei den originalen Zeilen ist sie leer. Ohne
diesen Vermerk ließe sich später nicht mehr unterscheiden, was das Modell
gesehen hat und was kopiert wurde.

Laufzeit: rund 3 Minuten auf dem Login-Node, keine GPU.

    $PYTHON erbe_fragen.py [--enrichment enrichment.parquet]
                           [--output enrichment_v2.parquet] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chunking import (config_hash, iter_chunks, list_markdown, make_point_id,
                      dedupe_key, select_unique_ids, selection_hash)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNKING_HASH = config_hash()
SELECTION_HASH = selection_hash()


def sammle_bestand(md_files):
    """Ein Durchlauf: je Gruppe der Vertreter, je Chunk seine Kenndaten."""
    vertreter: dict[str, str] = {}          # dedupe_key -> ID des ersten Chunks
    key_von_id: dict[str, str] = {}
    kopf_von_id: dict[str, str] = {}        # ID -> section_heading
    quelle_von_id: dict[str, tuple[str, int]] = {}   # ID -> (source_url, index)
    for _path, chunks in iter_chunks(md_files):
        for c in chunks:
            cid = make_point_id(c["source_url"], c["chunk_index"])
            k = dedupe_key(c)
            vertreter.setdefault(k, cid)
            key_von_id[cid] = k
            kopf_von_id[cid] = c.get("section_heading", "")
            quelle_von_id[cid] = (c["source_url"], c["chunk_index"])
    return vertreter, key_von_id, kopf_von_id, quelle_von_id


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("data/ingested"))
    p.add_argument("--enrichment", type=Path, default=Path("enrichment.parquet"))
    p.add_argument("--output", type=Path, default=Path("enrichment_v2.parquet"))
    p.add_argument("--dry-run", action="store_true",
                   help="nur prüfen und zählen, nichts schreiben")
    args = p.parse_args()

    if not args.enrichment.exists():
        logger.error("Anreicherung nicht gefunden: %s", args.enrichment)
        sys.exit(1)
    if args.output.exists() and not args.dry_run:
        logger.error("%s existiert bereits — erst wegräumen.", args.output)
        sys.exit(1)

    md_files = list_markdown(args.input_dir)
    logger.info("Dateien: %d | Grenzen %s | Auswahl %s",
                len(md_files), CHUNKING_HASH, SELECTION_HASH)

    logger.info("Chunking-Durchlauf 1/2 (Gruppen und Vertreter) …")
    vertreter, key_von_id, kopf_von_id, quelle_von_id = sammle_bestand(md_files)
    logger.info("Chunks %d, Gruppen %d", len(key_von_id), len(vertreter))

    logger.info("Chunking-Durchlauf 2/2 (Behaltemenge) …")
    keep_ids = select_unique_ids(md_files)
    if keep_ids is None:
        logger.error("DEDUPE_CHUNKS ist aus — dann gibt es nichts zu erben.")
        sys.exit(1)
    logger.info("Behaltemenge: %d Punkte", len(keep_ids))

    tabelle = pq.read_table(args.enrichment)
    spalten = {n: tabelle.column(n).to_pylist() for n in tabelle.schema.names}
    alte_ids = spalten["id"]
    logger.info("Anreicherung: %d Zeile(n) aus %s", len(alte_ids), args.enrichment)

    # ── Prüfung 1: sitzen die Fragen noch an derselben Textstelle? ────────
    unbekannt = [i for i in alte_ids if i not in key_von_id]
    if unbekannt:
        logger.error("%d angereicherte ID(s) kommen im heutigen Chunking nicht "
                     "mehr vor — die Grenzen haben sich verschoben. Abbruch.",
                     len(unbekannt))
        sys.exit(1)
    if "section_heading" in spalten:
        abweichend = sum(1 for i, h in zip(alte_ids, spalten["section_heading"])
                         if kopf_von_id[i] != h)
        if abweichend:
            logger.error("%d Zeile(n) mit abweichendem section_heading — die "
                         "Chunk-Grenzen haben sich verschoben. Die Fragen "
                         "gehören dann zu anderem Text. Abbruch.", abweichend)
            sys.exit(1)
        logger.info("Grenzen geprüft: %d von %d Zeilen mit identischem "
                    "section_heading.", len(alte_ids), len(alte_ids))
    else:
        logger.warning("Spalte section_heading fehlt — die Grenzen sind nicht "
                       "überprüfbar.")

    # ── Prüfung 2: ist die neue Menge eine Obermenge der alten? ───────────
    alt_set = set(alte_ids)
    verloren = alt_set - keep_ids
    if verloren:
        logger.error("%d angereicherte ID(s) fallen aus der neuen Behaltemenge "
                     "heraus. Dann ist das keine reine Erweiterung und die "
                     "Anreicherung müsste neu gerechnet werden. Abbruch.",
                     len(verloren))
        sys.exit(1)
    logger.info("Obermengen-Eigenschaft: alle %d alten IDs weiterhin behalten.",
                len(alt_set))

    # ── Erben ─────────────────────────────────────────────────────────────
    fragen_von_id = {i: q for i, q, ok in
                     zip(alte_ids, spalten["questions"], spalten["ok"]) if ok and q}
    neu = sorted(keep_ids - alt_set)
    logger.info("Neue Punkte ohne eigene Zeile: %d", len(neu))

    geerbt: list[dict] = []
    ohne_quelle = 0
    for cid in neu:
        quelle = vertreter[key_von_id[cid]]
        fragen = fragen_von_id.get(quelle)
        if not fragen:
            ohne_quelle += 1
            continue
        geerbt.append({"id": cid, "quelle": quelle, "questions": fragen})
    logger.info("davon geerbt: %d | ohne verwertbare Quelle: %d",
                len(geerbt), ohne_quelle)

    if args.dry_run:
        logger.info("--dry-run: nichts geschrieben.")
        return

    # ── Schreiben ─────────────────────────────────────────────────────────
    # Alte Zeilen bekommen die neuen Fingerabdrücke: Die Grenzen sind oben
    # zeichengenau nachgewiesen, die Auswahlregel ist die dieses Laufs.
    n_alt = len(alte_ids)
    spalten["chunking_config_hash"] = [CHUNKING_HASH] * n_alt
    spalten["selection_hash"] = [SELECTION_HASH] * n_alt
    spalten["geerbt_von"] = [""] * n_alt

    vorlage = {n: None for n in spalten}
    for g in geerbt:
        cid = g["id"]
        url, idx = quelle_von_id[cid]
        neue_zeile = {
            "id": cid,
            "source_url": url,
            "chunk_index": idx,
            "questions": g["questions"],
            "section_heading": kopf_von_id.get(cid, ""),
            "raw": "",
            "model": "",
            "ok": True,
            "chunking_config_hash": CHUNKING_HASH,
            "selection_hash": SELECTION_HASH,
            "geerbt_von": g["quelle"],
        }
        for name in spalten:
            spalten[name].append(neue_zeile.get(name, vorlage[name]))

    felder = []
    for feld in tabelle.schema:
        felder.append(feld)
    vorhanden = set(tabelle.schema.names)
    if "selection_hash" not in vorhanden:
        felder.append(pa.field("selection_hash", pa.string()))
    felder.append(pa.field("geerbt_von", pa.string()))
    schema = pa.schema(felder)

    aus = pa.table({f.name: spalten[f.name] for f in schema}, schema=schema)
    pq.write_table(aus, args.output, compression="zstd", compression_level=3)
    logger.info("Geschrieben: %s — %d Zeile(n) (%d original, %d geerbt), %.1f MB",
                args.output, aus.num_rows, n_alt, len(geerbt),
                args.output.stat().st_size / 1e6)
    logger.info("Der Vektorisierer ist jetzt mit --enrichment %s aufzurufen.",
                args.output)


if __name__ == "__main__":
    main()
