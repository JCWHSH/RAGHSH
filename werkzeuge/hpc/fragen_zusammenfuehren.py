"""Delta-Anreicherung in den Bestand einfügen — ohne GPU.

Warum es das gibt
-----------------
Nach einem Teil-Crawl reichert ``enricher.py --urls delta.txt`` nur die
geänderten Seiten an und schreibt eine kleine ``enrichment_delta.parquet``.
Der Fragenfilter (``filter_fragen.py``) zählt aber, an wie vielen
VERSCHIEDENEN Texten eine Frage hängt — das ist eine Eigenschaft des ganzen
Korpus. Gefiltert werden darf deshalb nur die vollständige, ungefilterte
Anreicherung. Dieses Skript baut sie:

    Bestand (ungefiltert)  minus alle Zeilen der Delta-URLs
                           plus  alle Zeilen der Delta-Anreicherung

Alle Zeilen einer Delta-URL werden ersetzt, nicht nur die mit gleicher ID:
Wird eine Seite kürzer, gäbe es sonst Fragen zu Chunks, die nicht mehr
existieren; ändert sich ihr Text bei gleicher chunk_index, hingen alte Fragen
an neuem Text.

Prüfungen (Abbruch statt stiller Fehler)
----------------------------------------
1. Beide Dateien müssen denselben chunking_config_hash tragen. Sonst zeigen
   gleiche IDs auf verschiedene Textstellen (siehe chunking.config_hash()).
2. Der Hash muss außerdem zum chunking.py im aktuellen Verzeichnis passen —
   dem, mit dem anschließend gefiltert und vektorisiert wird.
3. Jede Zeile der Delta-Datei muss zu einer URL der Liste gehören.

Der selection_hash wird nur verglichen und gemeldet: Eine geänderte
Auswahlregel verschiebt keine IDs (siehe chunking.selection_hash()).

Aufruf (auf dem Login-Node, ein bis zwei Minuten):

    $PYTHON fragen_zusammenfuehren.py \\
        --bestand enrichment_v2.parquet \\
        --delta   enrichment_delta.parquet \\
        --urls    delta.txt \\
        --output  enrichment_v3.parquet

Danach wie gewohnt:

    $PYTHON filter_fragen.py enrichment_v3.parquet \\
        --ausgabe enrichment_v3_wdh10.parquet --schreiben
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from chunking import config_hash, selection_hash


def lies_urls(pfad: Path) -> set[str]:
    """URL-Liste lesen: eine je Zeile, Leerzeilen und #-Kommentare übergehen."""
    return {z.strip() for z in pfad.read_text(encoding="utf-8").splitlines()
            if z.strip() and not z.strip().startswith("#")}


def lies_anreicherung(pfad: Path) -> pa.Table:
    """Eine Anreicherung lesen — Datei oder .parts-Verzeichnis von enricher.py."""
    if pfad.is_dir():
        return pa.concat_tables(pq.read_table(t) for t in sorted(pfad.glob("part_*.parquet")))
    return pq.read_table(pfad)


def einziger_wert(tabelle: pa.Table, spalte: str, name: str) -> str | None:
    """Den einen Wert einer Fingerabdruck-Spalte liefern; bei mehreren abbrechen."""
    if spalte not in tabelle.schema.names:
        return None
    werte = set(tabelle.column(spalte).to_pylist())
    if len(werte) != 1:
        sys.exit(f"{name}: {spalte} ist nicht einheitlich ({sorted(map(str, werte))}). "
                 "Die Datei mischt Läufe verschiedener Chunking-Stände.")
    return werte.pop()


def auf_schema_bringen(tabelle: pa.Table, schema: pa.Schema) -> pa.Table:
    """Spalten an das Schema des Bestands angleichen.

    Der Bestand (enrichment_v2.parquet) hat die Spalte 'geerbt_von' aus
    erbe_fragen.py, eine frische Delta-Anreicherung nicht. Fehlende Spalten
    werden mit Nullwerten ergänzt, damit beide Teile zusammenpassen.
    """
    spalten = []
    for feld in schema:
        if feld.name in tabelle.schema.names:
            spalten.append(tabelle.column(feld.name).cast(feld.type))
        else:
            spalten.append(pa.nulls(tabelle.num_rows, type=feld.type))
    return pa.Table.from_arrays(spalten, schema=schema)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bestand", type=Path, required=True,
                    help="UNGEFILTERTE Gesamtanreicherung, z.B. enrichment_v2.parquet")
    ap.add_argument("--delta", type=Path, required=True,
                    help="Ausgabe von enricher.py --urls (Datei oder .parts-Verzeichnis)")
    ap.add_argument("--urls", type=Path, required=True,
                    help="dieselbe URL-Liste wie beim Anreichern")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.output.resolve() in (args.bestand.resolve(), args.delta.resolve()):
        sys.exit("--output darf keine der Eingabedateien sein.")

    urls = lies_urls(args.urls)
    bestand = lies_anreicherung(args.bestand)
    delta = lies_anreicherung(args.delta)

    # ── Prüfung 1 und 2: Chunk-Grenzen ────────────────────────────────────
    h_bestand = einziger_wert(bestand, "chunking_config_hash", "Bestand")
    h_delta = einziger_wert(delta, "chunking_config_hash", "Delta")
    h_jetzt = config_hash()
    if not (h_bestand == h_delta == h_jetzt):
        sys.exit(f"chunking_config_hash passt nicht: Bestand {h_bestand}, "
                 f"Delta {h_delta}, chunking.py hier {h_jetzt}.\n"
                 "Ein Delta ist nur gegen einen Bestand mit identischem Chunking "
                 "möglich — sonst ist ein Volldurchlauf nötig (docs/hpc.md).")

    s_bestand = einziger_wert(bestand, "selection_hash", "Bestand")
    s_delta = einziger_wert(delta, "selection_hash", "Delta")
    if not (s_bestand == s_delta == selection_hash()):
        print(f"WARNUNG: selection_hash weicht ab (Bestand {s_bestand}, Delta "
              f"{s_delta}, hier {selection_hash()}). IDs bleiben gültig, die "
              "Punktmengen decken sich aber evtl. nicht vollständig.")

    # ── Prüfung 3: Delta gehört zur Liste ─────────────────────────────────
    fremd = set(delta.column("source_url").to_pylist()) - urls
    if fremd:
        sys.exit(f"{len(fremd)} URL(s) der Delta-Datei stehen nicht in "
                 f"{args.urls.name}, z.B. {sorted(fremd)[0]}")

    # ── Zusammenfügen ─────────────────────────────────────────────────────
    url_set = pa.array(sorted(urls), type=pa.string())
    weg = pc.is_in(bestand.column("source_url"), value_set=url_set)
    rest = bestand.filter(pc.invert(weg))
    n_weg = bestand.num_rows - rest.num_rows

    neu = pa.concat_tables([rest, auf_schema_bringen(delta, bestand.schema)])

    # Doppelte IDs wären ein Programmfehler (Delta-URLs wurden oben entfernt).
    if len(set(neu.column("id").to_pylist())) != neu.num_rows:
        sys.exit("Doppelte IDs im Ergebnis — abgebrochen, nichts geschrieben.")

    # Atomar schreiben: erst .partial, dann umbenennen.
    tmp = args.output.with_name(args.output.name + ".partial")
    pq.write_table(neu, tmp, compression="zstd", compression_level=3)
    os.replace(tmp, args.output)

    print(f"Bestand          {bestand.num_rows:>8} Zeilen")
    print(f"  davon ersetzt  {n_weg:>8} Zeilen ({len(urls)} URL(s) in der Liste)")
    print(f"Delta            {delta.num_rows:>8} Zeilen")
    print(f"Ergebnis         {neu.num_rows:>8} Zeilen  → {args.output}")
    print(f"chunking_config_hash {h_jetzt}")


if __name__ == "__main__":
    main()
