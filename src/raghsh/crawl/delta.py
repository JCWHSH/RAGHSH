"""
Delta-Ermittlung — welche Dokumente müssen (neu) vektorisiert werden?

Kurzbeschreibung
----------------
Vergleicht den aktuellen Textstand jeder Markdown-Datei mit dem Stand, der
zuletzt vektorisiert wurde, und gibt die Differenz als URL-Liste aus. Diese
Liste ist die Eingabe für alle folgenden Pipeline-Schritte:

    python -m raghsh.crawl.delta > delta.txt
    python hpc_vectorizer.py --urls delta.txt --output delta.parquet
    python -m raghsh.index.local_importer --input delta.parquet
    python -m raghsh.index.build_doc_index --urls delta.txt

Wie die Erkennung funktioniert
------------------------------
Jede .md-Datei trägt einen `content_hash` über ihren Body (vom Crawler
geschrieben). In doc_index.db stehen zwei Hashes pro Dokument:

    content_hash      — Stand der Datei beim letzten Indexlauf
    vectorized_hash   — Stand, der zuletzt vektorisiert und importiert wurde

Ein Dokument ist offen, wenn die beiden auseinanderlaufen oder
`vectorized_hash` noch NULL ist (nie vektorisiert). Der Vergleich läuft
bewusst gegen den Hash **in der Datei**, nicht gegen den in der Datenbank —
so ist das Ergebnis auch dann korrekt, wenn build_doc_index.py seit dem
letzten Crawl noch nicht gelaufen ist.

Damit ist der Zustand unterbrechungsfest: Bricht der Transfer zum HPC ab oder
scheitert der Import, bleibt `vectorized_hash` stehen und das Dokument taucht
beim nächsten Lauf wieder auf. Es gibt keine separate Zustandsdatei.

WICHTIG — wann der Delta-Betrieb NICHT genügt
---------------------------------------------
Die Punkt-IDs in Qdrant sind UUID5 über `source_url + chunk_index`. Ändern
sich die Chunking-Parameter in hpc_vectorizer.py (CHUNK_SIZE, CHUNK_OVERLAP,
HEADERS_TO_SPLIT), verschieben sich die chunk_index-Werte und damit die IDs.
Ein Delta-Import erzeugt dann Karteileichen statt Überschreibungen. Dasselbe
gilt beim Wechsel des Embedding-Modells. In beiden Fällen:

    python delete_qdrant.py && python -m raghsh.index.local_importer   (Volldurchlauf)

Aufruf
------
  python -m raghsh.crawl.delta                  # URL-Liste auf stdout
  python -m raghsh.crawl.delta --count          # nur die Anzahl
  python -m raghsh.crawl.delta --stats          # Aufschlüsselung nach Grund
  python -m raghsh.crawl.delta -o delta.txt     # direkt in eine Datei
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from raghsh import konfig
from raghsh.crawl.crawler_utils import read_header

INGESTED_DIR = konfig.KORPUS
DB_PATH      = konfig.DOC_INDEX


def collect_delta(ingested_dir: Path, db_path: Path) -> dict[str, list[str]]:
    """Ermittelt offene Dokumente, gruppiert nach Grund.

    Gibt {"neu": [...], "geaendert": [...]} zurück — beides URLs.
    "neu" = noch nie vektorisiert, "geaendert" = Inhalt seither verändert.
    """
    if not db_path.exists():
        print(f"Fehler: {db_path} nicht gefunden — zuerst build_doc_index.py laufen lassen.",
              file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # Die Spalte legt build_doc_index.open_db() an. Fehlt sie, wurde die
    # Zustandsverfolgung noch nie initialisiert — dann ist jede Aussage über
    # den Delta-Umfang wertlos (alles sähe nach "nie vektorisiert" aus).
    cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
    if "vectorized_hash" not in cols:
        conn.close()
        print(
            "Fehler: Spalte 'vectorized_hash' fehlt in doc_index.db.\n"
            "  Die Zustandsverfolgung ist noch nicht initialisiert. Einmalig:\n"
            "      python -m raghsh.index.build_doc_index --rebuild\n"
            "  Das legt die Spalte an und markiert alle Dokumente, die bereits\n"
            "  Chunks in Qdrant haben, als vektorisiert. Erst danach liefert\n"
            "  delta.py sinnvolle Ergebnisse.",
            file=sys.stderr,
        )
        sys.exit(1)

    known = {
        row[0]: row[1]
        for row in conn.execute("SELECT source_url, vectorized_hash FROM documents")
    }
    conn.close()

    neu: list[str] = []
    geaendert: list[str] = []

    for path in sorted(ingested_dir.glob("*.md")):
        meta = read_header(path)
        url  = meta.get("source_url")
        if not url:
            continue
        file_hash = meta.get("content_hash") or ""
        vec_hash  = known.get(url)

        if vec_hash is None:
            # Weder indexiert noch vektorisiert, oder nie markiert
            neu.append(url)
        elif file_hash and file_hash != vec_hash:
            geaendert.append(url)

    return {"neu": neu, "geaendert": geaendert}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Listet Dokumente, die (neu) vektorisiert werden müssen."
    )
    parser.add_argument("--count", action="store_true", help="Nur die Anzahl ausgeben")
    parser.add_argument("--stats", action="store_true",
                        help="Aufschlüsselung nach Grund statt der URL-Liste")
    parser.add_argument("-o", "--output", type=Path, metavar="FILE",
                        help="URL-Liste in diese Datei schreiben statt auf stdout")
    args = parser.parse_args()

    if not INGESTED_DIR.exists():
        print(f"Fehler: {INGESTED_DIR} nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    delta = collect_delta(INGESTED_DIR, DB_PATH)
    alle  = delta["neu"] + delta["geaendert"]

    if args.stats:
        gesamt = len(list(INGESTED_DIR.glob("*.md")))
        print(f"Dokumente im Korpus      : {gesamt}")
        print(f"Nie vektorisiert         : {len(delta['neu'])}")
        print(f"Seither geändert         : {len(delta['geaendert'])}")
        print(f"─────────────────────────────────")
        print(f"Offen insgesamt          : {len(alle)}")
        if alle:
            print(f"\nNächster Schritt: python -m raghsh.crawl.delta -o delta.txt")
        return

    if args.count:
        print(len(alle))
        return

    text = "\n".join(alle)
    if args.output:
        args.output.write_text(text + ("\n" if text else ""), encoding="utf-8")
        print(f"{len(alle)} URL(s) → {args.output}", file=sys.stderr)
    else:
        if text:
            print(text)


if __name__ == "__main__":
    main()
