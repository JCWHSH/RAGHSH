"""
Dokumentindex-Builder — Erstellt eine SQLite-Datenbank aller ingested Dokumente.

Kurzbeschreibung
----------------
Baut einen persistenten Index (doc_index.db) auf, der Metadaten, Markdown-
Headings, Dateipfade und Qdrant-Chunk-IDs für alle Dokumente der Wissensbasis
zusammenführt. Dieser Index ermöglicht dem Chatbot schnelle Lookups ohne
wiederholtes Dateisystem-Scanning oder Qdrant-Scrolling.

Verarbeitungsablauf
-------------------
1. Qdrant scrollen
   Alle Punkte der Collection 'hsh_knowledge' werden gelesen.
   Die Chunk-IDs werden nach source_url gruppiert und nach chunk_index sortiert.

2. Markdown-Dateien scannen
   Alle *.md-Dateien in var/korpus/ werden eingelesen.
   Pro Datei werden extrahiert:
     - YAML-Frontmatter: source_url, title, crawl_date, content_type, faculty
     - Markdown-Headings (H1–H3) mit Level als JSON-Array
     - Wortanzahl des Dokumenttexts (ohne Frontmatter)
     - Absoluter Dateipfad + Änderungszeitpunkt (mtime)

3. JOIN auf source_url
   Beide Quellen werden zusammengeführt. Dokumente die nur in Qdrant oder nur
   als Datei existieren, werden ebenfalls aufgenommen (chunk_count=0 bzw.
   file_path='').

4. SQLite schreiben
   INSERT OR REPLACE — idempotent, kann beliebig oft erneut ausgeführt werden.
   Inkrementelles Rebuild: Dateien deren mtime unverändert ist werden
   übersprungen (--rebuild für vollständiges Rebuild).

Datenbankschema (doc_index.db)
------------------------------
Tabelle 'documents':
  source_url   TEXT PRIMARY KEY
  title        TEXT
  file_path    TEXT
  crawl_date   TEXT
  content_type TEXT
  faculty      TEXT
  headings     TEXT  — JSON: [{"level": 1, "text": "..."}, ...]
  word_count   INTEGER
  chunk_count  INTEGER
  chunk_ids    TEXT  — JSON: ["uuid1", "uuid2", ...]
  file_mtime   REAL
  indexed_at   TEXT

Aufruf
------
  python -m raghsh.index.build_doc_index           # Inkrementell (nur geänderte Dateien)
  python -m raghsh.index.build_doc_index --rebuild # Vollständig neu aufbauen

Abhängigkeiten: qdrant-client (kein fastembed nötig)
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models

from raghsh import konfig

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "hsh_knowledge"
INGESTED_DIR    = konfig.KORPUS
DB_PATH         = konfig.DOC_INDEX
# Optional, von der Erzeugerseite geliefert — siehe lade_vertretung().
VERTRETUNG_FILE = konfig.VEKTOREN / "chunk_vertretung.parquet"

# Maximale Heading-Tiefe die gespeichert wird (H1–H3 reicht für Auswahl)
MAX_HEADING_LEVEL = 3
# Maximale Anzahl Headings pro Dokument (verhindert riesige JSON-Blobs bei PDFs)
MAX_HEADINGS = 50

# ---------------------------------------------------------------------------
# YAML-Frontmatter + Heading-Parser
# ---------------------------------------------------------------------------


def extract_faculty(url: str) -> str:
    """Leitet die Fakultät aus der URL ab (z.B. 'f1' für f1.hs-hannover.de)."""
    host = urlparse(url).hostname or ""
    m = re.match(r'^(f\d+)\.hs-hannover\.de$', host)
    return m.group(1) if m else ""


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extrahiert YAML-Frontmatter und gibt (meta_dict, body) zurück.

    Erwartet das Format:
        ---
        key: "value"
        ---
        Dokumenttext…
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    yaml_block = text[3:end].strip()
    body = text[end + 4:].strip()

    meta: dict[str, str] = {}
    for line in yaml_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")

    return meta, body


def extract_headings(body: str) -> list[dict]:
    """Extrahiert Markdown-Überschriften H1–H{MAX_HEADING_LEVEL} mit Level.

    Gibt eine Liste von {"level": int, "text": str} zurück,
    begrenzt auf MAX_HEADINGS Einträge.
    """
    headings: list[dict] = []
    pattern = re.compile(r'^(#{1,' + str(MAX_HEADING_LEVEL) + r'})\s+(.*)')

    for line in body.splitlines():
        m = pattern.match(line)
        if m:
            text = m.group(2).strip()
            # Inline-Markdown entfernen (Links, Fettdruck etc.)
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
            text = re.sub(r'[*_`]', '', text)
            if text:
                headings.append({"level": len(m.group(1)), "text": text})
        if len(headings) >= MAX_HEADINGS:
            break

    return headings


# ---------------------------------------------------------------------------
# Qdrant-Scrolling
# ---------------------------------------------------------------------------


def scroll_all_chunks(client: QdrantClient,
                      only_urls: set[str] | None = None) -> dict[str, list[dict]]:
    """Scrollt Punkte aus Qdrant und gruppiert sie nach source_url.

    Gibt {source_url: [{"id": str, "chunk_index": int}, ...]} zurück,
    sortiert nach chunk_index.

    Ist *only_urls* gesetzt, filtert Qdrant serverseitig auf diese source_urls
    (Delta-Betrieb). Der Filter wird in Blöcken angewendet, weil MatchAny bei
    sehr vielen Werten sonst unhandlich wird.
    """
    url_to_chunks: dict[str, list[dict]] = {}
    total = 0

    # Ohne Filter: ein Durchlauf über alles. Mit Filter: Blöcke à 128 URLs.
    if only_urls is None:
        filter_batches: list[models.Filter | None] = [None]
    else:
        urls = sorted(only_urls)
        filter_batches = [
            models.Filter(must=[models.FieldCondition(
                key="source_url", match=models.MatchAny(any=urls[i:i + 128])
            )])
            for i in range(0, len(urls), 128)
        ]

    for scroll_filter in filter_batches:
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=scroll_filter,
                offset=offset,
                limit=500,
                with_payload=True,
                with_vectors=False,
            )

            for point in result:
                payload    = point.payload or {}
                source_url = payload.get("source_url", "")
                if not source_url:
                    continue
                if source_url not in url_to_chunks:
                    url_to_chunks[source_url] = []
                url_to_chunks[source_url].append({
                    "id":            str(point.id),
                    "chunk_index":   payload.get("chunk_index", 0),
                    # Aus der Payload übernommen, nicht hier neu berechnet: Der
                    # Importer hat den Anzeigetitel bereits gebaut, und zwei
                    # Berechnungen desselben Titels könnten auseinanderlaufen.
                    "title_anzeige": payload.get("title_anzeige", ""),
                })
                total += 1

            print(f"  … {total} Chunks gelesen", end="\r")

            if next_offset is None:
                break
            offset = next_offset

    print()

    # Nach chunk_index sortieren
    for url in url_to_chunks:
        url_to_chunks[url].sort(key=lambda x: x["chunk_index"])

    return url_to_chunks


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            source_url    TEXT PRIMARY KEY,
            title         TEXT NOT NULL DEFAULT '',
            file_path     TEXT NOT NULL DEFAULT '',
            crawl_date    TEXT NOT NULL DEFAULT '',
            content_type  TEXT NOT NULL DEFAULT '',
            faculty       TEXT NOT NULL DEFAULT '',
            headings      TEXT NOT NULL DEFAULT '[]',
            word_count    INTEGER NOT NULL DEFAULT 0,
            chunk_count   INTEGER NOT NULL DEFAULT 0,
            chunk_ids     TEXT NOT NULL DEFAULT '[]',
            file_mtime    REAL,
            indexed_at    TEXT NOT NULL DEFAULT '',
            content_hash  TEXT,
            vectorized_at TEXT,
            imported_at   TEXT,
            enriched_at   TEXT,
            vectorized_hash TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_url ON documents(source_url)"
    )

    # Vertretungstabelle — welcher Punkt hält den Text welcher Dokumentstelle.
    #
    # Die Deduplizierung behält je Gruppe wortgleicher Chunks einen Vertreter.
    # 29,4 % der Dokumente haben dadurch Lücken in ihrer Chunkfolge; im
    # Extremfall liegen 3 von 242 Positionen unter ihrer eigenen URL. Ohne diese
    # Tabelle kann die zweite LLM-Runde ein solches Dokument nur als Fragment
    # ausgeben — mit ihr lässt es sich vollständig zusammensetzen, weil der Text
    # unter einer anderen source_url wortgleich vorliegt.
    #
    # vertreter_url ist redundant (sie steht in der Payload des Vertreters),
    # aber sie erspart beim Zusammensetzen einen zweiten Qdrant-Zugriff nur, um
    # zu erfahren, welche Fundstellen in der Quellenangabe zu nennen sind.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_vertretung (
            source_url      TEXT NOT NULL,
            chunk_index     INTEGER NOT NULL,
            vertreten_durch TEXT NOT NULL,
            vertreter_url   TEXT NOT NULL DEFAULT '',
            selbst          INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (source_url, chunk_index)
        )
    """)
    # Migration: neue Spalten zu bestehenden Datenbanken hinzufügen
    _new_cols = [
        # Lesbarer Titel für die LLM-Dokumentauswahl. `title` bleibt daneben
        # stehen — bei PDFs ist er der Dateiname und damit für die Auswahl
        # wertlos, aber er verbindet den Eintrag mit der Erzeugerseite.
        ("title_anzeige", "TEXT"),
        ("content_hash",  "TEXT"),
        ("vectorized_at", "TEXT"),
        ("imported_at",   "TEXT"),
        ("enriched_at",   "TEXT"),
        # Stand des Textes, der zuletzt vektorisiert wurde. Weicht er von
        # content_hash ab, ist das Dokument neu zu vektorisieren — das ist die
        # Grundlage der Delta-Erkennung in delta.py. Wird bewusst NICHT von
        # upsert_document() geschrieben, sondern nur von mark_vectorized().
        ("vectorized_hash", "TEXT"),
    ]
    for col_name, col_type in _new_cols:
        try:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Spalte existiert bereits
    conn.commit()
    return conn


def get_existing_mtimes(conn: sqlite3.Connection) -> dict[str, float]:
    """Gibt {source_url: file_mtime} für alle bereits indexierten Dokumente zurück."""
    rows = conn.execute(
        "SELECT source_url, file_mtime FROM documents WHERE file_mtime IS NOT NULL"
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def upsert_document(conn: sqlite3.Connection, doc: dict) -> None:
    # ON CONFLICT DO UPDATE statt INSERT OR REPLACE, damit vectorized_at,
    # imported_at und enriched_at bei Re-Index nicht auf NULL zurückgesetzt werden.
    conn.execute("""
        INSERT INTO documents
            (source_url, title, title_anzeige, file_path, crawl_date, content_type, faculty,
             headings, word_count, chunk_count, chunk_ids, file_mtime, indexed_at,
             content_hash)
        VALUES
            (:source_url, :title, :title_anzeige, :file_path, :crawl_date, :content_type, :faculty,
             :headings, :word_count, :chunk_count, :chunk_ids, :file_mtime, :indexed_at,
             :content_hash)
        ON CONFLICT(source_url) DO UPDATE SET
            title         = excluded.title,
            title_anzeige = excluded.title_anzeige,
            file_path    = excluded.file_path,
            crawl_date   = excluded.crawl_date,
            content_type = excluded.content_type,
            faculty      = excluded.faculty,
            headings     = excluded.headings,
            word_count   = excluded.word_count,
            chunk_count  = excluded.chunk_count,
            chunk_ids    = excluded.chunk_ids,
            file_mtime   = excluded.file_mtime,
            indexed_at   = excluded.indexed_at,
            content_hash = excluded.content_hash
    """, doc)


def lade_vertretung(conn: sqlite3.Connection, pfad: Path,
                    url_zu_chunks: dict[str, list[dict]]) -> int:
    """Liest chunk_vertretung.parquet in die SQLite-Tabelle.

    Die Datei kommt von der Erzeugerseite und beschreibt für JEDEN der 196.934
    Chunks vor der Deduplizierung, unter welcher Punkt-ID sein Text im Index
    steht. Sie ist optional: Fehlt sie, bleibt die Tabelle leer und die
    Rekonstruktion in Runde 2 fällt auf die vorhandenen Chunks zurück.

    `vertreter_url` wird hier aufgelöst — aus den ohnehin gescrollten Punkten,
    ohne zusätzliche Qdrant-Abfrage.
    """
    if not pfad.exists():
        print(f"  ⚠ {pfad.name} nicht gefunden — Vertretungstabelle bleibt leer.")
        print("    Runde 2 kann Dokumente dann nicht vollständig zusammensetzen.")
        return 0

    import pyarrow.parquet as pq

    id_zu_url = {c["id"]: url for url, chunks in url_zu_chunks.items() for c in chunks}

    daten = pq.read_table(pfad).to_pydict()
    zeilen = []
    unbekannt = 0
    for su, ci, vd, selbst in zip(daten["source_url"], daten["chunk_index"],
                                  daten["vertreten_durch"], daten["selbst_im_index"]):
        vurl = id_zu_url.get(vd, "")
        if not vurl:
            unbekannt += 1
        zeilen.append((su, int(ci), vd, vurl, 1 if selbst else 0))

    conn.execute("DELETE FROM chunk_vertretung")
    conn.executemany(
        "INSERT OR REPLACE INTO chunk_vertretung "
        "(source_url, chunk_index, vertreten_durch, vertreter_url, selbst) "
        "VALUES (?,?,?,?,?)", zeilen)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vertretung_url "
                 "ON chunk_vertretung(source_url, chunk_index)")
    if unbekannt:
        print(f"  ⚠ {unbekannt} Vertreter-IDs sind nicht in der Collection — "
              "Vektordatei und Vertretungstabelle passen nicht zusammen.")
    return len(zeilen)


def entferne_verwaiste(conn: sqlite3.Connection, aktuelle_urls: set[str]) -> int:
    """Löscht Dokumente, die weder als Datei noch in Qdrant existieren.

    `--rebuild` schrieb bisher nur mit ON CONFLICT DO UPDATE und löschte nie.
    Gemessen am 17.08.2026 standen dadurch 12.114 Zeilen in der Datenbank,
    während der Korpus 5.616 Dokumente hat: 6.498 Leichen aus früheren Ständen,
    deren chunk_ids auf Punkte zeigen, die es nicht mehr gibt. Praktisch
    harmlos, weil Kandidaten immer aus Qdrant kommen — aber `--rebuild` soll
    tun, was der Name sagt.
    """
    vorhanden = {r[0] for r in conn.execute("SELECT source_url FROM documents")}
    verwaist  = vorhanden - aktuelle_urls
    if verwaist:
        conn.executemany("DELETE FROM documents WHERE source_url = ?",
                         [(u,) for u in verwaist])
    return len(verwaist)


def mark_vectorized(conn: sqlite3.Connection, urls: list[str], now: str) -> int:
    """Setzt vectorized_hash/-at und imported_at für Dokumente mit Chunks in Qdrant.

    Nur Dokumente, die tatsächlich Chunks in der Collection haben, gelten als
    verarbeitet. vectorized_hash wird auf den aktuellen content_hash gesetzt —
    ab da meldet delta.py sie nicht mehr als offen, bis sich die Datei ändert.
    """
    if not urls:
        return 0
    placeholders = ",".join("?" * len(urls))
    cur = conn.execute(
        f"""UPDATE documents
               SET vectorized_hash = content_hash,
                   vectorized_at   = ?,
                   imported_at     = ?
             WHERE source_url IN ({placeholders})
               AND chunk_count > 0""",
        [now, now, *urls],
    )
    return cur.rowcount


def load_url_filter(path: Path) -> set[str]:
    """Liest eine URL-Liste (eine pro Zeile, # = Kommentar) für --urls."""
    if not path.exists():
        print(f"Fehler: URL-Liste nicht gefunden — {path}")
        sys.exit(1)
    urls = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.add(line)
    return urls


# ---------------------------------------------------------------------------
# Hauptlogik
# ---------------------------------------------------------------------------


def scan_markdown_files(
    ingested_dir: Path,
    existing_mtimes: dict[str, float],
    rebuild: bool,
) -> dict[str, dict]:
    """Scannt alle Markdown-Dateien und gibt {source_url: file_info} zurück.

    Bei rebuild=False werden Dateien mit unveränderter mtime übersprungen
    (ihre source_url fehlt dann im Rückgabe-Dict).
    """
    md_files = sorted(ingested_dir.glob("*.md"))
    print(f"  {len(md_files)} Markdown-Dateien gefunden.")

    file_data: dict[str, dict] = {}
    skipped = 0

    for md_file in md_files:
        mtime = md_file.stat().st_mtime

        text = md_file.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_frontmatter(text)
        source_url = meta.get("source_url", "")
        if not source_url:
            continue

        # Inkrementell: überspringen wenn mtime unverändert
        if not rebuild and existing_mtimes.get(source_url) == mtime:
            skipped += 1
            continue

        faculty = meta.get("faculty", "") or extract_faculty(source_url)
        file_data[source_url] = {
            "title":        meta.get("title", ""),
            "file_path":    str(md_file.resolve()),
            "crawl_date":   meta.get("crawl_date", ""),
            "content_type": meta.get("content_type", ""),
            "faculty":      faculty,
            "headings":     json.dumps(extract_headings(body), ensure_ascii=False),
            "word_count":   len(body.split()),
            "file_mtime":   mtime,
            "content_hash": meta.get("content_hash", None),
        }

    if skipped:
        print(f"  {skipped} Dateien unverändert — übersprungen.")

    return file_data


def build_index(rebuild: bool = False, only_urls: set[str] | None = None) -> None:
    print("=" * 70)
    print("  Dokumentindex-Builder")
    print("=" * 70 + "\n")

    # ── Qdrant ────────────────────────────────────────────────────────────
    print("1/3  Verbinde mit Qdrant und scrolle Chunks…")
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=60)
        client.get_collections()
    except Exception as exc:
        print(f"Fehler: Qdrant nicht erreichbar — {exc}")
        print("  → docker compose up -d")
        sys.exit(1)

    url_to_chunks = scroll_all_chunks(client, only_urls)
    print(f"  → {len(url_to_chunks)} eindeutige source_urls in Qdrant.\n")

    # ── Datenbank öffnen / anlegen ────────────────────────────────────────
    conn = open_db(DB_PATH)
    existing_mtimes = {} if rebuild else get_existing_mtimes(conn)

    # ── Markdown-Dateien scannen ──────────────────────────────────────────
    print("2/3  Scanne Markdown-Dateien…")
    if not INGESTED_DIR.exists():
        print(f"Fehler: Verzeichnis nicht gefunden — {INGESTED_DIR}")
        sys.exit(1)

    file_data = scan_markdown_files(INGESTED_DIR, existing_mtimes, rebuild)
    if only_urls is not None:
        file_data = {u: fd for u, fd in file_data.items() if u in only_urls}
        print(f"  → auf {len(file_data)} Datei(en) der URL-Liste begrenzt.")
    print(f"  → {len(file_data)} Dateien (neu/geändert) werden indexiert.\n")

    # ── JOIN + SQLite schreiben ───────────────────────────────────────────
    print("3/3  Schreibe Index…")
    now        = datetime.now().isoformat()
    all_urls   = set(url_to_chunks.keys()) | set(file_data.keys())
    # Bei inkrementellem Rebuild nur URLs die neue Dateidaten haben ODER
    # erstmalig in Qdrant auftauchen (noch nicht in DB)
    if not rebuild:
        db_urls = set(existing_mtimes.keys())
        # Qdrant-only URLs die noch nicht in DB sind
        new_qdrant_urls = set(url_to_chunks.keys()) - db_urls
        urls_to_write   = set(file_data.keys()) | new_qdrant_urls
    else:
        urls_to_write = all_urls

    if only_urls is not None:
        urls_to_write &= only_urls

    written = 0
    for url in urls_to_write:
        fd     = file_data.get(url, {})
        chunks = url_to_chunks.get(url, [])
        ids    = [c["id"] for c in chunks]

        doc = {
            "source_url":   url,
            "title":        fd.get("title", ""),
            # Aus der Qdrant-Payload; für Dokumente ohne Chunks bleibt der
            # Frontmatter-Titel die einzige Angabe.
            "title_anzeige": (chunks[0].get("title_anzeige") if chunks else "")
                             or fd.get("title", ""),
            "file_path":    fd.get("file_path", ""),
            "crawl_date":   fd.get("crawl_date", ""),
            "content_type": fd.get("content_type", ""),
            "faculty":      fd.get("faculty", ""),
            "headings":     fd.get("headings", "[]"),
            "word_count":   fd.get("word_count", 0),
            "chunk_count":  len(ids),
            "chunk_ids":    json.dumps(ids),
            "file_mtime":   fd.get("file_mtime"),
            "indexed_at":   now,
            "content_hash": fd.get("content_hash"),
        }
        # Falls Datei schon in DB: chunk_count aktualisieren auch ohne neue Datei
        if not fd and url in existing_mtimes:
            conn.execute(
                "UPDATE documents SET chunk_count=?, chunk_ids=?, indexed_at=? "
                "WHERE source_url=?",
                (len(ids), json.dumps(ids), now, url),
            )
        else:
            upsert_document(conn, doc)
        written += 1

    # Dokumente, deren Chunks jetzt in Qdrant liegen, als vektorisiert und
    # importiert markieren. Erst hier — vorher stimmen die chunk_ids nicht.
    marked = mark_vectorized(conn, sorted(urls_to_write), now)

    # Verwaiste Einträge nur beim vollständigen Rebuild entfernen. Beim
    # inkrementellen Lauf ist `all_urls` nur ein Ausschnitt — dort zu löschen
    # würde alles Nichtbetrachtete mitreißen.
    entfernt = 0
    if rebuild and only_urls is None:
        entfernt = entferne_verwaiste(conn, all_urls)

    vertretungen = 0
    if only_urls is None:
        vertretungen = lade_vertretung(conn, VERTRETUNG_FILE, url_to_chunks)

    conn.commit()
    conn.close()

    # ── Statistik ─────────────────────────────────────────────────────────
    only_qdrant = len(set(url_to_chunks.keys()) - set(file_data.keys()))
    only_files  = len(set(file_data.keys()) - set(url_to_chunks.keys()))
    both        = len(set(url_to_chunks.keys()) & set(file_data.keys()))

    print(f"\n  {written} Einträge geschrieben, {marked} als vektorisiert markiert.")
    if entfernt:
        print(f"  {entfernt} verwaiste Einträge entfernt (nicht mehr im Korpus).")
    if vertretungen:
        print(f"  {vertretungen} Zeilen Vertretungstabelle geladen.")
    print(f"\nStatistik:")
    print(f"  Vollständig (Datei + Qdrant-Chunks) : {both}")
    print(f"  Nur als Datei (noch nicht vektori.) : {only_files}")
    print(f"  Nur in Qdrant (Datei fehlt/gelöscht): {only_qdrant}")
    print(f"\nIndex gespeichert: {DB_PATH}")


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baut den Dokumentindex (doc_index.db) auf."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Vollständiges Rebuild — alle Dateien neu verarbeiten.",
    )
    parser.add_argument(
        "--urls",
        type=Path,
        metavar="FILE",
        help="Nur diese source_urls verarbeiten (eine pro Zeile). Delta-Betrieb: "
             "begrenzt Qdrant-Scroll, Dateiscan und das Setzen von "
             "vectorized_hash auf die Liste.",
    )
    args = parser.parse_args()
    only = load_url_filter(args.urls) if args.urls else None
    build_index(rebuild=args.rebuild, only_urls=only)


if __name__ == "__main__":
    main()
