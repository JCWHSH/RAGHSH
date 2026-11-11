"""
Qualitätsprüfung — Analyse der gescrapten Markdown-Dateien.

Kurzbeschreibung
----------------
Liest alle Markdown-Dateien aus dem Verzeichnis var/korpus/ und gibt eine
tabellarische Übersicht über Wortanzahl, Tabellenanzahl und Content-Typ aus.
Empfiehlt zusätzlich das Löschen von Dateien, die vordefinierten Qualitäts-
schwellwerten nicht genügen.

Ausführliche Beschreibung
--------------------------
Nach jedem Lauf von main.py (dem Web-Spider) enthält var/korpus/ eine Sammlung
von Markdown-Dateien. Dieses Skript liest jede dieser Dateien, trennt den
YAML-Frontmatter-Header vom eigentlichen Textinhalt und berechnet folgende Kennzahlen:

1. Wortanzahl
   Zählt alle durch Leerzeichen getrennten Tokens im Textinhalt (ohne den
   YAML-Header). Gibt Auskunft darüber, ob eine Seite inhaltlich reich oder
   leer gescrapt wurde. Eine sehr geringe Wortanzahl deutet auf fehlerhaftes
   Crawling oder eine inhaltlich schwache Seite hin.

2. Anzahl Markdown-Tabellen
   Erkennt Tabellen-Trennzeilen (Muster: |---|) mittels regulärem Ausdruck.
   Nützlich um zu prüfen, ob strukturierte Daten (z.B. Stundenplan-Tabellen,
   Modullisten) korrekt als Markdown übernommen wurden.

3. Content-Typ
   Liest das Feld 'content_type' aus dem YAML-Header (entweder 'html' oder 'pdf').

Löschvorschläge (Qualitätskriterien)
--------------------------------------
Folgende Dateien werden als löschenswert markiert:

  - Kein gültiger YAML-Header:
    Datei wurde nicht korrekt gescrapt oder ist korrupt.

  - Zu wenig Text (HTML):  < MIN_WORDS_HTML Wörter (Standard: 30)
    Typisch für Fehlerseiten, leere Weiterleitungsseiten oder reine
    Navigationsseiten, die keinen inhaltlichen Mehrwert liefern.

  - Zu wenig Text (PDF):   < MIN_WORDS_PDF Wörter (Standard: 50)
    PDFs mit sehr wenig Inhalt sind oft Deckblätter, Formularvorlagen ohne
    ausgefüllte Daten oder scan-only-PDFs ohne OCR-Text.

  - Fehlerseitenmuster:
    Der Textinhalt enthält klassische Fehlermeldungen wie "404", "Seite nicht
    gefunden" oder "nicht verfügbar". Diese Seiten bieten keine nutzbaren
    Informationen für die RAG-Pipeline.

  - Duplikate (gleicher URL-Slug, mehrere Datumspräfixe):
    Wenn mehr als ein Datum für denselben URL vorhanden ist, werden alle älteren
    Versionen zum Löschen vorgeschlagen — nur die neueste Datei wird behalten.

Aufruf:
   python check_quality.py

Abhängigkeiten: Nur Python-Standardbibliothek (re, pathlib, sys)
"""

import re
import sys
from pathlib import Path

from raghsh import konfig

INGESTED_DIR = konfig.KORPUS

# ---------------------------------------------------------------------------
# Qualitätsschwellwerte
# ---------------------------------------------------------------------------

MIN_WORDS_HTML = 30   # HTML-Seiten mit weniger Wörtern → Löschvorschlag
MIN_WORDS_PDF  = 50   # PDF-Dateien mit weniger Wörtern → Löschvorschlag

# Schlüsselwörter, die auf eine Fehlerseite hinweisen (Kleinbuchstaben)
ERROR_PATTERNS = [
    "404",
    "seite nicht gefunden",
    "page not found",
    "nicht verfügbar",
    "fehler beim laden",
    "zugriff verweigert",
    "access denied",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TABLE_SEPARATOR_RE = re.compile(r"\|[\s\-:]+\|")


def parse_front_matter(text: str) -> dict[str, str]:
    """Extract key-value pairs from a YAML front-matter block (``---`` … ``---``)."""
    meta: dict[str, str] = {}
    if not text.startswith("---"):
        return meta
    end = text.find("---", 3)
    if end == -1:
        return meta
    block = text[3:end].strip()
    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
    return meta


def count_words(text: str) -> int:
    """Return the number of whitespace-delimited tokens in *text*."""
    return len(text.split())


def count_tables(text: str) -> int:
    """Count Markdown table separator rows (lines containing ``|---|``)."""
    return sum(1 for line in text.splitlines() if TABLE_SEPARATOR_RE.search(line))


def classify_file(filepath: Path) -> tuple[dict, str, int, int, list[str]]:
    """Parse a Markdown file and return (meta, body, words, tables, reasons).

    *reasons* is a list of human-readable strings explaining why the file
    should be deleted.  An empty list means the file passes quality checks.
    """
    text = filepath.read_text(encoding="utf-8")
    meta = parse_front_matter(text)
    reasons: list[str] = []

    if not meta:
        reasons.append("Kein gültiger YAML-Header")
        return meta, "", 0, 0, reasons

    # Strip front matter for content analysis
    body = text
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            body = text[end + 3:]

    words  = count_words(body)
    tables = count_tables(body)
    content_type = meta.get("content_type", "html")

    # ── Zu wenig Text ─────────────────────────────────────────────────────
    threshold = MIN_WORDS_PDF if content_type == "pdf" else MIN_WORDS_HTML
    if words < threshold:
        reasons.append(
            f"Zu wenig Text: {words} Wörter (Minimum für {content_type.upper()}: {threshold})"
        )

    # ── Fehlerseitenmuster ────────────────────────────────────────────────
    body_lower = body.lower()
    for pattern in ERROR_PATTERNS:
        if pattern in body_lower:
            reasons.append(f"Fehlerseitenmuster gefunden: \"{pattern}\"")
            break  # ein Treffer reicht

    return meta, body, words, tables, reasons


def find_duplicates(md_files: list[Path]) -> dict[str, list[Path]]:
    """Group files by URL slug; return slugs with more than one file.

    The filename format is ``YYYY-MM-DD_slug.md``.  Files that share the same
    slug but carry different date prefixes are duplicates.  For each group the
    newest file (latest date) should be kept; the rest are candidates for
    deletion.
    """
    from collections import defaultdict

    groups: dict[str, list[Path]] = defaultdict(list)
    for filepath in md_files:
        stem = filepath.stem            # e.g. "2026-02-26_studium-studienangebot"
        parts = stem.split("_", 1)
        if len(parts) == 2:
            slug = parts[1]
        else:
            slug = stem
        groups[slug].append(filepath)

    return {slug: paths for slug, paths in groups.items() if len(paths) > 1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not INGESTED_DIR.exists():
        print(f"Directory not found: {INGESTED_DIR}")
        sys.exit(1)

    md_files = sorted(INGESTED_DIR.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {INGESTED_DIR}")
        sys.exit(0)

    # ── Qualitätsanalyse aller Dateien ────────────────────────────────────
    results: list[tuple[Path, dict, int, int, list[str]]] = []
    for filepath in md_files:
        meta, _, words, tables, reasons = classify_file(filepath)
        results.append((filepath, meta, words, tables, reasons))

    # ── Duplikaterkennung ─────────────────────────────────────────────────
    duplicates = find_duplicates(md_files)
    # Build set of files to delete due to duplication (all but latest per slug)
    duplicate_delete: set[Path] = set()
    for paths in duplicates.values():
        # Sort by date prefix descending; keep first (newest), mark rest
        sorted_paths = sorted(paths, reverse=True)   # lexicographic = date order
        for old_path in sorted_paths[1:]:
            duplicate_delete.add(old_path)

    # ── Übersichtstabelle ausgeben ────────────────────────────────────────
    col_file = max(len(f.name) for f, *_ in results)
    col_file = max(col_file, len("Filename"))

    header = (
        f"{'Filename':<{col_file}}  {'Words':>7}  {'Tables':>7}  {'Type':<6}  {'Quality'}"
    )
    separator = "-" * len(header)

    print()
    print(header)
    print(separator)

    total_words  = 0
    total_tables = 0
    delete_candidates: list[tuple[Path, list[str]]] = []

    for filepath, meta, words, tables, reasons in results:
        content_type = meta.get("content_type", "?") if meta else "?"
        total_words  += words
        total_tables += tables

        all_reasons = list(reasons)
        if filepath in duplicate_delete:
            all_reasons.append("Duplikat (ältere Version)")

        quality = "OK" if not all_reasons else "LÖSCHEN"
        print(
            f"{filepath.name:<{col_file}}  {words:>7,}  {tables:>7}  "
            f"{content_type:<6}  {quality}"
        )

        if all_reasons:
            delete_candidates.append((filepath, all_reasons))

    print(separator)
    print(f"{'TOTAL':<{col_file}}  {total_words:>7,}  {total_tables:>7}")
    print()

    # ── Löschvorschläge ausgeben ──────────────────────────────────────────
    if not delete_candidates:
        print("Keine Löschvorschläge — alle Dateien erfüllen die Qualitätskriterien.")
        return

    print("=" * 70)
    print(f"  LÖSCHVORSCHLÄGE  ({len(delete_candidates)} von {len(md_files)} Dateien)")
    print("=" * 70)
    for filepath, reasons in delete_candidates:
        print(f"\n  {filepath.name}")
        for reason in reasons:
            print(f"    • {reason}")

    print()
    print("Zum Löschen (Linux/macOS):")
    print("  cd", INGESTED_DIR)
    for filepath, _ in delete_candidates:
        print(f"  rm \"{filepath.name}\"")
    print()


if __name__ == "__main__":
    main()
