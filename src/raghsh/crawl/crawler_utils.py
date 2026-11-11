"""
Gemeinsame Hilfsfunktionen und Konstanten für main.py und resume_crawler.py.

Alle Ausschlussregeln, Qualitätsprüfungen und Hash-Logik sind hier zentralisiert,
damit beide Crawler-Varianten konsistent dieselben Seiten ausschließen.
"""

import hashlib
import unicodedata
import logging
import os
import re
import tarfile
from pathlib import Path
from urllib.parse import urlparse

from slugify import slugify

logger = logging.getLogger(__name__)

# Basisdomain der Hochschule — Grundlage für host_slug(). main.py führt
# dieselbe Konstante als ALLOWED_DOMAIN für die Crawl-Grenze.
BASE_DOMAIN = "hs-hannover.de"

# ---------------------------------------------------------------------------
# Ausgeschlossene Domains
# ---------------------------------------------------------------------------

BLOCKED_DOMAINS: set[str] = {
    # Bestehende Ausschlüsse
    "serwiss.bib.hs-hannover.de",       # Bibliothekssystem (eigene Suche)
    "typo3backend-live.hs-hannover.de", # TYPO3-Backend, kein öffentlicher Inhalt

    # Java-API-Dokumentation — kein Mehrwert für RAG
    "graja.hs-hannover.de",

    # Bibliothekskatalog — Suchoberfläche, kein strukturierter Inhalt
    "katalog.bib.hs-hannover.de",

    # Login-gesperrte Systeme
    "moodle.hs-hannover.de",
    "campusmanagement.hs-hannover.de",
    "ticketsystem.hs-hannover.de",
    "webdisk.hs-hannover.de",
    "idp.hs-hannover.de",
    "accounts.hs-hannover.de",
    "mail.hs-hannover.de",
    "vpn.hs-hannover.de",
    "raumplanung.wp.hs-hannover.de",
    "chat.bib.hs-hannover.de",

    # Nur intern erreichbar
    "intranet.hs-hannover.de",
    "hsh-intern.hs-hannover.de",
    "f1.hsh-intern.hs-hannover.de",
    "f5.hsh-intern.hs-hannover.de",
    "personalrat.hs-hannover.de",

    # Kein crawlbarer Inhalt / technische Probleme
    "icms.hs-hannover.de",              # Nur Redirect, kein Inhalt
    "preview.hs-hannover.de",           # Staging-Server (SSL defekt)
    "mediathek-f3.hs-hannover.de",      # DNS existiert nicht
}

# ---------------------------------------------------------------------------
# Ausgeschlossene Dateiendungen
# ---------------------------------------------------------------------------

BLOCKED_EXTENSIONS: set[str] = {
    # Office-Dokumente
    ".docx", ".doc", ".odt", ".rtf",
    ".xlsx", ".xls", ".ods", ".csv",
    ".pptx", ".ppt", ".odp",
    # Archive
    ".zip", ".tar", ".gz", ".7z", ".rar",
    # Bilder
    ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".webp", ".ico", ".bmp", ".tiff",
    # Medien
    ".mp4", ".mp3", ".avi", ".mov", ".wmv", ".wav", ".ogg",
    # Web-Assets
    ".css", ".js", ".json", ".xml",
    # Executables
    ".exe", ".msi", ".dmg",
}

# ---------------------------------------------------------------------------
# Qualitätsschwellwerte (identisch mit check_quality.py)
# ---------------------------------------------------------------------------

MIN_WORDS_HTML = 30
MIN_WORDS_PDF  = 50

# Formulierungen, die eine Seite als Fehlerseite ausweisen. Sie werden als
# Teilstring über den gesamten Body geprüft und müssen deshalb spezifisch
# genug sein, um in normalem Fließtext nicht vorzukommen.
#
# Bewusst NICHT in dieser Liste:
#   "404"             — steckt als Ziffernfolge in DOIs (978-3-658-40448-2),
#                       Seitenangaben ("S. 396-404"), PDF-Objektreferenzen und
#                       Base64-Bilddaten. Als HTTP-Status formulierte Treffer
#                       fängt stattdessen _GONE_MESSAGE_RE (siehe unten) ab.
#   "nicht verfügbar" — gewöhnliche deutsche Wendung ("Die Aufzeichnung ist
#                       nicht verfügbar"), die inhaltsreiche Seiten verwirft.
#                       Echte Fehlerseiten der HsH tragen ohnehin den Titel
#                       "Seite nicht gefunden".
ERROR_PATTERNS: list[str] = [
    "seite nicht gefunden",
    "page not found",
    "fehler beim laden",
    "zugriff verweigert",
    "access denied",
]

# ---------------------------------------------------------------------------
# HTTP-Status: Seite existiert nicht mehr
# ---------------------------------------------------------------------------

# Statuscodes, bei denen eine bereits gecachte Datei gelöscht wird.
# Nur 404/410 — 403/500/503 sind vorübergehend und dürfen nichts löschen.
GONE_STATUS: set[int] = {404, 410}

# Fallback für crawl4ai-Versionen, die keinen status_code liefern: nur explizit
# als HTTP-Status formulierte Meldungen zählen. Ein blosses "404" irgendwo im
# Text (z.B. in einer URL) darf keine Löschung auslösen.
_GONE_MESSAGE_RE = re.compile(
    r"(?:\bhttp\b|\bstatus(?:[ _-]?code)?\b)[^0-9]{0,12}(404|410)\b",
    re.IGNORECASE,
)


def is_gone_response(status_code: int | None, error_message: str = "") -> bool:
    """Gibt True zurück wenn die Seite serverseitig nicht mehr existiert (404/410).

    Ist *status_code* gesetzt, entscheidet ausschliesslich dieser. Nur wenn die
    verwendete crawl4ai-Version keinen Status liefert, wird ersatzweise die
    Fehlermeldung geprüft — dann aber nur bei eindeutiger HTTP-Formulierung.
    """
    if status_code is not None:
        return status_code in GONE_STATUS
    return bool(error_message) and bool(_GONE_MESSAGE_RE.search(error_message))


# ---------------------------------------------------------------------------
# URL-Filter
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Entfernt Fragment und Query, normalisiert Trailing-Slash, http→https.

    Zentral hier, damit main.py und resume_crawler.py garantiert dieselbe
    Normalisierung verwenden — sonst greifen Cache-Treffer und Löschabgleich
    zwischen den beiden Crawler-Varianten nicht zuverlässig.

    Der Trailing-Slash wird **auch am Root** entfernt: ``https://f4…de`` und
    ``https://f4…de/`` sind dieselbe Seite, landeten aber als zwei URLs im
    visited-Set und danach — gleicher Host, gleicher Slug "index" — auf
    demselben Dateinamen. Das waren die letzten 6 Kollisionen nach Phase A.
    """
    try:
        parsed = urlparse(url)
        scheme = "https" if parsed.scheme == "http" else parsed.scheme
        path = parsed.path
        if path.endswith("/"):
            path = path.rstrip("/")
        return parsed._replace(
            scheme=scheme, path=path, query="", fragment="", params=""
        ).geturl()
    except ValueError:
        return url


def slug_for_url(url: str) -> str:
    """Gibt den Dateinamen-Slug einer URL zurück (Pfad ohne führende Slashes).

    Bewusst **ohne** Host: der Slug ist der Glob-Schlüssel in
    ``cached_files_for_url()``. Er muss breiter sein als der Dateiname, damit
    auch Dateien nach altem Namensschema (ohne Host) noch gefunden werden.
    Die Präzision liefert ``matches_url()`` über die source_url im Header.
    """
    path = urlparse(url).path.strip("/")
    return slugify(path) if path else "index"


def host_slug(url: str) -> str:
    """Kürzel des Hosts für den Dateinamen — 'f5', 'www', 'service-it'.

    Für Subdomains von hs-hannover.de wird das Suffix abgeschnitten, alles
    andere (inkl. der Apex-Domain selbst) wird vollständig slugifiziert.
    Apex ergibt damit 'hs-hannover-de' und ist von 'www' unterscheidbar —
    beide sind eigene source_urls und dürfen nicht auf denselben Dateinamen
    fallen, sonst kehrt genau die Kollision zurück, die der Host beheben soll.
    """
    host = (urlparse(url).hostname or "").lower()
    suffix = "." + BASE_DOMAIN
    label = host[: -len(suffix)] if host.endswith(suffix) else host
    return slugify(label) or "host"


def filename_for_url(url: str, today: str) -> str:
    """Dateiname einer URL: ``<datum>_<host>_<pfad-slug>.md``.

    Einzige Stelle, an der ein Dateiname entsteht — main.py, resume_crawler.py
    und touch_cached_file() rufen alle hierher. Vorher bauten alle drei den
    Namen selbst, ohne Host: derselbe Pfad auf verschiedenen Subdomains ergab
    dieselbe Datei, die spätere überschrieb die frühere (129 belegte Verluste,
    siehe verifiziert.md).
    """
    return f"{today}_{host_slug(url)}_{slug_for_url(url)}.md"


def is_english_url(url: str) -> bool:
    """Gibt True zurück wenn die URL eine englischsprachige Seite ist (/en/…)."""
    path = urlparse(url).path
    return path == "/en" or path.startswith("/en/")


def is_typo3_id_url(url: str) -> bool:
    """Gibt True zurück für alte TYPO3-URLs mit ?id=… (z.B. index.php?id=4888).

    Diese URLs verlieren ihre Seiten-ID durch normalize_url (Query-Strip) und
    würden dann zur falschen Seite führen. Sie werden übersprungen, da die
    modernen Entsprechungen mit sauberen URLs bereits gecrawlt werden.
    """
    parsed = urlparse(url)
    return parsed.path.endswith("index.php") and "id=" in parsed.query


def has_blocked_extension(url: str) -> bool:
    """Gibt True zurück wenn die URL eine nicht unterstützte Dateiendung hat."""
    path = urlparse(url).path.lower()
    _, ext = os.path.splitext(path)
    return ext in BLOCKED_EXTENSIONS


def is_blocked_domain(url: str, allowed_domain: str) -> bool:
    """Gibt True zurück wenn die URL zu einer gesperrten Domain gehört."""
    netloc = urlparse(url).netloc.lower()
    if netloc in BLOCKED_DOMAINS:
        return True
    return not (netloc == allowed_domain or netloc.endswith("." + allowed_domain))


# ---------------------------------------------------------------------------
# Qualitätsprüfung
# ---------------------------------------------------------------------------


def is_low_quality(content_type: str, body: str) -> tuple[bool, str]:
    """Prüft ob ein Inhalt die Mindestziele für Qualität erfüllt.

    Gibt (True, Grund) zurück wenn der Inhalt verworfen werden soll,
    sonst (False, "").
    Entspricht den Löschkriterien aus check_quality.py.
    """
    words = len(body.split())
    threshold = MIN_WORDS_PDF if content_type == "pdf" else MIN_WORDS_HTML
    if words < threshold:
        return True, f"zu wenig Text ({words} Wörter, min. {threshold})"

    body_lower = body.lower()
    for pattern in ERROR_PATTERNS:
        if pattern in body_lower:
            return True, f'Fehlerseitenmuster: "{pattern}"'

    # Als HTTP-Status formulierte 404/410-Meldungen (Soft-404: Server liefert
    # 200, zeigt aber eine Fehlerseite). Dieselbe Regex wie is_gone_response,
    # damit ein blosses "404" in einer DOI oder Seitenzahl folgenlos bleibt.
    m = _GONE_MESSAGE_RE.search(body)
    if m:
        return True, f'Fehlerseitenmuster: HTTP-Status {m.group(1)}'

    return False, ""


# ---------------------------------------------------------------------------
# Content-Hash
# ---------------------------------------------------------------------------


_DIAERESE_VOR = re.compile("\u00a8([aouAOU])")
_UMLAUT = {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"}


def repariere_diakritika(text: str) -> str:
    """Zerrissene Umlaute aus der PDF-Konvertierung zusammensetzen.

    pymupdf4llm liefert bei manchen PDFs (LaTeX-Erzeugnisse, ältere Satz-
    programme) den Trema als eigenes Zeichen VOR dem Vokal — „Pr¨ufung",
    „m¨ussen" (U+00A8 + u) — oder als kombinierendes Zeichen danach
    (u + U+0308). Für BM25 ist „Pr¨ufung" ein anderes Token als „Prüfung",
    das Embedding sieht Buchstabensalat; am 21.09.2026 waren 38 Dateien von
    der ersten und 89 von der zweiten Form betroffen (g18 in eval_gold: kein
    Arm fand den Prüfungsplan).

    NFC fasst die kombinierende Form zusammen; die vorangestellte muss von
    Hand gedreht werden. Beides ist idempotent.
    """
    text = unicodedata.normalize("NFC", text)
    return _DIAERESE_VOR.sub(lambda m: _UMLAUT[m.group(1)], text)


def compute_hash(body: str) -> str:
    """SHA-256 der ersten 16 Hex-Zeichen über den Markdown-Body (ohne YAML)."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Bestehende Cache-Datei suchen
# ---------------------------------------------------------------------------


def split_frontmatter(text: str) -> tuple[list[str] | None, str]:
    """Zerlegt einen Dateiinhalt in YAML-Header-Zeilen und Rest.

    Gibt (None, text) zurück wenn kein gültiger Frontmatter-Block vorhanden ist.
    Der zurückgegebene Rest enthält alles nach der schliessenden ``---``-Zeile
    unverändert — inklusive der Leerzeile davor, damit ein Zusammensetzen den
    Body byteweise erhält (der content_hash bleibt gültig).
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[1:i], "".join(lines[i + 1:])
    return None, text


def _parse_header_fields(header_lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """Gibt die Header-Felder als dict und ihre ursprüngliche Reihenfolge zurück."""
    meta: dict[str, str] = {}
    order: list[str] = []
    for line in header_lines:
        key, sep, value = line.partition(":")
        key = key.strip()
        if sep and key:
            meta[key] = value.strip().strip('"')
            order.append(key)
    return meta, order


def read_header(filepath: Path) -> dict[str, str]:
    """Liest die YAML-Frontmatter-Felder einer Datei. Leeres dict bei Fehler."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except OSError:
        return {}
    header_lines, _ = split_frontmatter(text)
    if header_lines is None:
        return {}
    meta, _ = _parse_header_fields(header_lines)
    return meta


def matches_url(filepath: Path, url: str) -> bool:
    """Prüft ob die source_url einer Datei auf *url* zeigt (nach Normalisierung).

    Nötig, weil verschiedene Hosts denselben Slug erzeugen können — z.B. ergibt
    die Startseite jeder Subdomain den Slug "index". Ohne diese Prüfung würde
    ein Lösch- oder Aufräumlauf fremde Dateien treffen.
    """
    stored = read_header(filepath).get("source_url", "")
    return bool(stored) and normalize_url(stored) == normalize_url(url)


def cached_files_for_url(url: str, output_dir: Path) -> list[Path]:
    """Alle gecachten Dateien die tatsächlich zu *url* gehören, neueste zuerst."""
    slug = slug_for_url(url)
    matches = sorted(output_dir.glob(f"*_{slug}.md"), reverse=True)
    return [p for p in matches if matches_url(p, url)]


def find_cached_file(url: str, output_dir: Path) -> Path | None:
    """Gibt den Pfad zur neuesten gecachten Datei für eine URL zurück, oder None."""
    matches = cached_files_for_url(url, output_dir)
    return matches[0] if matches else None


def read_stored_hash(url: str, output_dir: Path) -> str | None:
    """Liest den gespeicherten content_hash aus dem YAML-Header der gecachten Datei."""
    cached = find_cached_file(url, output_dir)
    if cached is None:
        return None
    return read_header(cached).get("content_hash") or None


def touch_cached_file(url: str, output_dir: Path, today: str) -> Path | None:
    """Datiert eine inhaltlich unveränderte Datei auf *today* um.

    Ohne diesen Schritt behielte eine Seite, deren Inhalt sich nie ändert, für
    immer ihr altes crawl_date und würde bei jedem Lauf erneut als veraltet
    eingestuft und abgeholt.

    ``crawl_date``   = wann zuletzt geprüft   (wird hier aktualisiert)
    ``content_date`` = wann zuletzt inhaltlich geändert (bleibt unangetastet)

    Der Body wird byteweise übernommen, der content_hash bleibt damit gültig.
    Gibt den neuen Pfad zurück, oder None wenn nichts zu tun war.
    """
    cached = find_cached_file(url, output_dir)
    if cached is None:
        return None
    try:
        text = cached.read_text(encoding="utf-8")
    except OSError:
        return None

    header_lines, rest = split_frontmatter(text)
    if header_lines is None:
        return None

    meta, order = _parse_header_fields(header_lines)
    if meta.get("crawl_date") == today and cached.name.startswith(today + "_"):
        return cached

    # Migration älterer Dateien: content_date fehlt noch → das bisherige
    # crawl_date war das letzte tatsächliche Schreibdatum.
    if "content_date" not in meta:
        meta["content_date"] = meta.get("crawl_date", "")
        insert_at = order.index("crawl_date") + 1 if "crawl_date" in order else len(order)
        order.insert(insert_at, "content_date")
    meta["crawl_date"] = today

    new_header = "---\n" + "".join(f'{k}: "{meta[k]}"\n' for k in order) + "---\n"

    target = output_dir / filename_for_url(url, today)
    if target != cached and target.exists() and not matches_url(target, url):
        # Fremde Datei mit gleichem Slug (anderer Host) — nicht überschreiben.
        target = cached

    try:
        target.write_text(new_header + rest, encoding="utf-8")
        if target != cached:
            cached.unlink()
    except OSError as exc:
        logger.warning("Umdatieren fehlgeschlagen für %s: %s", url, exc)
        return None
    return target


def delete_cached_files(url: str, output_dir: Path) -> int:
    """Löscht alle gecachten Dateien zu *url* (Seite existiert nicht mehr).

    Betrifft nur Dateien, deren source_url tatsächlich auf *url* zeigt.
    Gibt die Anzahl gelöschter Dateien zurück.
    """
    removed = 0
    for filepath in cached_files_for_url(url, output_dir):
        try:
            filepath.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("Löschen fehlgeschlagen: %s — %s", filepath, exc)
    return removed


def cleanup_old_slug_files(url: str, output_dir: Path, keep: Path) -> None:
    """Löscht ältere Versionen derselben URL (Duplikate nach Re-Crawl).

    Der Abgleich läuft über die source_url im Header, nicht nur über den Slug —
    sonst würde z.B. die Startseite einer Fakultäts-Subdomain die Startseite
    einer anderen löschen (beide haben den Slug "index").
    """
    for old in cached_files_for_url(url, output_dir):
        if old != keep:
            try:
                old.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Delta-Erfassung: welche Dateien hat dieser Lauf geschrieben?
# ---------------------------------------------------------------------------

# Ein Crawl-Lauf berührt tausende Seiten, schreibt aber nur die tatsächlich
# geänderten. Genau diese Teilmenge muss anschließend zum HPC — nicht das
# ganze Korpus. Die Liste wird von save_markdown_if_changed() gefüllt (beide
# Crawler-Varianten haben eine eigene Kopie dieser Funktion) und am Ende des
# Laufs zu einem einzigen .tar.xz gepackt.
#
# Modul-global statt durchgereicht, weil die Schreibstellen tief in
# process_html()/process_pdf() sitzen und sonst jede Signatur der Aufrufkette
# einen zusätzlichen Parameter bräuchte.

_WRITTEN_FILES: list[Path] = []


def record_written(filepath: Path) -> None:
    """Vermerkt eine neu geschriebene oder geänderte Markdown-Datei."""
    _WRITTEN_FILES.append(filepath)


def written_files() -> list[Path]:
    """Gibt die in diesem Lauf geschriebenen Dateien zurück (ohne Duplikate)."""
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in _WRITTEN_FILES:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def reset_written() -> None:
    """Leert die Liste — für Tests und aufeinanderfolgende Läufe im selben Prozess."""
    _WRITTEN_FILES.clear()


def pack_delta_archive(archive_path: Path,
                       paths: list[Path] | None = None) -> tuple[int, int]:
    """Packt die geänderten Markdown-Dateien in ein .tar.xz.

    Die Dateien liegen flach im Archiv (ohne Verzeichnisanteil), damit sie sich
    auf dem HPC in ein beliebiges Zielverzeichnis entpacken lassen:

        tar xJf delta_2026-08-05.tar.xz -C var/korpus_delta/

    Übersprungen werden Dateien, die zwischenzeitlich wieder verschwunden sind
    (z.B. von cleanup_old_slug_files entfernte Vorgängerversionen).

    Gibt (Anzahl Dateien, Archivgröße in Bytes) zurück. Existiert keine
    geänderte Datei, wird kein Archiv angelegt und (0, 0) zurückgegeben.
    """
    files = [p for p in (paths if paths is not None else written_files()) if p.exists()]
    if not files:
        logger.info("Keine geänderten Seiten — kein Delta-Archiv erzeugt.")
        return 0, 0

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    # preset=6 statt des Defaults 9: bei Markdown bringt die höchste Stufe nur
    # wenige Prozent, kostet aber ein Vielfaches an Zeit.
    with tarfile.open(archive_path, "w:xz", preset=6) as tar:
        for p in files:
            tar.add(p, arcname=p.name)

    size = archive_path.stat().st_size
    logger.info("Delta-Archiv: %s  (%d Datei(en), %.1f KB)",
                archive_path, len(files), size / 1024)
    return len(files), size
