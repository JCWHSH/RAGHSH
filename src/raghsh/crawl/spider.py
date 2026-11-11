"""
HsH Web-Spider — Automatischer Crawler für die Website der Hochschule Hannover.

Kurzbeschreibung
----------------
Crawlt die gesamte Website der Hochschule Hannover (hs-hannover.de) und speichert
jede Seite als strukturierte Markdown-Datei mit YAML-Metadaten-Header. Bildet die
erste Stufe der RAG-Pipeline (Retrieval-Augmented Generation).

Ausführliche Beschreibung
--------------------------
Das Programm arbeitet als Breadth-First-Spider: Ausgehend von einer oder mehreren
Startseiten (SEED_URLS) werden alle internen Links entdeckt, in eine Warteschlange
eingereiht und nacheinander abgearbeitet.

Wesentliche Funktionen:

1. BFS-Crawling (Breadth-First Search)
   Die Hauptschleife in main() verwaltet eine deque als Warteschlange und ein
   visited-Set zur Duplikaterkennung. Gefundene interne Links werden normalisiert
   (Fragment und Query-Parameter entfernt, Slash-Normalisierung) und nur dann
   eingereiht, wenn sie zur erlaubten Domain (ALLOWED_DOMAIN) gehören.

2. Cache-Prüfung (is_fresh)
   Vor jedem Crawl prüft is_fresh(), ob bereits eine aktuell genug gecachte
   Markdown-Datei für diese URL existiert. Ist die Datei jünger als MAX_AGE_DAYS,
   wird die Seite trotzdem heruntergeladen (für Link-Extraktion), aber nicht neu
   gespeichert. PDFs werden bei frischem Cache vollständig übersprungen.

2b. Änderungserkennung und Umdatierung
   Jede Datei trägt einen content_hash über ihren Markdown-Body. Stimmt der Hash
   des neuen Abrufs mit dem gespeicherten überein, wird nicht neu geschrieben —
   die vorhandene Datei wird lediglich auf das heutige crawl_date umdatiert
   (Dateiname und Header). Ohne diesen Schritt bliebe eine nie geänderte Seite
   dauerhaft "veraltet" und würde bei jedem Lauf erneut abgeholt.
   Das Feld content_date hält fest, wann sich der Inhalt zuletzt wirklich änderte.

2c. Entfernte Seiten (404/410)
   Antwortet eine URL mit 404 oder 410, wird die zugehörige gecachte Datei
   gelöscht. So verschwinden von der Website entfernte Seiten auch aus dem
   Korpus. Andere Fehlerstatus (403, 5xx) gelten als vorübergehend und lösen
   niemals eine Löschung aus.
   Achtung: Seiten, die nirgends mehr verlinkt sind, werden nie besucht und
   liefern daher auch keinen 404 — die bleiben bis auf Weiteres im Korpus.

3. HTML-Verarbeitung (process_html)
   Nutzt die Bibliothek Crawl4AI mit einem AsyncWebCrawler. Eine CrawlerRunConfig
   mit CSS-Selektoren und Ausschluss-Regeln filtert Boilerplate (Navigation, Footer,
   Cookie-Banner, Sidebars) heraus und extrahiert nur den Hauptinhalt. Das Ergebnis
   wird als Markdown gespeichert. Über result.links werden neue URLs entdeckt.

4. PDF-Verarbeitung (process_pdf)
   PDF-Dateien werden per httpx heruntergeladen und mit pymupdf4llm in Markdown
   konvertiert. Die Erkennung erfolgt über Dateiendung oder Content-Type-Header.

5. Ausgabeformat
   Jede gespeicherte Datei beginnt mit einem YAML-Frontmatter-Block:
     ---
     source_url: "https://..."
     title: "Seitentitel"
     crawl_date: "2026-03-08"     # zuletzt geprüft
     content_date: "2026-01-14"   # zuletzt inhaltlich geändert
     content_type: "html"
     faculty: "f1"
     content_hash: "a1b2c3d4e5f60718"
     ---
   Dateinamen folgen dem Schema: YYYY-MM-DD_url-pfad-slug.md

6. Fehlerprotokoll (write_error_report)
   Alle fehlgeschlagenen URLs werden mit Fehlermeldung und Referrer (die Seite,
   auf der der Link gefunden wurde) in einer Excel-Datei (YYYY-MM-DD_fehler.xlsx)
   im Ausgabeverzeichnis gespeichert.

Konfiguration (Konstanten am Anfang der Datei):
   SEED_URLS        — Startseiten des Crawls
   MAX_PAGES        — Maximale Anzahl zu crawlender Seiten
   ALLOWED_DOMAIN   — Nur Links innerhalb dieser Domain werden verfolgt
   MAX_AGE_DAYS     — Wie alt eine gecachte Datei sein darf, bevor neu gecrawlt wird
   OUTPUT_DIR       — Zielverzeichnis für Markdown-Dateien
   RATE_LIMIT_SECONDS — Pause zwischen zwei Requests (Server schonen)

Abhängigkeiten: crawl4ai, httpx, pymupdf4llm, python-slugify, openpyxl
"""

import asyncio
import logging
import os
import tempfile
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

import httpx
import openpyxl
import pymupdf4llm
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from slugify import slugify

from raghsh import konfig
from raghsh.crawl.crawler_utils import (
    BLOCKED_DOMAINS,
    cleanup_old_slug_files,
    compute_hash,
    delete_cached_files,
    filename_for_url,
    has_blocked_extension,
    is_english_url,
    is_gone_response,
    is_low_quality,
    is_typo3_id_url,
    normalize_url,
    pack_delta_archive,
    read_stored_hash,
    record_written,
    touch_cached_file,
    repariere_diakritika,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED_URLS = ["https://www.hs-hannover.de/"]
MAX_PAGES = 10000
ALLOWED_DOMAIN = "hs-hannover.de"
MAX_AGE_DAYS = 7  # Re-crawl pages older than this many days

OUTPUT_DIR = konfig.KORPUS
# Delta-Archive liegen neben var/korpus, nicht darin — sonst würden sie
# vom Vectorizer und von build_doc_index.py als Korpusdateien mitgescannt.
DELTA_DIR  = konfig.DATEN
RATE_LIMIT_SECONDS = 2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_filename(url: str, today: str) -> str:
    """Create a filename like ``2026-02-26_www_studium-studienangebot.md``.

    Delegiert an crawler_utils.filename_for_url — der Host gehört in den Namen,
    sonst kollidieren gleiche Pfade auf verschiedenen Subdomains.
    """
    return filename_for_url(url, today)


def extract_faculty(url: str) -> str:
    """Leitet die Fakultät aus der URL ab.

    Gibt z.B. 'f1' für 'https://f1.hs-hannover.de/...' zurück,
    oder '' wenn die URL keiner Fakultät zugeordnet ist.
    """
    import re
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    m = re.match(r'^(f\d+)\.hs-hannover\.de$', host)
    return m.group(1) if m else ""


def build_yaml_header(
    source_url: str,
    title: str,
    crawl_date: str,
    content_type: str = "html",
    content_hash: str = "",
    content_date: str = "",
) -> str:
    """Return a YAML front-matter block.

    ``crawl_date``   — wann die Seite zuletzt geprüft wurde
    ``content_date`` — wann sich ihr Inhalt zuletzt tatsächlich geändert hat
    """
    faculty = extract_faculty(source_url)
    return (
        "---\n"
        f'source_url: "{source_url}"\n'
        f'title: "{title}"\n'
        f'crawl_date: "{crawl_date}"\n'
        f'content_date: "{content_date or crawl_date}"\n'
        f'content_type: "{content_type}"\n'
        f'faculty: "{faculty}"\n'
        f'content_hash: "{content_hash}"\n'
        "---\n"
    )


def save_markdown_if_changed(
    url: str, filepath: Path, header: str, body: str, today: str
) -> bool:
    """Schreibt Datei nur wenn sich der Inhalt geändert hat.

    Vergleicht den Hash des neuen Bodys mit dem gespeicherten Hash.
    Gibt True zurück wenn geschrieben wurde, False wenn unverändert.

    Bei unverändertem Inhalt wird die vorhandene Datei auf das heutige Datum
    umdatiert (crawl_date), damit sie nicht bei jedem Lauf erneut als veraltet
    gilt. Löscht ältere Versionen derselben URL nach erfolgreichem Schreiben.
    """
    stored = read_stored_hash(url, OUTPUT_DIR)
    new_hash = compute_hash(body)
    if stored == new_hash:
        touched = touch_cached_file(url, OUTPUT_DIR, today)
        logger.info("Inhalt unverändert — %s: %s",
                    "umdatiert" if touched else "übersprungen", url)
        return False
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(header + "\n" + body, encoding="utf-8")
    cleanup_old_slug_files(url, OUTPUT_DIR, keep=filepath)
    record_written(filepath)
    logger.info("Gespeichert (%s): %s", "neu" if stored is None else "geändert", filepath)
    return True


def is_same_domain(url: str) -> bool:
    """Return True only if *url* belongs to hs-hannover.de or a subdomain,
    and is not in the BLOCKED_DOMAINS list."""
    netloc = urlparse(url).netloc.lower()
    if netloc in BLOCKED_DOMAINS:
        return False
    return netloc == ALLOWED_DOMAIN or netloc.endswith("." + ALLOWED_DOMAIN)


def is_fresh(url: str) -> bool:
    """Return True if a recent enough cached file exists for *url*.

    Searches OUTPUT_DIR for files matching ``*_<slug>.md``, checks the date,
    and verifies that the stored source_url belongs to the same hostname —
    preventing false cache hits when different subdomains share the slug "index".
    """
    parsed = urlparse(url)
    url_host = parsed.hostname or ""
    path = parsed.path.strip("/")
    slug = slugify(path) if path else "index"
    cutoff = date.today() - timedelta(days=MAX_AGE_DAYS)
    for filepath in OUTPUT_DIR.glob(f"*_{slug}.md"):
        stem = filepath.stem
        date_part = stem.split("_")[0]
        try:
            file_date = date.fromisoformat(date_part)
        except ValueError:
            continue
        if file_date < cutoff:
            continue
        try:
            text = filepath.read_text(encoding="utf-8")
            cached_host = ""
            for line in text.splitlines()[:6]:
                if line.startswith("source_url:"):
                    cached_host = urlparse(line.split('"')[1]).hostname or ""
                    break
            if cached_host != url_host:
                continue
        except OSError:
            continue
        logger.info("Skipping (fresh cache from %s): %s", file_date, url)
        return True
    return False


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------


async def process_pdf(url: str, today: str) -> bool:
    """Download a PDF and convert it to Markdown via pymupdf4llm.

    Gibt True zurück wenn das PDF serverseitig nicht mehr existiert (404/410)
    und eine vorhandene gecachte Datei deshalb gelöscht wurde.
    """
    logger.info("Downloading PDF: %s", url)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url)
        if is_gone_response(resp.status_code):
            removed = delete_cached_files(url, OUTPUT_DIR)
            logger.info("HTTP %d — PDF existiert nicht mehr, %d Datei(en) gelöscht: %s",
                        resp.status_code, removed, url)
            return True
        resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        md_text = repariere_diakritika(pymupdf4llm.to_markdown(tmp_path))
    finally:
        os.unlink(tmp_path)

    low_q, reason = is_low_quality("pdf", md_text)
    if low_q:
        logger.info("PDF übersprungen (%s): %s", reason, url)
        return False

    title = Path(urlparse(url).path).stem or "PDF Document"
    new_hash = compute_hash(md_text)
    header = build_yaml_header(url, title, today, content_type="pdf",
                               content_hash=new_hash)
    filename = make_filename(url, today)
    save_markdown_if_changed(url, OUTPUT_DIR / filename, header, md_text, today)
    return False


# ---------------------------------------------------------------------------
# HTML handling
# ---------------------------------------------------------------------------


async def process_html(
    crawler: AsyncWebCrawler,
    url: str,
    today: str,
    config: CrawlerRunConfig,
    *,
    save: bool = True,
):
    """Crawl an HTML page and save its Markdown representation.

    Gibt ``(result, gone)`` zurück. ``gone=True`` bedeutet: die Seite liefert
    404/410, eine vorhandene gecachte Datei wurde gelöscht und es wurden keine
    Links extrahiert (eine Fehlerseite verlinkt nichts Brauchbares).
    """
    logger.info("Crawling: %s", url)
    result = await crawler.arun(url=url, config=config)

    status = getattr(result, "status_code", None)
    if is_gone_response(status, result.error_message or ""):
        removed = delete_cached_files(url, OUTPUT_DIR)
        logger.info("HTTP %s — Seite existiert nicht mehr, %d Datei(en) gelöscht: %s",
                    status if status is not None else "404/410", removed, url)
        return result, True

    if not result.success:
        raise RuntimeError(result.error_message or "Crawl failed (unknown reason)")

    # Prefer markdown_v2.raw_markdown if available, fall back to .markdown
    md_content = ""
    if hasattr(result, "markdown_v2") and result.markdown_v2:
        md_content = getattr(result.markdown_v2, "raw_markdown", "") or ""
    if not md_content:
        md_content = result.markdown or ""

    title = ""
    if hasattr(result, "metadata") and result.metadata:
        title = result.metadata.get("title", "") or ""
    if not title:
        # Fallback: first Markdown heading
        for line in md_content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
    if not title:
        title = url

    if save:
        low_q, reason = is_low_quality("html", md_content)
        if low_q:
            logger.info("HTML übersprungen (%s): %s", reason, url)
            return result, False
        new_hash = compute_hash(md_content)
        header = build_yaml_header(url, title, today, content_type="html",
                                   content_hash=new_hash)
        filename = make_filename(url, today)
        save_markdown_if_changed(url, OUTPUT_DIR / filename, header, md_content, today)
    return result, False


# ---------------------------------------------------------------------------
# Error report
# ---------------------------------------------------------------------------


def write_error_report(errors: list[dict], today: str) -> None:
    """Write a list of crawl errors to an Excel file in OUTPUT_DIR."""
    if not errors:
        logger.info("No errors — skipping error report.")
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Fehler"

    headers = ["URL", "Gefunden auf (Referrer)", "Fehlermeldung"]
    ws.append(headers)

    # Bold header row
    for cell in ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)

    for entry in errors:
        ws.append([entry["url"], entry["referrer"], entry["error"]])

    # Auto-fit column widths (heuristic)
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 80)

    filepath = OUTPUT_DIR / f"{today}_fehler.xlsx"
    wb.save(filepath)
    logger.info("Error report saved: %s (%d entries)", filepath, len(errors))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_run_config() -> CrawlerRunConfig:
    """Die CrawlerRunConfig für hs-hannover.de (TYPO3-Struktur).

    Als eigene Funktion herausgezogen, damit recrawl_urls.py exakt dieselbe
    Konfiguration verwendet — sonst driften Content-Targeting und Boilerplate-
    Ausschluss zwischen Voll- und Nachcrawl auseinander und erzeugen für
    dieselbe Seite unterschiedliches Markdown (und damit content_hash).
    """
    return CrawlerRunConfig(
        # ── Content targeting (HsH TYPO3 structure) ───────────────
        css_selector="main, .content-main, #content, .frame-default",

        # ── Exclude boilerplate elements ──────────────────────────
        excluded_tags=[
            "nav", "header", "footer", "aside", "form",
            "iframe", "script", "style", "noscript",
        ],
        excluded_selector=(
            # Navigation
            "nav.main-menu, .main-menu, .main-menu__mainmenu, "
            "ul#tabmenu1, ul[role='menubar'], "
            # Header & Quicklinks
            "header, .quicklinks, "
            # Breadcrumb
            ".breadcrumb, .breadcrumbs, [aria-label='breadcrumb'], "
            # Footer
            "footer, .footer, .site-footer, "
            # Cookiebot cookie-consent banner
            "#CybotCookiebotDialog, #CybotCookiebotDialogBody, "
            "[id*='Cookiebot'], [class*='cookiebot'], "
            # Sidebar & misc
            "aside, .sidebar, .widget, "
            ".search-form, form[role='search'], "
            ".social-media, .share-buttons"
        ),

        # ── Wait for main content to render ───────────────────────
        # wait_for="css:main" entfernt: Seiten ohne <main>-Element
        # (z.B. karriere.hs-hannover.de) liefen sonst in einen 60s-Timeout.
        # Das Content-Targeting übernimmt css_selector zuverlässig allein.
    )


async def is_pdf_url(url: str) -> bool:
    """Check if *url* points to a PDF — by extension or Content-Type header."""
    if url.lower().endswith(".pdf"):
        return True
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.head(url, timeout=10)
            content_type = resp.headers.get("content-type", "")
            return "application/pdf" in content_type.lower()
    except httpx.HTTPError:
        return False


async def main() -> None:
    today = date.today().isoformat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    success_count = 0
    fail_count = 0
    skip_count = 0
    gone_count = 0
    errors: list[dict] = []
    referrers: dict[str, str] = {}

    config = build_run_config()

    # BFS state
    visited: set[str] = set()
    queue: deque[str] = deque()

    for seed in SEED_URLS:
        norm = normalize_url(seed)
        if norm not in visited:
            visited.add(norm)
            queue.append(norm)
            referrers[norm] = "(seed)"

    pages_processed = 0

    async with AsyncWebCrawler() as crawler:
        while queue and pages_processed < MAX_PAGES:
            url = queue.popleft()
            pages_processed += 1

            try:
                if has_blocked_extension(url) or is_english_url(url):
                    skip_count += 1
                    continue
                fresh = is_fresh(url)
                if fresh:
                    skip_count += 1
                if await is_pdf_url(url):
                    if not fresh:
                        if await process_pdf(url, today):
                            gone_count += 1
                        else:
                            success_count += 1
                else:
                    result, gone = await process_html(crawler, url, today, config,
                                                      save=not fresh)
                    if gone:
                        # Fehlerseite — nicht zählen und keine Links verfolgen.
                        gone_count += 1
                    else:
                        if not fresh:
                            success_count += 1
                        # Extract internal links and enqueue new ones
                        links_data = getattr(result, "links", {}) or {}
                        for link_dict in links_data.get("internal", []):
                            href = link_dict.get("href", "")
                            if not href or not href.startswith("http"):
                                continue
                            norm_href = normalize_url(href)
                            if (norm_href not in visited
                                    and is_same_domain(norm_href)
                                    and not is_english_url(norm_href)
                                    and not has_blocked_extension(norm_href)
                                    and not is_typo3_id_url(href)):
                                visited.add(norm_href)
                                queue.append(norm_href)
                                referrers[norm_href] = url
            except Exception as exc:
                logger.error("Error processing %s: %s", url, exc)
                fail_count += 1
                errors.append({
                    "url": url,
                    "referrer": referrers.get(url, "(unknown)"),
                    "error": str(exc),
                })

            # Rate-limit between requests (skip after the last one)
            if queue and pages_processed < MAX_PAGES:
                await asyncio.sleep(RATE_LIMIT_SECONDS)

    write_error_report(errors, today)

    # Nur die tatsächlich geänderten Seiten als eine Datei bündeln — das ist
    # genau die Menge, die anschließend zum HPC muss.
    n_delta, _ = pack_delta_archive(DELTA_DIR / f"delta_{today}.tar.xz")

    logger.info(
        "Done. %d succeeded, %d failed, %d skipped (fresh cache), "
        "%d gone (404/410 — Datei gelöscht) out of %d URLs visited.",
        success_count,
        fail_count,
        skip_count,
        gone_count,
        pages_processed,
    )
    if n_delta:
        logger.info("Nächster Schritt: Delta-Archiv zum HPC übertragen — "
                    "siehe PIPELINE.md")


if __name__ == "__main__":
    asyncio.run(main())
