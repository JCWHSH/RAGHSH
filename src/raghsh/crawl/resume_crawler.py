"""
Resume-Crawler — Setzt einen unterbrochenen Crawl auf Basis vorhandener Markdown-Dateien fort.

Kurzbeschreibung
----------------
Analysiert die bereits gescrapten Markdown-Dateien in var/korpus/, extrahiert alle
darin enthaltenen Links und ermittelt, welche URLs noch nicht gescrapet wurden oder
veraltet sind. Startet dann den Crawl-Prozess nur für diese fehlenden URLs.

Ablauf
------
1. Alle Markdown-Dateien in var/korpus/ einlesen.
2. Aus dem YAML-Header: source_url + crawl_date → Menge der bekannten URLs aufbauen.
3. Aus dem Markdown-Body: alle internen Links extrahieren (Ziel-URLs entdecken).
4. Drei Listen erstellen und ausgeben:
     a) Bereits gescrapt & aktuell  (jünger als MAX_AGE_DAYS)
     b) Bereits gescrapt & veraltet (älter als MAX_AGE_DAYS → wird neu gecrawlt)
     c) Noch nicht gescrapt         (in Links gefunden, aber keine Datei vorhanden)
5. Nur Listen b) und c) werden gecrawlt (kein Doppel-Crawl aktueller Dateien).

Pflege des Bestands während des Laufs
-------------------------------------
- Unveränderter Inhalt (gleicher content_hash): die Datei wird nicht neu
  geschrieben, sondern nur auf das heutige crawl_date umdatiert. Sonst bliebe
  sie dauerhaft "veraltet" und würde bei jedem Lauf erneut abgeholt.
- HTTP 404/410: die gecachte Datei wird gelöscht — die Seite existiert nicht
  mehr. Andere Fehlerstatus (403, 5xx) sind vorübergehend und löschen nie.

Aufruf:
    python -m raghsh.crawl.resume_crawler            # Analyse + Crawl
    python -m raghsh.crawl.resume_crawler --dry-run  # Nur Analyse, kein Crawl

Abhängigkeiten: crawl4ai, httpx, pymupdf4llm, python-slugify, openpyxl
"""

import argparse
import asyncio
import logging
import os
import re
import sys
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
# Konfiguration — identisch mit main.py
# ---------------------------------------------------------------------------

SEED_URLS      = ["https://www.hs-hannover.de/"]
MAX_PAGES      = 100000
ALLOWED_DOMAIN = "hs-hannover.de"
MAX_AGE_DAYS   = 60

OUTPUT_DIR         = konfig.KORPUS
# Delta-Archive liegen neben var/korpus, nicht darin — sonst würden sie
# vom Vectorizer und von build_doc_index.py als Korpusdateien mitgescannt.
DELTA_DIR          = konfig.DATEN
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
# URL-Hilfsfunktionen (aus main.py übernommen)
# ---------------------------------------------------------------------------


def is_same_domain(url: str) -> bool:
    """Gibt True zurück wenn url zur erlaubten Domain gehört und nicht geblockt ist."""
    netloc = urlparse(url).netloc.lower()
    if netloc in BLOCKED_DOMAINS:
        return False
    return netloc == ALLOWED_DOMAIN or netloc.endswith("." + ALLOWED_DOMAIN)


def make_filename(url: str, today: str) -> str:
    """Wie main.make_filename — gemeinsame Implementierung in crawler_utils."""
    return filename_for_url(url, today)


def extract_faculty(url: str) -> str:
    """Leitet die Fakultät aus der URL ab (z.B. 'f1' für f1.hs-hannover.de)."""
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
    """Erzeugt den YAML-Frontmatter-Block (identisch mit main.py).

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


def save_markdown_if_changed(url: str, filepath: Path, header: str, body: str,
                             today: str) -> bool:
    """Schreibt Datei nur wenn sich der Inhalt geändert hat. Gibt True zurück wenn geschrieben.

    Bei unverändertem Inhalt wird die vorhandene Datei auf das heutige crawl_date
    umdatiert, damit sie nicht bei jedem Lauf erneut als veraltet gilt.
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


# ---------------------------------------------------------------------------
# Analyse der vorhandenen Markdown-Dateien
# ---------------------------------------------------------------------------

# Regex für Markdown-Links: [Text](https://...) und rohe https://-URLs
_MD_LINK_RE  = re.compile(r'\[.*?\]\((https?://[^)\s]+)\)')
_RAW_LINK_RE = re.compile(r'(?<![(\["])https?://[^\s)"\'<>]+')


def parse_frontmatter(text: str) -> dict:
    """Extrahiert Schlüssel-Wert-Paare aus einem YAML-Frontmatter-Block."""
    meta = {}
    if not text.startswith("---"):
        return meta
    parts = text.split("---", 2)
    if len(parts) < 3:
        return meta
    for line in parts[1].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
    return meta


def extract_links_from_body(body: str) -> set[str]:
    """Findet alle http(s)-URLs im Markdown-Body."""
    found: set[str] = set()
    for m in _MD_LINK_RE.finditer(body):
        found.add(m.group(1))
    for m in _RAW_LINK_RE.finditer(body):
        url = m.group(0).rstrip(".,;:!?)")
        # Eckige Klammern in der URL deuten auf IPv6-ähnliche Syntax hin → überspringen
        if "[" not in url and "]" not in url:
            found.add(url)
    return found


def analyse_ingested_dir() -> tuple[dict, dict, set[str]]:
    """Analysiert var/korpus/ und gibt drei Strukturen zurück:

    fresh    : {norm_url: filepath}   — aktuell gescrapte URLs
    stale    : {norm_url: filepath}   — veraltete gescrapte URLs
    discovered: set[norm_url]         — in Links gefundene, noch nicht gescrapte URLs
    """
    cutoff = date.today() - timedelta(days=MAX_AGE_DAYS)

    fresh:      dict[str, Path] = {}
    stale:      dict[str, Path] = {}
    all_links:  set[str]        = set()

    md_files = sorted(OUTPUT_DIR.glob("*.md"))
    logger.info("Analysiere %d Markdown-Dateien …", len(md_files))

    for filepath in md_files:
        try:
            text = filepath.read_text(encoding="utf-8")
        except Exception:
            continue

        meta = parse_frontmatter(text)
        source_url = meta.get("source_url", "")
        crawl_date_str = meta.get("crawl_date", "")

        if not source_url:
            continue

        norm = normalize_url(source_url)

        # Frische-Prüfung anhand crawl_date im Header
        try:
            file_date = date.fromisoformat(crawl_date_str)
            is_fresh  = file_date >= cutoff
        except ValueError:
            is_fresh = False

        if is_fresh:
            fresh[norm] = filepath
        else:
            stale[norm] = filepath

        # Links aus dem Body extrahieren
        if "---" in text:
            parts = text.split("---", 2)
            body = parts[2] if len(parts) == 3 else ""
        else:
            body = text

        for link in extract_links_from_body(body):
            try:
                if is_typo3_id_url(link):
                    continue
                norm_link = normalize_url(link)
                if is_same_domain(norm_link) and not has_blocked_extension(norm_link):
                    all_links.add(norm_link)
            except Exception:
                continue

    # URLs die in Links auftauchen, aber nicht als fresh bekannt sind
    all_scraped = set(fresh) | set(stale)
    discovered = all_links - all_scraped

    return fresh, stale, discovered


def print_report(fresh: dict, stale: dict, discovered: set) -> None:
    """Gibt eine übersichtliche Analyse auf der Konsole aus."""
    total = len(fresh) + len(stale) + len(discovered)
    print()
    print("=" * 70)
    print("  Crawl-Analyse")
    print("=" * 70)
    print(f"  Bereits gescrapt & aktuell   : {len(fresh):6d} URLs")
    print(f"  Bereits gescrapt & veraltet  : {len(stale):6d} URLs  → werden neu gecrawlt")
    print(f"  In Links entdeckt, fehlt noch: {len(discovered):6d} URLs  → werden gecrawlt")
    print(f"  ──────────────────────────────────────")
    print(f"  Gesamt bekannte URLs          : {total:6d}")
    print("=" * 70)

    if stale:
        print(f"\nVeraltete Dateien (erste 10 von {len(stale)}):")
        for url in list(stale)[:10]:
            print(f"  {url}")
        if len(stale) > 10:
            print(f"  … und {len(stale) - 10} weitere")

    if discovered:
        print(f"\nFehlende URLs (erste 10 von {len(discovered)}):")
        for url in list(discovered)[:10]:
            print(f"  {url}")
        if len(discovered) > 10:
            print(f"  … und {len(discovered) - 10} weitere")
    print()


# ---------------------------------------------------------------------------
# Crawl-Logik (aus main.py übernommen)
# ---------------------------------------------------------------------------


async def is_pdf_url(url: str) -> bool:
    if url.lower().endswith(".pdf"):
        return True
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.head(url, timeout=10)
            return "application/pdf" in resp.headers.get("content-type", "").lower()
    except httpx.HTTPError:
        return False


async def process_pdf(url: str, today: str) -> bool:
    """Lädt ein PDF und konvertiert es nach Markdown.

    Gibt True zurück wenn das PDF nicht mehr existiert (404/410) und eine
    vorhandene gecachte Datei deshalb gelöscht wurde.
    """
    logger.info("PDF herunterladen: %s", url)
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
    save_markdown_if_changed(url, OUTPUT_DIR / make_filename(url, today), header,
                             md_text, today)
    return False


async def process_html(
    crawler: AsyncWebCrawler,
    url: str,
    today: str,
    config: CrawlerRunConfig,
    *,
    save: bool = True,
):
    logger.info("Crawling: %s", url)
    result = await crawler.arun(url=url, config=config)

    status = getattr(result, "status_code", None)
    if is_gone_response(status, result.error_message or ""):
        removed = delete_cached_files(url, OUTPUT_DIR)
        logger.info("HTTP %s — Seite existiert nicht mehr, %d Datei(en) gelöscht: %s",
                    status if status is not None else "404/410", removed, url)
        return result, True

    if not result.success:
        raise RuntimeError(result.error_message or "Crawl fehlgeschlagen")

    md_content = ""
    if hasattr(result, "markdown_v2") and result.markdown_v2:
        md_content = getattr(result.markdown_v2, "raw_markdown", "") or ""
    if not md_content:
        md_content = result.markdown or ""

    title = ""
    if hasattr(result, "metadata") and result.metadata:
        title = result.metadata.get("title", "") or ""
    if not title:
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
        save_markdown_if_changed(url, OUTPUT_DIR / make_filename(url, today), header,
                                 md_content, today)

    return result, False


def write_error_report(errors: list[dict], today: str) -> None:
    if not errors:
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Fehler"
    ws.append(["URL", "Gefunden auf (Referrer)", "Fehlermeldung"])
    for cell in ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)
    for entry in errors:
        ws.append([entry["url"], entry["referrer"], entry["error"]])
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 80)
    filepath = OUTPUT_DIR / f"{today}_fehler.xlsx"
    wb.save(filepath)
    logger.info("Fehlerbericht gespeichert: %s (%d Einträge)", filepath, len(errors))


# ---------------------------------------------------------------------------
# Haupt-Crawl-Schleife
# ---------------------------------------------------------------------------


async def run_crawl(to_crawl: set[str]) -> None:
    """Crawlt alle URLs in to_crawl (stale + discovered)."""
    today   = date.today().isoformat()
    errors: list[dict] = []

    config = CrawlerRunConfig(
        css_selector="main, .content-main, #content, .frame-default",
        excluded_tags=["nav", "header", "footer", "aside", "form",
                       "iframe", "script", "style", "noscript"],
        excluded_selector=(
            "nav.main-menu, .main-menu, .main-menu__mainmenu, "
            "ul#tabmenu1, ul[role='menubar'], "
            "header, .quicklinks, "
            ".breadcrumb, .breadcrumbs, [aria-label='breadcrumb'], "
            "footer, .footer, .site-footer, "
            "#CybotCookiebotDialog, #CybotCookiebotDialogBody, "
            "[id*='Cookiebot'], [class*='cookiebot'], "
            "aside, .sidebar, .widget, "
            ".search-form, form[role='search'], "
            ".social-media, .share-buttons"
        ),
    )

    # Bekannte, aktuell gescrapte URLs ermitteln (Duplikat-Schutz während des Laufs)
    cutoff = date.today() - timedelta(days=MAX_AGE_DAYS)
    visited: set[str] = set()
    for filepath in OUTPUT_DIR.glob("*.md"):
        try:
            text = filepath.read_text(encoding="utf-8")
            meta = parse_frontmatter(text)
            src  = meta.get("source_url", "")
            crawl_date_str = meta.get("crawl_date", "")
            if src:
                norm = normalize_url(src)
                try:
                    if date.fromisoformat(crawl_date_str) >= cutoff:
                        visited.add(norm)   # frisch → nicht nochmal crawlen
                except ValueError:
                    pass
        except Exception:
            continue

    # Queue mit den zu crawlenden URLs befüllen (englische Seiten überspringen)
    queue: deque[str] = deque()
    referrers: dict[str, str] = {}
    skipped_en = 0
    for url in to_crawl:
        norm = normalize_url(url)
        if is_english_url(norm):
            skipped_en += 1
            continue
        if norm not in visited:
            queue.append(norm)
            referrers[norm] = "(resume)"
    if skipped_en:
        logger.info("Englische Seiten übersprungen: %d URLs (/en/…)", skipped_en)

    total     = len(queue)
    done      = 0
    success   = 0
    fail      = 0
    gone      = 0

    logger.info("Starte Resume-Crawl: %d URLs in der Warteschlange", total)

    async with AsyncWebCrawler() as crawler:
        while queue and done < MAX_PAGES:
            url  = queue.popleft()
            done += 1

            pct = 100.0 * done / max(total, 1)
            logger.info("[%d/%d  %5.1f%%] %s", done, total, pct, url)

            try:
                if has_blocked_extension(url):
                    logger.debug("Übersprungen (nicht unterstütztes Format): %s", url)
                    visited.add(url)
                    continue
                if await is_pdf_url(url):
                    is_gone = await process_pdf(url, today)
                else:
                    result, is_gone = await process_html(crawler, url, today, config,
                                                         save=True)
                    if not is_gone:
                        # Neu entdeckte interne Links in die Queue aufnehmen
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
                                total += 1

                visited.add(normalize_url(url))
                if is_gone:
                    gone += 1
                else:
                    success += 1

            except Exception as exc:
                logger.error("Fehler bei %s: %s", url, exc)
                fail += 1
                errors.append({
                    "url":      url,
                    "referrer": referrers.get(url, "(unbekannt)"),
                    "error":    str(exc),
                })

            if queue and done < MAX_PAGES:
                await asyncio.sleep(RATE_LIMIT_SECONDS)

    write_error_report(errors, today)
    logger.info(
        "Fertig. %d erfolgreich, %d fehlgeschlagen, %d entfernt (404/410 — Datei "
        "gelöscht) von %d verarbeiteten URLs.",
        success, fail, gone, done,
    )


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-Crawler: analysiert vorhandene MD-Dateien und crawlt fehlende URLs."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Nur Analyse ausgeben, keinen Crawl starten."
    )
    args = parser.parse_args()

    if not OUTPUT_DIR.exists():
        logger.error("Ausgabeverzeichnis nicht gefunden: %s", OUTPUT_DIR)
        sys.exit(1)

    # ── Analyse ───────────────────────────────────────────────────────────
    fresh, stale, discovered = analyse_ingested_dir()
    print_report(fresh, stale, discovered)

    to_crawl = set(stale) | discovered

    if not to_crawl:
        logger.info("Alle bekannten URLs sind aktuell. Kein Crawl notwendig.")
        return

    if args.dry_run:
        logger.info("--dry-run: Crawl wird nicht gestartet.")
        return

    # ── Crawl ─────────────────────────────────────────────────────────────
    await run_crawl(to_crawl)

    # Nur die tatsächlich geänderten Seiten als eine Datei bündeln — das ist
    # genau die Menge, die anschließend zum HPC muss.
    from datetime import date as _date
    n_delta, _ = pack_delta_archive(DELTA_DIR / f"delta_{_date.today().isoformat()}.tar.xz")
    if n_delta:
        logger.info("Nächster Schritt: Delta-Archiv zum HPC übertragen — "
                    "siehe PIPELINE.md")


if __name__ == "__main__":
    asyncio.run(main())
