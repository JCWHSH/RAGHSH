"""
Gezieltes Nachcrawlen einzelner URLs aus einer Textdatei.

Kurzbeschreibung
----------------
Crawlt genau die URLs aus einer Liste — ohne BFS, ohne Linkverfolgung, ohne
Frische-Prüfung. Gedacht für Seiten, die ein vorheriger Lauf zu Unrecht
verworfen hat (z.B. durch einen zu groben Fehlerseiten-Filter) und die deshalb
in keiner .md-Datei stehen. Da sie nie gespeichert wurden, greift die normale
MAX_AGE_DAYS-Logik bei ihnen nicht.

Es wird bewusst alles aus main.py wiederverwendet — dieselbe CrawlerRunConfig,
dieselben Qualitäts- und Hash-Prüfungen, dieselbe Dateibenennung. Damit sind
die erzeugten Dateien von denen eines regulären Crawls nicht unterscheidbar.

Aufruf:
    python recrawl_urls.py urls.txt
    python recrawl_urls.py urls.txt --dry-run    # nur prüfen, nichts schreiben

Eingabeformat: eine URL pro Zeile. Leerzeilen und #-Kommentare werden ignoriert.
"""

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path

from crawl4ai import AsyncWebCrawler

from raghsh.crawl import spider as crawler_main
from raghsh.crawl.crawler_utils import is_low_quality, normalize_url, pack_delta_archive

RATE_LIMIT_SECONDS = crawler_main.RATE_LIMIT_SECONDS
logger = crawler_main.logger


def load_urls(path: Path) -> list[str]:
    """Liest die URL-Liste und normalisiert sie (Duplikate fallen weg)."""
    if not path.exists():
        logger.error("Datei nicht gefunden: %s", path)
        sys.exit(1)

    seen: set[str] = set()
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        norm = normalize_url(line)
        if norm not in seen:
            seen.add(norm)
            urls.append(norm)
    return urls


async def recrawl(urls: list[str], dry_run: bool) -> dict[str, int]:
    today  = date.today().isoformat()
    config = crawler_main.build_run_config()
    stats  = {"gespeichert": 0, "verworfen": 0, "fehler": 0, "gone": 0}

    crawler_main.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(urls, 1):
            logger.info("[%d/%d] %s", i, len(urls), url)
            try:
                if await crawler_main.is_pdf_url(url):
                    if dry_run:
                        logger.info("  (dry-run) PDF — würde geladen")
                    elif await crawler_main.process_pdf(url, today):
                        stats["gone"] += 1
                        continue
                    else:
                        stats["gespeichert"] += 1
                    continue

                result, gone = await crawler_main.process_html(
                    crawler, url, today, config, save=not dry_run
                )
                if gone:
                    stats["gone"] += 1
                    continue

                md = ""
                if getattr(result, "markdown_v2", None):
                    md = getattr(result.markdown_v2, "raw_markdown", "") or ""
                if not md:
                    md = result.markdown or ""

                low_q, reason = is_low_quality("html", md)
                if low_q:
                    logger.info("  verworfen (%s)", reason)
                    stats["verworfen"] += 1
                else:
                    stats["gespeichert"] += 1
                    if dry_run:
                        logger.info("  (dry-run) würde gespeichert (%d Zeichen)", len(md))

            except Exception as exc:
                logger.warning("  FEHLER: %s", str(exc)[:160])
                stats["fehler"] += 1

            await asyncio.sleep(RATE_LIMIT_SECONDS)

    return stats


async def amain() -> None:
    parser = argparse.ArgumentParser(
        description="Crawlt gezielt die URLs aus einer Textdatei nach."
    )
    parser.add_argument("urlfile", type=Path, help="Textdatei mit einer URL pro Zeile")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur prüfen, keine Dateien schreiben")
    args = parser.parse_args()

    urls = load_urls(args.urlfile)
    logger.info("%d eindeutige URL(s) aus %s", len(urls), args.urlfile)
    if args.dry_run:
        logger.info("DRY-RUN — es wird nichts geschrieben.")

    stats = await recrawl(urls, args.dry_run)

    n_delta = 0
    if not args.dry_run:
        n_delta, _ = pack_delta_archive(
            crawler_main.DELTA_DIR / f"delta_{date.today().isoformat()}.tar.xz"
        )

    logger.info("─" * 60)
    logger.info("Fertig. gespeichert=%d  verworfen=%d  404/410=%d  Fehler=%d",
                stats["gespeichert"], stats["verworfen"], stats["gone"], stats["fehler"])
    logger.info("─" * 60)
    if n_delta:
        logger.info("Nächster Schritt: Delta-Archiv zum HPC übertragen — siehe PIPELINE.md")


if __name__ == "__main__":
    asyncio.run(amain())
