"""Chunking-Änderungen ohne HPC-Lauf messen: Golddokumente in einer Testkopie
der Collection neu vektorisieren.

Wozu
----
Eine Änderung an chunking.py (22.09.2026: Tabellen zeilenatomar, Kontextkopf
im Einbettungstext) kostet vollständig 1,7 GPU-Stunden doc2query plus
Vektorisierung. Bevor das läuft, soll bekannt sein, ob sie im Retrieval
überhaupt etwas bewegt. Dafür genügt es, die Golddokumente der Fallbasis neu
zu chunken und lokal (CPU) zu vektorisieren — in einer Kopie der Collection,
gegen die eine zweite Serverinstanz läuft.

Was das misst und was nicht
---------------------------
Eine **Obergrenze**: Nur die Golddokumente bekommen den Kontextkopf, ihre
Konkurrenten im Korpus nicht. Gewinne sind deshalb optimistisch, Verluste
und Nullbefunde dagegen belastbar. Die `questions`-Vektoren werden aus der
Produktion übernommen, wo der Chunktext unverändert ist (gleiche ID, gleicher
Text); Chunks mit neuen Grenzen (Tabellen) haben keinen Fragen-Arm — ein
kleiner Nachteil für genau die Fälle, um die es geht.

Ablauf
------
    # 1. Testkopie (einmalig): Snapshot der Produktion, Wiederherstellung unter neuem Namen
    curl -X POST localhost:6333/collections/hsh_knowledge/snapshots
    curl -X PUT  localhost:6333/collections/hsh_knowledge_v2test/snapshots/recover \\
         -H 'Content-Type: application/json' -d '{"location":"file:///qdrant/snapshots/hsh_knowledge/<name>"}'

    # 2. Golddokumente neu vektorisieren und in die Testkopie importieren
    python auswertung/gold/testindex.py --vorbereiten     # urls.txt + enrichment_teil.parquet
    python hpc_vectorizer.py --cpu --urls auswertung/testindex/urls.txt \\
        --output auswertung/testindex/vektoren.parquet \\
        --enrichment auswertung/testindex/enrichment_teil.parquet \\
        --avg-len 88.3 --avg-len-expanded 107.1 --avg-len-sample 0
    HSH_COLLECTION=hsh_knowledge_v2test python -m raghsh.index.local_importer --input auswertung/testindex/vektoren.parquet

    # 3. Zweite Serverinstanz und Retrieval-Lauf dagegen
    HSH_COLLECTION=hsh_knowledge_v2test HSH_MCP_PORT=8766 python -m raghsh.server &
    python auswertung/gold/lauf.py --nur-retrieval --mcp-url http://127.0.0.1:8766/mcp --notiz "…"

Die Fallbasis trägt `beleg_punkte` mit Produktions-IDs; in Tabellendokumenten
verschieben sich die IDs. Die Auswertung fällt dort auf die URL zurück
(bewerte_fall._rang_chunk), die Kandidatenränge sind davon unberührt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent
from raghsh import konfig  # noqa: E402
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE.parent / "gold"))   # vertrag.py

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from raghsh.crawl.crawler_utils import normalize_url  # noqa: E402
from vertrag import lade_fallbasis  # noqa: E402

ORDNER = _BASE
INGESTED = konfig.KORPUS
COLLECTION_PROD = "hsh_knowledge"


def gold_urls(faelle: list[dict]) -> set[str]:
    urls: set[str] = set()
    for f in faelle:
        for u in [f.get("source_url", ""), *(f.get("accepted_source_urls") or []),
                  *(f.get("beleg_urls") or [])]:
            if u:
                urls.add(normalize_url(u))
    return urls


def dateien_je_url() -> dict[str, Path]:
    """source_url (normalisiert) → Markdown-Datei, per Frontmatter."""
    treffer: dict[str, Path] = {}
    muster = re.compile(r'^source_url:\s*"?([^"\n]+)"?', re.M)
    for p in INGESTED.glob("*.md"):
        kopf = p.read_text(encoding="utf-8", errors="ignore")[:2000]
        m = muster.search(kopf)
        if m:
            treffer[normalize_url(m.group(1).strip())] = p
    return treffer


def vorbereiten() -> None:
    from raghsh.index import chunking
    from qdrant_client import QdrantClient

    ORDNER.mkdir(exist_ok=True)
    faelle = lade_fallbasis()
    urls = gold_urls(faelle)
    dateien = dateien_je_url()
    gefunden = {u: dateien[u] for u in urls if u in dateien}
    fehlend = sorted(urls - set(gefunden))
    print(f"{len(urls)} Gold-URLs, {len(gefunden)} mit Datei im Korpus, {len(fehlend)} ohne")

    # Die URL-Liste für hpc_vectorizer --urls: die ORIGINALEN source_urls aus
    # den Dateien (der Vektorisierer vergleicht wörtlich).
    originale = []
    for p in gefunden.values():
        m = re.search(r'^source_url:\s*"?([^"\n]+)"?', p.read_text(encoding="utf-8", errors="ignore")[:2000], re.M)
        originale.append(m.group(1).strip())
    (ORDNER / "urls.txt").write_text("\n".join(sorted(originale)) + "\n", encoding="utf-8")

    # Neue Chunks; Fragen übernehmen, wo ID und Text mit der Produktion übereinstimmen.
    hs, rs = chunking.make_splitters()
    neu: list[tuple[str, str, str]] = []
    n_tabelle = 0
    for p in gefunden.values():
        ergebnis = chunking.parse_markdown_file(p)
        if ergebnis is None:
            continue
        meta, body = ergebnis
        chunks = chunking.chunk_document(meta, body, hs, rs)
        if any(c["text"].lstrip().startswith("|") for c in chunks):
            n_tabelle += 1
        neu.extend((chunking.make_point_id(c["source_url"], c["chunk_index"]), c["source_url"], c["text"])
                   for c in chunks)
    # Fragen je Dokument über den Text zuordnen, nicht über die ID: In
    # Tabellendokumenten verschieben sich die Indizes, und die 113 umlaut-
    # reparierten Dateien lauten anders als der Index („Pr¨ufung"). Beide
    # Seiten werden deshalb repariert und auf Leerraum normalisiert verglichen.
    from raghsh.crawl.crawler_utils import repariere_diakritika
    from qdrant_client import models

    def norm(t: str) -> str:
        return " ".join(repariere_diakritika(t or "").split())

    client = QdrantClient(url="http://localhost:6333", timeout=60)
    fragen_je_url: dict[str, dict[str, list[str]]] = {}
    for u in gefunden:
        weiter, karte = None, {}
        while True:
            punkte, weiter = client.scroll(
                COLLECTION_PROD, scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="source_url", match=models.MatchValue(value=u))]),
                with_payload=["text", "questions"], with_vectors=False, limit=256, offset=weiter)
            for pt in punkte:
                pl = pt.payload or {}
                if pl.get("questions"):
                    karte[norm(pl.get("text", ""))] = list(pl["questions"])
            if weiter is None:
                break
        fragen_je_url[u] = karte
    zeilen_id, zeilen_q = [], []
    for pid, url, text in neu:
        q = fragen_je_url.get(normalize_url(url), {}).get(norm(text))
        if q:
            zeilen_id.append(pid)
            zeilen_q.append(q)
    tabelle = pa.table({"id": pa.array(zeilen_id, pa.string()),
                        "questions": pa.array(zeilen_q, pa.list_(pa.string()))})
    pq.write_table(tabelle, ORDNER / "enrichment_teil.parquet")
    print(f"{len(gefunden)} Dokumente → {len(neu)} neue Chunks ({n_tabelle} Dokumente mit Tabellen); "
          f"Fragen übernommen für {len(zeilen_id)} unveränderte Chunks "
          f"({len(neu) - len(zeilen_id)} ohne Fragen-Arm)")
    (ORDNER / "vorbereitung.json").write_text(json.dumps({
        "gold_urls": len(urls), "dokumente": len(gefunden), "fehlend": fehlend,
        "chunks_neu": len(neu), "mit_fragen": len(zeilen_id), "dokumente_mit_tabellen": n_tabelle,
        "chunking_config_hash": chunking.config_hash(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"→ {ORDNER / 'urls.txt'}, {ORDNER / 'enrichment_teil.parquet'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vorbereiten", action="store_true", help="urls.txt und enrichment_teil.parquet schreiben")
    args = ap.parse_args()
    if args.vorbereiten:
        vorbereiten()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
