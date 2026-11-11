"""
Lokaler Qdrant-Importer — Parquet-Datei in Qdrant-Datenbank laden.

Kurzbeschreibung
----------------
Liest die von hpc_vectorizer.py erzeugte Parquet-Datei (20 Spalten, vier
Retrieval-Arme je Chunk) und lädt Vektoren samt Metadaten per Upsert in die
lokale Qdrant-Instanz. Maßgeblich ist `Parquet-Schema_und_Qdrant-Import.md`.

Workflow:
    1. HPC:   python hpc_vectorizer.py   →  hsh_vectors_<variante>.parquet
                                         +  hsh_vectors_<variante>_sources.txt
    2. Lokal: python -m raghsh.index.local_importer   →  Qdrant befüllen

Collection-Schema (vier benannte Vektoren):
    Dense   "text"            — Jina v3, retrieval.passage, 1024 Dim., Cosine
    Dense   "questions"       — Jina v3, retrieval.query, über die generierten
                                Fragen; NICHT jeder Punkt hat diesen Vektor
    Sparse  "sparse"          — BM25 über den reinen Chunktext
    Sparse  "sparse_expanded" — BM25 über Chunktext + generierte Fragen

Beide Sparse-Vektoren werden mit `modifier=Modifier.IDF` angelegt. Das ist
Pflicht, kein Feinschliff: Die gespeicherten Gewichte enthalten nur die
Dokumentkomponente von BM25. Ohne IDF rankt Qdrant still nach reiner
Termfrequenz — häufige Allerweltswörter gewinnen, und zwar ohne Fehlermeldung.

Leere Fragenvektoren werden übersprungen, nicht mit Nullen gefüllt: Ein
Nullvektor hat keine definierte Cosine-Ähnlichkeit und würde den Punkt
entweder nie oder immer treffen.

Verhalten bei erneutem Aufruf:
    Punkt-IDs sind deterministisch (UUID5 aus source_url#chunk_index).
    Ein erneuter Lauf überschreibt vorhandene Einträge (Upsert), erzeugt
    keine Duplikate.

    Reines Upsert reicht dafür aber nicht: Wird eine Seite kürzer, überschreibt
    der Import nur noch die Chunks 0..n-1 — die alten Chunks n..m bleiben als
    Karteileichen mit veraltetem Inhalt in der Collection und tauchen weiter in
    Suchergebnissen auf. Deshalb wird der Altbestand vor dem Schreiben per
    Filter (source_url) gelöscht. Bevorzugt anhand der Sidecar-Datei
    `<parquet-stem>_sources.txt`, die ALLE verarbeiteten Seiten enthält — auch
    die, die wegen der Deduplizierung keinen eigenen Chunk mehr stellen und
    deshalb in der Parquet-Datei gar nicht vorkommen. Fehlt die Sidecar-Datei,
    wird ersatzweise je source_url beim ersten Auftreten gelöscht.

Konfiguration:
    PARQUET_FILE    — Vorgabe-Eingabedatei (überschreibbar mit --input)
    QDRANT_URL      — Adresse der lokalen Qdrant-Instanz
    COLLECTION_NAME — Name der Collection
    UPLOAD_BATCH    — Punkte pro upsert-Aufruf
    RESUME_FROM_ROW — Neustart ab Zeile N (0 = von Anfang an)

Abhängigkeiten (lokales VirtualEnv):
    pip install qdrant-client pyarrow
"""

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from raghsh import konfig
from raghsh.index.titel import anzeigetitel

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent

# Vorgabe für den Aufruf ohne --input. Der Variantenname gehört bewusst in den
# Dateinamen: Läufe mit anderem Fragenfilter unterscheiden sich ausschließlich
# in questions/sparse_expanded/vector_questions, teilen sich aber die Punkt-IDs
# und würden einander beim Upsert überschreiben. Zwei Varianten gehören in zwei
# Collections — deshalb ist die Datei explizit benannt, nicht generisch.
PARQUET_FILE    = konfig.VEKTOREN / "hsh_vectors_wdh10.parquet"
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = os.environ.get("HSH_COLLECTION", "hsh_knowledge")   # Testkopie: HSH_COLLECTION setzen
VECTOR_SIZE     = 1024

UPLOAD_BATCH    = 256  # Punkte pro upsert-Aufruf
PURGE_BATCH     = 128  # source_urls pro delete-by-filter-Aufruf

# Neustart nach Unterbrechung: 0-basierter Zeilenindex.
# 0 = von Anfang an. Bei Abbruch bei Zeile N: RESUME_FROM_ROW = N
RESUME_FROM_ROW = 0

# Die Payload führt die Fakultät ausgeschrieben, der Korpus (Frontmatter,
# doc_index.db, Weboberfläche) den Kurzschlüssel. Beides mitzuschreiben erspart
# jeder Empfängerseite die Übersetzung; die Tabelle ist fest und vollständig.
FACULTY_KEYS = {
    "Fakultät I":   "f1",
    "Fakultät II":  "f2",
    "Fakultät III": "f3",
    "Fakultät IV":  "f4",
    "Fakultät V":   "f5",
}

# Spalten, ohne die der Import nicht sinnvoll ist.
REQUIRED_COLUMNS = {
    "id", "vector", "source_url", "title", "crawl_date", "content_type",
    "faculty", "organisation", "section_heading", "text", "chunk_index",
    "total_chunks", "sparse_indices", "sparse_values",
    "sparse_expanded_indices", "sparse_expanded_values",
    "questions", "vector_questions",
    "chunking_config_hash", "selection_hash",
}

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


def ensure_collection(client: QdrantClient) -> bool:
    """Erstellt die Collection und Payload-Indizes falls noch nicht vorhanden.

    Gibt True zurück wenn die Collection neu angelegt wurde — dann ist sie leer
    und das Aufräumen alter Chunks (purge_documents) kann entfallen.

    Schema: zwei benannte Dense- und zwei benannte Sparse-Vektoren für die
    Prefetch-basierte Hybrid-Suche. Benannte Vektoren sind zwingend — Qdrant
    erlaubt entweder einen unbenannten oder mehrere benannte Dense-Vektoren,
    eine bestehende Collection lässt sich nicht nachträglich erweitern.

    Payload-Indizes:
      source_url, chunk_index  — Context Augmentation (Nachbar-Chunks)
      faculty, organisation,
      content_type             — Filter ohne Full-Scan
      text (Volltext)          — Keyword-Boost via MatchText im MCP-Server
    """
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        logger.info("Collection '%s' existiert bereits — Upsert-Modus", COLLECTION_NAME)
        return False

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text": models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
            "questions": models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            # modifier=IDF ist PFLICHT — die gespeicherten Gewichte enthalten
            # kein IDF, das entsteht erst zur Query-Zeit aus den
            # Collection-Statistiken.
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
            "sparse_expanded": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )
    logger.info(
        "Collection '%s' angelegt — dense: text, questions (%d Dim.); "
        "sparse: sparse, sparse_expanded (BM25, modifier=IDF)",
        COLLECTION_NAME, VECTOR_SIZE,
    )

    # Volltext-Index auf 'text' — MULTILINGUAL für deutsches Stemming
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="text",
        field_schema=models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.MULTILINGUAL,
            min_token_len=2,
            max_token_len=50,
            lowercase=True,
        ),
    )
    logger.info("Volltext-Index auf 'text' angelegt (MULTILINGUAL-Tokenizer)")

    # Keyword-Indizes für Filter. 'organisation' ist neu und bewusst von
    # 'faculty' getrennt: Bibliothek, Karriere und International sind keine
    # Fakultäten und machten jeden Fakultätsfilter unsauber.
    for feld in ("source_url", "faculty", "faculty_key", "organisation", "content_type"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=feld,
            field_schema=models.KeywordIndexParams(type="keyword"),
        )

    # Bool-Index auf 'text_vollstaendig' — erlaubt es, Fragmente in einer
    # Abfrage abzuwerten oder auszuschließen, ohne Full-Scan.
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="text_vollstaendig",
        field_schema=models.BoolIndexParams(type="bool"),
    )

    # Integer-Index auf 'chunk_index' für Context Augmentation
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="chunk_index",
        field_schema=models.IntegerIndexParams(type="integer", lookup=True, range=True),
    )
    logger.info("Payload-Indizes angelegt: source_url, faculty, faculty_key, "
                "organisation, content_type, chunk_index, text_vollstaendig")
    return True


def purge_documents(client: QdrantClient, urls: list[str]) -> None:
    """Löscht alle vorhandenen Punkte zu diesen source_urls.

    Entfernt Chunks, die nach einer inhaltlichen Änderung überzählig geworden
    sind (Seite ist kürzer als beim letzten Import). Läuft mit wait=True —
    andernfalls könnte die asynchrone Löschung die unmittelbar danach
    geschriebenen neuen Punkte wieder erwischen.
    """
    for start in range(0, len(urls), PURGE_BATCH):
        batch = urls[start:start + PURGE_BATCH]
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_url",
                            match=models.MatchAny(any=batch),
                        )
                    ]
                )
            ),
            wait=True,
        )


def read_sidecar(parquet_file: Path) -> list[str]:
    """Liest die Sidecar-Datei mit allen verarbeiteten source_urls.

    Sie enthält mehr URLs als die Parquet-Datei: Seiten, deren Inhalt wortgleich
    anderswo steht, stellen nach der Deduplizierung keinen eigenen Chunk mehr.
    Genau die müssen beim Delta-Import aber gelöscht werden — sonst bleibt ihr
    Altbestand als Karteileiche liegen.
    """
    sidecar = parquet_file.with_name(parquet_file.stem + "_sources.txt")
    if not sidecar.exists():
        return []
    urls = [z.strip() for z in sidecar.read_text(encoding="utf-8").splitlines() if z.strip()]
    logger.info("Sidecar-Datei %s: %d source_urls", sidecar.name, len(urls))
    return urls


def dokument_kennzahlen(parquet_file: Path) -> dict[str, dict]:
    """Vorlauf über die Datei: je source_url alles, was das ganze Dokument betrifft.

    Zwei Angaben lassen sich beim zeilenweisen Schreiben nicht ermitteln, weil
    sie die gesamte Datei kennen müssen:

    **Vollständigkeit.** `chunk_index` und `total_chunks` beziehen sich auf das
    Dokument VOR der Deduplizierung. Gemessen haben 1.559 der 5.306 Dokumente
    Lücken, 65.763 Positionen fehlen, im Extremfall liegen 3 von 242 Chunks vor.
    Ohne Kennzeichen gibt die zweite LLM-Runde ein solches Fragment als
    vollständigen Beleg aus und antwortet „laut Prüfungsordnung …" auf Basis von
    einem Prozent des Textes. Der Text ist nicht verloren — er steht wortgleich
    unter einer anderen source_url (siehe chunk_vertretung.parquet) — aber ohne
    diese Zahlen weiß die Empfängerseite nicht, wann sie ihn holen muss.

    **Anzeigetitel.** Er soll für alle Chunks eines Dokuments gleich sein und
    stützt sich auf die section_heading des ERSTEN Chunks; die steht erst fest,
    wenn die Datei einmal gelesen ist.

    Der Vorlauf liest nur fünf schmale Spalten, keine Vektoren.
    """
    tabelle = pq.read_table(
        parquet_file,
        columns=["source_url", "chunk_index", "total_chunks", "title", "section_heading"],
    ).to_pydict()

    vorhanden: Counter = Counter(tabelle["source_url"])
    kopf: dict[str, tuple[int, str, str]] = {}   # url -> (chunk_index, title, heading)
    gesamt: dict[str, int] = {}

    for url, idx, total, titel, heading in zip(
        tabelle["source_url"], tabelle["chunk_index"], tabelle["total_chunks"],
        tabelle["title"], tabelle["section_heading"],
    ):
        gesamt[url] = total
        if url not in kopf or idx < kopf[url][0]:
            kopf[url] = (idx, titel, heading)

    kennzahlen = {}
    for url, anzahl in vorhanden.items():
        _, titel, heading = kopf[url]
        kennzahlen[url] = {
            "chunks_vorhanden":  anzahl,
            "chunks_dokument":   gesamt[url],
            "text_vollstaendig": anzahl == gesamt[url],
            "title_anzeige":     anzeigetitel(url, heading, titel),
        }

    unvollstaendig = sum(1 for k in kennzahlen.values() if not k["text_vollstaendig"])
    logger.info("Vorlauf: %d Dokumente, davon %d unvollständig (%.1f %%)",
                len(kennzahlen), unvollstaendig, 100 * unvollstaendig / max(len(kennzahlen), 1))
    return kennzahlen


def build_points(rows: list[dict], kennzahlen: dict[str, dict]) -> list[models.PointStruct]:
    """Wandelt Parquet-Zeilen in Qdrant-Punkte mit bis zu vier Vektoren."""
    points: list[models.PointStruct] = []
    for r in rows:
        vektoren: dict = {
            "text": r["vector"],
            "sparse": models.SparseVector(
                indices=r["sparse_indices"],
                values=r["sparse_values"],
            ),
        }

        # Fall A (Lauf ganz ohne --enrichment): sparse_expanded ist leer.
        # Ein leerer Sparse-Vektor wäre wertlos und macht den Punkt nur größer.
        if r["sparse_expanded_indices"]:
            vektoren["sparse_expanded"] = models.SparseVector(
                indices=r["sparse_expanded_indices"],
                values=r["sparse_expanded_values"],
            )

        # Leere Liste bedeutet: kein Fragenvektor. Nicht mit Nullen füllen —
        # ein Nullvektor hat keine definierte Cosine-Ähnlichkeit und würde den
        # Punkt entweder nie oder immer treffen.
        if r["vector_questions"]:
            vektoren["questions"] = r["vector_questions"]

        k = kennzahlen[r["source_url"]]

        points.append(models.PointStruct(
            id=r["id"],
            vector=vektoren,
            payload={
                "source_url":      r["source_url"],
                "title":           r["title"],
                # Lesbarer Titel für Anzeige und LLM-Dokumentauswahl. `title`
                # bleibt daneben stehen — er ist Teil des Dedup-Schlüssels und
                # damit die Verbindung zur Erzeugerseite.
                "title_anzeige":   k["title_anzeige"],
                "crawl_date":      r["crawl_date"],
                "content_type":    r["content_type"],
                "faculty":         r["faculty"],
                "faculty_key":     FACULTY_KEYS.get(r["faculty"], ""),
                "organisation":    r["organisation"],
                # Vollständigkeit dieses Dokuments im Index — siehe
                # dokument_kennzahlen().
                "chunks_vorhanden":  k["chunks_vorhanden"],
                "chunks_dokument":   k["chunks_dokument"],
                "text_vollstaendig": k["text_vollstaendig"],
                "section_heading": r["section_heading"],
                "kontext":         r.get("kontext", ""),     # fehlt in Parquets vor dem 22.09.2026
                "text":            r["text"],
                "chunk_index":     r["chunk_index"],
                "total_chunks":    r["total_chunks"],
                # Retrieval-Signal, kein Inhalt: die Fragen stehen so nicht auf
                # der Webseite und gehören nicht in den LLM-Kontext. Sie liegen
                # nur zur Nachvollziehbarkeit im Payload.
                "questions":       r["questions"],
                "chunking_config_hash": r["chunking_config_hash"],
                "selection_hash":  r["selection_hash"],
            },
        ))
    return points


def verify_import(client: QdrantClient, parquet_file: Path,
                  kennzahlen: dict[str, dict] | None = None) -> bool:
    """Selbsttest nach dem Import — Sollwerte aus der Parquet-Datei selbst.

    Die Sollwerte werden nachgezählt statt fest verdrahtet: So gilt der Test
    auch für Delta-Läufe und andere Filtervarianten.
    """
    logger.info("─" * 60)
    logger.info("Selbsttest")

    info = client.get_collection(COLLECTION_NAME)
    soll_punkte = pq.ParquetFile(parquet_file).metadata.num_rows
    ist_punkte  = info.points_count
    ok = True

    dense_namen  = sorted(info.config.params.vectors or {})
    sparse_namen = sorted(info.config.params.sparse_vectors or {})
    logger.info("  Dense-Vektoren : %s", dense_namen)
    logger.info("  Sparse-Vektoren: %s", sparse_namen)
    if dense_namen != ["questions", "text"] or sparse_namen != ["sparse", "sparse_expanded"]:
        logger.error("  ✗ Vektorschema weicht ab")
        ok = False

    logger.info("  Punkte: %d (Parquet: %d)", ist_punkte, soll_punkte)
    if ist_punkte != soll_punkte:
        logger.error("  ✗ Punktzahl weicht ab — %d fehlen", soll_punkte - ist_punkte)
        ok = False

    # Payload-Verteilung gegen die Parquet-Datei prüfen. Nur zwei Spalten
    # lesen, das kostet kaum Speicher.
    tabelle = pq.read_table(parquet_file, columns=["faculty", "organisation"])
    for feld in ("faculty", "organisation"):
        soll = Counter(tabelle.column(feld).to_pylist())
        for wert, anzahl in sorted(soll.items(), key=lambda kv: -kv[1]):
            if not wert:
                continue  # leere Werte sind nicht per MatchValue zählbar
            ist = client.count(
                COLLECTION_NAME,
                count_filter=models.Filter(must=[
                    models.FieldCondition(key=feld, match=models.MatchValue(value=wert))
                ]),
                exact=True,
            ).count
            zeichen = "✓" if ist == anzahl else "✗"
            logger.info("  %s %-12s %-18s %7d (Soll %d)", zeichen, feld, wert, ist, anzahl)
            if ist != anzahl:
                ok = False
    del tabelle

    # Ein Punkt mit Fragen — sind alle vier Arme belegt?
    beispiel, _ = client.scroll(
        COLLECTION_NAME,
        scroll_filter=models.Filter(must=[
            models.IsEmptyCondition(is_empty=models.PayloadField(key="questions"))
        ]),
        limit=1,
    )
    logger.info("  Punkte ohne Fragen vorhanden: %s", bool(beispiel))

    # Die neuen Felder: stimmt die Zahl der Fragment-Dokumente mit dem überein,
    # was der Vorlauf aus der Datei gezählt hat?
    if kennzahlen is None:                      # --nur-pruefen: Vorlauf nachholen
        kennzahlen = dokument_kennzahlen(parquet_file)
    soll_unvollstaendig = sum(
        1 for k in kennzahlen.values() if not k["text_vollstaendig"]
    )
    urls_unvollstaendig = set()
    weiter = None
    while True:
        punkte, weiter = client.scroll(
            COLLECTION_NAME,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="text_vollstaendig", match=models.MatchValue(value=False))]),
            limit=4096, offset=weiter, with_payload=["source_url"], with_vectors=False,
        )
        urls_unvollstaendig.update((p.payload or {}).get("source_url", "") for p in punkte)
        if weiter is None:
            break
    zeichen = "✓" if len(urls_unvollstaendig) == soll_unvollstaendig else "✗"
    logger.info("  %s unvollständige Dokumente %d (Soll %d)",
                zeichen, len(urls_unvollstaendig), soll_unvollstaendig)
    if len(urls_unvollstaendig) != soll_unvollstaendig:
        ok = False

    fehlt_titel = client.count(
        COLLECTION_NAME,
        count_filter=models.Filter(must=[
            models.IsEmptyCondition(is_empty=models.PayloadField(key="title_anzeige"))
        ]), exact=True,
    ).count
    logger.info("  %s Punkte ohne title_anzeige: %d (Soll 0)",
                "✓" if fehlt_titel == 0 else "✗", fehlt_titel)
    if fehlt_titel:
        ok = False

    logger.info("  Ergebnis: %s", "vollständig" if ok else "FEHLERHAFT")
    logger.info("─" * 60)
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importiert eine Vektor-Parquet-Datei nach Qdrant."
    )
    parser.add_argument(
        "--input", type=Path, metavar="FILE", default=PARQUET_FILE,
        help=f"Zu importierende Parquet-Datei (Standard: {PARQUET_FILE.name}). "
             "Beim Delta-Betrieb das Delta-Parquet angeben.",
    )
    parser.add_argument(
        "--nur-pruefen", action="store_true",
        help="Nichts importieren, nur den Selbsttest gegen die Collection fahren.",
    )
    args = parser.parse_args()
    parquet_file = args.input

    # ── 1. Parquet-Datei prüfen ───────────────────────────────────────────
    if not parquet_file.exists():
        logger.error("Datei nicht gefunden: %s", parquet_file)
        logger.error("Bitte zuerst hpc_vectorizer.py auf dem HPC ausführen und die Datei übertragen.")
        sys.exit(1)

    pf         = pq.ParquetFile(parquet_file)
    total_rows = pf.metadata.num_rows
    size_mb    = parquet_file.stat().st_size / 1_048_576
    logger.info("Parquet-Datei: %s  (%.1f MB, %d Zeilen)", parquet_file, size_mb, total_rows)

    schema_names = set(pf.schema_arrow.names)
    missing = REQUIRED_COLUMNS - schema_names
    if missing:
        logger.error("Fehlende Spalten in Parquet: %s", sorted(missing))
        logger.error("Erwartet wird das 20-Spalten-Schema aus "
                     "Parquet-Schema_und_Qdrant-Import.md.")
        sys.exit(1)

    # ── 2. Qdrant verbinden ───────────────────────────────────────────────
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=120)
        client.get_collections()
        logger.info("Verbunden mit Qdrant: %s", QDRANT_URL)
    except Exception as exc:
        logger.error("Verbindung zu Qdrant fehlgeschlagen: %s — %s", QDRANT_URL, exc)
        logger.error("Docker-Container starten:  docker compose up -d")
        sys.exit(1)

    if args.nur_pruefen:
        sys.exit(0 if verify_import(client, parquet_file) else 1)

    collection_is_new = ensure_collection(client)

    # ── 3. Altbestand entfernen ───────────────────────────────────────────
    # Vor dem Upsert, nicht danach: chunk_index hängt von der Chunkung des
    # gesamten Dokuments ab. Liefert eine neu gecrawlte Seite 7 statt 12 Chunks,
    # werden 0–6 überschrieben — 7–11 blieben mit veraltetem Text liegen.
    sidecar_urls: list[str] = []
    if collection_is_new:
        logger.info("Neue Collection — kein Aufräumen alter Chunks nötig.")
    else:
        sidecar_urls = read_sidecar(parquet_file)
        if sidecar_urls:
            logger.info("Lösche Altbestand zu %d source_urls…", len(sidecar_urls))
            purge_documents(client, sidecar_urls)
            logger.info("Altbestand entfernt.")
        else:
            logger.warning("Keine Sidecar-Datei gefunden — Altbestand wird je "
                           "source_url beim ersten Auftreten gelöscht. Seiten, "
                           "die nach der Deduplizierung keinen eigenen Chunk "
                           "mehr stellen, bleiben dabei unberücksichtigt.")

    # ── 4. Vorlauf: Kennzahlen je Dokument ────────────────────────────────
    kennzahlen = dokument_kennzahlen(parquet_file)

    # ── 5. Row-Group-weise einlesen und hochladen ─────────────────────────
    num_row_groups = pf.metadata.num_row_groups
    logger.info("Parquet enthält %d Row-Group(s)", num_row_groups)

    if RESUME_FROM_ROW > 0:
        logger.info("Überspringe die ersten %d Zeilen (RESUME_FROM_ROW=%d)",
                    RESUME_FROM_ROW, RESUME_FROM_ROW)

    uploaded     = 0
    skipped_rows = 0
    row_cursor   = 0
    mit_fragen   = 0
    # source_urls, deren Altbestand in diesem Lauf bereits gelöscht wurde.
    # Verhindert, dass ein über mehrere Batches verteiltes Dokument seine
    # eigenen, bereits hochgeladenen Chunks wieder löscht. Nur relevant, wenn
    # keine Sidecar-Datei vorlag.
    purged: set[str] = set()
    purge_je_batch = not collection_is_new and not sidecar_urls

    for rg_idx in range(num_row_groups):
        # Row-Group lesen — bei Beschädigung überspringen statt abbrechen
        try:
            table = pf.read_row_group(rg_idx)
        except Exception as exc:
            rg_rows = pf.metadata.row_group(rg_idx).num_rows
            logger.warning("Row-Group %d/%d beschädigt — übersprungen (%d Zeilen): %s",
                           rg_idx + 1, num_row_groups, rg_rows, exc)
            skipped_rows += rg_rows
            row_cursor   += rg_rows
            continue

        # RESUME: Zeilen vor RESUME_FROM_ROW überspringen
        rg_start   = row_cursor
        rg_end     = row_cursor + len(table)
        row_cursor = rg_end

        if rg_end <= RESUME_FROM_ROW:
            del table
            continue
        if rg_start < RESUME_FROM_ROW:
            table = table.slice(RESUME_FROM_ROW - rg_start)
            rg_start = RESUME_FROM_ROW

        # Row-Group in UPLOAD_BATCH-große Häppchen aufteilen. Die Umwandlung
        # nach Python passiert erst je Häppchen — eine ganze Row-Group als
        # Python-Objekte wären mehrere hundert MB nur an Float-Objekten.
        for offset in range(0, len(table), UPLOAD_BATCH):
            rows = table.slice(offset, UPLOAD_BATCH).to_pylist()
            abs_row = rg_start + offset

            if purge_je_batch:
                new_urls = []
                for r in rows:
                    src = r["source_url"]
                    if src and src not in purged:
                        purged.add(src)
                        new_urls.append(src)
                if new_urls:
                    try:
                        purge_documents(client, new_urls)
                    except Exception as exc:
                        logger.error("Aufräumen alter Chunks fehlgeschlagen: %s", exc)
                        logger.error("Zum Fortsetzen: RESUME_FROM_ROW = %d", abs_row)
                        sys.exit(1)

            points = build_points(rows, kennzahlen)
            mit_fragen += sum(1 for r in rows if r["vector_questions"])

            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
            except Exception as exc:
                logger.error("Upsert fehlgeschlagen bei Zeile %d: %s", abs_row, exc)
                logger.error("Zum Fortsetzen: RESUME_FROM_ROW = %d", abs_row)
                sys.exit(1)

            uploaded += len(points)
            del points, rows

        del table

        pct = 100.0 * (uploaded + skipped_rows) / max(total_rows, 1)
        logger.info("  Row-Group %d/%d  [%5.1f%%]  %d Punkte hochgeladen (gesamt)",
                    rg_idx + 1, num_row_groups, pct, uploaded)

    logger.info("─" * 60)
    logger.info("Fertig. %d Punkte hochgeladen, %d Zeilen übersprungen (beschädigt).",
                uploaded, skipped_rows)
    logger.info("Davon mit Fragenvektor: %d  (ohne: %d)", mit_fragen, uploaded - mit_fragen)
    if purge_je_batch:
        logger.info("%d Dokumente vor dem Schreiben von Altbestand bereinigt.", len(purged))
    logger.info("Collection: %s", COLLECTION_NAME)

    verify_import(client, parquet_file, kennzahlen)

    logger.info("Nächster Schritt: python -m raghsh.index.build_doc_index --rebuild")
    logger.info("  → doc_index.db enthält Qdrant-Chunk-IDs und ist nach einem "
                "Neuimport veraltet.")


if __name__ == "__main__":
    main()
