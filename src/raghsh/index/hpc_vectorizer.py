"""
HPC-Vektorisierung — Markdown-Dateien vektorisieren und als Parquet exportieren.

Kurzbeschreibung
----------------
Dieses Skript ist für den Einsatz auf einem HPC-Cluster (z.B. GWDG/KISSKI mit
NVIDIA H100) vorgesehen. Es hat keine Qdrant-Abhängigkeit. Stattdessen werden
alle erzeugten Vektoren und Metadaten in eine komprimierte Parquet-Datei
geschrieben, die anschließend auf den lokalen Rechner übertragen und dort mit
local_importer.py in Qdrant geladen werden kann.

Workflow:
    0. Login-Node: python hpc_vectorizer.py --download-only   (Modelle in den Cache)
    1. Compute:    python enricher.py                         →  enrichment.parquet
    2. Compute:    python hpc_vectorizer.py --require-gpu \
                       --enrichment enrichment.parquet        →  hsh_vectors.parquet
    3. Transfer:   scp hsh_vectors.parquet user@localhost:~/RAGHSH/var/vektoren/
    4. Lokal:      python -m raghsh.index.local_importer

    Schritt 1 ist optional; ohne --enrichment bleiben die Fragen-Spalten leer.
    Die Reihenfolge ist aber zwingend: die Anreicherung muss VOR der
    Vektorisierung fertig sein, und beide müssen dieselbe chunking.py sehen.
    Letzteres prüft der chunking_config_hash beim Join.

Anreicherung (doc2query, --enrichment):
    enricher.py erzeugt je Chunk Fragen, die der Abschnitt beantwortet, und
    schließt damit die Formulierungslücke zwischen Nutzersprache und
    Verwaltungsdeutsch. Geschrieben werden ALLE Varianten nebeneinander:

      sparse_*            BM25 über den reinen Chunk-Text
      sparse_expanded_*   BM25 über 'Text + Fragen' (doc2query)
      vector              Dense über den Text, unberührt von der Anreicherung
      vector_questions    Dense über die Fragen, QUERY-Adapter — Frage gegen
                          Frage ist ein symmetrischer Vergleich, deshalb nicht
                          'passage'. Ob das besser trägt, ist eine Hypothese
                          und gehört gemessen.

    Damit fällt die Entscheidung, welche Arme das Retrieval nutzt, erst in der
    Query — nicht hier und nicht beim Import. Der Grund ist Umkehrbarkeit: Ein
    einziger, expandierter BM25-Arm ließe sich nachträglich nicht mehr in
    seine Bestandteile zerlegen. Zwei benannte Sparse-Vektoren in Qdrant kosten
    dagegen nur Speicher.

    Abschaltbar über --no-expanded-sparse und --no-question-vector, wenn die
    Rechenzeit oder der Speicher knapp ist.

    Die Payload-Spalte 'text' bleibt in jedem Fall der Originaltext — sonst
    bekäme das antwortende LLM später die Fragen als Kontext serviert.

BM25-Feinheiten (Sprache und avg_len):
    fastembed setzt für Qdrant/bm25 die Sprache 'english' und avg_len=256.0.
    Beides passt hier nicht:

    - Mit englischem Stemmer bleiben 'Anträge' und 'Antrag' getrennte Terme,
      während 'der', 'die', 'und' als bedeutungstragend im Index landen und das
      IDF verzerren. --sparse-language setzt 'german' (Vorgabe). Die
      QUERY-SEITE muss denselben Wert setzen, sonst treffen sich flektierte
      Formen nie mehr.
    - avg_len ist die mittlere Dokumentlänge in TOKENS und steuert die
      Längennormalisierung. Bei ~150 Tokens je Chunk gegen 256 gilt jedes
      Dokument als kürzer als der Durchschnitt. Bei einem Arm ist das ein
      konstanter Faktor; bei zweien bekäme der systematisch längere
      expandierte Arm einen unverdienten Vorteil, den ein A/B-Vergleich als
      doc2query-Effekt fehldeuten würde. Deshalb misst ein Vorlauf an einer
      Stichprobe (--avg-len-sample) je einen Wert pro Arm.

GPU-Nutzung (wichtig):
    fastembed wählt CUDA nur, wenn ONNX Runtime den Provider anbietet. Das
    Paket 'onnxruntime' (CPU-only) bietet ihn nie und fällt lautlos auf die CPU
    zurück — auf einer H100 sieht der Job dann normal aus und rechnet Tage
    statt Minuten. Empfohlener Installationsweg ist 'fastembed-gpu' (zieht
    onnxruntime-gpu als Abhängigkeit); das manuelle Austauschen von
    onnxruntime unter einem normalen 'fastembed' funktioniert meist, hinterlässt
    aber eine aus pip-Sicht inkonsistente Umgebung.

    Geprüft wird auf ZWEI Ebenen, weil die erste nicht ausreicht:
      a) bietet ONNX Runtime überhaupt einen CUDAExecutionProvider an?
      b) benutzt die tatsächlich erzeugte Inferenz-Session ihn auch?
    (b) kann fehlschlagen, obwohl (a) grün ist — etwa wenn onnxruntime-gpu
    andere cuDNN-/CUDA-Versionen erwartet als die im venv installierten
    (typisch neben einem PyTorch-Build). ORT registriert den Provider dann
    beim Session-Aufbau nicht und rechnet still auf der CPU.
    '--require-gpu' bricht bei beiden Fällen ab.

Task-Adapter (jina-embeddings-v3):
    jina-v3 nutzt task-spezifische LoRA-Adapter; für die Indexierung ist
    'retrieval.passage' korrekt, für Suchanfragen 'retrieval.query'.
    fastembeds embed() nimmt **kwargs entgegen und verschluckt unbekannte
    Parameternamen kommentarlos — ein falsch benannter Task-Parameter führt
    also nicht zu einem Fehler, sondern zu einem still falsch eingebetteten
    Korpus. Das Skript ermittelt den Parameternamen deshalb beim Start
    empirisch (siehe resolve_task_kwarg) und bricht ab, wenn es ihn nicht
    verifizieren kann. Override: --task-kwarg.

Speicherverwaltung und GPU-Auslastung:
    Chunks werden über Dateigrenzen hinweg gepuffert und erst bei
    FLUSH_CHUNKS als eine Parquet-Row-Group geschrieben. Das ist der
    wesentliche Unterschied zur früheren dateiweisen Verarbeitung: einzelne
    HsH-Seiten liefern oft nur 3–20 Chunks, die GPU bekam damit Mini-Batches
    weit unterhalb von DENSE_BATCH_SIZE und wartete mehr, als sie rechnete.
    Peak-RAM ≈ Dense-Modell (~500 MB) + Sparse-Modell (~50 MB)
    + FLUSH_CHUNKS × (Text + VECTOR_SIZE × 4 Byte)  ≈ wenige hundert MB.

Parquet-Schema:
    id              string          — deterministischer UUID5 (source_url + chunk_index)
    vector          list<float32>   — 1024-dimensionaler Dense-Einbettungsvektor (Jina)
    source_url      string
    title           string
    crawl_date      string
    content_type    string          — 'html' oder 'pdf'
    faculty         string          — NUR Fakultäten ('Fakultät I' … 'Fakultät V')
    organisation    string          — zentrale Einrichtungen (Bibliothek, Karriere, …)
    section_heading string
    kontext         string           — Anzeigetitel › Überschrift; geht mit in Dense/BM25 ein (22.09.2026)
    text            string          — Volltext des Chunks
    chunk_index     int32
    total_chunks    int32
    sparse_indices  list<int32>     — Token-IDs der BM25-Terme (nur Text)
    sparse_values   list<float32>   — BM25-Termgewichte OHNE IDF (s.u.)
    sparse_expanded_indices list<int32>   — BM25 über Text + Fragen
    sparse_expanded_values  list<float32> — leer ohne --enrichment
    questions       list<string>    — generierte Fragen (leer ohne --enrichment)
    vector_questions list<float32>  — Dense-Vektor der Fragen, QUERY-Adapter;
                                      leere Liste = keine Fragen für den Chunk
    chunking_config_hash string      — Fingerabdruck der Chunk-Grenzen
    selection_hash  string           — Fingerabdruck der Dedup-Auswahlregel

    ACHTUNG, Schema-Änderung: 'organisation' ist neu. Bisher landeten
    'Bibliothek', 'Karriere' und 'International' im Feld 'faculty' — das sind
    keine Fakultäten und machen jeden Fakultätsfilter unsauber. local_importer.py
    muss das neue Feld in die Payload übernehmen.

Zwei Punkte, die dieses Skript NICHT lösen kann (Importer-Seite):
    1. BM25/IDF — fastembeds 'Qdrant/bm25' liefert Termgewichte ohne
       IDF-Komponente. Die wird erst zur Query-Zeit in Qdrant angewandt, aber
       nur, wenn der Sparse-Index mit modifier=models.Modifier.IDF angelegt
       wurde. Fehlt das, ist das Ranking still falsch.
    2. Verwaiste Punkte — chunk_index hängt von der Chunkung des gesamten
       Dokuments ab. Liefert eine neu gecrawlte Seite 7 statt 12 Chunks, werden
       0–6 überschrieben, 7–11 bleiben mit veraltetem Text in Qdrant liegen.
       Der Importer muss vor dem Upsert per Filter auf source_url löschen.
       Dieses Skript schreibt dafür eine Sidecar-Datei <output>_sources.txt
       mit allen verarbeiteten source_urls.

Konfiguration:
    CACHE_DIR          — Modell-Cache (Vorgabe aus $FASTEMBED_CACHE_PATH)
    FLUSH_CHUNKS       — Chunks pro Parquet-Row-Group und Embedding-Durchgang
    DENSE_BATCH_SIZE   — Batch innerhalb eines Durchgangs (H100: 64 aufwärts)
    SPARSE_BATCH_SIZE  — Batch für BM25 (CPU-only, größere Batches OK)
    OUTPUT_FILE        — Ausgabedatei (Parquet)

Abhängigkeiten (HPC-VirtualEnv):
    pip install fastembed-gpu langchain-text-splitters pyarrow
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow as pa
import pyarrow.parquet as pq
from fastembed import SparseTextEmbedding, TextEmbedding
try:
    from raghsh import konfig as _konfig
    from raghsh.index.chunking import (  # gemeinsame Chunking-Logik
    chunk_document,
        config_hash,
        einbettungstext,
        iter_chunks,
        list_markdown,
        make_point_id,
        make_splitters,
        parse_markdown_file,
        read_frontmatter,
        select_unique_ids,
        selection_hash,
    )
except ImportError:  # auf dem HPC-Cluster: liegt neben chunking.py, titel.py und data/ingested
    _konfig = None
    from chunking import (
    chunk_document,
        config_hash,
        einbettungstext,
        iter_chunks,
        list_markdown,
        make_point_id,
        make_splitters,
        parse_markdown_file,
        read_frontmatter,
        select_unique_ids,
        selection_hash,
    )

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

_HIER = Path(__file__).parent
INGESTED_DIR = _konfig.KORPUS if _konfig else _HIER / "data" / "ingested"
OUTPUT_FILE  = (_konfig.VEKTOREN if _konfig else _HIER) / "hsh_vectors.parquet"

DENSE_MODEL  = "jinaai/jina-embeddings-v3"
SPARSE_MODEL = "Qdrant/bm25"

# Sprache für Stemmer und Stoppwortliste des BM25-Modells. fastembed setzt hier
# 'english' als Vorgabe — auf deutschem Text bliebe damit 'Anträge' von
# 'Antrag' getrennt, während 'der', 'die', 'und' als bedeutungstragende Terme
# im Index landen und das IDF verzerren. Der deutsche Snowball-Stemmer faltet
# zugleich die Umlaute (Prüfung → prufung); das betrifft Dokument- und
# Query-Seite gleichermaßen und ist deshalb unschädlich.
#
# WICHTIG: Die Query-Seite (local_importer.py bzw. der Retrieval-Pfad) muss
# dieselbe Sprache setzen. Stemmt die Anfrage englisch gegen einen deutsch
# gestemmten Index, treffen sich flektierte Formen nie mehr — das wäre
# schlimmer als der Ausgangszustand, in dem beide Seiten gleich falsch waren.
SPARSE_LANGUAGE = "german"

# Stichprobe für die avg_len-Messung (siehe estimate_avg_len). Deterministisch
# über einen festen Schritt durch die sortierte Dateiliste gezogen, damit
# Wiederholungsläufe dieselben Termgewichte erzeugen.
AVG_LEN_SAMPLE_FILES = 400
VECTOR_SIZE  = 1024          # Ausgabedimension von jina-embeddings-v3

# Task-Adapter von jina-v3. 'passage' für den Index, 'query' für die Suchseite
# — und für den Fragenvektor 'vector_questions'.
TASK_PASSAGE = "retrieval.passage"
TASK_QUERY   = "retrieval.query"

# fastembed liest FASTEMBED_CACHE_PATH selbst, bekommt cache_dir hier aber
# explizit übergeben. Die Variable wird deshalb hier ausgewertet, damit beide
# Wege auf dasselbe Verzeichnis zeigen.
CACHE_DIR = os.environ.get(
    "FASTEMBED_CACHE_PATH", str(_konfig.CACHE / "fastembed" if _konfig else _HIER / "fastembed_cache"))

FLUSH_CHUNKS       = int(os.environ.get("HSH_FLUSH_CHUNKS", "4096"))   # Chunks je Row-Group; lokal kleiner, damit Fortschritt sichtbar wird
# Auf CPU zählt der Batch doppelt: fastembed füllt auf die längste Sequenz im
# Batch auf, und die Attention wächst mit Batch × Länge². Tabellenchunks werden
# in Tokens lang; bei 64 waren es 7,3 GB und der OOM-Killer (22.09.2026, 8 GB
# Rechner). Für lokale Läufe deshalb HSH_DENSE_BATCH=4 setzen.
DENSE_BATCH_SIZE   = int(os.environ.get("HSH_DENSE_BATCH", "64"))   # A100/H100: 64–128
SPARSE_BATCH_SIZE  = 512     # BM25 läuft auf der CPU, größere Batches sind billig
GC_EVERY_N_FLUSHES = 10      # gegen Fragmentierung bei langen Läufen

# ---------------------------------------------------------------------------
# Parquet-Schema
# ---------------------------------------------------------------------------

PARQUET_SCHEMA = pa.schema([
    pa.field("id",              pa.string()),
    pa.field("vector",          pa.list_(pa.float32())),
    pa.field("source_url",      pa.string()),
    pa.field("title",           pa.string()),
    pa.field("crawl_date",      pa.string()),
    pa.field("content_type",    pa.string()),
    pa.field("faculty",         pa.string()),
    pa.field("organisation",    pa.string()),
    pa.field("section_heading", pa.string()),
    pa.field("kontext",         pa.string()),   # Anzeigetitel › Überschrift (eingebettet, nicht im LLM-Kontext)
    pa.field("text",            pa.string()),
    pa.field("chunk_index",     pa.int32()),
    pa.field("total_chunks",    pa.int32()),
    pa.field("sparse_indices",  pa.list_(pa.int32())),
    pa.field("sparse_values",   pa.list_(pa.float32())),
    # ── Anreicherung (--enrichment); ohne sie leer bzw. leere Listen ────────
    # Zweiter BM25-Arm über 'Text + Fragen'. Bewusst NEBEN dem reinen Textarm
    # gespeichert statt an seiner Stelle: Qdrant kann mehrere benannte
    # Sparse-Vektoren je Punkt halten, damit lässt sich die doc2query-Variante
    # per Query zu- und abschalten, ohne irgendetwas neu zu rechnen.
    pa.field("sparse_expanded_indices", pa.list_(pa.int32())),
    pa.field("sparse_expanded_values",  pa.list_(pa.float32())),
    pa.field("questions",       pa.list_(pa.string())),
    pa.field("vector_questions", pa.list_(pa.float32())),
    pa.field("chunking_config_hash", pa.string()),
    pa.field("selection_hash",       pa.string()),
])

CHUNKING_HASH  = config_hash()
SELECTION_HASH = selection_hash()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def chunks_to_record_batch(chunks: list[dict],
                           dense_vectors: list,
                           sparse_indices: list[list[int]],
                           sparse_values: list[list[float]],
                           sparse_exp_indices: list[list[int]],
                           sparse_exp_values: list[list[float]],
                           questions: list[list[str]],
                           question_vectors: list[list[float]]) -> pa.RecordBatch:
    """Erstellt einen PyArrow-RecordBatch aus Chunks, Dense- und Sparse-Vektoren."""
    return pa.record_batch(
        {
            "id":              [c["_id"]             for c in chunks],
            "vector":          [v.tolist() for v in dense_vectors],
            "source_url":      [c["source_url"]      for c in chunks],
            "title":           [c["title"]           for c in chunks],
            "crawl_date":      [c["crawl_date"]      for c in chunks],
            "content_type":    [c["content_type"]    for c in chunks],
            "faculty":         [c["faculty"]         for c in chunks],
            "organisation":    [c["organisation"]    for c in chunks],
            "section_heading": [c["section_heading"] for c in chunks],
            "kontext":         [c.get("kontext", "") for c in chunks],
            "text":            [c["text"]            for c in chunks],
            "chunk_index":     [c["chunk_index"]     for c in chunks],
            "total_chunks":    [c["total_chunks"]    for c in chunks],
            "sparse_indices":  sparse_indices,
            "sparse_values":   sparse_values,
            "sparse_expanded_indices": sparse_exp_indices,
            "sparse_expanded_values":  sparse_exp_values,
            "questions":       questions,
            "vector_questions": question_vectors,
            "chunking_config_hash": [CHUNKING_HASH] * len(chunks),
            "selection_hash":        [SELECTION_HASH] * len(chunks),
        },
        schema=PARQUET_SCHEMA,
    )


# ---------------------------------------------------------------------------
# Laufzeitprüfungen: benutzt die Session wirklich die GPU, wirkt der Task?
# ---------------------------------------------------------------------------


def session_providers(embedder) -> list[str] | None:
    """Provider der tatsächlich erzeugten ONNX-Session — None, wenn nicht auffindbar.

    ort.get_available_providers() sagt nur, was das Paket anbietet. Ob die
    Session den CUDA-Provider auch registriert hat, steht allein hier.
    Der Attributpfad ist fastembed-intern und versionsabhängig, deshalb
    defensiv über mehrere Kandidaten.
    """
    for path in (("model", "model"), ("model", "session"), ("model",)):
        obj = embedder
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "get_providers"):
            try:
                return list(obj.get_providers())
            except Exception:  # noqa: BLE001 — rein diagnostisch
                continue
    return None


def _task_id_pair(embedder) -> tuple | None:
    """Ermittelt die numerischen Task-IDs für passage/query.

    In fastembed 0.7.x ist jina-v3 die Klasse JinaEmbeddingV3, die die IDs als
    Klassenkonstanten PASSAGE_TASK/QUERY_TASK führt; der ONNX-Graph hat einen
    eigenen 'task_id'-Input. Ältere Versionen boten stattdessen ein
    get_task_type_dict() — als Fallback mitgeprüft.
    """
    model = getattr(embedder, "model", None)
    holders = [h for h in (model, type(model) if model is not None else None, embedder)
               if h is not None]

    for holder in holders:
        passage = getattr(holder, "PASSAGE_TASK", None)
        query   = getattr(holder, "QUERY_TASK", None)
        # Bewusst keine Typprüfung: ob die Werte brauchbar sind, entscheidet der
        # Wirksamkeitstest in resolve_task_kwarg. Ein isinstance-Filter würde
        # brauchbare Werte anderen Typs still verwerfen.
        if passage is not None and query is not None and passage != query:
            return passage, query

    for holder in holders:
        getter = getattr(holder, "get_task_type_dict", None)
        if not callable(getter):
            continue
        try:
            mapping = getter()
        except Exception:  # noqa: BLE001
            continue
        if isinstance(mapping, dict) and TASK_PASSAGE in mapping and TASK_QUERY in mapping:
            return mapping[TASK_PASSAGE], mapping[TASK_QUERY]
    return None


def resolve_task_kwarg(embedder, forced: str,
                       passage_id: int | None = None,
                       query_id: int | None = None) -> tuple[dict, dict]:
    """Ermittelt, wie der jina-Task an fastembed übergeben werden muss.

    Testverfahren: denselben Text einmal als 'passage' und einmal als 'query'
    einbetten. Unterscheiden sich die Vektoren, greift der Parameter. Sind sie
    identisch, wurde er über **kwargs verschluckt — dann wäre der ganze Korpus
    ohne Passage-Adapter eingebettet, ohne dass irgendwo ein Fehler aufträte.

    Rückgabe ist ein Paar (passage, query): Der Chunk-Text wird mit dem
    Passage-Adapter eingebettet, die generierten Fragen dagegen mit dem
    Query-Adapter — sie werden später gegen echte Nutzerfragen verglichen, das
    ist ein symmetrischer Vergleich.

    Nebeneffekt: der erste echte Inferenzlauf passiert hier, die Session ist
    danach warm.
    """
    if forced == "none":
        logger.warning(
            "--task-kwarg none: es wird KEIN Task-Adapter gesetzt. Der Korpus wird "
            "mit dem fastembed-Default eingebettet; Query-Seite entsprechend anpassen."
        )
        return {}, {}

    probe = ["Die Hochschule Hannover bietet Studiengänge in der Fakultät I an."]

    candidates: list[tuple[str, object, object]] = []

    # task_id zuerst: das ist der Parameter, den fastembed 0.7.x tatsächlich
    # entgegennimmt. 'task' bleibt als Kandidat für andere Versionen stehen.
    if forced in ("auto", "task_id"):
        pair = (passage_id, query_id) if passage_id is not None and query_id is not None \
            else _task_id_pair(embedder)
        if pair:
            candidates.append(("task_id", pair[0], pair[1]))
            logger.info("Task-IDs: passage=%s query=%s", *pair)
        elif forced == "task_id":
            logger.error(
                "--task-kwarg task_id verlangt, aber keine Task-IDs auffindbar. "
                "Mit --task-passage-id und --task-query-id explizit setzen."
            )
            sys.exit(1)
    if forced in ("auto", "task"):
        candidates.append(("task", TASK_PASSAGE, TASK_QUERY))

    for kw, passage, query in candidates:
        try:
            v_pass = next(iter(embedder.embed(probe, **{kw: passage})))
            v_query = next(iter(embedder.embed(probe, **{kw: query})))
        except Exception as exc:  # noqa: BLE001 — Kandidat scheidet aus
            logger.info("Task-Parameter '%s' nicht nutzbar: %s", kw, exc)
            continue
        if not np.allclose(v_pass, v_query, atol=1e-5):
            logger.info("Task-Parameter verifiziert: %s=%r wirkt.", kw, passage)
            return {kw: passage}, {kw: query}
        logger.info("Task-Parameter '%s' bleibt wirkungslos (passage == query).", kw)

    logger.error(
        "Kein wirksamer Task-Parameter gefunden. Der Korpus würde ohne den "
        "'%s'-Adapter eingebettet, und zwar ohne Fehlermeldung. Bitte die "
        "fastembed-Version prüfen und den Parameternamen mit --task-kwarg "
        "setzen; --task-kwarg none übergeht diese Prüfung bewusst.",
        TASK_PASSAGE,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Embedding + Schreiben
# ---------------------------------------------------------------------------


def build_sparse(embedder, texts: list[str], parallel: int | None
                 ) -> tuple[list[list[int]], list[list[float]]]:
    """Rechnet einen BM25-Arm und zerlegt ihn in die beiden Parquet-Spalten."""
    indices: list[list[int]] = []
    values: list[list[float]] = []
    for emb in embedder.embed(texts, batch_size=SPARSE_BATCH_SIZE, parallel=parallel):
        indices.append(emb.indices.tolist())
        values.append(emb.values.tolist())
    return indices, values


def embed_and_write(chunks: list[dict], dense_embedder, sparse_embedder,
                    sparse_expanded_embedder, writer: pq.ParquetWriter,
                    task_kwargs: dict, query_task_kwargs: dict,
                    sparse_parallel: int | None,
                    enrichment: dict[str, list[str]] | None,
                    question_vector: bool) -> tuple[int, int]:
    """Bettet einen Chunk-Puffer ein und schreibt ihn als eine Row-Group.

    Rückgabe: (geschriebene Zeilen, davon mit Fragen).

    Geschrieben werden bis zu vier Vektoren je Chunk:

      vector                    Dense über den Chunk-Text, PASSAGE-Adapter.
                                Von der Anreicherung unberührt.
      sparse_*                  BM25 über den reinen Chunk-Text.
      sparse_expanded_*         BM25 über 'Text + Fragen' (doc2query). Nur bei
                                --enrichment, sonst leere Listen.
      vector_questions          Dense über die zusammengefügten Fragen,
                                QUERY-Adapter. Nur bei --enrichment und ohne
                                --no-question-vector.

    Die beiden BM25-Arme stehen NEBENEINANDER statt einer an der Stelle des
    anderen. Der Grund ist Umkehrbarkeit: Die Termgewichte lassen sich später
    nicht mehr auftrennen, ein einziger expandierter Arm wäre also eine
    Einbahnstraße. Zwei benannte Sparse-Vektoren in Qdrant kosten dagegen nur
    Speicher, und welche Kombination gesucht wird, entscheidet die Query.

    In der Payload bleibt 'text' immer der Originaltext — sonst bekäme das
    antwortende LLM später die Fragen als Kontext serviert.
    """
    # Einmal berechnen: die ID ist zugleich Join-Schlüssel zur Anreicherung
    # und Punkt-ID in Qdrant. Zwei getrennte Berechnungen wären zwei Stellen,
    # an denen sie auseinanderlaufen können.
    for c in chunks:
        c["_id"] = make_point_id(c["source_url"], c["chunk_index"])

    # Eingebettet wird Kontextkopf + Text (chunking.einbettungstext, seit
    # 22.09.2026): Titel und Überschrift lagen vorher nur in der Payload, und
    # ein Chunk aus der Mitte eines Stundenplans wusste nicht, zu welchem
    # Studiengang er gehört. `text` in der Payload bleibt der reine Chunk.
    texts = [einbettungstext(c) for c in chunks]

    if enrichment is None:
        questions = [[] for _ in chunks]
    else:
        questions = [enrichment.get(c["_id"], []) for c in chunks]
    mit_fragen = sum(1 for q in questions if q)

    # ── Dense über den Chunk-Text ─────────────────────────────────────────
    # fastembed batcht selbst — die frühere manuelle range()-Schleife erzeugte
    # nur zusätzliche Aufrufe.
    dense_vecs = list(
        dense_embedder.embed(texts, batch_size=DENSE_BATCH_SIZE, **task_kwargs)
    )

    # ── BM25 über den reinen Text ─────────────────────────────────────────
    sparse_indices, sparse_values = build_sparse(
        sparse_embedder, texts, sparse_parallel)

    # ── BM25 über Text + Fragen ───────────────────────────────────────────
    if sparse_expanded_embedder is not None:
        sparse_exp_indices, sparse_exp_values = build_sparse(
            sparse_expanded_embedder, expanded_texts(texts, questions),
            sparse_parallel)
    else:
        sparse_exp_indices = [[] for _ in chunks]
        sparse_exp_values = [[] for _ in chunks]

    # ── Zweiter Dense-Vektor aus den Fragen ───────────────────────────────
    question_vectors: list[list[float]] = [[] for _ in chunks]
    if question_vector and enrichment is not None:
        # Nur die Chunks mit Fragen einbetten und positionsweise zurückschreiben.
        # Leere Texte durch das Modell zu schicken kostet Rechenzeit und liefert
        # einen Vektor, der überall mittelmäßig ähnlich ist — genau die Art
        # Treffer, die ein Retrieval verdirbt.
        idx_mit = [i for i, q in enumerate(questions) if q]
        if idx_mit:
            frage_texte = [" ".join(questions[i]) for i in idx_mit]
            q_vecs = list(dense_embedder.embed(frage_texte,
                                               batch_size=DENSE_BATCH_SIZE,
                                               **query_task_kwargs))
            if len(q_vecs) != len(idx_mit):
                raise RuntimeError(
                    f"Fragenvektoren: {len(q_vecs)} zurück für {len(idx_mit)} Eingaben")
            for i, v in zip(idx_mit, q_vecs):
                question_vectors[i] = v.tolist()

    # Die Arme werden getrennt erzeugt und positionsweise zusammengeführt.
    # Ein Längenunterschied wäre eine stille Fehlzuordnung von Vektor zu Text.
    if not (len(dense_vecs) == len(sparse_indices) == len(sparse_exp_indices)
            == len(chunks)):
        raise RuntimeError(
            f"Längen weichen ab: chunks={len(chunks)} dense={len(dense_vecs)} "
            f"sparse={len(sparse_indices)} sparse_exp={len(sparse_exp_indices)}"
        )

    batches = [
        chunks_to_record_batch(
            chunks[i:i + DENSE_BATCH_SIZE],
            dense_vecs[i:i + DENSE_BATCH_SIZE],
            sparse_indices[i:i + DENSE_BATCH_SIZE],
            sparse_values[i:i + DENSE_BATCH_SIZE],
            sparse_exp_indices[i:i + DENSE_BATCH_SIZE],
            sparse_exp_values[i:i + DENSE_BATCH_SIZE],
            questions[i:i + DENSE_BATCH_SIZE],
            question_vectors[i:i + DENSE_BATCH_SIZE],
        )
        for i in range(0, len(chunks), DENSE_BATCH_SIZE)
    ]
    table = pa.Table.from_batches(batches, schema=PARQUET_SCHEMA)
    writer.write_table(table)
    return len(table), mit_fragen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def expanded_texts(texts: list[str], questions: list[list[str]]) -> list[str]:
    """'Text + Fragen' für den doc2query-Arm — an EINER Stelle definiert.

    Die avg_len-Messung und die eigentliche Vektorisierung müssen exakt
    dieselbe Verkettung verwenden, sonst misst der Vorlauf eine andere
    Dokumentlänge als die später gewichtete.
    """
    return [t + "\n" + " ".join(q) if q else t for t, q in zip(texts, questions)]


def count_tokens(embedder, texts: list[str]) -> int:
    """Zählt Tokens mit der BM25-eigenen Tokenisierung.

    Wichtig, dass es genau diese ist: doc_len in der BM25-Formel zählt die
    Tokens NACH Stemming und Stoppwortentfernung. Eine eigene Näherung über
    text.split() läge im Deutschen deutlich daneben und würde avg_len
    systematisch verschieben.
    """
    for besitzer in (embedder, getattr(embedder, "model", None)):
        zaehler = getattr(besitzer, "token_count", None)
        if callable(zaehler):
            return int(zaehler(texts))
    raise RuntimeError(
        "Diese fastembed-Version stellt token_count() nicht bereit — avg_len "
        "kann nicht gemessen werden. Bitte --avg-len und --avg-len-expanded "
        "manuell setzen (mittlere Tokenzahl je Chunk, nicht Zeichen)."
    )


def estimate_avg_len(files: list[Path], probe, enrichment: dict[str, list[str]] | None,
                     sample_files: int, url_filter: set[str] | None,
                     keep_ids: set[str] | None = None
                     ) -> tuple[float, float]:
    """Misst die mittlere Chunk-Länge in Tokens — je einmal pro BM25-Arm.

    Warum überhaupt: fastembed setzt avg_len auf 256.0. Deine Chunks liegen bei
    grob 150 Tokens, gelten damit durchgehend als 'kürzer als der Durchschnitt'
    und werden gleichmäßig angehoben. Bei EINEM Arm ist das ein konstanter
    Faktor ohne Rankingwirkung. Bei zweien wird es zum Problem: Die
    expandierten Dokumente sind systematisch länger, und mit gemeinsamem
    avg_len bekäme der doc2query-Arm einen unverdienten Vorteil. Der A/B-
    Vergleich würde dann teilweise dieses Artefakt messen statt den Effekt der
    Fragen.

    Die Stichprobe läuft über einen festen Schritt durch die sortierte
    Dateiliste, nicht zufällig: Wiederholungsläufe sollen dieselben
    Termgewichte erzeugen, sonst sind zwei Läufe nicht mehr vergleichbar.
    """
    schritt = max(1, len(files) // sample_files)
    stichprobe = files[::schritt][:sample_files]
    logger.info("Messe avg_len an %d von %d Datei(en) …", len(stichprobe), len(files))

    texte: list[str] = []
    fragen: list[list[str]] = []
    for _path, chunks in iter_chunks(stichprobe):
        for c in chunks:
            if url_filter is not None and c["source_url"] not in url_filter:
                continue
            # Duplikate zählen nicht mit: avg_len soll den Korpus beschreiben,
            # der tatsächlich im Index landet. Eine 234-fach wiederholte
            # Navigationszeile würde den Mittelwert sonst nach unten ziehen.
            if keep_ids is not None and make_point_id(
                    c["source_url"], c["chunk_index"]) not in keep_ids:
                continue
            texte.append(einbettungstext(c))
            ident = make_point_id(c["source_url"], c["chunk_index"])
            fragen.append(enrichment.get(ident, []) if enrichment else [])

    if not texte:
        logger.warning("Stichprobe leer — es bleibt bei der fastembed-Vorgabe 256.")
        return 256.0, 256.0

    avg_text = count_tokens(probe, texte) / len(texte)
    if enrichment is None:
        return avg_text, avg_text

    avg_exp = count_tokens(probe, expanded_texts(texte, fragen)) / len(texte)
    return avg_text, avg_exp


def load_enrichment(path: Path, ignore_hash: bool = False) -> dict[str, list[str]]:
    """Lädt id → Fragen aus enrichment.parquet (oder dem .parts-Verzeichnis).

    Geprüft werden zwei Fingerabdrücke, und sie haben verschiedenes Gewicht:

    chunking_config_hash — die Chunk-GRENZEN. Weicht er ab, wurde zwischen
    Anreicherung und Vektorisierung an den Splitter-Parametern oder den
    Bereinigungsmustern geschraubt. Die IDs blieben syntaktisch gültig und
    würden weiterhin joinen — nur zeigten sie auf andere Textstellen. Das
    Ergebnis wäre ein Index, in dem Fragen an fremden Abschnitten hängen, ohne
    Fehlermeldung. Deshalb Abbruch.

    selection_hash — die AUSWAHL, welche Chunks die Deduplizierung behält.
    Weicht er ab, zeigt jede ID weiterhin auf denselben Text; es decken sich
    nur die Punktmengen nicht. Praktisch heißt das: für neu hinzugekommene
    Chunks liegen noch keine Fragen vor. Das ist kein stiller Fehler, sondern
    an der Trefferquote unten direkt ablesbar — deshalb nur eine Warnung.

    Zeilen ohne Fragen (ok=False) werden nicht aufgenommen; sie stehen in der
    Datei nur für die Fehlerquote.
    """
    if not path.exists():
        logger.error("Anreicherungsdatei nicht gefunden: %s", path)
        sys.exit(1)

    spalten = ["id", "questions"]
    schema = pq.read_schema(path) if path.is_file() else None
    hat_hash = schema is not None and "chunking_config_hash" in schema.names
    if hat_hash:
        spalten.append("chunking_config_hash")
    hat_auswahl = schema is not None and "selection_hash" in schema.names
    if hat_auswahl:
        spalten.append("selection_hash")

    # PyArrow liest ein Verzeichnis mit Teildateien genauso wie eine Datei —
    # so lässt sich auch ohne vorheriges --merge vektorisieren.
    table = pq.read_table(path, columns=spalten)
    logger.info("Anreicherung: %d Zeile(n) aus %s", table.num_rows, path)

    if hat_hash:
        hashes = set(table.column("chunking_config_hash").to_pylist())
        hashes.discard(None)
        if hashes and hashes != {CHUNKING_HASH}:
            meldung = (
                f"Chunking-Fingerabdruck weicht ab: Anreicherung {sorted(hashes)}, "
                f"dieser Lauf {CHUNKING_HASH}. Die Chunk-Grenzen haben sich "
                f"zwischen den Läufen verschoben — die Fragen gehörten dann zu "
                f"anderen Textstellen. Anreicherung neu erzeugen (oder "
                f"--ignore-enrichment-hash, wenn du sicher weißt, was du tust). "
                f"Wenn die Datei von VOR dem 14.08.2026 stammt: Damals ging die "
                f"Dedup-Regel noch in diesen Hash ein, die Grenzen selbst haben "
                f"sich nicht verschoben — dann ist enrichment_v2.parquet aus "
                f"erbe_fragen.py die richtige Datei, nicht ein neuer Lauf."
            )
            if ignore_hash:
                logger.warning(meldung)
            else:
                logger.error(meldung)
                sys.exit(1)
    else:
        logger.warning(
            "Anreicherung ohne chunking_config_hash (ältere enricher.py) — die "
            "Übereinstimmung der Chunk-Grenzen ist nicht überprüfbar. Die "
            "Trefferquote unten ist der einzige Hinweis darauf.")

    if hat_auswahl:
        auswahl = set(table.column("selection_hash").to_pylist())
        auswahl.discard(None)
        if auswahl and auswahl != {SELECTION_HASH}:
            logger.warning(
                "Auswahlregel weicht ab: Anreicherung %s, dieser Lauf %s. Die "
                "Chunk-Grenzen sind identisch, die Fragen sitzen also richtig "
                "— aber die Deduplizierung behält eine andere Punktmenge. Für "
                "hinzugekommene Chunks fehlen Fragen; die Trefferquote unten "
                "zeigt, wie viele.", sorted(auswahl), SELECTION_HASH)
    else:
        logger.info(
            "Anreicherung ohne selection_hash — sie stammt aus einer Version "
            "vor der Trennung von Grenzen und Auswahl.")

    mapping: dict[str, list[str]] = {}
    for ident, fragen in zip(table.column("id").to_pylist(),
                             table.column("questions").to_pylist()):
        if fragen:
            mapping[ident] = list(fragen)
    logger.info("Davon mit Fragen: %d", len(mapping))
    return mapping


def load_url_filter(path: Path) -> set[str]:
    """Liest eine URL-Liste (eine pro Zeile, # = Kommentar) für --urls."""
    if not path.exists():
        logger.error("URL-Liste nicht gefunden: %s", path)
        sys.exit(1)
    urls = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.add(line)
    return urls


def require_explicit_output(out_file: Path, reason: str) -> None:
    """Verhindert, dass ein Teillauf die Volldatei ersetzt.

    Betrifft --urls und --resume-from gleichermaßen: beide schreiben nur einen
    Ausschnitt, und der ParquetWriter beginnt immer mit einer leeren Datei.
    Ginge das auf die Standard-Zieldatei, ersetzte der Ausschnitt am Ende den
    gesamten Bestand.
    """
    if out_file.resolve() == OUTPUT_FILE.resolve():
        logger.error(
            "%s zusammen mit der Standard-Ausgabe %s. Das würde die Volldatei "
            "durch den Teillauf ersetzen. Bitte --output explizit auf einen "
            "anderen Pfad setzen, z.B. --output hsh_vectors_delta.parquet",
            reason, OUTPUT_FILE.name,
        )
        sys.exit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vektorisiert Markdown-Dateien (Dense + BM25) nach Parquet."
    )
    parser.add_argument(
        "--urls", type=Path, metavar="FILE",
        help="Nur Dateien mit diesen source_urls vektorisieren (eine pro Zeile). "
             "Delta-Betrieb nach einem Teil-Crawl. Verlangt --output.",
    )
    parser.add_argument(
        "--output", type=Path, metavar="FILE", default=OUTPUT_FILE,
        help=f"Zieldatei (Standard: {OUTPUT_FILE.name}).",
    )
    parser.add_argument(
        "--input-dir", type=Path, metavar="DIR", default=INGESTED_DIR,
        help=f"Quellverzeichnis (Standard: {INGESTED_DIR.name}). Für ein "
             "entpacktes Delta-Archiv auf dem HPC.",
    )
    parser.add_argument(
        "--resume-from", type=int, metavar="N", default=1,
        help="Ab Dateinummer N weiterrechnen. Erzeugt eine Datei, die NUR die "
             "Dateien ab N enthält — Parquet kann nicht fortgeschrieben werden. "
             "Verlangt deshalb --output; die Teildateien anschließend als "
             "Parquet-Dataset gemeinsam einlesen.",
    )

    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument(
        "--require-gpu", action="store_true",
        help="Abbrechen, wenn die Inferenz-Session nicht auf CUDA läuft. Auf dem "
             "HPC immer setzen — sonst rechnet der Job unbemerkt auf der CPU.",
    )
    gpu.add_argument(
        "--cpu", action="store_true",
        help="Erzwingt CPU-Ausführung (nur für lokale Tests sinnvoll).",
    )

    parser.add_argument(
        "--task-kwarg", choices=("auto", "task", "task_id", "none"), default="auto",
        help="Name des jina-Task-Parameters. 'auto' probiert die Kandidaten durch "
             "und verifiziert sie (Standard). 'none' setzt keinen Task-Adapter.",
    )
    parser.add_argument(
        "--task-passage-id", type=int, metavar="N", default=None,
        help="Numerische Task-ID für 'retrieval.passage'. Überschreibt die aus dem "
             "Modell gelesene ID. Nur zusammen mit --task-query-id sinnvoll.",
    )
    parser.add_argument(
        "--task-query-id", type=int, metavar="N", default=None,
        help="Numerische Task-ID für 'retrieval.query' — wird nur für die "
             "Wirksamkeitsprüfung gebraucht, nicht für die Indexierung.",
    )
    parser.add_argument(
        "--enrichment", type=Path, metavar="FILE", default=None,
        help="enrichment.parquet aus enricher.py (oder das .parts-Verzeichnis). "
             "Die Fragen gehen als Payload mit, in einen zweiten BM25-Arm "
             "(sparse_expanded_*) und in einen zweiten Dense-Vektor "
             "(vector_questions).",
    )
    parser.add_argument(
        "--no-question-vector", action="store_true",
        help="Den Dense-Vektor aus den Fragen NICHT berechnen. Spart GPU-Zeit, "
             "wenn nur der doc2query-Effekt auf der BM25-Seite interessiert.",
    )
    parser.add_argument(
        "--no-expanded-sparse", action="store_true",
        help="Den zweiten BM25-Arm über 'Text + Fragen' NICHT berechnen.",
    )
    parser.add_argument(
        "--sparse-language", default=SPARSE_LANGUAGE, metavar="LANG",
        help=f"Sprache für Stemmer und Stoppwortliste des BM25-Modells "
             f"(Vorgabe: {SPARSE_LANGUAGE}). fastembed selbst würde 'english' "
             f"nehmen. Die Query-Seite muss denselben Wert setzen.",
    )
    parser.add_argument(
        "--avg-len", type=float, default=None, metavar="N",
        help="avg_len des BM25-Arms über den reinen Text, in TOKENS. Ohne "
             "Angabe an einer Stichprobe gemessen (siehe --avg-len-sample).",
    )
    parser.add_argument(
        "--avg-len-expanded", type=float, default=None, metavar="N",
        help="avg_len des BM25-Arms über 'Text + Fragen', in Tokens.",
    )
    parser.add_argument(
        "--avg-len-sample", type=int, default=AVG_LEN_SAMPLE_FILES, metavar="N",
        help=f"Wie viele Dateien der Messvorlauf anfasst (Vorgabe: "
             f"{AVG_LEN_SAMPLE_FILES}). 0 schaltet die Messung ab und belässt "
             f"es bei der fastembed-Vorgabe 256.",
    )
    parser.add_argument(
        "--ignore-enrichment-hash", action="store_true",
        help="Abweichenden chunking_config_hash nur warnen statt abbrechen. "
             "Nur sinnvoll, wenn die Änderung an chunking.py die Chunk-Grenzen "
             "nachweislich nicht verschoben hat. Für eine geänderte "
             "Dedup-Auswahlregel wird der Schalter NICHT gebraucht — die "
             "steht seit dem 14.08.2026 im eigenen selection_hash und warnt "
             "ohnehin nur.",
    )
    parser.add_argument(
        "--sparse-parallel", type=int, metavar="N", default=None,
        help="Worker-Prozesse für BM25. Ohne Angabe single-process.",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Nur die Modelle in den Cache laden und beenden. Auf dem Login-Node "
             "ausführen — Compute-Nodes haben keinen Internetzugang.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    ingested_dir = args.input_dir
    out_file     = args.output

    # ── 1. Provider-Vorprüfung (Paketebene) ───────────────────────────────
    available = ort.get_available_providers()
    has_cuda  = "CUDAExecutionProvider" in available
    logger.info("ONNX Runtime %s — verfügbare Provider: %s", ort.__version__, available)

    if args.require_gpu and not has_cuda:
        logger.error(
            "Kein CUDAExecutionProvider verfügbar (--require-gpu gesetzt). "
            "In der HPC-venv 'fastembed-gpu' installieren:\n"
            "    pip uninstall -y fastembed onnxruntime && pip install fastembed-gpu"
        )
        sys.exit(1)

    use_cuda = has_cuda and not args.cpu
    logger.info("Dense-Ausführung angefordert: %s", "GPU (CUDA)" if use_cuda else "CPU")

    # ── 2. Modelle laden ──────────────────────────────────────────────────
    cache_path = Path(CACHE_DIR)
    if cache_path.resolve().is_relative_to(Path.home().resolve()):
        logger.warning(
            "Modell-Cache liegt unter $HOME (%s). Auf GWDG-Systemen ist die "
            "Home-Quota knapp; FASTEMBED_CACHE_PATH besser ins Projektverzeichnis legen.",
            cache_path,
        )
    # Threadzahl EXPLIZIT setzen. Ohne Angabe legt ONNX Runtime einen Pool über
    # alle Kerne des Knotens an und versucht, jeden Thread an einen festen Kern
    # zu binden — in einer Slurm-cgroup zeigen diese Masken auf nicht zugeteilte
    # Kerne, was hunderte 'pthread_setaffinity_np failed'-Fehler erzeugt.
    # sched_getaffinity() liefert die tatsächlich erlaubten CPUs, auch wenn
    # SLURM_CPUS_PER_TASK nicht gesetzt ist (z.B. srun ohne --cpus-per-task).
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        threads = int(slurm_cpus)
    else:
        threads = len(os.sched_getaffinity(0))
    logger.info("ONNX-Threads: %s", threads)

    logger.info("Lade Dense-Modell '%s' (Cache: %s) …", DENSE_MODEL, cache_path)
    dense_embedder = TextEmbedding(
        model_name=DENSE_MODEL,
        cache_dir=str(cache_path),
        cuda=use_cuda,
        device_ids=[0] if use_cuda else None,
        threads=threads,
    )

    logger.info("Lade Sparse-Modell '%s' (BM25, Sprache=%s, CPU-only, threads=%s) …",
                SPARSE_MODEL, args.sparse_language, threads)
    # Sonde: dient nur der Tokenzählung für die avg_len-Messung. token_count()
    # hängt nicht von avg_len ab, deshalb ist die Vorgabe hier gleichgültig.
    sparse_probe = SparseTextEmbedding(
        model_name=SPARSE_MODEL,
        cache_dir=str(cache_path),
        threads=threads,
        language=args.sparse_language,
    )
    logger.info("Beide Modelle bereit.")

    if args.download_only:
        logger.info("--download-only: Modelle liegen im Cache, Ende.")
        return

    # ── 3. Provider-Nachprüfung (Session-Ebene) ───────────────────────────
    # Das ist die Prüfung, auf die es ankommt: onnxruntime-gpu kann installiert
    # sein und den Provider anbieten, ihn beim Session-Aufbau aber wegen
    # CUDA-/cuDNN-Versionskonflikten still verwerfen.
    providers = session_providers(dense_embedder)
    if providers is None:
        logger.warning(
            "Session-Provider nicht auslesbar (fastembed-Interna geändert?) — "
            "GPU-Nutzung nicht verifizierbar. Bitte nvidia-smi während des Laufs prüfen."
        )
    else:
        logger.info("Session-Provider (Dense): %s", providers)
        on_cuda = "CUDAExecutionProvider" in providers
        if args.require_gpu and not on_cuda:
            logger.error(
                "CUDA war verfügbar, die Session läuft aber auf der CPU. Typische "
                "Ursache: onnxruntime-gpu erwartet andere CUDA-/cuDNN-Versionen als "
                "die im venv installierten. Abbruch wegen --require-gpu."
            )
            sys.exit(1)
        if not on_cuda and not args.cpu:
            logger.warning("Läuft auf der CPU — bei ~200k Chunks bedeutet das Tage statt Minuten.")

    # ── 4. Task-Adapter verifizieren ──────────────────────────────────────
    task_kwargs, query_task_kwargs = resolve_task_kwarg(
        dense_embedder, args.task_kwarg, args.task_passage_id, args.task_query_id)

    # ── 4b. Anreicherung laden ────────────────────────────────────────────
    logger.info("Fingerabdruck Grenzen: %s | Auswahl: %s",
                CHUNKING_HASH, SELECTION_HASH)
    enrichment = None
    if args.enrichment:
        enrichment = load_enrichment(args.enrichment, args.ignore_enrichment_hash)
        if not args.no_question_vector and not query_task_kwargs:
            logger.warning(
                "Kein Task-Adapter gesetzt (--task-kwarg none): der Fragenvektor "
                "wird mit dem fastembed-Default eingebettet statt mit '%s'.",
                TASK_QUERY)

    # ── 5. Dateien einlesen und filtern ───────────────────────────────────
    md_files = list_markdown(ingested_dir)
    if not md_files:
        logger.error("Keine .md-Dateien gefunden in %s", ingested_dir)
        sys.exit(1)
    logger.info("Gefunden: %d .md-Datei(en) in %s", len(md_files), ingested_dir)

    # ── Globale Deduplizierung ────────────────────────────────────────────
    # ZWINGEND vor --urls und --resume-from: select_unique_ids() wählt den
    # Vertreter jeder Duplikatgruppe in Dateireihenfolge. Über eine Teilmenge
    # berechnet fiele die Wahl anders aus als im Anreicherer, und der Join
    # liefe auf Chunks, die dort nie angereichert wurden.
    #
    # Kostet auch im Delta-Betrieb einen vollständigen Chunking-Durchlauf
    # (~1 min für 5.600 Dateien). Das ist der Preis dafür, dass die Punktmenge
    # unabhängig vom Aufrufweg dieselbe ist.
    keep_ids = select_unique_ids(md_files)
    if keep_ids is not None:
        logger.info("Deduplizierung aktiv: %d eindeutige Chunk(s) werden "
                    "vektorisiert.", len(keep_ids))

    if args.urls:
        require_explicit_output(out_file, "--urls (Delta-Betrieb)")
        wanted = load_url_filter(args.urls)
        md_files = [p for p in md_files if read_frontmatter(p).get("source_url", "") in wanted]
        logger.info("Delta-Betrieb: %d von %d URL(s) der Liste im Verzeichnis gefunden.",
                    len(md_files), len(wanted))
        if not md_files:
            logger.error("Keine der gelisteten URLs liegt in %s — abgebrochen.", ingested_dir)
            sys.exit(1)

    if args.resume_from > 1:
        require_explicit_output(out_file, f"--resume-from {args.resume_from}")
        skip = args.resume_from - 1
        logger.info("Überspringe die ersten %d Datei(en) (--resume-from %d)",
                    skip, args.resume_from)
        md_files = md_files[skip:]
        if not md_files:
            logger.error("--resume-from %d liegt hinter der letzten Datei.", args.resume_from)
            sys.exit(1)

    # ── 5b. avg_len messen und die BM25-Arme bauen ────────────────────────
    # Erst hier, weil die Messung die endgültige Dateiliste braucht: --urls und
    # --resume-from ändern den Korpus, und avg_len soll den beschreiben, der
    # tatsächlich vektorisiert wird.
    if args.avg_len is not None and (enrichment is None
                                     or args.avg_len_expanded is not None
                                     or args.no_expanded_sparse):
        avg_text = args.avg_len
        avg_exp = args.avg_len_expanded if args.avg_len_expanded is not None else avg_text
        logger.info("avg_len gesetzt: Text %.1f, erweitert %.1f (Tokens)", avg_text, avg_exp)
    elif args.avg_len_sample <= 0:
        avg_text = avg_exp = 256.0
        logger.info("avg_len-Messung abgeschaltet — fastembed-Vorgabe 256.0.")
    else:
        avg_text, avg_exp = estimate_avg_len(
            md_files, sparse_probe, enrichment, args.avg_len_sample,
            wanted if args.urls else None, keep_ids)
        if args.avg_len is not None:
            avg_text = args.avg_len
        if args.avg_len_expanded is not None:
            avg_exp = args.avg_len_expanded
        logger.info("avg_len gemessen: Text %.1f, erweitert %.1f Tokens "
                    "(fastembed-Vorgabe wäre 256.0)", avg_text, avg_exp)

    sparse_embedder = SparseTextEmbedding(
        model_name=SPARSE_MODEL, cache_dir=str(cache_path), threads=threads,
        language=args.sparse_language, avg_len=avg_text,
    )
    # Eigene Instanz mit eigenem avg_len. Beide Arme dieselbe Zahl zu geben
    # wäre der Fehler, den die Messung gerade vermeidet: die expandierten
    # Dokumente sind systematisch länger und bekämen sonst einen unverdienten
    # Vorteil, den ein A/B-Vergleich als doc2query-Effekt fehldeutet.
    sparse_expanded_embedder = None
    if enrichment is not None and not args.no_expanded_sparse:
        sparse_expanded_embedder = SparseTextEmbedding(
            model_name=SPARSE_MODEL, cache_dir=str(cache_path), threads=threads,
            language=args.sparse_language, avg_len=avg_exp,
        )

    # ── 6. Splitter vorbereiten (zustandslos, einmal genügt) ──────────────
    header_splitter, rec_splitter = make_splitters()

    # ── 7. Parquet-Writer öffnen ──────────────────────────────────────────
    # Geschrieben wird in eine .partial-Datei, die erst nach erfolgreichem
    # Durchlauf über die Zieldatei geschoben wird. Grund: ParquetWriter leert
    # sein Ziel bereits beim Öffnen. Ohne diesen Umweg vernichtet jeder Abbruch
    # ab hier — auch ein OOM — die vorhandene Ausgabe.
    tmp_file = out_file.with_name(out_file.name + ".partial")
    writer = pq.ParquetWriter(
        tmp_file,
        schema=PARQUET_SCHEMA,
        compression="zstd",
        compression_level=3,
    )
    logger.info("Schreibe nach: %s  (final: %s)", tmp_file, out_file)

    total_points  = 0
    total_enriched = 0
    skipped       = 0
    flushes       = 0
    pending: list[dict] = []
    sources_seen: set[str] = set()
    total_files   = len(md_files)
    completed_ok  = False

    try:
        for files_done, path in enumerate(md_files, start=1):
            result = parse_markdown_file(path)
            if result is None:
                skipped += 1
                continue

            meta, body = result
            chunks = chunk_document(meta, body, header_splitter, rec_splitter)

            # VOR dem Dedup-Filter vermerken, nicht danach. Die Sidecar-Liste
            # dient dazu, beim Delta-Lauf die Punkte einer Seite zu löschen,
            # bevor die neuen kommen. Eine Seite, deren Chunks diesmal alle
            # wortgleich anderswo stehen, bekommt keinen eigenen Punkt — ihr
            # ALTBESTAND in der Collection muss aber trotzdem weg. Stand sie
            # nicht in der Liste, blieben dort Punkte mit veraltetem Text
            # liegen, die weiterhin in Treffern auftauchen. Betrifft 298 der
            # 5.604 Seiten (gemessen 17.08.2026).
            if chunks:
                sources_seen.add(meta.get("source_url", ""))

            if keep_ids is not None:
                chunks = [c for c in chunks
                          if make_point_id(c["source_url"],
                                           c["chunk_index"]) in keep_ids]
            if not chunks:
                skipped += 1
                continue

            pending.extend(chunks)

            # Erst bei FLUSH_CHUNKS einbetten. Dateiweise wären das oft nur
            # 3–20 Chunks pro GPU-Aufruf gewesen.
            if len(pending) >= FLUSH_CHUNKS:
                written, enriched = embed_and_write(
                    pending, dense_embedder, sparse_embedder,
                    sparse_expanded_embedder, writer, task_kwargs,
                    query_task_kwargs, args.sparse_parallel, enrichment,
                    not args.no_question_vector)
                total_points += written
                total_enriched += enriched
                flushes += 1
                pct = 100.0 * files_done / total_files
                logger.info(
                    "  [%d/%d  %5.1f%%]  Row-Group %d geschrieben: %d Chunk(s)  (gesamt: %d)",
                    files_done, total_files, pct, flushes, written, total_points,
                )
                pending = []
                if flushes % GC_EVERY_N_FLUSHES == 0:
                    gc.collect()

        if pending:
            written, enriched = embed_and_write(
                pending, dense_embedder, sparse_embedder,
                sparse_expanded_embedder, writer, task_kwargs,
                query_task_kwargs, args.sparse_parallel, enrichment,
                not args.no_question_vector)
            total_points += written
            total_enriched += enriched
            flushes += 1
            logger.info("  [%d/%d 100.0%%]  Row-Group %d (Rest): %d Chunk(s)  (gesamt: %d)",
                        total_files, total_files, flushes, written, total_points)
            pending = []

        completed_ok = True

    finally:
        writer.close()
        if not completed_ok:
            logger.warning(
                "Lauf abgebrochen. %s enthält unvollständige Daten und wurde NICHT "
                "nach %s übernommen. Datei löschen, bevor sie mit einer gültigen "
                "Ausgabe verwechselt wird.", tmp_file, out_file,
            )

    if total_points == 0:
        logger.error("Kein verwendbarer Inhalt — abgebrochen. %s bleibt unangetastet.", out_file)
        tmp_file.unlink(missing_ok=True)
        sys.exit(1)

    # Erst jetzt die Zieldatei ersetzen — os.replace ist innerhalb eines
    # Dateisystems atomar, es gibt also keinen Moment mit halber Datei.
    os.replace(tmp_file, out_file)

    # Sidecar für den Importer: Vor dem Upsert müssen die Punkte dieser URLs
    # per Filter gelöscht werden, sonst bleiben Chunks aus einer früheren,
    # längeren Fassung der Seite als Waisen in Qdrant zurück.
    sources_file = out_file.with_name(out_file.stem + "_sources.txt")
    sources_file.write_text("\n".join(sorted(u for u in sources_seen if u)) + "\n",
                            encoding="utf-8")

    size_mb = out_file.stat().st_size / 1_048_576
    logger.info("─" * 60)
    logger.info("Ausgabe      : %s  (%.1f MB)", out_file, size_mb)
    logger.info("Quell-URLs   : %s  (%d)", sources_file.name, len(sources_seen))
    logger.info("Dateien      : %d  (übersprungen: %d)", total_files - skipped, skipped)
    logger.info("Chunks gesamt: %d  in %d Row-Group(s)", total_points, flushes)

    if enrichment is not None:
        quote = 100.0 * total_enriched / total_points
        logger.info("Mit Fragen   : %d  (%.1f%%)", total_enriched, quote)
        # Eine niedrige Quote bedeutet fast immer, dass die IDs nicht passen —
        # etwa weil die Anreicherung auf einem anderen Stand von var/korpus
        # lief. Das ist die einzige Stelle, an der so ein Fehler auffällt.
        if quote < 50.0:
            logger.warning(
                "Nur %.1f%% der Chunks haben Fragen bekommen. Erwartbar sind "
                "Verluste durch --min-chars und leere Modellantworten, aber "
                "nicht in dieser Größenordnung. Prüfe, ob Anreicherung und "
                "Vektorisierung denselben Bestand in var/korpus gesehen haben.",
                quote)
    logger.info("─" * 60)
    logger.info("Nächster Schritt: Datei auf lokalen Rechner übertragen")
    logger.info("  scp %s %s user@localhost:~/RAGHSH/var/vektoren/", out_file, sources_file)
    logger.info("Im local_importer.py sicherstellen:")
    logger.info("  • Sparse-Index mit modifier=models.Modifier.IDF anlegen (sonst falsches Ranking)")
    logger.info("  • Punkte je source_url vor dem Upsert löschen (Waisen aus früheren Crawls)")
    logger.info("  • Query-Seite mit '%s' einbetten, nicht mit '%s'", TASK_QUERY, TASK_PASSAGE)
    logger.info("  • Neues Payload-Feld 'organisation' übernehmen")
    logger.info("  • BM25 der Query-Seite mit language='%s' erzeugen — sonst "
                "stemmt die Anfrage anders als der Index", args.sparse_language)
    if enrichment is not None:
        logger.info("  • Payload-Feld 'questions' übernehmen")
        if sparse_expanded_embedder is not None:
            logger.info("  • Zweiten Sparse-Index 'sparse_expanded' anlegen, "
                        "ebenfalls mit modifier=IDF, und 'sparse_expanded_*' "
                        "dorthin laden")
        if not args.no_question_vector:
            logger.info("  • Zweiten benannten Dense-Vektor 'questions' anlegen "
                        "(%d Dim., Cosine) und 'vector_questions' dorthin laden — "
                        "leere Listen bedeuten: kein Fragenvektor, den Punkt dort "
                        "nicht eintragen", VECTOR_SIZE)
        logger.info("  • Retrieval als RRF-Fusion; welche Arme mitspielen, "
                    "entscheidet die Query — nichts davon ist einbetoniert")


if __name__ == "__main__":
    main()
