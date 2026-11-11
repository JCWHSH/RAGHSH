"""
Hybrid-Suche — Semantische und schlüsselwortbasierte Suche in der Wissensdatenbank.

Kurzbeschreibung
----------------
Durchsucht die Qdrant-Collection 'hsh_knowledge' mit einer kombinierten
Hybrid-Suche über vier Retrieval-Arme, die per Qdrant-nativer
Prefetch+RRF-Fusion zusammengeführt werden. Kann als eigenständige CLI oder
als importiertes Modul vom Chatbot verwendet werden.

Ausführliche Beschreibung
--------------------------
Eine einfache Vektorsuche findet semantisch ähnliche Texte, übersieht aber
exakte Schlüsselwörter (z.B. Modulnummern, Namen). Eine reine Keyword-Suche
findet exakte Begriffe, versteht aber keine Bedeutungen. Die Hybrid-Suche
kombiniert beide Verfahren und erreicht so höhere Treffergenauigkeit.

Sucharchitektur:

1. Vier Retrieval-Arme (Prefetch)
   Jeder Chunk liegt in vier Repräsentationen in Qdrant (siehe
   Parquet-Schema_und_Qdrant-Import.md):

     "text"            Dense über den Chunktext        (jina v3, passage)
     "questions"       Dense über generierte Fragen    (jina v3, query)
     "sparse"          BM25 über den Chunktext
     "sparse_expanded" BM25 über Chunktext + Fragen

   Die Nutzeranfrage wird einmal dense und einmal sparse eingebettet und gegen
   alle in ARMS aufgeführten Arme geprüft. "questions" ist der Arm, in dem
   Frageformulierung auf Frageformulierung trifft; nicht jeder Punkt hat ihn.
   Über ARMS lassen sich Arme für einen A/B-Vergleich abschalten, ohne die
   Collection neu zu bauen.

2. Qdrant-native Fusion (Prefetch + FusionQuery), Verfahren: DBSF
   Alle Arme laufen als Prefetch in einem einzigen query_points()-Aufruf,
   die Fusion erfolgt serverseitig — kein Python-Fusionscode.

   DBSF (Distribution-Based Score Fusion) normalisiert die Score-Verteilung
   jedes Arms auf eine gemeinsame Skala und addiert. Die Alternative RRF
   (Reciprocal Rank Fusion) verwirft die Scores und wertet nur Plätze; warum
   das hier schadet, steht bei FUSION weiter unten.

   Punkte ohne Fragenvektor werden vom "questions"-Arm schlicht nicht gefunden
   und bleiben trotzdem im Ergebnis — sie bekommen nur keinen Beitrag aus ihm.

4. Deduplizierung nach Quell-URL (max_per_url=2)
   Nach der Fusion: maximal MAX_PER_URL Chunks pro source_url werden behalten.
   Dies verhindert, dass viele Chunks desselben Dokuments alle Top-Plätze
   belegen, erlaubt aber einen zweiten relevanten Absatz pro Quelle.

5. Context Augmentation (Nachbar-Chunks, Top-N)
   Für die AUGMENT_TOP_N am höchsten gerankten Treffer werden die unmittelbar
   benachbarten Chunks (chunk_index - 1 und chunk_index + 1) derselben source_url
   nachgeladen. Dies verhindert, dass Informationen an Chunk-Grenzen zerrissen werden.
   Nachbar-Chunks werden dem Kontext vorangestellt/angehängt, aber nicht
   als eigenständige Top-Treffer gezählt.

Konfiguration:
   ARMS            — aktive Retrieval-Arme (Standard: alle vier)
   FUSION          — Fusionsverfahren: DBSF oder RRF (Standard: DBSF)
   TOP_K           — Anzahl der finalen Ergebnisse (Standard: 8)
   CANDIDATE_LIMIT — Kandidaten pro Sucharm (Standard: 100)
   DEDUP_BUFFER    — Faktor für Überabtastung vor URL-Dedup (Standard: 4)
   MAX_PER_URL     — Max. Chunks pro source_url nach Dedup (Standard: 2)
   AUGMENT_TOP_N   — Für wie viele Top-Treffer Nachbar-Chunks geladen werden (Standard: 3)

Voraussetzungen:
   - Qdrant läuft lokal, Collection 'hsh_knowledge' befüllt mit den vier Armen
     (Ausgabe von hpc_vectorizer.py + local_importer.py)

Abhängigkeiten: qdrant-client, fastembed
"""

import os
import re
import sys

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_URL      = os.environ.get("HSH_QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.environ.get("HSH_COLLECTION", "hsh_knowledge")   # Testkopie: HSH_COLLECTION setzen
DENSE_MODEL     = "jinaai/jina-embeddings-v3"
SPARSE_MODEL    = "Qdrant/bm25"
SPARSE_LANGUAGE = "german"   # muss zum Index passen — siehe load_sparse_embedder()

# URLs die in Suchergebnissen nicht angezeigt werden sollen
BLOCKED_URL_PREFIXES = (
    "https://serwiss.bib.",
    "http://serwiss.bib.",
)

# Aktive Retrieval-Arme. Der Index hält alle vier bereit, die Abfrage
# entscheidet, welche mitspielen — ein A/B-Vergleich braucht deshalb keinen
# zweiten Vektorisierungslauf, nur andere Werte hier.
#
#   Baseline              ("text",) / ("sparse",)
#   nur doc2query sparse  ("text",) / ("sparse_expanded",)      bis 22.09.2026
#   doc2query beidseitig  ("text", "questions") / ("sparse_expanded",)  ← eingestellt
#   alles                 ("text", "questions") / ("sparse", "sparse_expanded")
#
# Ein Sparse-Arm ist Absicht, nicht Sparsamkeit: Jeder Arm geht mit gleichem
# Gewicht in die Fusion ein. Laufen beide Sparse-Arme mit, stimmen sie fast
# immer gleich ab (dasselbe Verfahren auf fast demselben Text) und wiegen zu
# zweit schwerer als die Vektorseite. Gemessen an „Wie melde ich mich zur
# Prüfung an?": mit allen vier Armen gewannen ein Veranstaltungstechnik-
# Handbuch und eine Prüfungsordnung von 2009, während die passende
# Kurzanleitung nach unten rutschte. eval_gold 21.09.2026 bestätigt das:
# vier Arme unter DBSF 44/73, drei Arme 53/73.
#
# Der zweite Dense-Arm dagegen zahlt sich aus (eval_gold, 75 Fälle, Vertrag
# 529351a): "questions" findet g21 und g30 als einziger Arm und hebt die
# Kandidatenquote von 50 auf 53; zusammen mit TOP_DOCS 12 im Server auf 59.
# Im Volllauf mit GPT-5.6 Luna: Gold in den Quellen 50 → 53, Zeit und Token
# unverändert. Details: auswertung/gold/README.md, „Volllauf V0/V1/V2".
DENSE_ARMS  = ("text", "questions")
SPARSE_ARMS = ("sparse_expanded",)

# Fusionsverfahren. models.Fusion.RRF ist die Alternative.
#
# DBSF normalisiert die Score-Verteilung jedes Arms und addiert; RRF verwirft
# die Scores und wertet nur Plätze. Für diesen Index ist das der Unterschied
# zwischen brauchbar und zufällig, weil die Spitze beider Arme flach ist —
# gemessen über die besten zwanzig Treffer:
#
#     Sparse-Scores  13,43 … 12,01     Abstand Platz 1→2: 0,21  (1,6 %)
#     Dense-Scores    0,675 …  0,594   Abstand Platz 1→2: 0,012 (1,8 %)
#
# Die zwanzig besten Treffer sind praktisch gleichwertig. Qdrants RRF gewichtet
# aber mit 1/(1+Platz), also 0,500 / 0,333 / 0,250: Ein Rückfall von Platz 1 auf
# Platz 2 kostet ein Drittel der Stimme — für 1,6 % realen Unterschied. Damit
# entscheidet Rauschen. Unter RRF gewann bei obiger Frage die Prüfungsordnung
# von 2009, unter DBSF stehen auf den ersten sechs Plätzen ausschließlich
# Prüfungs-FAQ und Kurzanleitungen zur Prüfungsanmeldung.
#
# Preis: DBSF normalisiert über die zurückgelieferte Kandidatenmenge und hängt
# damit an CANDIDATE_LIMIT, während RRF davon unabhängig ist. Wer CANDIDATE_LIMIT
# ändert, verschiebt hier also auch das Ranking.
FUSION = models.Fusion.DBSF

TOP_K           = 8    # finale Ergebnisse nach URL-Dedup
CANDIDATE_LIMIT = 200  # Kandidaten pro Sucharm (Prefetch limit)
DEDUP_BUFFER    = 4    # TOP_K * DEDUP_BUFFER = Ergebnisse von Qdrant vor Dedup
MAX_PER_URL     = 2    # max. Chunks pro source_url nach Dedup
# Kein Cross-Encoder-Reranking: seit 21.09.2026 aus, seit S075 entfernt
# (Jina Reranker v2 auf dieser Hardware ≥ 17× zu langsam, Nutzen unbelegt;
# docs/wissen/cross-encoder-bewusst-inaktiv.md).
AUGMENT_TOP_N   = 3    # Nachbar-Chunks nur für die Top-N Treffer laden
GLEICHSTAND_RESERVE = 10  # so viele Treffer mehr holen, wie nach dem Tie-Break abgeschnitten werden


# ---------------------------------------------------------------------------
# Tie-Break bei Score-Gleichstand (S079)
# ---------------------------------------------------------------------------
# Qdrant liefert Treffer mit exakt gleichem Score in keiner festen Reihenfolge
# (Segmente werden parallel durchsucht); bei RRF sind Gleichstände häufig
# (gleiche Rangsumme). Dann flackerten Kandidatenlisten und Gold-Rang zwischen
# zwei Läufen auf demselben Code (S075/S076). Deshalb wird jede Trefferliste
# nachsortiert: Score absteigend, bei Gleichstand nach source_url, dann
# Punkt-ID. Abgeschnitten wird erst danach — wer `limit` Treffer braucht, holt
# `limit + GLEICHSTAND_RESERVE`, damit ein Gleichstand an der Schnittgrenze
# nicht davon abhängt, welche Punkte Qdrant zufällig zuerst geliefert hat.


def nach_score(treffer: list, limit: int | None = None) -> list:
    """Stabil nach (−Score, source_url, Punkt-ID) sortiert, dann auf `limit` gekürzt.

    Nimmt Qdrant-Punkte (ScoredPoint) und Ergebnis-Dicts ({"id", "score",
    "payload"}). Die Reihenfolge ungleicher Scores bleibt unberührt.
    """
    def schluessel(t) -> tuple:
        if isinstance(t, dict):
            score, payload, pid = t.get("score"), t.get("payload"), t.get("id")
        else:
            score, payload, pid = t.score, t.payload, t.id
        return (-float(score or 0.0), (payload or {}).get("source_url", ""), str(pid))

    return sorted(treffer, key=schluessel)[:limit]

# ---------------------------------------------------------------------------
# Suchkonfiguration je Aufruf
# ---------------------------------------------------------------------------
# Die Konstanten oben sind die Produktionswerte. Für Experimente (eval_gold)
# lassen sie sich je Aufruf überschreiben, ohne den Server neu zu starten und
# ohne dass Manifest und Wirklichkeit auseinanderlaufen können: `wirksame_
# konfiguration()` sagt, womit tatsächlich gesucht wurde.

ARME_DENSE  = ("text", "questions")
ARME_SPARSE = ("sparse", "sparse_expanded")
FUSIONEN    = {"dbsf": models.Fusion.DBSF, "rrf": models.Fusion.RRF}


def _fusion_name(f) -> str:
    return str(getattr(f, "value", f)).lower()


def wirksame_konfiguration(experiment: dict | None = None) -> dict:
    """Produktionswerte, überschrieben durch `experiment` — validiert.

    Erlaubte Schlüssel: dense_arms, sparse_arms, fusion, candidate_limit,
    dedup_buffer, max_per_url, collection. Unbekannte Schlüssel und unzulässige Werte sind
    ein Fehler, kein stilles Ignorieren — ein Experiment, das nicht tut, was
    sein Name sagt, ist schlimmer als keins.
    """
    k = {
        "DENSE_ARMS":      list(DENSE_ARMS),
        "SPARSE_ARMS":     list(SPARSE_ARMS),
        "FUSION":          _fusion_name(FUSION),
        "CANDIDATE_LIMIT": CANDIDATE_LIMIT,
        "DEDUP_BUFFER":    DEDUP_BUFFER,
        "MAX_PER_URL":     MAX_PER_URL,
        "COLLECTION_NAME": COLLECTION_NAME,
        "reranker_aktiv":  False,          # auswertung.py liest das Feld; seit S075 immer aus
    }
    if not experiment:
        return k
    erlaubt = {"dense_arms", "sparse_arms", "fusion", "candidate_limit",
               "dedup_buffer", "max_per_url", "collection"}
    fremd = set(experiment) - erlaubt
    if fremd:
        raise ValueError(f"unbekannte Experiment-Schlüssel: {sorted(fremd)}")
    if "dense_arms" in experiment:
        arme = list(experiment["dense_arms"])
        if any(a not in ARME_DENSE for a in arme):
            raise ValueError(f"dense_arms müssen aus {ARME_DENSE} sein: {arme}")
        k["DENSE_ARMS"] = arme
    if "sparse_arms" in experiment:
        arme = list(experiment["sparse_arms"])
        if any(a not in ARME_SPARSE for a in arme):
            raise ValueError(f"sparse_arms müssen aus {ARME_SPARSE} sein: {arme}")
        k["SPARSE_ARMS"] = arme
    if not k["DENSE_ARMS"] and not k["SPARSE_ARMS"]:
        raise ValueError("mindestens ein Arm muss aktiv sein")
    if "fusion" in experiment:
        f = str(experiment["fusion"]).lower()
        if f not in FUSIONEN:
            raise ValueError(f"fusion muss eines von {sorted(FUSIONEN)} sein: {f}")
        k["FUSION"] = f
    if "collection" in experiment:
        # Eine Testkopie der Collection (auswertung/testindex.py) — dieselben
        # Modelle, derselbe Server, nur ein anderer Index. Existenz prüft Qdrant
        # beim ersten Zugriff; ein Tippfehler ist dort ein Fehler, kein leerer
        # Treffer.
        name = str(experiment["collection"])
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError(f"collection: unzulässiger Name {name!r}")
        k["COLLECTION_NAME"] = name
    for name, obergrenze in (("candidate_limit", 1000), ("dedup_buffer", 20), ("max_per_url", 10)):
        if name in experiment:
            wert = int(experiment[name])
            if not 1 <= wert <= obergrenze:
                raise ValueError(f"{name} muss zwischen 1 und {obergrenze} liegen: {wert}")
            k[name.upper()] = wert
    return k


def suche_je_arm(
    client: QdrantClient,
    dense_embedder: TextEmbedding,
    sparse_embedder: SparseTextEmbedding,
    query: str,
    limit: int = 50,
    query_filter: models.Filter | None = None,
    collection: str = COLLECTION_NAME,
) -> dict[str, list[dict]]:
    """Jeder Arm einzeln, ohne Fusion — für die Diagnose, welcher Arm das Gold trägt.

    Liefert je Arm die Top-`limit` Treffer als (rang, source_url, score). Vier
    zusätzliche Qdrant-Anfragen, zusammen unter einer Sekunde; nur die
    Evaluation ruft das auf.
    """
    dense_vec  = embed_query_dense(dense_embedder, query)
    sparse_vec = embed_query_sparse(sparse_embedder, query)
    ergebnis: dict[str, list[dict]] = {}
    for arm, vec in [(a, dense_vec) for a in ARME_DENSE] + [(a, sparse_vec) for a in ARME_SPARSE]:
        punkte = nach_score(client.query_points(
            collection_name=collection, query=vec, using=arm,
            limit=limit + GLEICHSTAND_RESERVE, query_filter=query_filter,
            with_payload=["source_url"],
        ).points, limit)
        ergebnis[arm] = [
            {"rang": i, "source_url": (h.payload or {}).get("source_url", ""),
             "score": round(float(h.score), 5)}
            for i, h in enumerate(punkte, start=1)
        ]
    return ergebnis


# ---------------------------------------------------------------------------
# Embedding-Hilfsfunktionen
# ---------------------------------------------------------------------------


def load_sparse_embedder(cache_dir: str | None = None) -> SparseTextEmbedding:
    """BM25-Embedder mit derselben Sprache, mit der der Index gebaut wurde.

    language="german" ist zwingend, nicht kosmetisch: Stemmer und Stoppwortliste
    bestimmen die Token-IDs. Mit der Vorgabe "english" bleiben deutsche
    Stoppwörter stehen und die Stämme weichen ab — dieselbe Frage liefert dann
    andere IDs als die im Index gespeicherten, und die Treffer bleiben aus oder
    sind zufällig.
    """
    return SparseTextEmbedding(SPARSE_MODEL, cache_dir=cache_dir, language=SPARSE_LANGUAGE)


def embed_query_dense(embedder: TextEmbedding, query: str) -> list[float]:
    """Dense-Embedding für eine Suchanfrage — mit dem QUERY-Adapter.

    jina-v3 nutzt aufgabenspezifische LoRA-Adapter. Der Index wurde mit
    retrieval.passage gebaut, die Anfrage gehört nach retrieval.query.

    Dafür ist query_embed() nötig. Ein `embed(..., task="retrieval.query")`
    sieht richtig aus, wird von fastembed 0.8 aber wortlos in **kwargs
    verschluckt: JinaEmbeddingV3.embed() kennt nur `task_id` (int) und fällt
    sonst auf den Passage-Adapter zurück. Gemessen liegen die beiden Vektoren
    bei cos = 0.84 auseinander — das kostet vor allem den "questions"-Arm,
    der genau auf Query-Geometrie gebaut ist.
    """
    return list(embedder.query_embed(query))[0].tolist()


def embed_query_sparse(embedder: SparseTextEmbedding, query: str) -> models.SparseVector:
    """BM25-Sparse-Embedding für eine Suchanfrage.

    query_embed(), nicht embed(): embed() liefert Dokumentgewichte (~1.67), die
    mit dem IDF der Collection multipliziert die Scores verzerren. Die Anfrage
    braucht Gewichte von 1.0.
    """
    result = list(embedder.query_embed([query]))[0]
    return models.SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------


def _fetch_neighbor_chunk(
    client: QdrantClient,
    source_url: str,
    chunk_index: int,
    richtung: int,
    collection: str = COLLECTION_NAME,
) -> dict | None:
    """Lädt den nächsten tatsächlich vorhandenen Nachbarn eines Chunks.

    `richtung` ist -1 für davor und +1 für danach.

    Nicht `chunk_index ± 1` abfragen: Die Deduplizierung hat 29,4 % der
    Dokumente Lücken in die Chunkfolge gerissen — auf Position 9 kann direkt 14
    folgen. Eine Punktabfrage auf ±1 liefert dann nichts, und die Context
    Augmentation läuft still leer, statt den vorhandenen Nachbarn zu holen.

    Deshalb ein Bereichsfilter plus Sortierung: der größte Index unterhalb bzw.
    der kleinste oberhalb. Das setzt voraus, dass der Payload-Index auf
    `chunk_index` mit `range=True` angelegt ist (siehe local_importer).
    """
    if richtung < 0:
        bereich = models.Range(lt=chunk_index)
        ordnung = models.OrderBy(key="chunk_index", direction=models.Direction.DESC)
    else:
        bereich = models.Range(gt=chunk_index)
        ordnung = models.OrderBy(key="chunk_index", direction=models.Direction.ASC)

    hits, _ = client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_url",
                    match=models.MatchValue(value=source_url),
                ),
                models.FieldCondition(key="chunk_index", range=bereich),
            ]
        ),
        order_by=ordnung,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    return hits[0].payload if hits else None


def perform_hybrid_search(
    client: QdrantClient,
    dense_embedder: TextEmbedding,
    sparse_embedder: SparseTextEmbedding,
    query: str,
    top_k: int = TOP_K,
    query_filter: models.Filter | None = None,
    experiment: dict | None = None,
) -> list[dict]:
    """Hybrid-Suche via Qdrant-nativem Prefetch + Score-Fusion (FUSION).

    `experiment` überschreibt Arme, Fusion und Limits je Aufruf (siehe
    wirksame_konfiguration); ohne Angabe gelten die Produktionskonstanten.

    Alle aktiven Sucharme laufen in einem einzigen Qdrant-Aufruf.
    Die Fusion erfolgt serverseitig — kein Python-Fusionscode nötig.
    Nach der Fusion:
      1. URL-Dedup: max. MAX_PER_URL Chunks pro source_url.
      2. Context Augmentation: für die AUGMENT_TOP_N besten Treffer werden
         Nachbar-Chunks (chunk_index ± 1) nachgeladen und dem Payload angehängt.
    """
    konf = wirksame_konfiguration(experiment)
    dense_arms, sparse_arms = konf["DENSE_ARMS"], konf["SPARSE_ARMS"]
    fusion = FUSIONEN[konf["FUSION"]]
    candidate_limit, dedup_buffer, max_per_url = (
        konf["CANDIDATE_LIMIT"], konf["DEDUP_BUFFER"], konf["MAX_PER_URL"])
    collection = konf["COLLECTION_NAME"]

    dense_vec  = embed_query_dense(dense_embedder, query)  if dense_arms  else None
    sparse_vec = embed_query_sparse(sparse_embedder, query) if sparse_arms else None

    # ── Qdrant-native Prefetch + Fusion ───────────────────────────────────
    # Ein Prefetch je aktivem Arm. Derselbe Anfragevektor geht an beide
    # Dense- bzw. beide Sparse-Arme — sie unterscheiden sich in dem, was
    # indexseitig darin steckt, nicht in der Anfrage.
    prefetch = [
        models.Prefetch(
            query=dense_vec,
            using=arm,
            limit=candidate_limit,
            filter=query_filter,
        )
        for arm in dense_arms
    ] + [
        models.Prefetch(
            query=sparse_vec,
            using=arm,
            limit=candidate_limit,
            filter=query_filter,
        )
        for arm in sparse_arms
    ]

    if len(prefetch) == 1:
        # Ein einzelner Arm braucht keine Fusion — Qdrant verlangt für
        # FusionQuery mindestens zwei Prefetches. Direkt abfragen.
        einzel = prefetch[0]
        raw_results = client.query_points(
            collection_name=collection,
            query=einzel.query, using=einzel.using, query_filter=query_filter,
            limit=top_k * dedup_buffer + GLEICHSTAND_RESERVE,
            with_payload=True,
        ).points
    else:
        raw_results = client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=fusion),
            limit=top_k * dedup_buffer + GLEICHSTAND_RESERVE,
            with_payload=True,
        ).points
    # Gleichstände fest ordnen, erst dann auf die bisherige Länge kürzen
    raw_results = nach_score(raw_results, top_k * dedup_buffer)

    # ── Geblockte URLs herausfiltern ───────────────────────────────────────
    raw_results = [
        h for h in raw_results
        if not (h.payload or {}).get("source_url", "").startswith(BLOCKED_URL_PREFIXES)
    ]

    # ── URL-Dedup: max. MAX_PER_URL Chunks pro source_url ─────────────────
    url_counts: dict[str, int] = {}
    results: list[dict] = []

    for hit in raw_results:
        source_url = (hit.payload or {}).get("source_url", str(hit.id))
        count = url_counts.get(source_url, 0)
        if count < max_per_url:
            url_counts[source_url] = count + 1
            results.append({
                "id":      hit.id,
                "score":   hit.score,
                "payload": hit.payload,
            })
        if len(results) == top_k:
            break

    # ── Context Augmentation: Nachbar-Chunks für Top-N nachladen ──────────
    for i, result in enumerate(results):
        if i >= AUGMENT_TOP_N:
            break

        payload     = result["payload"] or {}
        source_url  = payload.get("source_url", "")
        chunk_index = payload.get("chunk_index")

        if source_url and chunk_index is not None:
            prev_text = ""
            next_text = ""

            # Keine Grenzprüfung gegen total_chunks mehr: Der Wert zählt die
            # Chunks des Dokuments VOR der Deduplizierung und liegt bei 1.559
            # Dokumenten über der Zahl der tatsächlich vorhandenen. Die
            # Bereichsabfrage liefert am Rand von selbst nichts.
            prev = _fetch_neighbor_chunk(client, source_url, chunk_index, -1, collection)
            if prev:
                prev_text = prev.get("text", "")

            nxt = _fetch_neighbor_chunk(client, source_url, chunk_index, +1, collection)
            if nxt:
                next_text = nxt.get("text", "")

            # Erweiterten Text im Payload hinterlegen
            core_text = payload.get("text", "")
            parts = []
            if prev_text:
                parts.append(prev_text)
            parts.append(core_text)
            if next_text:
                parts.append(next_text)
            result["payload"] = {**payload, "text": "\n\n".join(parts)}

    return results


# ---------------------------------------------------------------------------
# RAG context builder
# ---------------------------------------------------------------------------


def build_rag_context(results: list[dict]) -> str:
    """Wandelt Suchergebnisse in einen formatierten Kontext-String für das LLM um."""
    blocks = []
    for i, r in enumerate(results, 1):
        p = r["payload"]
        lines = [
            f"[Quelle {i}] {p.get('title', '(kein Titel)')}",
            f"URL: {p.get('source_url', '—')}",
            f"Stand: {p.get('crawl_date', 'unbekannt')}",
        ]
        if p.get("faculty"):
            lines.append(f"Fakultät: {p['faculty']}")
        if p.get("organisation"):
            lines.append(f"Organisation: {p['organisation']}")
        if p.get("section_heading"):
            lines.append(f"Abschnitt: {p['section_heading']}")
        lines.append("")
        lines.append(p.get("text", ""))
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[dict]) -> None:
    """Pretty-print ranked results to the terminal."""
    if not results:
        print("  Keine Treffer gefunden.")
        return

    for i, r in enumerate(results, 1):
        p       = r["payload"]
        score   = r["score"]
        url     = p.get("source_url", "—")
        faculty = p.get("faculty", "")
        heading = p.get("section_heading", "")
        date    = p.get("crawl_date", "")
        preview = p.get("text", "")[:200].replace("\n", " ")

        print(f"\n  ┌─ Treffer {i}  ({FUSION.upper()}-Score: {score:.5f})")
        print(f"  │  URL      : {url}")
        if faculty:
            print(f"  │  Fakultät : {faculty}")
        if heading:
            print(f"  │  Abschnitt: {heading}")
        if date:
            print(f"  │  Stand    : {date}")
        print(f"  │  Vorschau : {preview}…")
        print(f"  └{'─' * 62}")


# ---------------------------------------------------------------------------
# Main / interactive CLI
# ---------------------------------------------------------------------------


def main() -> None:

    # ── Connect ───────────────────────────────────────────────────────────
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        print(f"Fehler: Qdrant nicht erreichbar — {exc}")
        print("  → docker compose up -d")
        sys.exit(1)

    # ── Embedding-Modelle laden ───────────────────────────────────────────
    print(f"\nLade Dense-Modell '{DENSE_MODEL}'…")
    dense_embedder = TextEmbedding(model_name=DENSE_MODEL)

    print(f"Lade Sparse-Modell '{SPARSE_MODEL}' (language={SPARSE_LANGUAGE})…")
    sparse_embedder = load_sparse_embedder()

    print("Bereit.  Tippe eine Frage, Enter zum Suchen, Strg+C zum Beenden.\n")

    # ── Interactive loop ──────────────────────────────────────────────────
    while True:
        try:
            query = input("Frage> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAuf Wiedersehen.")
            break

        if not query:
            continue

        results = perform_hybrid_search(
            client, dense_embedder, sparse_embedder, query
        )
        print_results(results)


if __name__ == "__main__":
    main()
