"""
HsH RAG MCP Server

FastMCP HTTP Server (Streamable HTTP Transport) der die zweistufige
RAG-Pipeline als MCP-Tool exponiert.

Endpunkt: http://127.0.0.1:8765/mcp (Produktion, über nginx erreichbar)
Andere Ports (HSH_MCP_PORT, z. B. 8767 der Agenten) gelten als Test-/Agenteninstanz:
sie halten die GWDG-Reserve der Produktion ein (_gwdg_reserve_pruefen).

Ressourcen:
  hsh://models   — verfügbare GWDG-Modelle (5-Minuten-Cache)

Tools:
  frag_hsh(frage, modell, fakultaet?)
      Zweistufige RAG-Pipeline: Hybrid-Suche → Dokumentauswahl (LLM 1)
      → Volltext laden → Antwort generieren (LLM 2)
      Progress Notifications für jeden Pipeline-Schritt.
"""

import asyncio
import atexit
import gc
import json
import os
import re
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ONNX Runtime Arena auf "genau so viel wie angefordert" begrenzen —
# verhindert, dass der BFC-Allocator unbegrenzt Speicher akkumuliert.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("ONNXRUNTIME_ARENA_EXTEND_STRATEGY", "kSameAsRequested")

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from mcp.server.fastmcp import FastMCP, Context
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, BadRequestError, AuthenticationError
from qdrant_client import QdrantClient, models

from raghsh import konfig
from raghsh.suche.hybrid_search import (load_sparse_embedder, perform_hybrid_search,
                                        suche_je_arm, wirksame_konfiguration)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

load_dotenv(konfig.ENV_DATEI)

GWDG_API_KEY  = os.getenv("GWDG_API_KEY", "")
GWDG_API_BASE = os.getenv("GWDG_API_BASE", "https://chat-ai.academiccloud.de/v1")

# OpenRouter — kostenpflichtige Premium-Modelle (Claude, Gemini, ChatGPT).
# Der API-Key liegt in einer separaten Datei im Verzeichnis (nicht in git).
# Der Zugriff ist passwortgeschützt, da jede Anfrage echtes Geld kostet:
# frag_hsh akzeptiert Premium-Modelle nur mit korrektem Passwort.
OPENROUTER_KEY_FILE  = konfig.OPENROUTER_KEY_DATEI
OPENROUTER_API_BASE  = "https://openrouter.ai/api/v1"
# Das Passwort steht in .env (nicht in git). Fehlt es, bleibt FULL_ACCESS_PASSWORD
# leer und Premium-Modelle sind gesperrt — ein leeres Passwort gilt nie als korrekt.
FULL_ACCESS_PASSWORD = os.getenv("FULL_ACCESS_PASSWORD", "")
# Agenten-Instanz (nur 127.0.0.1, Messläufe): Premium-Modelle ohne Passwort,
# gesetzt von mcp_dev.sh. In der Produktion ist die Variable nie gesetzt.
PREMIUM_OHNE_PASSWORT = os.getenv("HSH_PREMIUM_OHNE_PASSWORT") == "1"

# Je ein Modell pro Anbieter — mittlerer Preis, relativ schnell.
# Die IDs sind die echten OpenRouter-Modell-IDs (frei anpassbar).
# Anzeigenamen werden in der Weboberfläche über format_func gesetzt.
OPENROUTER_MODELS: dict[str, str] = {
    "anthropic/claude-sonnet-4.5": "Claude Sonnet (Anthropic)",
    "google/gemini-2.5-flash":     "Gemini Flash (Google)",
    "openai/gpt-5.6-luna":         "ChatGPT (OpenAI GPT-5.6 Luna)",
    "openai/gpt-6-luna":           "ChatGPT (OpenAI GPT-6 Luna)",
    "z-ai/glm-5.3-flash":          "GLM-5.3 Flash (DeepInfra, fp4)",
}

# Feste Anbieterwahl je Modell (OpenRouter `provider`). Ohne Vorgabe routet
# OpenRouter je Anfrage zu einem beliebigen Anbieter — andere Quantisierung,
# anderer Preis, nicht vergleichbare Messungen (Christian, 01.10.2026).
OPENROUTER_ANBIETER: dict[str, dict] = {
    "z-ai/glm-5.3-flash": {"order": ["deepinfra/fp4"], "allow_fallbacks": False},
}

# Wartezeiten bei 429 von OpenRouter (Sekunden). Anbieter wie DeepInfra drosseln
# bei Lastspitzen für einige Sekunden bis ~30 s; die SDK-eigenen Wiederholungen
# (2 × < 1 s) überbrücken das nicht (01.10.2026: 4 Ausfälle in 54 Anfragen).
# DeepInfra drosselt zeitweise minutenlang (01.10. 17:30–17:45, S041) → zusätzlich
# 60 s und 120 s, zusammen knapp 4 min (Christian, 01.10.2026).
OPENROUTER_429_WARTEN = (5, 15, 30, 60, 120)


# ── GWDG-Kontingent (Christian, 04.10.2026) ──────────────────────────────────
# Produktion und Agenten teilen einen GWDG-Schlüssel; die Limits zählen Anfragen
# (30/min, 200/h, 1000/Tag, 3000/Monat). Jede GWDG-Antwort trägt die Restwerte in
# x-ratelimit-remaining-*; sie landen in einer gemeinsamen Datei, die Web-App,
# Dashboard und `gwdg-kontingent` lesen.
GWDG_STAND = Path(os.getenv("HSH_GWDG_STAND", "/var/lib/raghsh/gwdg_kontingent.json"))
_GWDG_FENSTER = {"minute": "minute", "hour": "stunde", "day": "tag", "month": "monat"}


def _gwdg_stand_merken(headers, quelle: str) -> None:
    try:
        rest = {de: int(headers[f"x-ratelimit-remaining-{en}"])
                for en, de in _GWDG_FENSTER.items() if headers.get(f"x-ratelimit-remaining-{en}")}
        if not rest:
            return
        limit = {de: int(headers[f"x-ratelimit-limit-{en}"])
                 for en, de in _GWDG_FENSTER.items() if headers.get(f"x-ratelimit-limit-{en}")}
        daten = {"zeit": datetime.now().isoformat(timespec="seconds"), "ts": time.time(),
                 "quelle": quelle, "rest": rest, "limit": limit}
        tmp = GWDG_STAND.with_name(f".{GWDG_STAND.name}.{os.getpid()}")
        tmp.write_text(json.dumps(daten, ensure_ascii=False), encoding="utf-8")
        os.chmod(tmp, 0o664)
        os.replace(tmp, GWDG_STAND)
    except Exception:                                   # Anzeige darf nie eine Antwort kosten
        pass


def gwdg_stand() -> dict:
    try:
        return json.loads(GWDG_STAND.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _gwdg_reserve_pruefen() -> None:
    """Agentenversion: unterhalb der Reserve keine GWDG-Anfragen mehr — der Rest gehört
    der Produktion. Reserve in /opt/raghsh-agent/grenzen.json (nicht ändern, nicht umgehen;
    REGELN.md). Die Produktion selbst (PRODUKTION) prüft nicht — für sie ist die Reserve da."""
    if PRODUKTION:
        return
    d = gwdg_stand()
    if not d.get("rest"):
        return
    try:
        g = json.loads(Path("/opt/raghsh-agent/grenzen.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        g = {}
    stand, jetzt = datetime.fromtimestamp(d["ts"]), datetime.now()
    knapp = []
    if stand.date() == jetzt.date() and d["rest"].get("tag", 10**9) < g.get("gwdg_reserve_tag", 400):
        knapp.append(f"Tag {d['rest']['tag']}")
    if (stand.year, stand.month) == (jetzt.year, jetzt.month) and \
            d["rest"].get("monat", 10**9) < g.get("gwdg_reserve_monat", 400):
        knapp.append(f"Monat {d['rest']['monat']}")
    if knapp:
        raise RuntimeError("GWDG-Kontingent: Reserve für die Produktion erreicht (noch "
                           + ", ".join(knapp) + " Anfragen) — die Agentenversion stellt keine "
                           "GWDG-Anfragen mehr.")

QDRANT_URL      = os.environ.get("HSH_QDRANT_URL", "http://localhost:6333")
# Über HSH_COLLECTION / HSH_MCP_PORT lässt sich eine zweite Instanz gegen eine
# Testkopie der Collection starten (eval_gold, Chunking-Experimente 22.09.2026),
# ohne die Produktion anzufassen. Ohne die Variablen: wie immer.
COLLECTION_NAME = os.environ.get("HSH_COLLECTION", "hsh_knowledge")
MCP_PORT        = int(os.environ.get("HSH_MCP_PORT", "8765"))
# Produktion = Standardport. Nur sie darf das GWDG-Kontingent bis zum Ende nutzen;
# jede andere Instanz (Agenten, Tests) stoppt an der Reserve (Christian, 11.10.2026).
PRODUKTION      = MCP_PORT == 8765
EMBED_MODEL     = "jinaai/jina-embeddings-v3"
SPARSE_MODEL    = "Qdrant/bm25"
DB_PATH         = konfig.DOC_INDEX
LOG_PATH        = konfig.QA_LOG
FASTEMBED_CACHE = Path.home() / ".cache" / "fastembed"
READY_FILE      = Path("/tmp/hsh-mcp.ready" if MCP_PORT == 8765 else f"/tmp/hsh-mcp-{MCP_PORT}.ready")

RAG_CANDIDATE_K        = 30
TOP_DOCS               = 12   # war 8; 12 bringt +5 Goldkandidaten bei gleicher Zeit
                              # (eval_gold 21.09.2026, Phase 3), bei Fakultätsfrage ×2
SELECT_N               = 4
MAX_CONTEXT_TOKENS     = 100_000
PROMPT_OVERHEAD_TOKENS = 1_000
TEMPERATURE            = 0.0
# Christian, 04.10.2026: 200 s statt 60 s, 1 statt 3 Wiederholungen. GLM (Standard seit
# 03.10.) denkt vor der Antwort und braucht bei großem Kontext 60–120 s; mit 60 s wurde
# jeder solche Aufruf abgebrochen und neu gestellt (04.10.: ~100 Wiederholungen je
# Sitzung/Messlauf, jede zählt gegen das GWDG-Kontingent; pk07 scheiterte 4 × 60 s).
GWDG_TIMEOUT           = 200.0  # Sekunden pro Versuch — siehe Kommentar in _lifespan
GWDG_MAX_RETRIES       = 1
MODEL_CACHE_TTL        = 300   # Sekunden
MAX_LOG_AGE_DAYS       = 365   # 12 Monate Aufbewahrungsfrist (DSGVO)

FACULTIES = [
    ("f1", "Fakultät 1 — Elektro- und Informationstechnik"),
    ("f2", "Fakultät 2 — Maschinenbau und Bioverfahrenstechnik"),
    ("f3", "Fakultät 3 — Medien, Information und Design"),
    ("f4", "Fakultät 4 — Wirtschaft und Informatik"),
    ("f5", "Fakultät 5 — Diakonie, Gesundheit und Soziales"),
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SELECTION_SYSTEM = f"""\
Du bist ein Dokumentenauswahl-Assistent der Hochschule Hannover.

Aufgabe: Analysiere die folgende Dokumentliste und wähle die bis zu {SELECT_N} \
Dokumente aus, die die Antwort auf die Nutzerfrage am wahrscheinlichsten enthalten.
Sortiere die Auswahl nach absteigender Relevanz — das wichtigste Dokument steht an erster Stelle.

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt, kein weiterer Text:
{{"selected": [<Nummer>, ...]}}
"""

ANSWER_SYSTEM = """\
Du bist der offizielle Assistent der Hochschule Hannover (HsH).

Zitierregeln — ZWINGEND EINZUHALTEN:
- Du MUSST jede sachliche Aussage mit einem wörtlichen Zitat aus dem Kontext belegen.
- Zitate EXAKT wie im Original übernehmen — keine Umformulierungen, keine Kürzungen.
- Anführungszeichen: ausschließlich „..." verwenden (öffnend: „  schließend: "),
  direkt gefolgt von der Dokumentnummer: „Zitat" [N]
- [N] ist die Nummer aus dem Abschnittskopf [Dokument N] im bereitgestellten Kontext.
- Die Kopfzeilen unter [Dokument N] (Titel, URL, Fakultät, Stand, Prüfungsordnung) sind keine Quelle zum Zitieren.
- Zitate kurz halten — ein Satz oder ein prägnanter Teilsatz genügt.
- Paraphrasen NUR zur Einleitung oder Verknüpfung zwischen Zitaten, nie als Ersatz.

Beispiel für eine korrekte Antwort:
  Die Regelstudienzeit beträgt „sechs Semester" [1]. Für die Zulassung gilt:
  „Bewerberinnen und Bewerber müssen die Hochschulzugangsberechtigung nachweisen" [1].
  Ausnahmen regelt „§ 3 Abs. 2 der Zulassungsordnung" [2].

Weitere Regeln:
- Antworte AUSSCHLIESSLICH auf Basis des bereitgestellten Kontextes.
- Erfinde keine Informationen und spekuliere nicht.
- Falls die Antwort im Kontext nicht enthalten ist, antworte wörtlich:
  „Dazu liegen mir keine Informationen aus den offiziellen Dokumenten der HsH vor."
- Schreibe klar, präzise und auf Deutsch. Die verschiedenen Fakultäten der
  Hochschule Hannover haben unterschiedliche Regelungen — formuliere so spezifisch
  wie möglich. Wenn die Fakultät nicht eindeutig hervorgeht, weise darauf hin.

Länge und Aufbau:
- Sprich die fragende Person immer mit „Sie" an, nie mit „du".
- Beantworte die Frage gleich im ersten Satz direkt.
- Halte dich kurz: Meist genügen zwei bis vier Sätze. Weitere Einzelheiten (Ausnahmen, Fristen, Hintergründe) nur ergänzen, wenn sie für die fragende Person die Antwort ändern.
- Verwende Begriffe, Bezeichnungen und Zahlen so, wie sie in der Quelle stehen — nicht umformulieren.
- Erwähne keine Regelungen, die erkennbar nicht auf die Frage zutreffen (z. B. andere Studiengänge oder andere Fassungen, wenn nach einer bestimmten Fassung gefragt ist), und keine Vermutungen darüber, was ein Formular oder Dokument andeuten könnte.
- Ausnahme: Unterscheiden sich mehrere geltende Fassungen einer Prüfungsordnung im gefragten Punkt und nennt die Frage keine Fassung, nenne den Wert je Fassung.

Abweichende Angaben in den Quellen — vor dem Schreiben einordnen:
- Neuere Regelung: Regeln zwei Ordnungen oder Richtlinien dasselbe Verfahren verschieden, gilt die jüngere (Jahr im Titel, Stand, Kopfzeile „nicht mehr gültig seit …“). Beginne die Antwort mit der geltenden Regelung; die ältere nur danach und nur als „Früher galt … [N]; seit … gilt … [M].“ Fragt die Person ausdrücklich nach einer Fassung, deren Kopfzeile „nicht mehr gültig“ zeigt, antworte aus dieser Fassung und schließe mit einem Satz ohne Datum, dass diese Fassung nicht mehr gilt.
- Anderer Gegenstand: Betreffen die Angaben verschiedene noch geltende Fassungen (auch „gilt übergangsweise bis …“), Studiengänge, Abschlüsse oder Personengruppen, ordne — sofern die Frage nicht genau einen davon meint — jeden Wert seinem Gegenstand zu („PO 2018: … [N], PO 2021: … [M]“). Das ist eine Unterscheidung, kein Gegensatz der Quellen.
- Nur wenn gleichrangige, gleich aktuelle Quellen zum selben Punkt verschiedene Werte nennen, nenne beide Werte, jeden mit seiner Marke.
- Ergänzen sich die Quellen oder sagt eine nur mehr als die andere, erwähne das nicht. Schreibe nie, dass die Quellen übereinstimmen oder einander nicht entgegenstehen.
"""

# Prompt-Versuche ohne Codeänderung: Datei, deren Text an ANSWER_SYSTEM angehängt
# wird (Christian, 01.10.2026: Längentest). Ohne Variable keine Änderung.
# Die Widerspruch-Taxonomie (S046, zusatz_v4.txt) steht seit S058 fest oben im
# Prompt — die Variable wirkt zusätzlich, zusatz_v4.txt also nicht mehr darüber laden.
if os.getenv("HSH_ANSWER_ZUSATZ_DATEI"):
    ANSWER_SYSTEM += "\n" + Path(os.environ["HSH_ANSWER_ZUSATZ_DATEI"]).read_text(encoding="utf-8").strip() + "\n"

# ---------------------------------------------------------------------------
# Globale Ressourcen (warm im RAM nach dem Start)
# ---------------------------------------------------------------------------

_embedder: TextEmbedding | None = None
_sparse_embedder: SparseTextEmbedding | None = None
_qdrant: QdrantClient | None = None
_index_conn: sqlite3.Connection | None = None
_gwdg_client: OpenAI | None = None
_openrouter_client: OpenAI | None = None

_model_cache: list[str] = []
_model_cache_ts: float = 0.0

# Die Lifespan wird von FastMCPs Streamable-HTTP-Transport NICHT einmal je
# Prozess durchlaufen, sondern bei jeder Client-Session erneut betreten — und
# beim Trennen oft nicht sauber verlassen. Gemessen am 14.08.2026: neun
# Eintritte, zwei Austritte, und der Speicher des Dienstes wuchs binnen vierzehn
# Minuten von 136 MB auf 3,7 GB, weil jeder Eintritt einen weiteren Dense- und
# Sparse-Embedder anlegte, während die hängengebliebenen Kontextmanager die
# alten am Leben hielten. Auf 7,7 GB Maschinenspeicher endet das im OOM.
#
# Deshalb sind die Ressourcen prozessweit, nicht sessionweit: _ressourcen_laden()
# läuft genau einmal, jeder weitere Lifespan-Eintritt benutzt das Vorhandene
# weiter. Der Abbau hängt entsprechend am Prozessende (atexit), nicht am
# Lifespan-Austritt — sonst schlösse die erste endende Session die SQLite-
# Verbindung, die parallele Sessions noch benutzen.
_ressourcen_bereit = False
_ressourcen_lock = asyncio.Lock()
_cleanup_task: asyncio.Task | None = None

# ---------------------------------------------------------------------------
# Lifespan — Ressourcen prozessweit laden
# ---------------------------------------------------------------------------


def _ressourcen_freigeben() -> None:
    """Abbau am Prozessende. Über atexit registriert, siehe Kommentar oben."""
    READY_FILE.unlink(missing_ok=True)
    if _index_conn is not None:
        try:
            _index_conn.close()
        except Exception:
            pass


@asynccontextmanager
async def _lifespan(server: FastMCP):
    """Stellt sicher, dass die Ressourcen da sind — beim ersten Eintritt.

    Jeder weitere Eintritt ist ein reiner Durchlauf. Der Lock verhindert, dass
    zwei gleichzeitig eintreffende Sessions das Laden doppelt anstoßen.
    """
    async with _ressourcen_lock:
        if not _ressourcen_bereit:
            await _ressourcen_laden()
        else:
            print("Lifespan erneut betreten — vorhandene Ressourcen weiterverwendet.")
    yield
    # Kein Abbau: die Ressourcen überleben die Session bewusst.
    print("Client-Session beendet — Ressourcen bleiben geladen.")


async def _ressourcen_laden() -> None:
    """Legt die geteilten Ressourcen an. Läuft genau einmal je Prozess."""
    global _embedder, _sparse_embedder, _qdrant, _index_conn, _gwdg_client, _openrouter_client
    global _ressourcen_bereit, _cleanup_task

    print("HsH RAG MCP Server — Lade Ressourcen…")

    if not GWDG_API_KEY:
        raise RuntimeError("GWDG_API_KEY fehlt in .env")
    # Timeout und Retries explizit setzen: Die SDK-Defaults (600 s, 2 Retries)
    # lassen einen Aufruf bis zu 30 Minuten hängen. GWDG blockiert Anfragen
    # sporadisch minutenlang, und zwar bimodal — gemessen entweder ~0,3 s oder
    # Stillstand jenseits von 400 s, unabhängig von Kontextgröße und
    # Nebenläufigkeit. Bei dieser Verteilung ist früh abbrechen und neu fragen
    # deutlich billiger als warten: der Neuversuch trifft meist den schnellen Fall.
    _gwdg_client = OpenAI(
        api_key=GWDG_API_KEY,
        base_url=GWDG_API_BASE,
        timeout=GWDG_TIMEOUT,
        max_retries=GWDG_MAX_RETRIES,
    )

    # OpenRouter-Client nur aktivieren, wenn ein Key hinterlegt ist.
    # Fehlt der Key, bleiben die Premium-Modelle einfach deaktiviert.
    or_key = _load_openrouter_key()
    if or_key:
        _openrouter_client = OpenAI(api_key=or_key, base_url=OPENROUTER_API_BASE)
        print(f"  ✓ OpenRouter-Client bereit — Premium-Modelle: {', '.join(OPENROUTER_MODELS)}")
    else:
        print(f"  ⚠ Kein OpenRouter-Key ({OPENROUTER_KEY_FILE.name}) — Premium-Modelle deaktiviert")

    _qdrant = QdrantClient(url=QDRANT_URL, timeout=10)
    await asyncio.to_thread(_qdrant.get_collections)
    print("  ✓ Qdrant verbunden")

    if not DB_PATH.exists():
        raise RuntimeError(f"doc_index.db nicht gefunden: {DB_PATH}\n→ python -m raghsh.index.build_doc_index")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    _index_conn = conn
    print("  ✓ Dokumentindex geladen")

    FASTEMBED_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"  Lade Dense-Embedding-Modell '{EMBED_MODEL}'…")
    _embedder = await asyncio.to_thread(
        lambda: TextEmbedding(EMBED_MODEL, cache_dir=str(FASTEMBED_CACHE))
    )
    print("  ✓ Dense-Embedder bereit")

    # language="german" ist zwingend — sonst passen die Token-IDs der Anfrage
    # nicht zu denen im Index. load_sparse_embedder() hält das an einer Stelle.
    print(f"  Lade Sparse-Modell '{SPARSE_MODEL}' (language=german)…")
    _sparse_embedder = await asyncio.to_thread(
        lambda: load_sparse_embedder(cache_dir=str(FASTEMBED_CACHE))
    )
    print("  ✓ Sparse-Embedder bereit")

    # Veraltete Log-Einträge beim Start bereinigen (DSGVO: max. 12 Monate)
    removed = await asyncio.to_thread(_cleanup_old_log_entries)
    if removed:
        print(f"  🧹 {removed} Log-Einträge älter als 12 Monate gelöscht")
    else:
        print("  ✓ Log-Cleanup: keine veralteten Einträge")

    _cleanup_task = asyncio.create_task(_periodic_log_cleanup())

    _ressourcen_bereit = True
    print(f"HsH RAG MCP Server bereit — http://127.0.0.1:{MCP_PORT}/mcp\n")


# ---------------------------------------------------------------------------
# FastMCP-Instanz
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "HsH RAG",
    host="127.0.0.1",
    port=MCP_PORT,
    lifespan=_lifespan,
)

# ---------------------------------------------------------------------------
# Modell-Cache
# ---------------------------------------------------------------------------


def _load_openrouter_key() -> str:
    """Liest den OpenRouter-API-Key aus der Key-Datei. Leer, wenn nicht vorhanden."""
    try:
        return OPENROUTER_KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _get_models() -> list[str]:
    global _model_cache, _model_cache_ts
    now = time.monotonic()
    if _model_cache and now - _model_cache_ts < MODEL_CACHE_TTL:
        return _model_cache
    try:
        _model_cache = [m.id for m in _gwdg_client.models.list().data]
        _model_cache_ts = now
    except Exception as exc:
        if _model_cache:
            return _model_cache
        raise RuntimeError(f"Modellliste nicht abrufbar: {exc}") from exc
    return _model_cache


# ---------------------------------------------------------------------------
# Resource: verfügbare Modelle
# ---------------------------------------------------------------------------


@mcp.resource("hsh://kontingent")
def gwdg_kontingent() -> str:
    """Verbleibende GWDG-Anfragen (Minute/Stunde/Tag/Monat), Stand der letzten GWDG-Antwort."""
    return json.dumps(gwdg_stand(), ensure_ascii=False)


@mcp.resource("hsh://models")
def list_models() -> str:
    """Verfügbare GWDG-Modelle für das frag_hsh-Tool (5-Minuten-Cache)."""
    return json.dumps(_get_models(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Pipeline-Hilfsfunktionen
# ---------------------------------------------------------------------------


async def _fortschritt(ctx: Context, schritt: int, text: str) -> None:
    """Meldet einen Pipeline-Schritt IM Antwortstrom des laufenden Aufrufs.

    Nicht ctx.report_progress() benutzen: das lässt `related_request_id` weg,
    und der Streamable-HTTP-Transport legt Notifications ohne diese Angabe in
    den separaten GET-Stream (streamable_http.py: `request_stream_id =
    target_request_id if ... else GET_STREAM_KEY`). Ein Client, der nur den
    POST-Stream seines eigenen tools/call liest — und genau das tut der
    naheliegende Client —, sah deshalb nie einen Fortschritt, ohne dass
    irgendwo ein Fehler auftauchte. Gemessen am 17.08.2026: null Events im
    POST-Stream, alle fünf im parallel geöffneten GET-Stream.

    Der Client muss weiterhin `params._meta.progressToken` mitschicken. Fehlt
    das Token, gibt es keine Adresse für die Notification und wir senden nichts.
    """
    meta  = ctx.request_context.meta
    token = meta.progressToken if meta else None
    if token is None:
        return
    await ctx.session.send_progress_notification(
        progress_token=token,
        progress=schritt,
        total=5,
        message=text,
        related_request_id=ctx.request_id,
    )


def _format_heading_tree(headings_json: str) -> str:
    try:
        headings = json.loads(headings_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    lines = []
    for h in headings:
        level = h.get("level", 1)
        text  = h.get("text", "")
        lines.append("  " * (level - 1) + "#" * level + " " + text)
    return "\n".join(lines)


# Fakultätsvorzug: zwei Abfragen statt einer nachgelagerten Filterung.
#
# Bis zum 17.08.2026 lief die Suche ungefiltert, und erst die fertige
# Kandidatenliste wurde auf die gewählte Fakultät reduziert. Enthielt die Liste
# keine Dokumente dieser Fakultät, blieben nur die fakultätslosen Reste übrig.
# Gemessen an „Kann ich eine Praxisphase schon im Grundstudium machen?" mit
# Fakultät I: Von 24 Kandidaten überstanden 10 den Filter — und KEIN EINZIGER
# war ein F1-Dokument. Übrig blieben Verkündungsblätter fremder Studiengänge,
# weil die zentral abgelegt sind und deshalb keine Fakultät tragen. Für den
# Nutzer sah das aus, als ignoriere der Filter seine Auswahl.
#
# Filtert dagegen Qdrant schon während des Retrievals, stehen die
# Praxisphasenordnungen der Fakultät I auf Platz 1 und 2.
#
# Zwei Abfragen, weil eine harte Filterung zu weit ginge: 39.600 Punkte tragen
# gar keine Fakultät — Immatrikulation, Prüfungsamt, Semesterticket. Die sind
# für viele Fragen genau richtig und dürfen nicht verschwinden.
#
# `faculty_key` (f1…f5 bzw. leer) kommt seit dem Neuimport aus der Payload und
# ist indiziert.


def _mische_kandidaten(treffer_fak: list[dict], treffer_zentral: list[dict],
                       top_n: int, studiengang_erkannt: bool = False) -> list[str]:
    """Kandidaten-URLs aus beiden Armen, Fakultät bevorzugt, Zentrale garantiert.

    Die Fakultät bekommt die erste Hälfte der Plätze, die zentralen Seiten die
    zweite; bleibt eine Seite unter ihrem Kontingent, füllt die andere auf.

    Ein reines Voranstellen der Fakultätstreffer wäre zu grob: Bei „Wann ist die
    Bibliothek geöffnet?" mit Fakultät I gewännen schwache F1-Treffer gegen die
    tatsächlich richtige zentrale Bibliotheksseite. Die Hälftelung stellt
    sicher, dass beide Sichten in der Auswahlrunde vertreten sind.
    """
    haelfte = max(1, top_n // 2)
    urls: list[str] = []
    gesehen: set[str] = set()

    def nimm(treffer: list[dict], grenze: int) -> None:
        for r in treffer:
            if len(urls) >= grenze:
                return
            u = (r["payload"] or {}).get("source_url", "")
            if u and u not in gesehen:
                gesehen.add(u)
                urls.append(u)

    nimm(treffer_fak, haelfte)
    nimm(treffer_zentral, top_n)
    nimm(treffer_fak, top_n)       # Rest auffüllen, falls zu wenig Zentrale

    # Die beste zentrale URL rückt nach ihrem Fusionsscore zwischen die
    # Fakultätsplätze (S069). Sonst stand zentrales Gold frühestens auf Platz
    # top_n/2 + 1, auch wenn es die Fakultätstreffer deutlich überragte. Nur
    # diese eine URL, weil die Scores beider Suchen je Anfrage normiert und
    # damit nur grob vergleichbar sind; die übrigen Plätze bleiben, wie sie sind.
    # Ist ein Studiengang erkannt, enthält der Fakultätsarm dessen Dokumente,
    # und Platz 1 bleibt ihm: in PO-Fragen träfe die zentrale Suche sonst meist
    # ein Verkündungsblatt fremder Studiengänge und schöbe es davor.
    beste_zentral = next(((r, (r["payload"] or {}).get("source_url", ""))
                          for r in treffer_zentral
                          if (r["payload"] or {}).get("source_url")), None)
    if beste_zentral and beste_zentral[1] in gesehen:
        z_treffer, z_url = beste_zentral
        z_score = z_treffer.get("score") or 0.0
        fak_score: dict[str, float] = {}
        for r in treffer_fak:      # absteigend sortiert: erster Chunk = bester
            u = (r["payload"] or {}).get("source_url", "")
            if u:
                fak_score.setdefault(u, r.get("score") or 0.0)
        rest = [u for u in urls if u != z_url]
        pos = next((i for i, u in enumerate(rest)
                    if fak_score.get(u, float("-inf")) < z_score), len(rest))
        if studiengang_erkannt:
            pos = max(pos, min(1, len(rest)))
        # nur nach vorn holen: steht sie über den Fakultätsarm (Dokument des
        # Studiengangs ohne Fakultät) schon weiter oben, bleibt sie dort
        pos = min(pos, urls.index(z_url))
        urls = rest[:pos] + [z_url] + rest[pos:]
    return urls


def _chunks_fuer_diagnose(results: list[dict], auszug_zeichen: int = 300) -> list[dict]:
    """Die tatsächlich gefundenen Chunks, für die Auswertung.

    Beantwortet die Frage, die eine reine URL-Liste offenlässt: War die richtige
    TEXTSTELLE im Kontext, oder nur die richtige Seite? Bei 29 % Dokumenten mit
    Lücken in der Chunkfolge ist das ein Unterschied mit Folgen.

    Wird nur auf Anforderung geliefert (`mit_chunks`), damit die Antwort für die
    Weboberfläche schlank bleibt — die braucht das nicht und würde es nur
    übertragen.
    """
    chunks = []
    for rang, r in enumerate(results, start=1):
        p = r.get("payload") or {}
        chunks.append({
            "rang":              rang,
            "score":             round(float(r.get("score") or 0.0), 5),
            "id":                str(r.get("id", "")),
            "source_url":        p.get("source_url", ""),
            "title_anzeige":     p.get("title_anzeige", "") or p.get("title", ""),
            "section_heading":   p.get("section_heading", ""),
            "chunk_index":       p.get("chunk_index"),
            "text_vollstaendig": p.get("text_vollstaendig"),
            "auszug":            (p.get("text", "") or "")[:auszug_zeichen],
        })
    return chunks


def _deduplicate_to_docs(results: list[dict], top_n: int) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for r in results:
        url = (r["payload"] or {}).get("source_url", "")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
        if len(urls) == top_n:
            break
    return urls


def _lookup_documents(urls: list[str]) -> dict[str, dict]:
    if not urls:
        return {}
    placeholders = ",".join("?" * len(urls))
    rows = _index_conn.execute(
        f"SELECT * FROM documents WHERE source_url IN ({placeholders})", urls
    ).fetchall()
    return {row["source_url"]: dict(row) for row in rows}


# Die Qdrant-Payload führt die Fakultät ausgeschrieben ("Fakultät IV"), der
# Dokumentindex den Kurzschlüssel aus dem Markdown-Frontmatter ("f4"). Verglichen
# wird unten gegen den Kurzschlüssel — ohne Übersetzung fiele jedes Dokument, das
# nur über die Payload bekannt ist, aus jeder Fakultätsauswahl heraus.
_FACULTY_KEYS = {
    "Fakultät I":   "f1",
    "Fakultät II":  "f2",
    "Fakultät III": "f3",
    "Fakultät IV":  "f4",
    "Fakultät V":   "f5",
}


def _faculty_key(value: str) -> str:
    """Normalisiert eine Fakultätsangabe auf den Kurzschlüssel f1…f5."""
    return _FACULTY_KEYS.get(value, value)


def _build_document_list(
    docs_info: dict[str, dict],
    urls: list[str],
    boost_urls: set[str] | None = None,
) -> str:
    blocks = []
    for i, url in enumerate(urls, 1):
        info     = docs_info.get(url, {})
        # Der Anzeigetitel entscheidet diese Runde: Das Modell sieht den
        # Volltext nicht, nur diese Liste. Bei PDFs ist `title` der Dateiname
        # und trägt nichts bei — "f3 › Veranstaltungsmanagement › BVM
        # RiggingPoint" sagt dagegen, worum es geht.
        title    = info.get("title_anzeige") or info.get("title") or "(kein Titel)"
        faculty  = info.get("faculty", "")
        headings = _format_heading_tree(info.get("headings", "[]"))
        lines    = [f"[{i}] {title}", f"    URL: {url}"]
        if boost_urls and url in boost_urls:
            lines.append("    HINWEIS: Dieses Dokument enthält den gesuchten Begriff direkt.")
        if faculty:
            lines.append(f"    Fakultät: {faculty}")
        if headings:
            lines.append("    Inhaltsstruktur:")
            for hline in headings.splitlines():
                lines.append(f"      {hline}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_selected_indices(response: str, max_idx: int) -> list[int]:
    try:
        data    = json.loads(response.strip())
        indices = data.get("selected", [])
        valid   = [int(i) for i in indices if 1 <= int(i) <= max_idx]
        if valid:
            return valid[:SELECT_N]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = re.search(r'\{[^}]*"selected"\s*:\s*\[([^\]]+)\][^}]*\}', response)
    if m:
        try:
            nums  = [int(x.strip()) for x in m.group(1).split(",")]
            valid = [n for n in nums if 1 <= n <= max_idx]
            if valid:
                return valid[:SELECT_N]
        except ValueError:
            pass
    return list(range(1, min(SELECT_N, max_idx) + 1))


def _load_chunks_from_qdrant(chunk_ids: list[str]) -> str:
    """Setzt den Text aus den Chunks EINER source_url zusammen — Rückfallweg."""
    if not chunk_ids:
        return ""
    points = _qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=chunk_ids,
        with_payload=True,
        with_vectors=False,
    )
    points.sort(key=lambda p: (p.payload or {}).get("chunk_index", 0))
    return "\n\n".join(
        (p.payload or {}).get("text", "")
        for p in points
        if (p.payload or {}).get("text")
    )


def _vertretung_lesen(url: str) -> list[sqlite3.Row]:
    """Liest die Vertreterzeilen eines Dokuments — mit EIGENER Verbindung.

    Nicht `_index_conn` benutzen: Diese Funktion läuft über asyncio.to_thread in
    einem Worker-Thread, und SQLite-Verbindungen sind an den Thread gebunden, in
    dem sie erzeugt wurden ("SQLite objects created in a thread can only be used
    in that same thread"). Die übrigen Index-Zugriffe laufen im Event-Loop-
    Thread und dürfen `_index_conn` weiter verwenden.

    Eine Verbindung je Aufruf kostet unter einer Millisekunde und ist der
    sicherere Weg gegenüber `check_same_thread=False`, bei dem sich mehrere
    Threads dasselbe Verbindungsobjekt teilen würden.
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT chunk_index, vertreten_durch, vertreter_url "
            "FROM chunk_vertretung WHERE source_url = ? ORDER BY chunk_index",
            (url,),
        ).fetchall()
    finally:
        conn.close()


def _load_document_text(url: str, chunk_ids: list[str]) -> tuple[str, list[str]]:
    """Setzt ein Dokument VOLLSTÄNDIG zusammen, über Dokumentgrenzen hinweg.

    Die Deduplizierung behält je Gruppe wortgleicher Chunks nur einen Vertreter.
    29,4 % der Dokumente haben deshalb Lücken; im Extremfall liegen 3 von 242
    Positionen unter ihrer eigenen URL. Wer nur diese lädt, übergibt dem LLM ein
    Fragment und nennt es „das Dokument" — bei einem Modulhandbuch bleibt dann
    ein einziges Modul übrig, und die Antwort ist sauber belegt und trotzdem
    grob falsch.

    Die Tabelle `chunk_vertretung` löst das auf: Sie nennt für jede Position des
    Dokuments die Punkt-ID, unter der ihr Text im Index steht — auch wenn das
    ein Punkt einer anderen source_url ist.

    Gibt (Text, fremde Fundstellen) zurück. Die fremden URLs gehören in die
    Quellenangabe: Der Absatz stammt dann aus einem gleichlautenden Dokument,
    oft einer anderen Fakultät oder einem anderen Jahrgang.

    Fehlt die Tabelle (Vertretungsdatei nicht geladen), wird auf die vorhandenen
    Chunks zurückgefallen — dann ist der Text wieder lückenhaft, aber die
    Pipeline läuft.
    """
    zeilen = _vertretung_lesen(url)

    if not zeilen:
        return _load_chunks_from_qdrant(chunk_ids), []

    punkte = _qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[z["vertreten_durch"] for z in zeilen],
        with_payload=True,
        with_vectors=False,
    )
    # retrieve() gibt keine Reihenfolge zu — über die ID zuordnen, nicht über
    # die Position in der Antwort.
    nach_id = {str(p.id): p for p in punkte}

    teile: list[str] = []
    fremd: list[str] = []
    for z in zeilen:
        p = nach_id.get(z["vertreten_durch"])
        if p is None:
            continue
        text = (p.payload or {}).get("text", "")
        if not text:
            continue
        teile.append(text)
        vurl = z["vertreter_url"]
        if vurl and vurl != url and vurl not in fremd:
            fremd.append(vurl)

    return "\n\n".join(teile), fremd


def _estimate_tokens(text: str) -> int:
    # Deutscher Text hat ~3 Zeichen/Token (konservativer als englische 4-Zeichen-Faustregel)
    return max(1, len(text) // 3)


def _fit_to_budget(docs: list[dict]) -> tuple[list[dict], int]:
    """Wählt so viele Dokumente, wie ins Tokenbudget passen.

    Das erste Dokument kommt IMMER mit, sonst hätte das LLM gar keinen Kontext —
    aber es wird nötigenfalls GEKÜRZT. Diese Kürzung fehlte bis zum 17.08.2026,
    und solange Runde 2 nur die eigenen Chunks einer URL lud, fiel das nicht auf:
    Dokumente waren selten größer als ein paar hundert Kilobyte.

    Seit die Dokumente über die Vertretungstabelle vollständig zusammengesetzt
    werden, ist das anders. Gemessen an Fall q20 der Auswertungsbasis: vier
    ausgewählte Dokumente mit zusammen 4,4 Millionen Zeichen, das größte allein
    2,2 Millionen (ein Vorlesungsverzeichnis). Das erste ging ungekürzt an die
    API und lief in „maximum context length is 131072 tokens".

    Die Kürzung setzt eine sichtbare Marke ans Ende. Ein stillschweigend
    abgeschnittenes Dokument wäre derselbe Fehler wie ein stillschweigendes
    Fragment: Das Modell hielte es für vollständig.
    """
    budget = MAX_CONTEXT_TOKENS - PROMPT_OVERHEAD_TOKENS
    result: list[dict] = []
    used = 0
    for doc in docs:
        tokens = _estimate_tokens(doc["text"])
        if not result:
            if tokens > budget:
                doc = {**doc,
                       "text": _kuerze_auf_budget(doc["text"], budget),
                       "gekuerzt": True}
                tokens = _estimate_tokens(doc["text"])
            result.append(doc)
            used += tokens
            if tokens >= budget:
                break
        elif used + tokens <= budget:
            result.append(doc)
            used += tokens
    return result, used


def _kuerze_auf_budget(text: str, budget_tokens: int) -> str:
    """Kürzt einen Dokumenttext auf das Tokenbudget, an einer Absatzgrenze.

    Die Marke am Ende ist kein Schmuck: Ohne sie hielte das Modell die ersten
    dreihundert Kilobyte eines Vorlesungsverzeichnisses für das ganze Dokument
    und würde „darin steht nichts über X" antworten, obwohl X auf Seite 400
    steht.
    """
    marke = "\n\n[…] Dieses Dokument wurde gekürzt; der weitere Inhalt liegt nicht im Kontext."
    max_zeichen = max(budget_tokens * 3 - len(marke), 1000)
    if len(text) <= max_zeichen:
        return text
    schnitt = text.rfind("\n\n", 0, max_zeichen)
    if schnitt < max_zeichen // 2:      # keine brauchbare Absatzgrenze gefunden
        schnitt = max_zeichen
    return text[:schnitt] + marke


def _trim_snippet(text: str, max_chars: int = 2000) -> str:
    """Kürzt einen Textabschnitt auf max_chars, bevorzugt an Absatz- oder Satzgrenzen."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    idx = truncated.rfind("\n\n")
    if idx > max_chars // 2:
        return truncated[:idx]
    for sep in (". ", ".\n", "! ", "? "):
        idx = truncated.rfind(sep)
        if idx > max_chars // 2:
            return truncated[: idx + 1]
    return truncated


def _find_quote(quote: str, full_text: str) -> int:
    """Findet ein Zitat im Volltext: erst exakt, dann mit normalisiertem Whitespace.

    Gibt den Index in full_text zurück (-1 wenn nicht gefunden).
    Bei normalisiertem Match ist der Index eine Approximation der Position.
    """
    idx = full_text.find(quote)
    if idx >= 0:
        return idx
    norm_q = re.sub(r'\s+', ' ', quote).strip()
    norm_t = re.sub(r'\s+', ' ', full_text)
    idx_norm = norm_t.find(norm_q)
    return idx_norm if idx_norm >= 0 else -1


# ---------------------------------------------------------------------------
# Zitatbereinigung (S023): wörtliche Zitate „…" [N] gegen die Quellen prüfen
# ---------------------------------------------------------------------------
# Das Modell setzt gelegentlich Umformuliertes in Anführungszeichen oder
# ordnet ein Zitat der falschen Quelle zu. Je Zitat, der Reihe nach:
#   1  steht (kompakt normalisiert) in einer seiner Quellen N → unverändert
#   2  steht in einer anderen ausgewählten Quelle M → Verweis [N] → [M]
#   3  ähnlichste Stelle in Quelle N (Wortfolge, difflib) ≥ ZITAT_AEHNLICHKEIT
#      → Originalwortlaut einsetzen
#   4  sonst → Anführungszeichen entfernen (bleibt als Paraphrase stehen)
# Ein Kurzzitat (≤ 3 Wörter) ohne direkt folgende Marke ist eine
# Begriffsmarkierung („ein „freier Versuch" ist nicht vorgesehen"), kein
# Quellenzitat, und bleibt unberührt (Fall 0) — wie in der Belegprüfung.

ZITAT_AEHNLICHKEIT = 0.80   # Mindest-Ähnlichkeit (SequenceMatcher über Wörter) für Fall 3
ZITAT_MIN_WOERTER  = 4      # kürzere Zitate werden nie unscharf ersetzt

_ZB_ZITAT      = re.compile(r"„([^“”\"\n]{1,2000})([“”\"])")
_ZB_MARKE      = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_ZB_AUSLASSUNG = re.compile(r"\s*(?:…|\.\.\.|\[\s*…\s*\]|\(\s*…\s*\))\s*")
_ZB_SATZENDE   = re.compile(r"(?<=[^\d\s][.!?:])\s+(?=[A-ZÄÖÜ„])|\n")
_ZB_UMLAUTE    = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_ZB_MD_LINK    = re.compile(r"!?\[([^\]\n]*)\]\([^)\s]*\)")
_ZB_ZEILENRAND = re.compile(r"[ \t]*\n[ \t]*(?:[>#|]+[ \t]*)*")   # Umbruch + Markdown-Präfix


def _zb_norm(text: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKC", text).casefold().translate(_ZB_UMLAUTE)


def _zb_kompakt(text: str) -> str:
    """Ohne Leerraum und Interpunktion: fängt PDF-Umbrüche („de r") ab."""
    return re.sub(r"[^\w]|_", "", _zb_norm(_ZB_MD_LINK.sub(r"\1", text)))


def _zb_marken(abschnitt: str) -> list[int]:
    nummern: list[int] = []
    for m in _ZB_MARKE.finditer(abschnitt):
        for teil in m.group(1).split(","):
            von, *bis = re.split(r"[–-]", teil)
            nummern.extend(range(int(von), int(bis[0]) + 1) if bis else [int(von)])
    return nummern


def _zb_aehnlichste(teil: str, text: str) -> tuple[float, str, str, str]:
    """Ähnlichste Wortfolge zu `teil` in `text`.

    Rückgabe: (Ähnlichkeit, Originalwortlaut, vorn, hinten). `vorn`/`hinten`
    sind Randwörter des Zitats ohne Gegenstück in der Quelle (etwa ein
    angepasstes Verb); sie gehören zur Aussage und bleiben außerhalb der
    Anführungszeichen stehen, sofern die Quelle sie nicht selbst enthält.
    """
    import difflib
    from collections import Counter
    q = [_zb_norm(w) for w in re.findall(r"\w+", teil)]
    text = _ZB_MD_LINK.sub(r"\1", text)
    spans = [(m.start(), m.end(), _zb_norm(m.group())) for m in re.finditer(r"\w+", text)]
    k = len(q)
    if not q or len(spans) < max(2, k // 2):
        return 0.0, "", "", ""
    worte = [s[2] for s in spans]
    # Vorauswahl: Fenster der Länge k mit den meisten gemeinsamen Wörtern.
    qz, fenster, gemein, kandidaten = Counter(q), Counter(), 0, []
    for i, w in enumerate(worte):
        fenster[w] += 1
        if fenster[w] <= qz[w]:
            gemein += 1
        if i >= k:
            alt = worte[i - k]
            if fenster[alt] <= qz[alt]:
                gemein -= 1
            fenster[alt] -= 1
        if i >= k - 1:
            kandidaten.append((gemein, i - k + 1))
    kandidaten = sorted(kandidaten, reverse=True)[:15]
    # Maß: gemeinsame Wörter / max(Zitatlänge, Fensterlänge). Anders als
    # SequenceMatcher.ratio() belohnt es kein Fenster, das am Rand Wörter des
    # Zitats weglässt; bei Gleichstand gewinnt die Länge nahe am Zitat.
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(q)
    best = (0.0, 0, 0, 0)
    for _, start in kandidaten:
        for a in range(max(0, start - 2), start + 3):
            for laenge in range(max(1, k - 3), k + 4):
                if a + laenge > len(worte):
                    break
                sm.set_seq1(worte[a:a + laenge])
                gleich = sum(b.size for b in sm.get_matching_blocks())
                wert = (gleich / max(k, laenge), -abs(laenge - k))
                if wert > best[:2]:
                    best = (*wert, a, a + laenge)
    r, _, a, b = best
    if not r:
        return 0.0, "", "", ""
    # Randwörter ohne Gegenstück im Zitat abschneiden („BIS). Die …" → „Die …")
    sm.set_seq1(worte[a:b])
    bloecke = [x for x in sm.get_matching_blocks() if x.size]
    a, b = a + bloecke[0].a, a + bloecke[-1].a + bloecke[-1].size
    if sum(x.size for x in bloecke) == k and b - a - k <= 3:
        # Alle Zitatwörter in Reihenfolge, nur durch Satzreste getrennt
        # (Tabellenzeile, Seitenkopf im PDF): Das Zitat ist belegt.
        return 1.0, teil.strip(), "", ""
    q_spans = [m.span() for m in re.finditer(r"\w+", teil)]
    q_von, q_bis = bloecke[0].b, bloecke[-1].b + bloecke[-1].size
    im_fenster = set(worte[a:b])
    vorn   = teil[:q_spans[q_von][0]].strip() if q_von else ""
    hinten = teil[q_spans[q_bis - 1][1]:].strip(" .,;:") if q_bis < k else ""
    vorn   = "" if set(q[:q_von]) <= im_fenster else vorn
    hinten = "" if set(q[q_bis:]) <= im_fenster else hinten
    wortlaut = _ZB_ZEILENRAND.sub(" ", text[spans[a][0]:spans[b - 1][1]])
    if "|" in wortlaut:
        return 0.0, "", "", ""          # Tabellen-/PDF-Spaltensatz: kein zitierbarer Wortlaut
    wortlaut = " ".join(wortlaut.replace("**", "").replace("\xad", "").split())
    wortlaut = re.sub(r"[\[\]]", "", wortlaut).strip("() ")
    # Anführungszeichen im Wortlaut würden das Zitat beenden → einfache ‚…‘
    wortlaut = wortlaut.replace("„", "‚").replace("“", "‘").replace("”", "‘")
    teile = wortlaut.split('"')
    wortlaut = "".join(t + ("‚" if i % 2 == 0 else "‘") for i, t in enumerate(teile[:-1])) + teile[-1]
    return r, wortlaut, vorn, hinten


def _zitate_bereinigen(antwort: str, quelltexte: list[str]) -> tuple[str, list[dict]]:
    """Prüft jedes Zitat „…" [N] gegen die Quellen und repariert es (Fälle 1–4, s. o.).

    `quelltexte[n-1]` ist der Text der Quelle mit Marke [n]. Ohne Marke hinter
    dem Zitat gelten die Marken des Satzes, ohne jede Marke alle Quellen.
    Rückgabe: (bereinigte Antwort, je Zitat {fall, quellen, nach, aehnlichkeit,
    ersatz}). `ersatz` ist eingesetzter Quellwortlaut (öffentlich), nie Text
    der Antwort oder Frage.
    """
    if not quelltexte:
        return antwort, []
    kompakt = [_zb_kompakt(t or "") for t in quelltexte]
    alle = list(range(1, len(quelltexte) + 1))

    def belegt(teile: list[str], n: int) -> bool:
        return 1 <= n <= len(kompakt) and all(_zb_kompakt(t) in kompakt[n - 1] for t in teile)

    # Satzgrenzen (für Marken, die nicht direkt am Zitat stehen)
    grenzen = [0] + [m.end() for m in _ZB_SATZENDE.finditer(antwort)] + [len(antwort) + 1]

    def satz(pos: int) -> tuple[int, int]:
        for a, b in zip(grenzen, grenzen[1:]):
            if a <= pos < b:
                return a, b
        return 0, len(antwort)

    treffer = list(_ZB_ZITAT.finditer(antwort))
    aenderungen: list[tuple[int, int, str]] = []
    protokoll: list[dict] = []
    for i, m in enumerate(treffer):
        s_anf, s_end = satz(m.start())
        ende = min(treffer[i + 1].start() if i + 1 < len(treffer) else len(antwort), s_end)
        nummern = (_zb_marken(antwort[m.end():ende])
                   or _zb_marken(antwort[s_anf:s_end]) or alle)
        nummern = [n for n in dict.fromkeys(nummern) if 1 <= n <= len(quelltexte)] or alle
        zitat = m.group(1)
        teile = [t for t in _ZB_AUSLASSUNG.split(zitat) if re.search(r"\w", t)]
        direkt = _ZB_MARKE.match(antwort[m.end():].lstrip(" .,;:)"))
        eintrag = {"quellen": nummern}
        if not teile or (len(re.findall(r"\w+", zitat)) <= 3 and not direkt):
            protokoll.append({**eintrag, "fall": 0})
            continue
        if any(belegt(teile, n) for n in nummern):
            protokoll.append({**eintrag, "fall": 1})
            continue
        andere = [n for n in alle if n not in nummern and belegt(teile, n)]
        if andere:
            neu = f"[{andere[0]}]"
            mm = re.match(r"\s*" + _ZB_MARKE.pattern, antwort[m.end():])
            if mm and set(_zb_marken(mm.group())) <= set(nummern):
                aenderungen.append((m.end() + mm.start(), m.end() + mm.end(), " " + neu))
            else:
                aenderungen.append((m.end(), m.end(), " " + neu))
            protokoll.append({**eintrag, "fall": 2, "nach": andere[0]})
            continue
        # Fall 3: je Teilstück die ähnlichste Stelle in einer der Quellen N
        ersetzt, werte = [], []
        for t in teile:
            if any(belegt([t], n) for n in nummern):
                ersetzt.append(t.strip())
                continue
            if len(re.findall(r"\w+", t)) < ZITAT_MIN_WOERTER:
                break
            r, wortlaut, vorn, hinten = max(
                (_zb_aehnlichste(t, quelltexte[n - 1] or "") for n in nummern), key=lambda x: x[0])
            werte.append(round(r, 3))
            if r < ZITAT_AEHNLICHKEIT:
                break
            if r >= 1.0 and wortlaut == t.strip():
                werte.pop()
            else:
                eintrag.setdefault("ersatz", []).append(wortlaut)
            ersetzt.append(wortlaut)
        if len(ersetzt) == len(teile) and not werte:
            protokoll.append({**eintrag, "fall": 1})
            continue
        if len(ersetzt) == len(teile):
            neu = "„" + " … ".join(ersetzt) + m.group(2)
            if len(teile) == 1 and werte:      # Randwörter ohne Beleg bleiben stehen
                neu = " ".join(x for x in (vorn, neu, hinten) if x)
            aenderungen.append((m.start(), m.end(), neu))
            protokoll.append({**eintrag, "fall": 3, "aehnlichkeit": min(werte)})
            continue
        # Fall 4: Anführungszeichen entfernen
        aenderungen.append((m.start(), m.end(), zitat))
        protokoll.append({**eintrag, "fall": 4, "aehnlichkeit": min(werte) if werte else None})

    for a, b, neu in sorted(aenderungen, reverse=True):
        antwort = antwort[:a] + neu + antwort[b:]
    return antwort, protokoll


# ---------------------------------------------------------------------------
# Markenkorrektur (S041): falsche Quellenmarke nachträglich umsetzen
# ---------------------------------------------------------------------------
# Deterministisch, ohne LLM. Steht ein Zitat „…" bzw. eine Zahl nicht in den
# Quellen, die sein Satz nennt, aber wörtlich in **genau einer** anderen
# Kontextquelle [M], zeigt die Marke auf [M]; in mehreren oder keiner bleibt
# alles, wie es ist. Zuordnung wie in der Belegprüfung (eval_gold): ein Zitat
# gehört zu den Marken zwischen seinem Ende und dem nächsten Zitat bzw.
# Satzende, sonst zu den Marken des Satzes; eine Zahl zu den Marken ihres
# Satzes. Ohne Marke im Satz gilt die ganze Antwort als belegt — nichts zu tun.
# - Zitat mit genau einer Markengruppe direkt dahinter → Gruppe wird [M],
#   sofern dadurch keine andere Zahl des Satzes ihren Beleg verliert (sonst
#   unverändert); Zitat ohne eigene Marke → „ [M]" direkt dahinter.
# - Zahl → „ [M]" direkt hinter die Zahl (die übrigen Fakten des Satzes
#   behalten ihre Marke).
# „Steht in N" ist großzügig (kompakt oder Wortfolge mit kleinen Lücken), „steht
# wörtlich in M" streng (kompakt): umgesetzt wird nur, was sicher woanders steht.

_MK_SATZENDE    = re.compile(r"(?<=[^\d\s][.!?:])\s+(?=[A-ZÄÖÜ„])")
_MK_AUFZAEHLUNG = re.compile(r"^\s*(?:[-*]\s+)?\d{1,2}[.)]\s")
_MK_ZAHL        = re.compile(r"(?<![\w§])\d+(?:[.,:/]\d+)*(?!\w)")
_MK_URL         = re.compile(r"(?:https?://|www\.)\S+|\S+@\S+\.\w+|!?\[[^\]\n]*\]\([^)\s]*\)")
_MK_UMLAUT_ZERLEGT = re.compile(r"¨\s?([aouAOU])|([aouAOU])\s?¨")
_MK_UMLAUT_ZU   = {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"}
_MK_DOKUMENT_NR = re.compile(r"\b(?:Dokument|Dokumenten|Dokuments|Quelle|Quellen)\s+\d+\b")
MARKE_LUECKE    = 3          # fremde Wörter zwischen zwei Zitatwörtern, die „steht in N" noch zulässt


def _mk_text(text: str) -> str:
    """PDF-Text mit zerlegten Umlauten („Pr¨ufung") zusammenführen."""
    return _MK_UMLAUT_ZERLEGT.sub(lambda m: _MK_UMLAUT_ZU[m.group(1) or m.group(2)], text or "")


def _mk_kompakt(text: str) -> str:
    return _zb_kompakt(_mk_text(text).replace("&", " und "))


def _mk_woerter(text: str) -> list[str]:
    return re.findall(r"\w+", _zb_norm(_ZB_MD_LINK.sub(r"\1", _mk_text(text))))


def _mk_wortfolge(muster: list[str], quelle: list[str], index: dict[str, list[int]]) -> bool:
    """Alle Wörter von `muster` in Reihenfolge in `quelle`, je Lücke ≤ MARKE_LUECKE."""
    for start in index.get(muster[0], []):
        pos, ok = start, True
        for w in muster[1:]:
            naechste = next((i for i in range(pos + 1, min(pos + 2 + MARKE_LUECKE, len(quelle)))
                             if quelle[i] == w), None)
            if naechste is None:
                ok = False
                break
            pos = naechste
        if ok:
            return True
    return False


def _mk_zahlen(text: str) -> set[str]:
    """Zahlen samt ihren Teilen: „21.09.2026" belegt auch 21, 09 und 2026."""
    gefunden = set(_MK_ZAHL.findall(_mk_text(text)))
    return gefunden | {t for z in gefunden for t in re.split(r"[.,:/]", z) if t}


def _marken_korrigieren(antwort: str, quelltexte: list[str],
                        frage: str = "") -> tuple[str, list[dict]]:
    """Setzt falsche Quellenmarken um (s. o.). `quelltexte[n-1]` ist der Text
    der Quelle mit Marke [n]. Rückgabe: (Antwort, je Korrektur {art, von, nach}).
    Das Protokoll enthält nur Nummern, nie Antwort- oder Fragetext."""
    if len(quelltexte) < 2 or not antwort:
        return antwort, []
    kompakt = [_mk_kompakt(t) for t in quelltexte]
    woerter = [_mk_woerter(t) for t in quelltexte]
    indizes = []
    for ws in woerter:
        idx: dict[str, list[int]] = {}
        for i, w in enumerate(ws):
            idx.setdefault(w, []).append(i)
        indizes.append(idx)
    zahlen = [_mk_zahlen(t) for t in quelltexte]
    frage_zahlen = _mk_zahlen(frage)
    alle = range(1, len(quelltexte) + 1)

    def teile(zitat: str) -> list[str]:
        return [t for t in _ZB_AUSLASSUNG.split(_mk_text(zitat)) if re.search(r"\w", t)]

    def steht_in(tl: list[str], n: int) -> bool:          # großzügig
        for t in tl:
            if _mk_kompakt(t) in kompakt[n - 1]:
                continue
            muster = _mk_woerter(t)
            if not muster or not _mk_wortfolge(muster, woerter[n - 1], indizes[n - 1]):
                return False
        return True

    def woertlich_in(tl: list[str], n: int) -> bool:      # streng
        return all(_mk_kompakt(t) in kompakt[n - 1] for t in tl)

    def gueltig(nummern: list[int]) -> list[int]:
        return [n for n in dict.fromkeys(nummern) if 1 <= n <= len(quelltexte)]

    aenderungen: list[tuple[int, int, str]] = []
    protokoll: list[dict] = []
    pos = 0
    for zeile in antwort.split("\n"):
        z_anf, pos = pos, pos + len(zeile) + 1
        auf = _MK_AUFZAEHLUNG.match(zeile)
        s_anf = auf.end() if auf else 0
        grenzen = [s_anf] + [m.end() for m in _MK_SATZENDE.finditer(zeile, s_anf)] + [len(zeile)]
        # Zitate über eine Satzgrenze hinweg („… sein. Der Nachweis …") sieht die
        # Satzzerlegung nicht als Zitat — ihre Spannen gelten zeilenweise als gesperrt,
        # sonst landete eine Marke mitten im Zitat.
        zeilen_zitate = [(m.start(), m.end()) for m in _ZB_ZITAT.finditer(zeile)]
        for a, b in zip(grenzen, grenzen[1:]):
            satz = zeile[a:b]
            satz_marken = gueltig(_zb_marken(satz))
            if not satz_marken:
                continue
            gesperrt = [(m.start(), m.end()) for rx in (_ZB_MARKE, _MK_URL, _MK_DOKUMENT_NR)
                        for m in rx.finditer(satz)]
            # Zahlen außerhalb von Zitaten, Marken, Adressen
            zitate = list(_ZB_ZITAT.finditer(satz))
            gesperrt += [(m.start(), m.end()) for m in zitate]
            gesperrt += [(x - a, y - a) for x, y in zeilen_zitate if x < b and y > a]
            satz_zahlen = [m for m in _MK_ZAHL.finditer(satz)
                           if not any(x <= m.start() < y for x, y in gesperrt)
                           and m.group() not in frage_zahlen]

            def zahlen_belegt(nummern: list[int]) -> set[str]:
                return {m.group() for m in satz_zahlen
                        if any(m.group() in zahlen[n - 1] for n in nummern)}

            for i, m in enumerate(zitate):
                ende = zitate[i + 1].start() if i + 1 < len(zitate) else len(satz)
                bereich = satz[m.end():ende]
                gruppen = list(_ZB_MARKE.finditer(bereich))
                direkt = re.match(r"[\s.,;:)]*" + _ZB_MARKE.pattern, bereich)
                tl = teile(m.group(1))
                if not tl or (len(re.findall(r"\w+", m.group(1))) <= 3 and not direkt):
                    continue                    # Begriffsmarkierung, kein Quellenzitat
                nummern = gueltig(_zb_marken(bereich)) or satz_marken
                if any(steht_in(tl, n) for n in nummern):
                    continue
                andere = [n for n in alle if n not in nummern and woertlich_in(tl, n)]
                if len(andere) != 1:
                    continue
                neu = andere[0]
                if not gruppen:
                    aenderungen.append((z_anf + a + m.end(), z_anf + a + m.end(), f" [{neu}]"))
                elif len(gruppen) == 1 and direkt:
                    g = gruppen[0]
                    rest = gueltig(_zb_marken(satz[:m.end()] + satz[m.end() + g.end():]))
                    if zahlen_belegt(satz_marken) - zahlen_belegt(rest + [neu]):
                        continue                # Gruppe trägt noch eine Zahl des Satzes
                    s = z_anf + a + m.end()
                    aenderungen.append((s + g.start(), s + g.end(), f"[{neu}]"))
                else:
                    continue
                protokoll.append({"art": "zitat", "von": nummern, "nach": neu})

            for m in satz_zahlen:
                if m.group() in zahlen_belegt(satz_marken):
                    continue
                andere = [n for n in alle if n not in satz_marken and m.group() in zahlen[n - 1]]
                if len(andere) != 1:
                    continue
                aenderungen.append((z_anf + a + m.end(), z_anf + a + m.end(), f" [{andere[0]}]"))
                protokoll.append({"art": "zahl", "von": satz_marken, "nach": andere[0]})

    for a, b, neu in sorted(aenderungen, reverse=True):
        antwort = antwort[:a] + neu + antwort[b:]
    return antwort, protokoll


def _extract_verified_snippets(
    answer: str,
    selected_docs: list[dict],
    fallback_results: list[dict],
    max_chars: int = 2000,
    max_per_doc: int = 3,
) -> dict[str, dict]:
    """Verifiziert wörtliche Zitate aus der LLM-Antwort gegen den Originaltext.

    Für jede ausgewählte Quelle:
    - Sucht alle Zitate (in Anführungszeichen) aus der Antwort im Volltext.
    - Gibt den umgebenden Absatz als Snippet zurück (verifiziert=True).
    - Fällt auf Such-Chunks zurück, wenn kein Zitat im Text nachweisbar ist
      (verifiziert=False — Hinweis auf LLM-Paraphrase oder Halluzination).
    - Entnimmt den Abschnittsnamen (section_heading) aus den Such-Chunks.
    """
    quotes = re.findall(r'„([^„"]{10,500})"', answer)

    fallback_snippets: dict[str, list[str]] = {}
    section_by_url: dict[str, str] = {}
    for r in fallback_results:
        payload = r["payload"] or {}
        url     = payload.get("source_url", "")
        text    = payload.get("text", "").strip()
        heading = payload.get("section_heading", "")
        if not url:
            continue
        if heading and url not in section_by_url:
            section_by_url[url] = heading
        snips = fallback_snippets.setdefault(url, [])
        if len(snips) < max_per_doc and text and text not in snips:
            snips.append(_trim_snippet(text, max_chars))

    result: dict[str, dict] = {}
    for doc in selected_docs:
        url         = doc["url"]
        full_text   = doc["text"]
        snippets:    list[str] = []
        found_quotes: list[str] = []
        verified    = False

        for quote in quotes:
            norm_q = re.sub(r'\s+', ' ', quote).strip()
            idx    = _find_quote(quote, full_text)
            if idx < 0:
                continue
            verified = True
            found_quotes.append(norm_q)
            para_start  = full_text.rfind("\n\n", 0, min(idx, len(full_text)))
            para_start  = para_start + 2 if para_start >= 0 else 0
            para_end    = full_text.find("\n\n", idx + len(quote))
            para_end    = para_end if para_end >= 0 else len(full_text)
            snippet     = _trim_snippet(full_text[para_start:para_end].strip(), max_chars)
            if snippet and snippet not in snippets:
                snippets.append(snippet)
            if len(snippets) >= max_per_doc:
                break

        if not snippets:
            snippets = fallback_snippets.get(url, [])

        result[url] = {
            "snippets":    snippets,
            "zitate":      found_quotes,
            "verifiziert": verified,
            "abschnitt":   section_by_url.get(url, ""),
        }

    return result


def _ask_llm(model: str, messages: list[dict], frist_s: float | None = None) -> tuple[str, str]:
    """frist_s: harte Obergrenze für diesen Aufruf (Fassungsvergleich, S056) — eigener
    Client mit timeout=frist_s und ohne Retries, kein Warten nach 429. Ohne frist_s:
    Client-Einstellungen wie immer (Antwort- und Auswahlaufruf)."""
    # Premium-Modelle laufen über OpenRouter, alle anderen über GWDG ChatAI.
    if model in OPENROUTER_MODELS:
        client = _openrouter_client
        dienst = "OpenRouter"
        if client is None:
            raise RuntimeError("Premium-Modell nicht verfügbar — kein OpenRouter-API-Key hinterlegt.")
    else:
        client = _gwdg_client
        dienst = "GWDG"
    gwdg = client is _gwdg_client
    try:
        extra = ({"provider": OPENROUTER_ANBIETER[model]}
                 if client is _openrouter_client and model in OPENROUTER_ANBIETER else None)
        warten = list(OPENROUTER_429_WARTEN) if client is _openrouter_client else []
        if frist_s is not None:
            # with_options: eigener Client (teilt nur den Verbindungspool), Timeout
            # = Restzeit; sonst hängt ein verworfener Versuch samt SDK-Retries bis
            # zu GWDG_TIMEOUT × (GWDG_MAX_RETRIES + 1) weiter.
            client = client.with_options(timeout=max(frist_s, 1.0), max_retries=0)
            warten = []
        while True:
            try:
                if gwdg:
                    _gwdg_reserve_pruefen()
                    roh = client.chat.completions.with_raw_response.create(
                        model=model,
                        messages=messages,
                        temperature=TEMPERATURE,
                        extra_body=extra,
                    )
                    _gwdg_stand_merken(roh.headers, "produktion" if PRODUKTION else "agenten")
                    response = roh.parse()
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=TEMPERATURE,
                        extra_body=extra,
                    )
                break
            except RateLimitError as exc:
                if gwdg and getattr(exc, "response", None) is not None:
                    _gwdg_stand_merken(exc.response.headers, "produktion" if PRODUKTION else "agenten")
                # Begründung des Anbieters mitschreiben (ohne Anfrageinhalt).
                print(f"  ⚠ 429 von {dienst} ({model}): {str(getattr(exc, 'body', '') or exc)[:300]}"
                      + (f" — neuer Versuch in {warten[0]} s" if warten else " — aufgegeben"), flush=True)
                if not warten:
                    raise
                time.sleep(warten.pop(0))
    except AuthenticationError:
        raise RuntimeError(f"API-Authentifizierung fehlgeschlagen — {dienst}-API-Key ungültig oder abgelaufen.")
    except RateLimitError:
        raise RuntimeError(f"Rate-Limit der {dienst}-API erreicht — bitte kurz warten und erneut versuchen.")
    except APITimeoutError:
        raise RuntimeError(f"Zeitüberschreitung bei der {dienst}-API — der Dienst antwortet zu langsam.")
    except APIConnectionError as exc:
        raise RuntimeError(f"Verbindung zur {dienst}-API fehlgeschlagen: {exc}")
    except BadRequestError as exc:
        raise RuntimeError(f"Ungültige Anfrage an die {dienst}-API (Kontext zu lang?): {exc}")
    except Exception as exc:
        raise RuntimeError(f"Unerwarteter API-Fehler: {exc}")

    if not response.choices:
        raise RuntimeError(f"{dienst}-API hat eine leere Antwort zurückgegeben (keine choices).")

    msg = response.choices[0].message
    content = msg.content or ""
    if not content.strip():
        finish = getattr(response.choices[0], "finish_reason", "unbekannt")
        raise RuntimeError(
            f"LLM hat keinen Text zurückgegeben (finish_reason: {finish}). "
            "Mögliche Ursachen: Inhaltsfilter, Kontextüberschreitung oder Modellproblem."
        )

    reasoning = getattr(msg, "reasoning_content", "") or ""
    return content, reasoning


def _cleanup_old_log_entries() -> int:
    """Löscht Einträge älter als 12 Monate aus qa_log.jsonl. Gibt Anzahl gelöschter Einträge zurück."""
    if not LOG_PATH.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_LOG_AGE_DAYS)
    kept: list[str] = []
    removed = 0
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
                ts = datetime.fromisoformat(entry.get("ts", ""))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    kept.append(stripped)
                else:
                    removed += 1
            except (json.JSONDecodeError, ValueError):
                kept.append(stripped)
    if removed:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))
    return removed


async def _periodic_log_cleanup() -> None:
    """Läuft alle 24 Stunden und bereinigt veraltete Log-Einträge."""
    while True:
        await asyncio.sleep(86400)
        removed = await asyncio.to_thread(_cleanup_old_log_entries)
        if removed:
            print(f"  🧹 Cleanup: {removed} Log-Einträge älter als 12 Monate gelöscht")


def _log_qa(
    frage: str,
    modell: str,
    fakultaet: str,
    antwort: str,
    quellen: list[dict],
    tokens: int,
    source: str = "mcp",
) -> None:
    entry = {
        "ts":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source":    source,
        "modell":    modell,
        "fakultaet": fakultaet,
        "frage":     frage,
        "antwort":   antwort,
        "quellen":   quellen,
        "tokens":    tokens,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _empty_result(fehler: str) -> dict:
    # `zeiten` gehört mit in die Fehlerantwort, auch wenn es leer bleibt: Sonst
    # hat dasselbe Tool je nach Ausgang zwei verschiedene Antwortschemata, und
    # ein Client, der die Antwort gegen ein festes Schema prüft, scheitert
    # ausgerechnet im Fehlerfall.
    return {
        "fehler": fehler,
        "kandidaten": [],
        "auswahl_begruendung": "",
        "ausgewaehlte_docs": [],
        "token_budget": {"verwendet": 0, "maximum": MAX_CONTEXT_TOKENS},
        "antwort": "",
        "quellen": [],
        "zeiten": {},
    }


# ---------------------------------------------------------------------------
# Keyword-Boost: direkte Textsuche für Eigennamen / Modulkürzel
# ---------------------------------------------------------------------------


def _keyword_boost_urls(frage: str, existing_urls: set[str],
                        collection: str = COLLECTION_NAME) -> list[str]:
    """Findet Source-URLs die einen Eigennamen aus der Frage per exaktem
    Qdrant-Scroll enthalten und noch nicht im Kandidatenpool sind.

    Eigennamen-Heuristik: Wörter mit Bindestrichen oder Großbuchstaben-
    Sequenzen (z.B. 'Start-ING', 'EIT-119', 'IIM').
    """
    tokens = re.findall(r'\b(?:[A-Za-zÄÖÜäöü]+-[A-Za-zÄÖÜäöü]+|[A-ZÄÖÜ]{2,})\b', frage)
    tokens = [t for t in tokens if len(t) >= 4]
    if not tokens:
        return []

    new_urls: list[str] = []
    seen: set[str] = set(existing_urls)

    for token in tokens[:3]:
        hits, _ = _qdrant.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="text", match=models.MatchText(text=token))]
            ),
            limit=20,
            with_payload=True,
            with_vectors=False,
        )
        for h in hits:
            url = (h.payload or {}).get("source_url", "")
            if url and url not in seen:
                seen.add(url)
                new_urls.append(url)

    return new_urls


# ---------------------------------------------------------------------------
# Tool: frag_hsh
# ---------------------------------------------------------------------------

# PO-Pfad (Studiengang, Fassung, Vielfalt, Schranke, Vergleich, Rückfrage,
# Kopfzeilen) liegt in raghsh/po/pfad.py. Er braucht Qdrant, Embedder und den
# LLM-Aufruf dieses Moduls und holt sie zur Aufrufzeit über verbinden().
from raghsh.po import pfad as _po_pfad
from raghsh.po.pfad import (  # noqa: E402  PO-Pfad, siehe docs/po_logik.md
    PO_SCHRANKE_RANG,
    _fakultaet_aus_frage,
    _fakultaet_des_studiengangs,
    _po_aus_frage,
    _po_auswahl_ergaenzen,
    _po_entscheiden,
    _po_geltende,
    _po_gewaehlt_zeile,
    _po_hinweis,
    _po_kopfzeile,
    _po_plan,
    _po_rueckfrage,
    _po_schranke_rang,
    _po_tabellenurteil,
    _po_url_fassungen,
    _po_vielfalt,
    _studiengang_familie_aus_frage,
    _studiengang_gewaehlt,
    _studiengang_mehrdeutig,
    _studiengang_rueckfrage,
    _urls_des_studiengangs,
)

_po_pfad.verbinden(sys.modules[__name__])


def _filter_und(a: models.Filter | None, b: models.Filter | None) -> models.Filter | None:
    """Zwei Qdrant-Filter UND-verknüpfen (None = kein Filter)."""
    if a is None or b is None:
        return a or b
    return models.Filter(must=[a, b])


def _experiment_aufteilen(experiment: dict | None) -> tuple[dict, int, bool, dict, dict | None]:
    """→ (Suchexperiment, top_docs, keyword_boost, wirksame Konfiguration, PO-Plan).

    `top_docs`, `keyword_boost` und `po` wertet der Server aus,
    alles andere prüft hybrid_search.wirksame_konfiguration. Beides zusammen
    ist die wirksame Konfiguration, die im Ergebnis und damit im Manifest der
    Evaluation steht. Ohne `po` ist der PO-Plan None und der Codepfad der
    bisherige.
    """
    experiment = dict(experiment or {})
    top_docs = int(experiment.pop("top_docs", TOP_DOCS))
    if not 1 <= top_docs <= 30:
        raise ValueError(f"top_docs muss zwischen 1 und 30 liegen: {top_docs}")
    keyword_boost = bool(experiment.pop("keyword_boost", True))
    plan = _po_plan(experiment.pop("po", None))
    suche = experiment                                  # Rest prüft hybrid_search
    wirksam = wirksame_konfiguration(suche)
    wirksam.update({"RAG_CANDIDATE_K": RAG_CANDIDATE_K, "TOP_DOCS": top_docs,
                    "FAKULTAET_FAKTOR": 2,
                    "keyword_boost": keyword_boost, "SELECT_N": SELECT_N,
                    "MAX_CONTEXT_TOKENS": MAX_CONTEXT_TOKENS})
    if plan:
        wirksam["po"] = {"studiengang": plan["studiengang"], "fassung": plan["fassung"]}
    return suche, top_docs, keyword_boost, wirksam, plan


async def _kandidaten_bestimmen(frage: str, fakultaet: str, ctx: Context,
                                experiment: dict | None = None,
                                eigene_urls: set[str] | None = None) -> dict:
    """Schritte 1 und 2 der Pipeline: Hybrid-Suche und Kandidatenliste.

    Herausgelöst aus `frag_hsh`, damit die Evaluation (`nur_retrieval`) exakt
    denselben Code durchläuft wie der Volllauf — Fakultätsarme, Dedup auf
    Dokumente, Keyword-Boost, Fakultäts-Sicherheitsnetz. Eine Nachbildung
    über `hybrid_search` allein sähe die Dedup/Limit-Verluste nicht, an denen
    im Lauf vom 18.08.2026 zwei Fälle scheiterten.

    Gibt im Fehlerfall das Ergebnis von `_empty_result` zurück (erkennbar an
    `fehler`), sonst die Zwischenstände, die `frag_hsh` weiterverarbeitet.

    `experiment` überschreibt Arme, Fusion, Limits, top_docs und Keyword-Boost
    je Aufruf (Evaluation); ohne Angabe gelten die Produktionswerte.
    """
    suche_exp, top_docs_exp, keyword_boost, wirksam, po_plan = _experiment_aufteilen(experiment)
    po_flt = po_plan["filter"] if po_plan else None
    faculty_label = next(
        (label for key, label in FACULTIES if key == fakultaet),
        "Alle Fakultäten",
    )
    # Mit Fakultät laufen zwei Suchen (Fakultät + zentral), die Liste wird
    # halb/halb gemischt und dafür verlängert. Faktor 2 statt 3 seit
    # 21.09.2026: Bei top_docs = 12 ergab ×3 eine Liste von 36 Dokumenten, in
    # der Runde 1 das Gold auf Rang 4 nicht mehr wählte (eval_gold q17); mit
    # 24 hatte sie es gewählt. 24 = die Länge, die bei ×3 und top_docs = 8 galt.
    candidate_k = RAG_CANDIDATE_K * 2 if fakultaet else RAG_CANDIDATE_K
    top_docs    = top_docs_exp * 2    if fakultaet else top_docs_exp

    # Zeitmessung der Pipeline-Phasen (perf_counter = monotone Uhr in Sekunden).
    _t = time.perf_counter
    t_start = _t()

    # ── Schritt 1: Hybrid-Suche ───────────────────────────────────────────
    # Bei gewählter Fakultät filtert Qdrant bereits WÄHREND des Retrievals, in
    # zwei Armen (Fakultät + zentrale Seiten). Begründung samt Messung steht
    # über _mische_kandidaten().
    await _fortschritt(ctx, 1, "Hybrid-Suche läuft…")
    if fakultaet:
        fak_bed = [models.FieldCondition(
            key="faculty_key", match=models.MatchValue(value=fakultaet))]
        if eigene_urls:
            # Dokumente des erkannten Studiengangs liegen teils unter einer
            # anderen Fakultät (Vertreter-URLs, gemeinsame Ordnungen): sie
            # gehören trotzdem in den Fakultätsarm (S043).
            fak_bed.append(models.FieldCondition(
                key="source_url", match=models.MatchAny(any=sorted(eigene_urls))))
        nur_fak = _filter_und(models.Filter(should=fak_bed), po_flt)
        nur_zentral = _filter_und(models.Filter(must=[models.FieldCondition(
            key="faculty_key", match=models.MatchValue(value=""))]), po_flt)
        treffer_fak = await asyncio.to_thread(
            perform_hybrid_search, _qdrant, _embedder, _sparse_embedder, frage,
            candidate_k, nur_fak, suche_exp)
        treffer_zentral = await asyncio.to_thread(
            perform_hybrid_search, _qdrant, _embedder, _sparse_embedder, frage,
            candidate_k, nur_zentral, suche_exp)
        results = treffer_fak + treffer_zentral
    else:
        treffer_fak, treffer_zentral = [], []
        results = await asyncio.to_thread(
            perform_hybrid_search,
            _qdrant, _embedder, _sparse_embedder, frage,
            candidate_k, po_flt, suche_exp,
        )
    gc.collect()
    t_search = _t()
    if not results:
        return _empty_result("Keine passenden Dokumente in der Wissensdatenbank.")

    # ── Schritt 2: Kandidaten bestimmen ──────────────────────────────────
    if fakultaet:
        candidate_urls = _mische_kandidaten(treffer_fak, treffer_zentral, top_docs,
                                            bool(eigene_urls))
    else:
        candidate_urls = _deduplicate_to_docs(results, top_docs)

    # Keyword-Boost: Eigennamen / Modulkürzel per Scroll nachladen und direkt
    # in die Kandidatenliste einmischen — unabhängig vom Hybrid-Suchergebnis.
    if keyword_boost:
        boost_urls = await asyncio.to_thread(_keyword_boost_urls, frage, set(candidate_urls),
                                             wirksam.get("COLLECTION_NAME", COLLECTION_NAME))
    else:
        boost_urls = []
    candidate_urls = candidate_urls + boost_urls

    if po_plan:
        # Die Suche ist bereits gefiltert; der Keyword-Boost kommt aus einem
        # ungefilterten Scroll und wird hier nachgefiltert.
        andere = po_plan["andere"]
        boost_urls = [u for u in boost_urls if u not in andere]
        candidate_urls = [u for u in candidate_urls if u not in andere]

    docs_info      = _lookup_documents(candidate_urls)

    # Rückfall für Dokumente, die der Index nicht kennt: alles, was der
    # Dokumentindex sonst liefert, steht auch in der Qdrant-Payload. Fehlten
    # hier `title_anzeige` und `crawl_date`, träte der Unterschied erst weit
    # später zutage — als PDF-Dateiname in der Kandidatenliste und als leeres
    # Datum in der Quellenangabe.
    for url in candidate_urls:
        if url not in docs_info:
            for r in results:
                if (r["payload"] or {}).get("source_url") == url:
                    p = r["payload"] or {}
                    docs_info[url] = {
                        "title":          p.get("title", url),
                        "title_anzeige":  p.get("title_anzeige", ""),
                        "crawl_date":     p.get("crawl_date", ""),
                        "faculty":        p.get("faculty_key")
                                          or _faculty_key(p.get("faculty", "")),
                        "headings":       "[]",
                        "chunk_ids":      "[]",
                        "chunk_count":    0,
                    }
                    break

    if fakultaet:
        # Sicherheitsnetz für die Keyword-Boost-URLs: die kommen aus einem
        # ungefilterten Scroll und können jeder Fakultät angehören. Die Treffer
        # der beiden Suchen sind bereits gefiltert und passieren ohnehin.
        candidate_urls = [
            url for url in candidate_urls
            if not docs_info.get(url, {}).get("faculty")
            or docs_info.get(url, {}).get("faculty") == fakultaet
        ]
        if not candidate_urls:
            return _empty_result(f"Keine Dokumente der {faculty_label} gefunden.")

    kandidaten_out = [
        {
            "nr":        i + 1,
            "titel":     (docs_info.get(url, {}).get("title_anzeige")
                          or docs_info.get(url, {}).get("title") or url),
            "url":       url,
            "fakultaet": docs_info.get(url, {}).get("faculty", ""),
            "struktur":  _format_heading_tree(
                             docs_info.get(url, {}).get("headings", "[]")
                         ),
        }
        for i, url in enumerate(candidate_urls)
    ]

    t_candidates = _t()
    await _fortschritt(
        ctx, 2, f"{len(results)} Chunks → {len(candidate_urls)} Kandidaten"
    )


    return {
        "results":        results,
        "candidate_urls": candidate_urls,
        "boost_urls":     boost_urls,
        "docs_info":      docs_info,
        "kandidaten_out": kandidaten_out,
        "faculty_label":  faculty_label,
        "t_start":        t_start,
        "t_search":       t_search,
        "t_candidates":   t_candidates,
        "konfiguration":  wirksam,
    }


@mcp.tool()
async def frag_hsh(
    frage: str,
    modell: str,
    ctx: Context,
    fakultaet: str = "",
    speichern: bool = False,
    passwort: str = "",
    mit_chunks: bool = False,
    nur_retrieval: bool = False,
    experiment: dict | None = None,
    po: dict | None = None,
) -> dict:
    """
    Beantwortet eine Frage zur Hochschule Hannover via zweistufige RAG-Pipeline.

    Args:
        frage:     Nutzeranfrage auf Deutsch.
        modell:    GWDG-Modell-ID (siehe Resource hsh://models) oder eine der
                   kostenpflichtigen Premium-Modell-IDs (siehe OPENROUTER_MODELS).
        fakultaet: Optional — Filterung: f1…f5 oder leer für alle Fakultäten.
                   Leer und die Frage nennt genau eine Fakultät („Fakultät I",
                   „F4", „Wirtschaft und Informatik") → diese gilt; das
                   Ergebnis meldet sie unter `fakultaet_erkannt`.
        speichern: Einwilligung zur Datenspeicherung (DSGVO). Wenn True, werden
                   Frage und Antwort für maximal 12 Monate gespeichert. Standard: False.
        passwort:  Erforderlich für Premium-Modelle (OpenRouter). Wird für die
                   kostenlosen GWDG-Modelle ignoriert.
        mit_chunks: Liefert zusätzlich die gefundenen Qdrant-Chunks mit Rang,
                   Score und Textauszug zurück. Für die Evaluation gedacht
                   (auswertung/gold/), nicht für die Weboberfläche.
        nur_retrieval: Bricht nach Hybrid-Suche und Kandidatenliste ab — kein
                   LLM-Aufruf, `modell`, `speichern` und `passwort` werden
                   ignoriert. Liefert `kandidaten`, `chunks` und `zeiten`;
                   die Antwortfelder bleiben leer. Für die Evaluation des
                   Retrievals (auswertung/gold/lauf.py --nur-retrieval), damit
                   eine Konfigurationsänderung in Sekunden statt Minuten
                   gemessen werden kann.
        experiment: Überschreibt die Suchkonfiguration für diesen Aufruf:
                   dense_arms (aus text, questions), sparse_arms (aus sparse,
                   sparse_expanded), fusion (dbsf | rrf), candidate_limit,
                   dedup_buffer, max_per_url, top_docs,
                   keyword_boost und po ({"studiengang": "BIS", "fassung": 2018}).
                   Unzulässige Werte sind ein Fehler. Das Ergebnis trägt die
                   wirksame Konfiguration unter `konfiguration`. Für die
                   Evaluation; ohne Angabe gelten die Produktionswerte.
        po:        Optional — Prüfungsordnung, für die die Antwort gelten soll,
                   z. B. {"studiengang": "BIS", "fassung": 2018}. Andere
                   Fassungen desselben Studiengangs werden aus der Suche
                   ausgeschlossen (E006). Unbekannter Studiengang oder unbekannte Fassung →
                   `fehler` mit den vorhandenen Studiengängen bzw. Fassungen.
                   Den Gesprächszustand hält der Client; der Server bleibt
                   zustandslos. Hat Vorrang vor `experiment.po`.
                   Leer und die Frage nennt Studiengang und Fassung („BIS PO
                   2016“) oder einen Neubeginn („ich fange jetzt an“ → jüngste
                   aktuelle Fassung) → diese gilt; gemeldet unter `po_erkannt`.
                   Nur {"studiengang": "BBA"} ohne Fassung (Wahl aus der
                   Rückfrage nach dem Studiengang, S085) legt den Studiengang
                   fest; die Fassung gilt dann aus der Frage oder kommt per
                   PO-Rückfrage.

    Returns:
        Strukturiertes Ergebnis:
          kandidaten        — Dokumente aus der Hybrid-Suche
          auswahl_begruendung — LLM-Antwort aus Runde 1 (Dokumentauswahl)
          ausgewaehlte_docs — Vom LLM gewählte Dokumente
          token_budget      — Verwendete / maximale Token
          antwort           — Generierte Antwort
          quellen           — Verwendete Quelldokumente
          chunks            — nur bei mit_chunks=True: die gefundenen Chunks
          po_vielfalt       — mehrere PO-Fassungen desselben Studiengangs in
                              den Kandidaten: {studiengang, fassungen,
                              je_fassung: {Fassung: [Kandidatenrang, …]},
                              chunks: {Fassung: [Punkt-ID, …]}}, sonst null
          po_entscheidung   — bei Vielfalt ohne `po`: {urteil: verschieden|zusatzbedingung|
                              gleich|unklar|eine_geltende|schranke, quelle: tabelle|vergleich|
                              schranke|geltung, tabelle (Tabellenurteil gleich|verschieden|keins,
                              S088), punkt/beleg_url (nur quelle „tabelle“), verglichen, werte {Fassung: Wert},
                              zusatz {Fassung: Bedingung}, nicht_verglichen, abgelaufen,
                              dauer_s, versuche, grund?} — grund „zeit“ (über PO_VERGLEICH_GRENZE_S)
                              oder „fehler:<Art>“, Urteil dann „unklar“; „eine_geltende“:
                              nach Abzug der abgelaufenen Fassungen (S033) bleibt
                              höchstens eine, kein Vergleich; „schranke“ (S064): kein
                              PO-Dokument des Studiengangs unter den ersten
                              PO_SCHRANKE_RANG Kandidaten, kein Vergleich (schranke_rang)
          rueckfrage        — bei urteil „verschieden“: {frage, optionen:
                              [{studiengang, name, fassung, aktuell}]}; bei
                              studiengang_mehrdeutig (S085): {frage, optionen:
                              [{studiengang, name}]}; antwort ist dann leer
          studiengang_mehrdeutig — Kürzel zu einer mehrdeutigen Kurzform der Frage
                              („BWL im Bachelor“), nur ohne erkannten Studiengang und
                              mit Abschluss/PO in der Frage (S080); löst ohne
                              `po` die Rückfrage nach dem Studiengang aus (S085)
          fehler            — Fehlermeldung oder null
    """
    # ── Modell validieren ─────────────────────────────────────────────────
    if nur_retrieval:
        pass                                    # kein Modell nötig
    elif modell in OPENROUTER_MODELS:
        # Premium-Modell: nur mit korrektem Passwort und vorhandenem Key.
        # Die Passwortprüfung erfolgt serverseitig, damit der kostenpflichtige
        # Key auch bei direktem Zugriff auf den MCP-Endpunkt geschützt bleibt.
        if not PREMIUM_OHNE_PASSWORT and (not FULL_ACCESS_PASSWORD or passwort != FULL_ACCESS_PASSWORD):
            return _empty_result(
                "Dieses Premium-Modell ist passwortgeschützt. "
                "Bitte geben Sie das korrekte Passwort ein, um kostenpflichtige "
                "Modelle (Claude, Gemini, ChatGPT) zu nutzen."
            )
        if _openrouter_client is None:
            return _empty_result(
                "Premium-Modell nicht verfügbar — auf dem Server ist kein "
                "OpenRouter-API-Key hinterlegt."
            )
    else:
        available = await asyncio.to_thread(_get_models)
        if modell not in available:
            return _empty_result(
                f"Unbekanntes Modell '{modell}'. Verfügbar: {available}"
            )

    # Nennt die Frage eine Fakultät und hat der Aufrufer keine gewählt, gilt die
    # genannte. `fakultaet_erkannt` sagt im Ergebnis, ob das passiert ist.
    fakultaet_erkannt = ""
    if not fakultaet:
        fakultaet_erkannt = _fakultaet_aus_frage(frage)
        fakultaet = fakultaet_erkannt

    # `po` als eigener Parameter (S021) läuft über denselben Plan wie
    # `experiment.po` (S011); geprüft wird vorab, damit eine ungültige Angabe
    # als PO-Meldung und nicht als „Experiment abgelehnt“ zurückkommt.
    # Ohne `po` (auch nicht in `experiment`) gilt eine in der Frage genannte
    # PO (S022); `po_erkannt` sagt im Ergebnis, ob das passiert ist.
    # `po_angabe` ist die wirksame Angabe aus Parameter, Experiment oder Frage.
    # Die Studiengangsfamilie aus der Frage (S032) wird einmal bestimmt und an
    # alle Stellen durchgereicht, die sie brauchen.
    # `po` nur mit Studiengang (Wahl aus der Studiengangs-Rückfrage, S085)
    # ersetzt die Erkennung aus der Frage; die Fassung bleibt offen.
    if po and "fassung" not in po:
        try:
            familie = [_studiengang_gewaehlt(po.get("studiengang"))]
        except ValueError as exc:
            return _empty_result(f"Ungültige PO-Angabe: {exc}")
        po = None
    else:
        familie = _studiengang_familie_aus_frage(frage)
    studiengang_mehrdeutig = _studiengang_mehrdeutig(frage, familie)
    po_angabe = po or (experiment or {}).get("po")
    po_erkannt = None
    if not po_angabe:
        po_erkannt = _po_aus_frage(frage, familie)
        po = po_angabe = po_erkannt
    if po:
        try:
            _po_plan(po)
        except ValueError as exc:
            return _empty_result(f"Ungültige PO-Angabe: {exc}")
        experiment = {**(experiment or {}), "po": po}

    # Erkannter Studiengang ohne gewählte Fakultät (S043): Die Suche bleibt
    # bei seiner Fakultät (plus zentrale Seiten). Der PO-Filter
    # schließt nur die anderen Fassungen aus und ließ sonst alle Fakultäten
    # zu — bei MAT (f1) landeten so Merkblätter der Fakultät IV vorn.
    po_studiengang = (po_angabe or {}).get("studiengang")
    if not fakultaet:
        fakultaet_erkannt = _fakultaet_des_studiengangs(po_studiengang, familie)
        fakultaet = fakultaet_erkannt
    eigene_urls = _urls_des_studiengangs(po_studiengang, familie)

    _t = time.perf_counter
    try:
        k = await _kandidaten_bestimmen(frage, fakultaet, ctx, experiment, eigene_urls)
    except ValueError as exc:                          # ungültiges Experiment
        return _empty_result(f"Experiment abgelehnt: {exc}")
    if k.get("fehler"):
        return k
    # Familie ohne Auflösung (S032: gleicher Name, z. B. Bachelor und Master,
    # Frage nennt keinen Abschluss): Steht genau ein Mitglied mit PO in den
    # Kandidaten, gilt die PO aus der Frage für dieses — zweite Suche mit Filter.
    if not po_angabe and len(familie) > 1:
        zuordnung = _po_url_fassungen()
        in_kandidaten = {sg for kd in k["kandidaten_out"] for sg in zuordnung.get(kd["url"], {})}
        po_erkannt = _po_aus_frage(frage, familie, in_kandidaten)
        if po_erkannt:
            po = po_angabe = po_erkannt
            experiment = {**(experiment or {}), "po": po}
            t_erste = k["t_start"]
            try:
                k = await _kandidaten_bestimmen(frage, fakultaet, ctx, experiment, eigene_urls)
            except ValueError as exc:
                return _empty_result(f"Experiment abgelehnt: {exc}")
            if k.get("fehler"):
                return k
            k["t_start"] = t_erste
    results, candidate_urls, boost_urls = k["results"], k["candidate_urls"], k["boost_urls"]
    docs_info, kandidaten_out = k["docs_info"], k["kandidaten_out"]
    t_start, t_search, t_candidates = k["t_start"], k["t_search"], k["t_candidates"]
    # Mehrere PO-Fassungen desselben Studiengangs in den Kandidaten? (S026)
    # Mit `po` sind die anderen Fassungen weggefiltert.
    po_vielfalt = _po_vielfalt(familie, kandidaten_out, results)

    if nur_retrieval:
        # Evaluation des Retrievals ohne LLM: nach der Kandidatenliste ist
        # Schluss. Die Antwortfelder bleiben leer, das Schema bleibt dasselbe.
        # Dazu jeder Arm einzeln (Top-50, ohne Fusion): Welcher Arm trägt das
        # Gold, und sieht es überhaupt einer? Das beantwortet die Frage, die
        # eine fusionierte Liste allein nicht beantworten kann.
        arme = await asyncio.to_thread(
            suche_je_arm, _qdrant, _embedder, _sparse_embedder, frage, 50, None,
            k["konfiguration"].get("COLLECTION_NAME", COLLECTION_NAME))
        t_ende = _t()
        ergebnis = _empty_result("")
        ergebnis.update({
            "fehler":        None,
            "kandidaten":    kandidaten_out,
            "chunks":        _chunks_fuer_diagnose(results),
            "arme":          arme,
            "konfiguration": k["konfiguration"],
            "fakultaet":     fakultaet,
            "fakultaet_erkannt": fakultaet_erkannt,
            "po_erkannt":    po_erkannt,
            "po_vielfalt":   po_vielfalt,
            "studiengang_mehrdeutig": studiengang_mehrdeutig,
            "zeiten": {
                "hybrid_suche_s": round(t_search     - t_start,  2),
                "kandidaten_s":   round(t_candidates - t_search, 2),
                "arme_s":         round(t_ende       - t_candidates, 2),
                "gesamt_s":       round(t_ende       - t_start,  2),
            },
        })
        return ergebnis

    # ── Rückfrage nach dem Studiengang (S085) ───────────────────────────
    # Mehrdeutige Kurzform im PO-Kontext, kein Studiengang erkannt und keine
    # PO angegeben: nicht raten (F4), sondern fragen. Kein LLM-Aufruf.
    if studiengang_mehrdeutig and not po_angabe:
        ergebnis = _empty_result("")
        ergebnis.update({
            "fehler":          None,
            "antwort":         "",
            "rueckfrage":      _studiengang_rueckfrage(studiengang_mehrdeutig),
            "kandidaten":      kandidaten_out,
            "konfiguration":   k["konfiguration"],
            "fakultaet":       fakultaet,
            "fakultaet_erkannt": fakultaet_erkannt,
            "po_erkannt":      po_erkannt,
            "po_vielfalt":     po_vielfalt,
            "studiengang_mehrdeutig": studiengang_mehrdeutig,
            "zeiten": {
                "hybrid_suche_s": round(t_search     - t_start,  2),
                "kandidaten_s":   round(t_candidates - t_search, 2),
                "gesamt_s":       round(_t()         - t_start,  2),
            },
        })
        return ergebnis

    # ── Rückfrage bei PO-Vielfalt (S027) ────────────────────────────────
    # Nur ohne `po` (auch nicht aus der Frage erkannt): Unterscheiden sich die
    # Fassungen im gefragten Punkt, kommt statt der Antwort eine Rückfrage;
    # der Antwortschritt entfällt. „gleich“/„unklar“ → normale Antwort.
    po_entscheidung = None
    if po_vielfalt and not po_angabe:
        # Abgelaufene Fassungen vergleichen nicht mit (S033); bleibt nur eine
        # geltende übrig, wird direkt geantwortet.
        geltend, abgelaufen = _po_geltende(po_vielfalt)
        schranke_rang = _po_schranke_rang(po_vielfalt["studiengang"], kandidaten_out)
        # S088: zuerst die Wertetabelle; sie braucht kein PO-Dokument vorn (Schranke)
        t_tabelle = _t()
        tabelle = _po_tabellenurteil(frage, po_vielfalt["studiengang"], kandidaten_out, familie)
        if tabelle["urteil"] != "keins":
            print(f"  ⏱ PO-Vergleich: tabelle {tabelle['urteil']} ({tabelle['punkt']})", flush=True)
            po_entscheidung = {"urteil": tabelle["urteil"], "quelle": "tabelle",
                               "punkt": tabelle["punkt"],
                               "verglichen": sorted(tabelle["werte"]),
                               "werte": {f: " + ".join(w) for f, w in sorted(tabelle["werte"].items())},
                               "nicht_verglichen": tabelle["ohne_zeile"],
                               "beleg_url": tabelle["beleg_url"],
                               "dauer_s": round(_t() - t_tabelle, 2)}
        elif len(geltend) >= 2 and (schranke_rang is None or schranke_rang > PO_SCHRANKE_RANG):
            # S064: kein PO-Dokument des Studiengangs vorn → kein Vergleich, normale Antwort
            print(f"  ⏱ PO-Vergleich: schranke (PO-Dokument Rang {schranke_rang})", flush=True)
            po_entscheidung = {"urteil": "schranke", "quelle": "schranke", "verglichen": [],
                               "werte": {}, "nicht_verglichen": geltend, "dauer_s": 0.0,
                               "schranke_rang": schranke_rang}
        elif len(geltend) >= 2:
            po_entscheidung = await _po_entscheiden(
                frage, {**po_vielfalt, "fassungen": geltend}, modell)
            po_entscheidung["quelle"] = "vergleich"
        else:
            po_entscheidung = {"urteil": "eine_geltende", "quelle": "geltung", "verglichen": [],
                               "werte": {}, "nicht_verglichen": geltend, "dauer_s": 0.0}
        po_entscheidung["tabelle"] = tabelle["urteil"]
        po_entscheidung["abgelaufen"] = abgelaufen
        if po_entscheidung["urteil"] == "verschieden":
            ergebnis = _empty_result("")
            ergebnis.update({
                "fehler":          None,
                "antwort":         "",
                "rueckfrage":      _po_rueckfrage(po_vielfalt),
                "kandidaten":      kandidaten_out,
                "konfiguration":   k["konfiguration"],
                "fakultaet":       fakultaet,
                "fakultaet_erkannt": fakultaet_erkannt,
                "po_erkannt":      po_erkannt,
                "po_vielfalt":     po_vielfalt,
                "po_entscheidung": po_entscheidung,
                "studiengang_mehrdeutig": studiengang_mehrdeutig,
                "zeiten": {
                    "hybrid_suche_s": round(t_search     - t_start,  2),
                    "kandidaten_s":   round(t_candidates - t_search, 2),
                    "po_vergleich_s": po_entscheidung["dauer_s"],
                    "gesamt_s":       round(_t()         - t_start,  2),
                },
            })
            return ergebnis

    # ── Schritt 3: LLM Runde 1 — Dokumentauswahl ─────────────────────────
    await _fortschritt(ctx, 3, f"LLM wählt Dokumente aus ({modell})…")
    doc_list     = _build_document_list(docs_info, candidate_urls, boost_urls=set(boost_urls))
    po_wirksam   = k["konfiguration"].get("po")          # gewählt oder erkannt (S048)
    po_zeile     = _po_gewaehlt_zeile(po_wirksam)
    frage_mit_po = f"{frage}\n{po_zeile}" if po_zeile else frage
    sel_messages = [
        {"role": "system", "content": SELECTION_SYSTEM},
        {"role": "user",   "content": f"Frage: {frage_mit_po}\n\nDokumente:\n\n{doc_list}"},
    ]
    try:
        sel_response, _ = await asyncio.to_thread(_ask_llm, modell, sel_messages)
    except RuntimeError as exc:
        return {
            **_empty_result(f"Fehler bei der Dokumentauswahl (LLM Runde 1): {exc}"),
            "kandidaten": kandidaten_out,
        }
    t_select = _t()

    selected_indices = _parse_selected_indices(sel_response, len(candidate_urls))
    selected_indices = _po_auswahl_ergaenzen(selected_indices, candidate_urls, po_wirksam)
    # S088: Tabellenurteil „gleich“ → die Fundstelle der Tabelle ist Beleg der Antwort;
    # fehlt sie in der Auswahl, kommt sie an den Anfang (wie _po_auswahl_ergaenzen)
    beleg_url = (po_entscheidung or {}).get("beleg_url")
    if beleg_url in candidate_urls and candidate_urls.index(beleg_url) + 1 not in selected_indices:
        selected_indices = [candidate_urls.index(beleg_url) + 1] + selected_indices
    if not selected_indices:
        return {
            **_empty_result(
                "Die Dokumentenauswahl hat keine passenden Dokumente identifiziert. "
                "Bitte formulieren Sie die Frage um oder fassen Sie sie genauer."
            ),
            "kandidaten": kandidaten_out,
            "auswahl_begruendung": sel_response,
        }
    selected_urls    = [candidate_urls[i - 1] for i in selected_indices]

    ausgewaehlte_out = [
        {
            "nr":    idx,
            "titel": (docs_info.get(url, {}).get("title_anzeige")
                      or docs_info.get(url, {}).get("title") or url),
            "url":   url,
        }
        for idx, url in zip(selected_indices, selected_urls)
    ]

    # ── Schritt 4: Vollständige Dokumente aus Qdrant laden ────────────────
    await _fortschritt(ctx, 4, "Lade vollständige Dokumente aus Qdrant…")
    selected_docs = []
    for url in selected_urls:
        info      = docs_info.get(url, {})
        chunk_ids = json.loads(info.get("chunk_ids", "[]"))
        doc_text, fremde = await asyncio.to_thread(_load_document_text, url, chunk_ids)

        if not doc_text:
            fremde   = []
            fallback = [
                (r["payload"] or {}).get("text", "")
                for r in results
                if (r["payload"] or {}).get("source_url") == url
            ]
            doc_text = "\n\n".join(t for t in fallback if t)

        if doc_text:
            selected_docs.append({"url": url, "info": info, "text": doc_text,
                                  "fremde_fundstellen": fremde})

    if not selected_docs:
        return {
            **_empty_result(
                "Die ausgewählten Dokumente konnten nicht geladen werden. "
                "Möglicherweise sind die Inhalte nicht mehr im Index vorhanden."
            ),
            "kandidaten": kandidaten_out,
            "auswahl_begruendung": sel_response,
            "ausgewaehlte_docs": ausgewaehlte_out,
        }

    selected_docs, tokens_used = _fit_to_budget(selected_docs)
    t_load = _t()

    # ── Schritt 5: LLM Runde 2 — Antwort generieren ──────────────────────
    await _fortschritt(ctx, 5, "Generiere Antwort…")
    context_blocks = []
    for i, doc in enumerate(selected_docs, 1):
        info  = doc["info"]
        lines = [
            # title_anzeige statt title: Bei PDFs ist `title` der Dateiname
            # ("09-PO_MGB_BT_11_09_2008") und für das Modell wertlos.
            f"[Dokument {i}] {info.get('title_anzeige') or info.get('title') or doc['url']}",
            f"URL: {doc['url']}",
        ]
        if info.get("faculty"):
            lines.append(f"Fakultät: {info['faculty']}")
        if info.get("crawl_date"):
            lines.append(f"Stand: {info['crawl_date']}")
        if po_kopf := _po_kopfzeile(doc["url"]):
            lines.append(po_kopf)
        lines.append("")
        lines.append(doc["text"])
        context_blocks.append("\n".join(lines))
    context = "\n\n---\n\n".join(context_blocks)
    # Gleicher Wert, aber Zusatzbedingung in einer Fassung (S039): Hinweis
    # als Kontext für die Antwort, keine Regel im Systemprompt.
    hinweis = _po_hinweis(po_vielfalt["studiengang"], po_entscheidung) if po_vielfalt else ""
    if hinweis:
        context = f"{hinweis}\n\n---\n\n{context}"

    ans_messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user",   "content": (
            f"Kontext aus den offiziellen HsH-Dokumenten:\n\n{context}\n\n"
            f"---\n\nFrage: {frage_mit_po}"
        )},
    ]
    try:
        answer, _ = await asyncio.to_thread(_ask_llm, modell, ans_messages)
    except RuntimeError as exc:
        return {
            **_empty_result(f"Fehler bei der Antwortgenerierung (LLM Runde 2): {exc}"),
            "kandidaten":          kandidaten_out,
            "auswahl_begruendung": sel_response,
            "ausgewaehlte_docs":   ausgewaehlte_out,
        }

    t_answer = _t()

    # Falsche Quellenmarken deterministisch umsetzen (S041); Log nur mit Nummern.
    answer, marken_korrektur = _marken_korrigieren(
        answer,
        [f"{d['info'].get('title_anzeige') or d['info'].get('title') or ''}\n{d['text']}"
         for d in selected_docs],
        frage)
    if marken_korrektur:
        print("  ✎ Quellenmarken umgesetzt: " + ", ".join(
            f"{k['art']} {k['von']}→[{k['nach']}]" for k in marken_korrektur), flush=True)

    # Phasen-Zeiten (Sekunden). "quellen_s" = Zeit bis die Quellen feststehen
    # und geladen sind (Hybrid-Suche + Kandidaten + LLM-Auswahl + Volltext laden),
    # also alles VOR der Antwortgenerierung.
    zeiten = {
        "hybrid_suche_s":   round(t_search     - t_start,      2),
        "kandidaten_s":     round(t_candidates - t_search,     2),
        "llm_auswahl_s":    round(t_select     - t_candidates, 2),
        "docs_laden_s":     round(t_load       - t_select,     2),
        "quellen_s":        round(t_load       - t_start,      2),
        "antwort_llm_s":    round(t_answer     - t_load,       2),
        "gesamt_s":         round(t_answer     - t_start,      2),
    }
    print(
        f"  ⏱ Zeiten [{modell}]: Quellen {zeiten['quellen_s']}s "
        f"(Suche {zeiten['hybrid_suche_s']}s · Kandidaten {zeiten['kandidaten_s']}s · "
        f"LLM-Auswahl {zeiten['llm_auswahl_s']}s · Laden {zeiten['docs_laden_s']}s) "
        f"· Antwort {zeiten['antwort_llm_s']}s · Gesamt {zeiten['gesamt_s']}s",
        flush=True,
    )

    cite_data = _extract_verified_snippets(answer, selected_docs, results)

    quellen_out = [
        {
            "titel":       (doc["info"].get("title_anzeige")
                            or doc["info"].get("title") or doc["url"]),
            "url":         doc["url"],
            "crawl_date":  doc["info"].get("crawl_date", ""),
            # Absätze, die wortgleich unter einer anderen URL im Index liegen
            # und von dort geholt wurden. Gehört in die Quellenangabe: Der
            # Wortlaut stammt aus einem gleichlautenden Dokument, oft einer
            # anderen Fakultät oder einem anderen Jahrgang.
            "fremde_fundstellen": doc.get("fremde_fundstellen", []),
            "abschnitt":   cite_data.get(doc["url"], {}).get("abschnitt", ""),
            "snippets":    cite_data.get(doc["url"], {}).get("snippets", []),
            "zitate":      cite_data.get(doc["url"], {}).get("zitate", []),
            "verifiziert": cite_data.get(doc["url"], {}).get("verifiziert", False),
        }
        for doc in selected_docs
    ]
    if speichern:
        _log_qa(frage, modell, fakultaet, answer, quellen_out, tokens_used)

    ergebnis = {
        "fehler":             None,
        "kandidaten":         kandidaten_out,
        "auswahl_begruendung": sel_response,
        "ausgewaehlte_docs":  ausgewaehlte_out,
        "token_budget":       {"verwendet": tokens_used, "maximum": MAX_CONTEXT_TOKENS},
        "antwort":            answer,
        "quellen":            quellen_out,
        "zeiten":             zeiten,
        "konfiguration":      k["konfiguration"],
        "fakultaet":          fakultaet,
        "fakultaet_erkannt":  fakultaet_erkannt,
        "po_erkannt":         po_erkannt,
        "po_vielfalt":        po_vielfalt,
        "po_entscheidung":    po_entscheidung,
        "studiengang_mehrdeutig": studiengang_mehrdeutig,
        "marken_korrektur":   marken_korrektur,
    }
    if mit_chunks:
        ergebnis["chunks"] = _chunks_fuer_diagnose(results)
    return ergebnis


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

def main() -> None:
    # Bereitschaftssignal beim Prozessstart, nicht in der Lifespan.
    #
    # hsh-webapp.service wartet in ExecStartPre bis zu zehn Minuten auf diese
    # Datei. Entstünde sie erst in der Lifespan, entstünde sie erst mit der
    # ersten Client-Session — und der erste Client IST die Webapp. Sie wartete
    # damit auf sich selbst und startete erst nach dem Timeout. Die Datei sagt
    # deshalb "der Prozess läuft", nicht "die Modelle sind geladen"; letzteres
    # dauert nach dem ersten Verbindungsaufbau rund drei Sekunden.
    atexit.register(_ressourcen_freigeben)
    READY_FILE.write_text("ready")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
