"""
HsH-Web-App (MCP) — Streamlit-Oberfläche via MCP-Server der Hochschule Hannover.

Kommuniziert ausschließlich über das MCP-Protokoll (Streamable HTTP, Spec 2025-03-26)
mit dem HsH RAG MCP Server. Keine lokalen Abhängigkeiten (kein Qdrant, kein Embedder).

Aufruf:
    streamlit run web/app.py

Abhängigkeiten:
    pip install streamlit requests
"""

import html
import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# HsH-Logo — einmalig vom Dateisystem laden (inline SVG)
# ---------------------------------------------------------------------------

def _load_logo() -> str:
    path = Path(__file__).parent / "static" / "hsh_logo.svg"
    try:
        svg = path.read_text(encoding="utf-8")
        # XML-Deklaration und DOCTYPE entfernen (ungültig in inline-HTML)
        svg = re.sub(r'<\?xml[^?]*\?>', '', svg)
        svg = re.sub(r'<!DOCTYPE[^>]*>', '', svg)
        svg = svg.strip()
        # width/height-Attribute entfernen — Größe wird per CSS gesteuert
        svg = re.sub(r'\s+width="[^"]*"', '', svg, count=1)
        svg = re.sub(r'\s+height="[^"]*"', '', svg, count=1)
        # CSS-Klasse und ARIA-Label für den <svg>-Tag ergänzen
        svg = svg.replace("<svg ", '<svg class="hsh-logo-hero" role="img" aria-label="Hochschule Hannover" ', 1)
        return svg
    except OSError:
        return ""

_LOGO_SVG = _load_logo()

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

MCP_URL_REMOTE = "https://nemo.christianwill.de/hsh/mcp"
MCP_URL_LOCAL  = "http://127.0.0.1:8765/mcp"
# Agentenversion, nur im Testmodus erreichbar (raghsh-zyklus testen); Fragen werden dort nie gespeichert.
MCP_URL_AGENT  = "http://127.0.0.1:8767/mcp"

FAKULTAETEN: dict[str, str] = {
    "Alle Fakultäten":                                        "",
    "Fakultät I — Elektro- und Informationstechnik":          "f1",
    "Fakultät II — Maschinenbau und Bioverfahrenstechnik":    "f2",
    "Fakultät III — Medien, Information und Design":          "f3",
    "Fakultät IV — Soziale Arbeit":                           "f4",
    "Fakultät V — Diakonie, Gesundheit und Soziales":         "f5",
}

SCHRITTE_ICONS = {1: "🔍", 2: "📋", 3: "🤔", 4: "📄", 5: "✍️"}

# Modell-Empfehlungen für den RAG-Einsatz
# Kriterien: Deutschkenntnisse, Instruktionstreue, Kontextverarbeitung, Geschwindigkeit
RAG_EMPFEHLUNGEN: list[dict] = [
    {
        "modell":  "glm-4.7",
        "label":   "⭐ Beste Wahl",
        "grund":   "Standard — starkes Deutsch, sehr instruktionstreu",
    },
    {
        "modell":  "llama-3.3-70b-instruct",
        "label":   "✅ Sehr gut",
        "grund":   "Bewährt, gutes Deutsch, 128 K Kontext",
    },
    {
        "modell":  "qwen3.5-27b",
        "label":   "✅ Sehr gut",
        "grund":   "256 K Kontext, starke Mehrsprachigkeit",
    },
    {
        "modell":  "deepseek-r1-distill-llama-70b",
        "label":   "✅ Sehr gut",
        "grund":   "Für komplexe Fragen — mit Denkprozess",
    },
    {
        "modell":  "qwen3.5-122b-a10b",
        "label":   "✅ Sehr gut",
        "grund":   "Höchste Qualität, etwas langsamer",
    },
    {
        "modell":  "meta-llama-3.1-8b-instruct",
        "label":   "🚀 Schnell",
        "grund":   "Einfache Anfragen, minimale Latenz",
    },
]

_MODELL_LABELS: dict[str, str] = {rec["modell"]: rec["label"] for rec in RAG_EMPFEHLUNGEN}

# Reihenfolge der Vorauswahl im Modell-Dropdown: der erste Eintrag, den der
# Server tatsächlich anbietet, wird vorausgewählt.
DEFAULT_MODELL_PRIO: tuple[str, ...] = (
    "glm-4.7",
    "llama-3.3-70b",
    "qwen3.5-122b",
)

# Kostenpflichtige Premium-Modelle über OpenRouter — nur nach Passwort-Freischaltung.
# Die IDs MÜSSEN mit OPENROUTER_MODELS in server/mcp_server.py übereinstimmen.
# Das Passwort steht in .env (nicht in git), derselbe Wert wie im MCP-Server.
# Fehlt es, lässt sich der Premium-Zugang nicht freischalten.
from raghsh import konfig  # noqa: E402
load_dotenv(konfig.ENV_DATEI)
FULL_ACCESS_PASSWORD = os.getenv("FULL_ACCESS_PASSWORD", "")
PREMIUM_MODELS: dict[str, str] = {
    "anthropic/claude-sonnet-4.5": "💎 Claude Sonnet (Anthropic)",
    "google/gemini-2.5-flash":     "💎 Gemini Flash (Google)",
    "openai/gpt-5.6-luna":         "💎 ChatGPT (OpenAI GPT-5.6 Luna)",
    "openai/gpt-6-luna":           "💎 ChatGPT (OpenAI GPT-6 Luna)",
    "z-ai/glm-5.3-flash":          "💎 GLM-5.3 Flash (DeepInfra)",
}

REQUEST_TIMEOUT = 300  # Sekunden — frag_hsh kann 30–60 s brauchen

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# ---------------------------------------------------------------------------
# Quelltextnachweis — Ähnlichkeitsmarkierung
# ---------------------------------------------------------------------------

def _highlight_snippet(snippet: str, zitate: list[str]) -> str:
    """Markiert Zitate im Snippet mit <mark>.

    Sucht zuerst exakt, dann mit normalisiertem Whitespace (LLM schreibt
    manchmal Leerzeichen statt Zeilenumbruch). Arbeitet intern auf dem
    whitespace-normalisierten Text, da beide Seiten (Zitat + Snippet)
    dieselbe Normalisierung erfahren.
    """
    if not snippet:
        return ""
    norm_snippet = re.sub(r'\s+', ' ', snippet)
    relevant = sorted(
        (
            re.sub(r'\s+', ' ', q).strip()
            for q in zitate
            if len(q) >= 10 and re.sub(r'\s+', ' ', q).strip() in norm_snippet
        ),
        key=len,
        reverse=True,  # längste zuerst verhindert Partial-Match-Überschneidungen
    )
    if not relevant:
        return html.escape(snippet).replace("\n", "<br>")

    n    = len(norm_snippet)
    mask = bytearray(n)
    for norm_q in relevant:
        idx = norm_snippet.find(norm_q)
        if idx >= 0:
            for k in range(idx, min(idx + len(norm_q), n)):
                mask[k] = 1

    parts: list[str] = []
    i = 0
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            parts.append(f'<mark class="hsh-highlight">{html.escape(norm_snippet[i:j])}</mark>')
            i = j
        else:
            j = i + 1
            while j < n and not mask[j]:
                j += 1
            parts.append(html.escape(norm_snippet[i:j]))
            i = j

    return "".join(parts).replace("\n", "<br>")


def render_source_evidence(quellen: list[dict]) -> None:
    """Zeigt verifizierte Quelltextabschnitte mit exakt markierten Originalzitaten.

    Nur Quellen mit nachgewiesenen Zitaten (verifiziert=True) werden angezeigt.
    """
    verified = [q for q in quellen if q.get("verifiziert") and q.get("snippets")]
    if not verified:
        return

    with st.expander("📑 Quelltextnachweis — verifizierte Originalzitate", expanded=False):
        st.caption(
            "Gelb markierte Stellen sind wörtliche Zitate, die in der Antwort "
            "verwendet und im Originaltext nachgewiesen wurden."
        )
        for q in verified:
            snippets = q.get("snippets", [])
            zitate   = q.get("zitate", [])
            titel    = q.get("titel", q.get("url", "—"))
            url      = q.get("url", "")
            if url:
                header = (
                    f'<a href="{url}" target="_blank" '
                    f'style="color:var(--hsh-primary);font-weight:600;text-decoration:none">'
                    f'{html.escape(titel)}</a>'
                )
            else:
                header = f'<strong>{html.escape(titel)}</strong>'
            st.markdown(header, unsafe_allow_html=True)

            for snippet in snippets:
                highlighted = _highlight_snippet(snippet, zitate)
                st.markdown(
                    f'<div class="hsh-snippet">{highlighted}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("---")


def render_answer(text: str) -> None:
    """Zeigt <think>-Blöcke eingeklappt und den Rest als Markdown."""
    thinks = _THINK_RE.findall(text)
    clean  = _THINK_RE.sub("", text).strip()
    if thinks:
        with st.expander("💭 Gedanken", expanded=False):
            for block in thinks:
                st.markdown(block.strip())
    st.markdown(clean)

# ---------------------------------------------------------------------------
# Seitenconfig — muss vor allen anderen st.*-Aufrufen stehen
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HsH-Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# MCP-Client
# ---------------------------------------------------------------------------


class MCPClient:
    """Minimaler MCP-Client für Streamable HTTP (MCP Spec 2025-03-26)."""

    def __init__(self, url: str) -> None:
        self.url        = url
        self.session_id: str | None = None
        self._req_id    = 0

    def _id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept":       "application/json, text/event-stream",
        }
        if self.session_id:
            h["mcp-session-id"] = self.session_id
        return h

    def initialize(self) -> None:
        body = {
            "jsonrpc": "2.0",
            "id":      self._id(),
            "method":  "initialize",
            "params":  {
                "protocolVersion": "2025-03-26",
                "capabilities":    {},
                "clientInfo":      {"name": "hsh-web-app-mcp", "version": "2.0"},
            },
        }
        resp = requests.post(self.url, json=body, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        for k, v in resp.headers.items():
            if k.lower() == "mcp-session-id":
                self.session_id = v
                break

    def get_kontingent(self) -> dict:
        """Verbleibende GWDG-Anfragen (Ressource hsh://kontingent); {} bei älterem Server."""
        try:
            text = self._resource_text("hsh://kontingent")
            return json.loads(text) if text else {}
        except Exception:
            return {}

    def _resource_text(self, uri: str) -> str:
        body = {
            "jsonrpc": "2.0",
            "id":      self._id(),
            "method":  "resources/read",
            "params":  {"uri": uri},
        }
        resp = requests.post(self.url, json=body, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        if "text/event-stream" in resp.headers.get("content-type", ""):
            msg: dict = {}
            for raw in resp.iter_lines():
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if line.startswith("data:"):
                    msg = json.loads(line[5:].lstrip())
                    break
        else:
            msg = resp.json()
        contents = msg.get("result", {}).get("contents", [])
        return contents[0].get("text", "") if contents else ""

    def get_models(self) -> list[str]:
        body = {
            "jsonrpc": "2.0",
            "id":      self._id(),
            "method":  "resources/read",
            "params":  {"uri": "hsh://models"},
        }
        resp = requests.post(self.url, json=body, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        if "text/event-stream" in resp.headers.get("content-type", ""):
            msg: dict = {}
            for raw in resp.iter_lines():
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if line.startswith("data:"):
                    msg = json.loads(line[5:].lstrip())
                    break
        else:
            msg = resp.json()
        contents = msg.get("result", {}).get("contents", [])
        if contents:
            return sorted(json.loads(contents[0].get("text", "[]")))
        return []

    def frag_hsh(
        self,
        frage: str,
        modell: str,
        fakultaet: str = "",
        speichern: bool = False,
        passwort: str = "",
        po: dict | None = None,
    ) -> Iterator[dict]:
        """
        Ruft frag_hsh via MCP auf. Liefert strukturierte Events:

          {"type": "progress", "step": n, "total": 5, "message": "..."}
          {"type": "result",   "data": {...}}
          {"type": "error",    "message": "..."}
        """
        req_id = self._id()
        body = {
            "jsonrpc": "2.0",
            "id":      req_id,
            "method":  "tools/call",
            "params":  {
                # Ohne progressToken sendet der Server GAR KEINE
                # Fortschritts-Notifications: FastMCP verwirft die Meldung
                # stillschweigend, wenn der Client keinen Token angefragt hat.
                # Bis zum 17.08.2026 fehlte das hier — die Fortschrittsanzeige
                # unten wartete deshalb auf Ereignisse, die nie kamen, und der
                # Balken blieb bis zum Ergebnis stehen.
                "_meta":     {"progressToken": f"frag-{req_id}"},
                "name":      "frag_hsh",
                "arguments": {
                    "frage":     frage,
                    "modell":    modell,
                    "fakultaet": fakultaet,
                    "speichern": speichern,
                    "passwort":  passwort,
                    # Nur mitschicken, wenn gewählt — ohne `po` verhält sich
                    # frag_hsh wie vor der PO-Rückfrage.
                    **({"po": po} if po else {}),
                },
            },
        }
        with requests.post(
            self.url,
            json=body,
            headers=self._headers(),
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            if "text/event-stream" in resp.headers.get("content-type", ""):
                yield from _parse_sse(resp.iter_lines())
            else:
                yield _jsonrpc_to_event(resp.json())


def _parse_sse(lines) -> Iterator[dict]:
    """Parst SSE-Bytestream und liefert strukturierte Events."""
    buf: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if line.startswith("data:"):
            buf.append(line[5:].lstrip())
        elif not line and buf:
            text = "\n".join(buf)
            buf = []
            try:
                yield _jsonrpc_to_event(json.loads(text))
            except json.JSONDecodeError:
                pass


def _jsonrpc_to_event(msg: dict) -> dict:
    """Wandelt eine JSON-RPC-Nachricht in ein strukturiertes Event um."""
    if msg.get("method") == "notifications/progress":
        p = msg.get("params", {})
        return {
            "type":    "progress",
            "step":    p.get("progress", 0),
            "total":   p.get("total", 5),
            "message": p.get("message", ""),
        }
    if "error" in msg:
        e = msg["error"]
        return {"type": "error", "message": e.get("message", str(e))}
    if "result" in msg:
        result   = msg["result"]
        contents = result.get("content", [])
        if contents and contents[0].get("type") == "text":
            try:
                return {"type": "result", "data": json.loads(contents[0]["text"])}
            except json.JSONDecodeError:
                pass
        return {"type": "result", "data": result}
    return {"type": "unknown", "raw": msg}


# ---------------------------------------------------------------------------
# Gecachte Ressourcen
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner="Verbinde mit MCP-Server …")
def fetch_server_info(url: str) -> dict:
    try:
        client = MCPClient(url)
        client.initialize()
        models = client.get_models()
        return {
            "online": True,
            "models": sorted(models) if models else ["llama-3.3-70b-instruct"],
        }
    except Exception:
        return {
            "online": False,
            "models": ["llama-3.3-70b-instruct", "meta-llama-3.1-8b-instruct"],
        }


@st.cache_data(ttl=60, show_spinner=False)
def fetch_kontingent(url: str) -> dict:
    try:
        client = MCPClient(url)
        client.initialize()
        return client.get_kontingent()
    except Exception:
        return {}


def render_kontingent(url: str) -> None:
    """Verbleibende GWDG-Anfragen anzeigen (Produktion und Agenten teilen den Schlüssel)."""
    d = fetch_kontingent(url)
    rest, limit = d.get("rest") or {}, d.get("limit") or {}
    if "tag" not in rest:
        return
    try:
        from zoneinfo import ZoneInfo
        stand = datetime.fromtimestamp(d["ts"], ZoneInfo("Europe/Berlin"))
    except Exception:
        return
    jetzt = datetime.now(stand.tzinfo)
    tag = rest["tag"] if stand.date() == jetzt.date() else limit.get("tag", rest["tag"])
    monat = rest.get("monat")
    if monat is not None and (stand.year, stand.month) != (jetzt.year, jetzt.month):
        monat = limit.get("monat", monat)
    st.markdown('<p class="hsh-sidebar-label">GWDG-Kontingent</p>', unsafe_allow_html=True)
    if limit.get("tag"):
        st.progress(max(0.0, min(1.0, tag / limit["tag"])))
    zeile = f"Noch **{tag}** Anfragen heute"
    if monat is not None:
        zeile += f" · **{monat}** in diesem Monat"
    st.caption(zeile + f" (Stand {stand:%H:%M}). Eine Frage braucht etwa 2–4 Anfragen.")


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ---- CSS-Variablen ---- */
:root {
    --hsh-primary:        #00559a;
    --hsh-primary-light:  #e8f0fb;
    --hsh-primary-medium: #c8dff3;
    --hsh-red:            #e30613;
    --hsh-radius:         8px;
    --hsh-text:           #1a1a1a;
    --hsh-muted:          #6b7280;
}

/* ---- Layout ---- */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 1rem;
}

/* ---- HsH-Logo ---- */
.hsh-logo-hero {
    max-width: 200px;
    width: 100%;
    height: auto;
    display: block;
    margin: 0 auto 1rem;
}

/* ---- Hero Card ---- */
.hsh-hero-card {
    background: linear-gradient(135deg, #f0f6fc 0%, var(--hsh-primary-light) 100%);
    border: 1px solid var(--hsh-primary-medium);
    border-radius: 16px;
    padding: 2rem 1.5rem 1.25rem;
    max-width: 760px;
    margin: 0 auto 1.5rem;
    text-align: center;
}
.hsh-hero-card h1 {
    font-size: 2rem;
    font-weight: 800;
    color: var(--hsh-primary);
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.hsh-hero-card .hsh-desc {
    color: #4a4a4a;
    font-size: 0.94rem;
    line-height: 1.6;
    margin-bottom: 1.1rem;
}

/* ---- Badges ---- */
.hsh-notices {
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
}
.hsh-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.32rem 0.75rem;
    border-radius: 999px;
    font-size: 0.77rem;
    font-weight: 500;
    line-height: 1.35;
}
.hsh-badge-green { background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }
.hsh-badge-amber { background: #fff8e1; color: #6d4c00; border: 1px solid #ffe082; }

/* ---- Stats-Zeile ---- */
.hsh-stats-row {
    display: flex;
    gap: 2rem;
    justify-content: center;
    flex-wrap: wrap;
    border-top: 1px solid var(--hsh-primary-medium);
    padding-top: 1rem;
    margin-top: 0.25rem;
}
.hsh-stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.1rem;
}
.hsh-stat-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--hsh-primary);
}
.hsh-stat-label {
    font-size: 0.68rem;
    color: var(--hsh-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---- Chat-Input ---- */
[data-testid="stBottom"] {
    border-top: 1px solid #e0e0e0;
    box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.06);
    padding-top: 0.5rem;
}

/* ---- User-Chat-Bubble ---- */
.hsh-chat-user {
    background: var(--hsh-primary-light);
    border: 1px solid var(--hsh-primary-medium);
    border-radius: 14px 14px 4px 14px;
    padding: 0.65rem 1rem;
    line-height: 1.6;
    color: var(--hsh-text);
}

/* ---- Sidebar: Abschnittslabels ---- */
.hsh-sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #999;
    margin: 0.9rem 0 0.1rem;
    padding: 0;
    line-height: 1.2;
}

/* ---- Status-Pill ---- */
.hsh-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 500;
    margin-top: 0.3rem;
}
.hsh-status-online  { background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }
.hsh-status-offline { background: #ffebee; color: #b71c1c; border: 1px solid #ef9a9a; }

/* ---- Quelltextnachweis ---- */
.hsh-highlight {
    background: #FFD700;
    padding: 0 2px;
    border-radius: 2px;
}
.hsh-snippet {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.55rem;
    font-size: 0.87rem;
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ---- Quellen als Cards ---- */
.hsh-source-card {
    border-left: 3px solid var(--hsh-primary);
    padding: 0.5rem 0.75rem;
    background: #f4f8fd;
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.45rem;
    font-size: 0.9rem;
    line-height: 1.5;
    transition: background 0.15s ease, transform 0.15s ease, border-color 0.15s ease;
}
.hsh-source-card:hover {
    background: var(--hsh-primary-light);
    transform: translateX(3px);
    border-left-color: #003d6b;
}
.hsh-source-card a {
    color: var(--hsh-primary);
    text-decoration: none;
    font-weight: 500;
}
.hsh-source-card a:hover { text-decoration: underline; }
.hsh-source-domain {
    display: inline-block;
    font-size: 0.67rem;
    color: var(--hsh-muted);
    background: #dde8f3;
    border-radius: 4px;
    padding: 0.05rem 0.4rem;
    margin-left: 0.4rem;
    vertical-align: middle;
    font-family: monospace;
    letter-spacing: 0.02em;
}
/* Herkunftshinweis: Absätze, die wortgleich unter einer anderen URL im Index
   liegen und von dort geholt wurden. Gehört sichtbar in die Quellenangabe. */
.hsh-source-fremd {
    margin-top: 0.35rem;
    font-size: 0.78rem;
    line-height: 1.45;
    color: var(--hsh-muted);
}
.hsh-source-fremd a { font-weight: 400; }
</style>
"""

def _build_hero_html() -> str:
    logo_block = _LOGO_SVG if _LOGO_SVG else '<span style="font-size:2.5rem">🎓</span>'
    return f"""
<div class="hsh-hero-card">
  {logo_block}
  <h1>HsH-Chatbot</h1>
  <p class="hsh-desc">
    Stellen Sie Fragen zur Hochschule Hannover auf Basis offizieller Dokumente.<br>
    Wählen Sie links die <strong>Fakultät</strong> und das <strong>KI-Modell</strong> aus.
  </p>
  <div class="hsh-notices">
    <span class="hsh-badge hsh-badge-amber">💾 Speicherung aktiv — max. 12 Monate (DSGVO)</span>
    <span class="hsh-badge hsh-badge-green">🔒 Deaktivierbar in den Einstellungen</span>
  </div>
  <div class="hsh-stats-row">
    <div class="hsh-stat">
      <span class="hsh-stat-value">10.000+</span>
      <span class="hsh-stat-label">Dokumente</span>
    </div>
    <div class="hsh-stat">
      <span class="hsh-stat-value">5</span>
      <span class="hsh-stat-label">Fakultäten</span>
    </div>
    <div class="hsh-stat">
      <span class="hsh-stat-value">Dense + Sparse</span>
      <span class="hsh-stat-label">Hybrid-Suche</span>
    </div>
    <div class="hsh-stat">
      <span class="hsh-stat-value">DSGVO ✓</span>
      <span class="hsh-stat-label">GWDG ChatAI</span>
    </div>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _merke(schluessel: str, wert):
    """Legt eine Sidebar-Auswahl in einem eigenen session_state-Schlüssel ab.

    Streamlit verwirft den Zustand eines Widgets, das im aktuellen Durchlauf
    nicht gezeichnet wird. Die Seite "Beispielfragen" rendert die Sidebar der
    Chatseite nicht — ohne diese Ablage fiele beim Zurückspringen jede
    Einstellung auf ihren Startwert zurück, allen voran das Modell. Ein `key`
    am Widget allein genügt dafür nicht, weil auch keyed State mit dem Widget
    verschwindet; gemerkt wird deshalb daneben, nicht im Widget.
    """
    st.session_state[schluessel] = wert
    return wert


def _startindex(schluessel: str, optionen: list, fallback: int = 0) -> int:
    """Index der zuletzt gemerkten Auswahl, sonst der übergebene Fallback."""
    gemerkt = st.session_state.get(schluessel)
    if gemerkt in optionen:
        return optionen.index(gemerkt)
    return fallback


def render_sidebar() -> tuple[str, str, str, bool, bool, str]:
    with st.sidebar:
        st.markdown("## ⚙️ Einstellungen")

        st.markdown('<p class="hsh-sidebar-label">MCP-Server</p>', unsafe_allow_html=True)
        server_optionen = ["Remote (nemo.christianwill.de)", "Lokal (127.0.0.1:8765)",
                           "Agentenversion (Test, 127.0.0.1:8767)", "Eigene URL"]
        server_wahl = _merke("srv_wahl", st.radio(
            "server_wahl",
            server_optionen,
            index=_startindex("srv_wahl", server_optionen),
            label_visibility="collapsed",
        ))
        if server_wahl.startswith("Remote"):
            mcp_url = MCP_URL_REMOTE
        elif server_wahl.startswith("Lokal"):
            mcp_url = MCP_URL_LOCAL
        elif server_wahl.startswith("Agentenversion"):
            mcp_url = MCP_URL_AGENT
            st.caption("Nur im Testmodus erreichbar (`raghsh-zyklus testen`). Fragen werden nicht gespeichert.")
        else:
            mcp_url = _merke("url_wahl", st.text_input(
                "MCP-URL", value=st.session_state.get("url_wahl", MCP_URL_REMOTE)
            ))

        server_info = fetch_server_info(mcp_url)
        if server_info["online"]:
            st.markdown(
                '<span class="hsh-status-pill hsh-status-online">● Online</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="hsh-status-pill hsh-status-offline">○ Nicht erreichbar</span>',
                unsafe_allow_html=True,
            )

        # ── Premium-Zugang (kostenpflichtige Modelle via OpenRouter) ─────
        st.markdown('<p class="hsh-sidebar-label">🔑 Premium-Zugang</p>', unsafe_allow_html=True)
        if st.session_state.get("full_access"):
            st.caption("💎 Premium-Modelle freigeschaltet — Claude · Gemini · ChatGPT")
            if st.button("Sperren", use_container_width=True):
                st.session_state.full_access    = False
                st.session_state.full_access_pw = ""
                st.rerun()
        else:
            pw = st.text_input(
                "Passwort für Premium-Modelle",
                type="password",
                placeholder="Passwort für Premium-Modelle …",
                label_visibility="collapsed",
                help=(
                    "Schaltet kostenpflichtige Modelle (Claude, Gemini, ChatGPT) "
                    "über OpenRouter frei. Jede Anfrage verursacht Kosten."
                ),
            )
            if pw:
                if FULL_ACCESS_PASSWORD and pw == FULL_ACCESS_PASSWORD:
                    st.session_state.full_access    = True
                    st.session_state.full_access_pw = pw
                    st.rerun()
                else:
                    st.caption("❌ Falsches Passwort.")

        passwort = st.session_state.get("full_access_pw", "")

        st.markdown('<p class="hsh-sidebar-label">Modell</p>', unsafe_allow_html=True)
        modelle = list(server_info["models"])
        if st.session_state.get("full_access"):
            modelle += list(PREMIUM_MODELS.keys())
        # Bevorzugte Vorauswahl: GLM-4.7, sonst der erste Treffer aus der
        # Prioritätsliste, sonst das erste Modell, das der Server anbietet.
        default_idx = next(
            (
                i
                for kandidat in DEFAULT_MODELL_PRIO
                for i, m in enumerate(modelle)
                if kandidat in m
            ),
            0,
        )

        def _modell_label(m: str) -> str:
            if m in PREMIUM_MODELS:
                return PREMIUM_MODELS[m]
            if m in _MODELL_LABELS:
                return f"{_MODELL_LABELS[m]}  {m}"
            return m

        modell = _merke("modell_wahl", st.selectbox(
            "GWDG-Modell",
            modelle,
            index=_startindex("modell_wahl", modelle, default_idx),
            format_func=_modell_label,
        ))
        st.caption(
            "📖 [Modellbeschreibungen & Empfehlungen](https://nemo.christianwill.de/hsh-app/Dokumentation)"
        )
        if modell not in PREMIUM_MODELS and server_info["online"]:
            render_kontingent(mcp_url)

        st.markdown('<p class="hsh-sidebar-label">Fakultät</p>', unsafe_allow_html=True)
        fak_optionen = list(FAKULTAETEN.keys())
        fak_label = _merke("fak_wahl", st.selectbox(
            "Fakultät", fak_optionen, index=_startindex("fak_wahl", fak_optionen)
        ))
        fakultaet = FAKULTAETEN[fak_label]

        st.markdown('<p class="hsh-sidebar-label">Anzeige</p>', unsafe_allow_html=True)
        show_steps = _merke("schritte_wahl", st.toggle(
            "Zwischenschritte anzeigen",
            value=st.session_state.get("schritte_wahl", True),
        ))

        st.divider()
        st.markdown('<p class="hsh-sidebar-label">🔒 Datenschutz (DSGVO)</p>', unsafe_allow_html=True)
        speichern = _merke("speichern_wahl", st.toggle(
            "Frage & Antwort speichern",
            value=st.session_state.get("speichern_wahl", True),
            help=(
                "Mit Ihrer Einwilligung werden Frage, Antwort und verwendete Quellen "
                "zur Qualitätssicherung und Evaluation gespeichert. "
                "Die Daten werden nach spätestens 12 Monaten automatisch gelöscht. "
                "Ohne Einwilligung werden keine personenbezogenen Daten erfasst."
            ),
        ))
        if speichern:
            st.caption("✅ Einwilligung erteilt — Daten werden max. 12 Monate gespeichert.")
        else:
            st.caption("🚫 Keine Speicherung — Daten werden nicht protokolliert.")

        st.divider()
        st.info(
            "**HsH-Chatbot**\n\n"
            "Beantwortet Fragen auf Basis offizieller HsH-Dokumente "
            "via RAG-Pipeline (Hybrid-Suche + LLM).\n\n"
            f"Server: `{mcp_url}`"
        )
        if st.button("Chat zurücksetzen", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("pending_po", None)
            st.rerun()

    if mcp_url == MCP_URL_AGENT:
        speichern = False   # Testfragen dürfen nicht im qa_log der Agenten landen
    return mcp_url, modell, fakultaet, show_steps, speichern, passwort


# ---------------------------------------------------------------------------
# Zwischenschritte und Quellen rendern (für History-Replay)
# ---------------------------------------------------------------------------


def render_zeiten(zeiten: dict) -> None:
    """Zeigt die Dauer der Pipeline-Phasen — Fokus: Zeit bis die Quellen feststehen."""
    if not zeiten:
        return
    quellen = zeiten.get("quellen_s", 0)
    antwort = zeiten.get("antwort_llm_s", 0)
    gesamt  = zeiten.get("gesamt_s", 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("⏱ Quellen", f"{quellen:.1f} s", help=(
        "Zeit bis die Quellen feststehen und geladen sind — Hybrid-Suche + "
        "Kandidaten + LLM-Dokumentauswahl + Volltext laden (ohne Antwortgenerierung)."
    ))
    c2.metric("✍️ Antwort", f"{antwort:.1f} s", help="Dauer der finalen Antwortgenerierung (LLM Runde 2).")
    c3.metric("Σ Gesamt", f"{gesamt:.1f} s")
    st.caption(
        f"Aufschlüsselung Quellen: Hybrid-Suche {zeiten.get('hybrid_suche_s', 0):.1f} s · "
        f"Kandidaten {zeiten.get('kandidaten_s', 0):.1f} s · "
        f"LLM-Auswahl {zeiten.get('llm_auswahl_s', 0):.1f} s · "
        f"Volltext laden {zeiten.get('docs_laden_s', 0):.1f} s"
    )


def render_zwischenschritte(meta: dict) -> None:
    """Zeigt die Zwischenschritte aus gespeicherten Metadaten (eingeklappt)."""
    if not meta:
        return
    with st.expander("🔎 Zwischenschritte", expanded=False):
        # Fortschritt
        steps = meta.get("fortschritt", [])
        if steps:
            for s in steps:
                icon = SCHRITTE_ICONS.get(s["step"], "▶")
                st.markdown(f"{icon} **Schritt {s['step']}/{s['total']}** — {s['message']}")
            st.divider()

        # Kandidaten
        kandidaten = meta.get("kandidaten", [])
        if kandidaten:
            st.markdown(f"**{len(kandidaten)} Kandidaten gefunden:**")
            for k in kandidaten:
                fak = f" · Fak. {k['fakultaet']}" if k.get("fakultaet") else ""
                st.markdown(f"**{k['nr']}.** [{k['titel']}]({k['url']}){fak}")
                if k.get("struktur"):
                    with st.expander("Inhaltsstruktur", expanded=False):
                        st.code(k["struktur"], language=None)
            st.divider()

        # Dokumentauswahl
        ausgewaehlte = meta.get("ausgewaehlte_docs", [])
        if ausgewaehlte:
            st.markdown("**Vom LLM ausgewählte Dokumente:**")
            for d in ausgewaehlte:
                st.markdown(f"- **{d['nr']}.** [{d['titel']}]({d['url']})")
            st.divider()

        # Token-Budget
        budget = meta.get("token_budget", {})
        if budget:
            verwendet = budget.get("verwendet", 0)
            maximum   = budget.get("maximum", 100_000)
            pct       = min(1.0, verwendet / maximum) if maximum else 0.0
            st.markdown(
                f"**Token-Budget:** {verwendet:,} / {maximum:,} "
                f"({round(pct * 100, 1)} %)"
            )
            st.progress(pct)

        # Zeitmessung der Phasen
        zeiten = meta.get("zeiten", {})
        if zeiten:
            st.divider()
            st.markdown("**⏱ Dauer**")
            render_zeiten(zeiten)


def _domain_chip(url: str) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url).hostname or ""
        if host.startswith("www."):
            host = host[4:]
        return f'<span class="hsh-source-domain">{html.escape(host)}</span>'
    except Exception:
        return ""


def _format_crawl_date(raw: str) -> str:
    """Wandelt YYYY-MM-DD in DD.MM.YYYY um; gibt raw zurück wenn Format unbekannt."""
    try:
        from datetime import date
        d = date.fromisoformat(raw)
        return d.strftime("%-d.%-m.%Y")
    except (ValueError, TypeError):
        return raw


def _shorten_url(url: str) -> str:
    """Kürzt eine URL auf Host und letztes Pfadsegment — für Fließtexthinweise.

    Die vollständige URL steht weiterhin im href; nur die Beschriftung wird
    gekürzt, damit ein Herkunftshinweis die Quellenkarte nicht sprengt.
    """
    try:
        teile = urlparse(url)
        host  = (teile.hostname or "").removeprefix("www.")
        if not host:                      # kein absoluter Link — unverändert lassen
            return url
        letzt = [s for s in teile.path.split("/") if s]
        return f"{host}/…/{letzt[-1]}" if letzt else host
    except Exception:
        return url


def render_sources(quellen: list[dict]) -> None:
    """Zeigt die verwendeten Quellen — samt fremder Fundstellen.

    `fremde_fundstellen` nennt die URLs, unter denen Absätze dieses Dokuments
    tatsächlich im Index liegen: Die Deduplizierung behält je Gruppe wortgleicher
    Chunks nur einen Vertreter, der Text kann deshalb aus einem gleichlautenden
    Dokument einer anderen Fakultät oder eines anderen Jahrgangs stammen. Bleibt
    das unerwähnt, führt die Quellenangabe auf ein Dokument, das den zitierten
    Wortlaut so nicht enthalten muss.
    """
    if not quellen:
        return
    with st.expander("📚 Verwendete Quellen", expanded=True):
        cards = ""
        for i, q in enumerate(quellen, 1):
            titel      = q.get("titel", q.get("url", "—"))
            url        = q.get("url", "")
            crawl_date = q.get("crawl_date", "")
            fremde     = q.get("fremde_fundstellen", []) or []
            link       = f'<a href="{url}" target="_blank">{html.escape(titel)}</a>' if url else html.escape(titel)
            datum      = f', abgerufen: {_format_crawl_date(crawl_date)}' if crawl_date else ""
            hinweis    = ""
            if fremde:
                fremd_links = " · ".join(
                    f'<a href="{f}" target="_blank">{html.escape(_shorten_url(f))}</a>'
                    for f in fremde
                )
                hinweis = (
                    '<div class="hsh-source-fremd">↪ Einzelne Absätze stammen wortgleich aus: '
                    f'{fremd_links}</div>'
                )
            cards += (
                f'<div class="hsh-source-card"><strong>[{i}]</strong> {link}{datum}'
                f'{hinweis}</div>'
            )
        st.markdown(cards, unsafe_allow_html=True)


def _waehle_po(option: dict, frage: str) -> None:
    """Button-Callback der Rückfrage: auslösende Frage einmal mit `po` neu stellen.

    Option mit Fassung (PO-Rückfrage) → `po` mit Studiengang und Fassung;
    ohne Fassung (Rückfrage nach dem Studiengang, S085) → nur der Studiengang,
    die Fassung kommt dann aus der Frage oder per PO-Rückfrage. Die Langform
    liefert der Server im Feld `name` der Option.

    Den Gesprächszustand hält der Client (ZIEL.md). Die gewählte Fassung gilt
    nur für genau diese eine Wiederholung (pending_po wird beim Senden
    verbraucht); jede neue freie Eingabe geht wieder ohne `po` an frag_hsh —
    eine Folgefrage kann einen anderen Studiengang oder eine andere PO meinen.
    """
    name = option.get("name") or option["studiengang"]
    if "fassung" in option:
        st.session_state.pending_po = {"studiengang": option["studiengang"],
                                       "fassung": option["fassung"]}
        st.session_state.pending_anzeige = f"PO {option['fassung']} · {name}"
    else:
        st.session_state.pending_po = {"studiengang": option["studiengang"]}
        st.session_state.pending_anzeige = name
    st.session_state.pending_frage = frage


def render_rueckfrage(msg: dict, auswaehlbar: bool, key: str) -> None:
    """Zeigt die Rückfrage von frag_hsh mit den Optionen als Buttons —
    Fassungen (PO-Rückfrage) oder Studiengänge (S085).

    Nur die jüngste Rückfrage im Verlauf ist anklickbar; ältere bleiben als
    Text stehen.
    """
    rueckfrage = msg["rueckfrage"]
    optionen   = rueckfrage.get("optionen", [])
    text       = rueckfrage.get("frage", "")   # nennt die Langform schon selbst (S084)
    st.markdown(f"**Rückfrage:** {text}")
    if not auswaehlbar or not optionen:
        return
    spalten = st.columns(min(len(optionen), 4))
    for i, o in enumerate(optionen):
        if "fassung" in o:
            label = f"PO {o['fassung']}" + (" (aktuell)" if o.get("aktuell") else "")
        else:
            label = o.get("name") or o["studiengang"]
        spalten[i % len(spalten)].button(
            label,
            key=f"{key}_po_{o['studiengang']}_{o.get('fassung', '')}",
            on_click=_waehle_po,
            args=(o, msg.get("frage", "")),
            use_container_width=True,
        )


def render_stored_message(msg: dict, show_steps: bool,
                          auswaehlbar: bool = False, key: str = "") -> None:
    """Rendert eine gespeicherte Assistent-Antwort aus dem Chat-Verlauf."""
    if show_steps and msg.get("meta"):
        render_zwischenschritte(msg["meta"])
    if msg.get("rueckfrage"):
        render_rueckfrage(msg, auswaehlbar, key)
        return
    render_answer(msg["content"])
    render_sources(msg.get("quellen", []))
    render_source_evidence(msg.get("quellen", []))


# ---------------------------------------------------------------------------
# Live-Streaming einer Anfrage
# ---------------------------------------------------------------------------


def stream_and_render(
    mcp_url: str,
    frage: str,
    modell: str,
    fakultaet: str,
    show_steps: bool,
    speichern: bool = False,
    passwort: str = "",
    po: dict | None = None,
) -> dict:
    """
    Öffnet eine MCP-Session, ruft frag_hsh auf und rendert den Verlauf live.

    Returns:
        {"antwort": str, "quellen": [...], "meta": {...}, "rueckfrage": dict | None}
    """
    fortschritt: list[dict] = []
    result_data: dict       = {}

    try:
        client = MCPClient(mcp_url)
        client.initialize()
    except Exception as exc:
        st.error(f"MCP-Server nicht erreichbar ({mcp_url}): {exc}")
        return {"antwort": "", "quellen": [], "meta": {}}

    # st.status zeigt Fortschritt live und klappt nach Abschluss ein/aus
    with st.status("Verarbeite Anfrage …", expanded=show_steps) as status:
        try:
            for event in client.frag_hsh(frage, modell, fakultaet, speichern, passwort, po):
                etype = event.get("type")

                if etype == "progress":
                    fortschritt.append({
                        "step":    event["step"],
                        "total":   event["total"],
                        "message": event["message"],
                    })
                    if show_steps:
                        icon = SCHRITTE_ICONS.get(event["step"], "▶")
                        st.write(f"{icon} {event['message']}")

                elif etype == "result":
                    data = event.get("data", {})
                    result_data = data

                    if data.get("fehler"):
                        status.update(
                            label="Fehler bei der Verarbeitung",
                            state="error",
                            expanded=True,
                        )
                        break

                    if show_steps:
                        # Kandidaten
                        kandidaten = data.get("kandidaten", [])
                        if kandidaten:
                            st.divider()
                            st.markdown(f"**{len(kandidaten)} Kandidaten gefunden:**")
                            for k in kandidaten:
                                fak = f" · Fak. {k['fakultaet']}" if k.get("fakultaet") else ""
                                st.markdown(
                                    f"**{k['nr']}.** [{k['titel']}]({k['url']}){fak}"
                                )
                                if k.get("struktur"):
                                    with st.expander("Inhaltsstruktur", expanded=False):
                                        st.code(k["struktur"], language=None)

                        # Dokumentauswahl
                        ausgewaehlte = data.get("ausgewaehlte_docs", [])
                        if ausgewaehlte:
                            st.divider()
                            st.markdown("**Vom LLM ausgewählte Dokumente:**")
                            for d in ausgewaehlte:
                                st.markdown(f"- **{d['nr']}.** [{d['titel']}]({d['url']})")

                        # Token-Budget
                        budget = data.get("token_budget", {})
                        if budget:
                            verwendet = budget.get("verwendet", 0)
                            maximum   = budget.get("maximum", 100_000)
                            pct       = min(1.0, verwendet / maximum) if maximum else 0.0
                            st.divider()
                            st.markdown(
                                f"**Token-Budget:** {verwendet:,} / {maximum:,} "
                                f"({round(pct * 100, 1)} %)"
                            )
                            st.progress(pct)

                    status.update(
                        label="✓ Verarbeitung abgeschlossen",
                        state="complete",
                        expanded=False,
                    )
                    break

                elif etype == "error":
                    st.write(f"❌ {event.get('message', 'Unbekannter Fehler')}")
                    status.update(label="Fehler", state="error", expanded=True)
                    return {"antwort": "", "quellen": [], "meta": {"fortschritt": fortschritt}}

        except requests.Timeout:
            status.update(label="Zeitüberschreitung", state="error", expanded=True)
            st.write("Der MCP-Server hat nicht rechtzeitig geantwortet.")
            return {"antwort": "", "quellen": [], "meta": {"fortschritt": fortschritt}}
        except requests.ConnectionError as exc:
            status.update(label="Verbindungsfehler", state="error", expanded=True)
            st.write(f"Verbindungsfehler: {exc}")
            return {"antwort": "", "quellen": [], "meta": {"fortschritt": fortschritt}}
        except Exception as exc:
            status.update(label="Fehler", state="error", expanded=True)
            st.write(f"Unerwarteter Fehler: {exc}")
            return {"antwort": "", "quellen": [], "meta": {"fortschritt": fortschritt}}

    # Fehler-Rückgabe wenn kein Ergebnis
    if not result_data:
        return {"antwort": "", "quellen": [], "meta": {"fortschritt": fortschritt}}

    if result_data.get("fehler"):
        st.error(f"Fehler: {result_data['fehler']}")
        return {"antwort": "", "quellen": [], "meta": {"fortschritt": fortschritt}}

    meta = {
        "fortschritt":         fortschritt,
        "kandidaten":          result_data.get("kandidaten", []),
        "ausgewaehlte_docs":   result_data.get("ausgewaehlte_docs", []),
        "auswahl_begruendung": result_data.get("auswahl_begruendung", ""),
        "token_budget":        result_data.get("token_budget", {}),
        "zeiten":              result_data.get("zeiten", {}),
    }

    # PO-Rückfrage statt Antwort — die Auswahl rendert main() im Verlauf
    if result_data.get("rueckfrage"):
        return {"antwort": "", "quellen": [], "meta": meta,
                "rueckfrage": result_data["rueckfrage"]}

    # Antwort — Gedanken eingeklappt, Rest sichtbar
    antwort = result_data.get("antwort", "")
    render_answer(antwort)
    antwort = _THINK_RE.sub("", antwort).strip()

    # Quellen — immer sichtbar
    quellen = result_data.get("quellen", [])
    render_sources(quellen)

    # Quelltextnachweis — relevante Originalpassagen mit Ähnlichkeitsmarkierung
    render_source_evidence(quellen)

    # Dauer der Pipeline-Phasen (Fokus: Zeit bis die Quellen feststehen)
    render_zeiten(result_data.get("zeiten", {}))

    return {"antwort": antwort, "quellen": quellen, "meta": meta}


# ---------------------------------------------------------------------------
# Haupt-App
# ---------------------------------------------------------------------------


def main() -> None:
    mcp_url, modell, fakultaet, show_steps, speichern, passwort = render_sidebar()

    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_build_hero_html(), unsafe_allow_html=True)
    st.caption(f"Modell: **{modell}** · Server: `{mcp_url}`")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat-Verlauf rendern
    letzte = len(st.session_state.messages) - 1
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                safe = html.escape(msg["content"]).replace("\n", "<br>")
                st.markdown(f'<div class="hsh-chat-user">{safe}</div>', unsafe_allow_html=True)
            else:
                render_stored_message(msg, show_steps, auswaehlbar=(i == letzte), key=f"msg{i}")

    # Nutzereingabe — eine auf der Seite "Beispielfragen" angeklickte Frage
    # liegt in st.session_state.pending_frage und wird hier übernommen.
    # Nach einer PO-Auswahl steht dort die ursprüngliche Frage, angezeigt wird
    # nur die gewählte Fassung (pending_anzeige).
    question = st.chat_input("Stellen Sie Ihre Frage zur Hochschule Hannover …")
    anzeige  = question
    po       = None
    if question:
        # Freie Eingabe: eine liegengebliebene Auswahl verfällt
        st.session_state.pop("pending_po", None)
    else:
        question = st.session_state.pop("pending_frage", None)
        anzeige  = st.session_state.pop("pending_anzeige", None) or question
        po       = st.session_state.pop("pending_po", None)
    if not question:
        return

    with st.chat_message("user"):
        safe = html.escape(anzeige).replace("\n", "<br>")
        st.markdown(f'<div class="hsh-chat-user">{safe}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": anzeige})

    # Antwort generieren und live anzeigen
    with st.chat_message("assistant"):
        result = stream_and_render(mcp_url, question, modell, fakultaet, show_steps,
                                   speichern, passwort, po)

    if result.get("rueckfrage"):
        st.session_state.messages.append({
            "role":       "assistant",
            "content":    "",
            "rueckfrage": result["rueckfrage"],
            "frage":      question,
            "quellen":    [],
            "meta":       result["meta"],
        })
        st.rerun()                      # Verlauf neu zeichnen, Auswahl-Buttons aktiv
    elif result["antwort"]:
        st.session_state.messages.append({
            "role":    "assistant",
            "content": result["antwort"],
            "quellen": result["quellen"],
            "meta":    result["meta"],
        })


if __name__ == "__main__":
    main()
