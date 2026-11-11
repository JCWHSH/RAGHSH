"""
Führt die Fallbasis gegen den laufenden MCP-Server und schreibt Rohdaten.

Was dieser Runner anders macht als eval/run_rag.py
--------------------------------------------------
`eval/run_rag.py` prüft das Golddokument gegen `quellen` — und das ist die Liste
NACH der LLM-Dokumentauswahl. Eine Trefferquote von dort sagt nur: „gefunden UND
ausgewählt". Sie kann nicht unterscheiden, ob ein Fehlschlag am Retrieval lag
oder daran, dass Runde 1 das richtige Dokument in der Hand hatte und verwarf.
Das sind zwei völlig verschiedene Reparaturen.

Deshalb speichert dieser Lauf die Zwischenstufen roh und wertet NICHT selbst:
Kandidaten (vor der Auswahl), Quellen (nach der Auswahl), die gefundenen Chunks,
Zeiten, Token-Budget. Die Diagnose macht auswertung.py auf diesen Rohdaten —
so lässt sich die Auswertung ändern, ohne die Läufe zu wiederholen.

Das Manifest hält die vollständige Retrieval-Konfiguration fest. Ohne sie ist
ein Laufvergleich wertlos: Am 17.08.2026 stieg unsere Trefferquote von 25 % auf
39 %, während sechs Dinge gleichzeitig anders waren — Index, Query-Adapter,
BM25-Sprache, Fusion, Armzahl und Titel. Solche Zahlen sind eine Beobachtung,
keine Messung.

Aufrufe
-------
    python auswertung/gold/lauf.py --modell meta-llama-3.1-8b-instruct --auswerten --bericht
    python auswertung/gold/lauf.py --faelle chargy_beidawi --limit 5
    python auswertung/gold/lauf.py --faelle q06,q11,q19
    python auswertung/gold/lauf.py --fortsetzen laeufe/2026-08-17_1830_llama31
    python auswertung/gold/lauf.py --nur-retrieval --auswerten     # kein LLM, ~1 min
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

_BASE = Path(__file__).resolve().parent
from raghsh import konfig  # noqa: E402

from mcp import ClientSession                              # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402

from raghsh.suche import hybrid_search                                        # noqa: E402

sys.path.insert(0, str(_BASE))
from vertrag import hash_fallbasis, lade_fallbasis           # noqa: E402
from vertrag import kurz as hash_kurz                        # noqa: E402

MCP_URL      = "http://127.0.0.1:8765/mcp"      # --mcp-url für eine Testinstanz
FALLBASIS    = _BASE / "fallbasis.json"
LAEUFE       = konfig.LAEUFE
STANDARDMODELL = "meta-llama-3.1-8b-instruct"


def richte_log_ein(lauf_dir: Path) -> logging.Logger:
    """Hängt einen Dateihandler an die Wurzel und gibt den Lauf-Logger zurück.

    Was die Bibliotheken loggen — MCP-Aufrufe, HTTP, Fehlschläge — ging bisher
    nur auf die Konsole und landete allein dann in einer Datei, wenn jemand von
    Hand umleitete. Genau so entstanden zwei herrenlose Logs neben den
    Laufordnern. Jetzt schreibt jeder Lauf sein eigenes; angehängt wird, damit
    --fortsetzen den bisherigen Verlauf behält.
    """
    handler = logging.FileHandler(lauf_dir / "lauf.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    wurzel = logging.getLogger()
    wurzel.setLevel(logging.INFO)
    wurzel.addHandler(handler)
    return logging.getLogger("lauf")


def melde(logger: logging.Logger, text: str = "") -> None:
    """Auf die Konsole und ins Laufprotokoll — der Fortschritt gehört in beides."""
    print(text)
    if text.strip():
        logger.info(text.strip())


# ---------------------------------------------------------------------------
# Manifest — der Zustand, gegen den gemessen wurde
# ---------------------------------------------------------------------------


def sammle_konfiguration() -> dict:
    """Liest die Retrieval-Konfiguration aus dem Modul, das der Server benutzt.

    Bewusst aus `hybrid_search` importiert statt abgetippt: Was hier steht, ist
    dann garantiert das, was auch gesucht hat.
    """
    konf = {
        name: getattr(hybrid_search, name, None)
        for name in ("DENSE_ARMS", "SPARSE_ARMS", "TOP_K", "CANDIDATE_LIMIT",
                     "DEDUP_BUFFER", "MAX_PER_URL", "USE_RERANKER",
                     "AUGMENT_TOP_N", "SPARSE_LANGUAGE", "COLLECTION_NAME")
    }
    fusion = getattr(hybrid_search, "FUSION", None)
    konf["FUSION"] = str(fusion.value if hasattr(fusion, "value") else fusion)
    konf["DENSE_ARMS"] = list(konf["DENSE_ARMS"] or [])
    konf["SPARSE_ARMS"] = list(konf["SPARSE_ARMS"] or [])

    try:
        from raghsh.server import mcp_server as srv
        konf.update({n: getattr(srv, n, None)
                     for n in ("RAG_CANDIDATE_K", "TOP_DOCS", "SELECT_N",
                               "MAX_CONTEXT_TOKENS")})
    except Exception:
        pass  # Server-Modul nicht importierbar — Konfiguration bleibt teilweise

    try:
        from qdrant_client import QdrantClient
        info = QdrantClient(url="http://localhost:6333", timeout=20).get_collection(
            os.environ.get("HSH_COLLECTION") or konf.get("COLLECTION_NAME") or "hsh_knowledge")
        konf["points_count"] = info.points_count
    except Exception as exc:
        konf["points_count"] = f"unbekannt ({exc})"

    try:
        konf["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=konfig.WURZEL,
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        konf["git_commit"] = None
    return konf


# ---------------------------------------------------------------------------
# Fallauswahl
# ---------------------------------------------------------------------------


def baue_experiment(args) -> dict:
    """Aus den CLI-Optionen das Experiment-Objekt für frag_hsh; leer = Produktion."""
    e: dict = {}
    if args.dense is not None:
        e["dense_arms"] = [a for a in args.dense.split(",") if a]
    if args.sparse is not None:
        e["sparse_arms"] = [a for a in args.sparse.split(",") if a]
    if args.fusion:
        e["fusion"] = args.fusion
    for name in ("candidate_limit", "dedup_buffer", "max_per_url", "top_docs"):
        wert = getattr(args, name)
        if wert is not None:
            e[name] = wert
    if args.kein_boost:
        e["keyword_boost"] = False
    if args.collection:
        e["collection"] = args.collection
    # Dieselbe Prüfung wie im Server, nur früher: ein Tippfehler im Armnamen
    # soll hier scheitern, nicht nach 75 Fehlerzeilen im Lauf.
    try:
        hybrid_search.wirksame_konfiguration(
            {k: v for k, v in e.items() if k not in ("top_docs", "keyword_boost")})
    except ValueError as exc:
        raise SystemExit(f"Experiment ungültig: {exc}")
    if "top_docs" in e and not 1 <= e["top_docs"] <= 30:
        raise SystemExit(f"Experiment ungültig: top_docs muss zwischen 1 und 30 liegen")
    return e


def experiment_kuerzel(e: dict) -> str:
    """Kurzname fürs Verzeichnis: t+q_se_rrf_db6 — lesbar in der Übersicht."""
    kurz = {"text": "t", "questions": "q", "sparse": "s", "sparse_expanded": "se"}
    teile = []
    if "dense_arms" in e or "sparse_arms" in e:
        d = "+".join(kurz[a] for a in e.get("dense_arms", ["text"])) or "-"
        sp = "+".join(kurz[a] for a in e.get("sparse_arms", ["sparse_expanded"])) or "-"
        teile.append(f"{d}_{sp}")
    if "fusion" in e:
        teile.append(e["fusion"])
    for name, k in (("candidate_limit", "cl"), ("dedup_buffer", "db"),
                    ("max_per_url", "mu"), ("top_docs", "td")):
        if name in e:
            teile.append(f"{k}{e[name]}")
    if "collection" in e:
        teile.append(e["collection"].replace("hsh_knowledge_", ""))
    if e.get("keyword_boost") is False:
        teile.append("noboost")
    return "_".join(teile) or "exp"


def lade_faelle(auswahl: str) -> list[dict]:
    if not FALLBASIS.exists():
        raise SystemExit(f"Fehlt: {FALLBASIS} — erst baue_fallbasis.py laufen lassen.")
    faelle = json.loads(FALLBASIS.read_text(encoding="utf-8"))
    if auswahl in ("alle", ""):
        return faelle
    if auswahl in ("chargy_beidawi", "nutzer_gold"):
        return [f for f in faelle if f["herkunft"] == auswahl]
    gewuenscht = {t.strip() for t in auswahl.split(",") if t.strip()}
    treffer = [f for f in faelle if f["id"] in gewuenscht]
    fehlend = gewuenscht - {f["id"] for f in treffer}
    if fehlend:
        raise SystemExit(f"Unbekannte Fall-IDs: {', '.join(sorted(fehlend))}")
    return treffer


# ---------------------------------------------------------------------------
# Ein Fall
# ---------------------------------------------------------------------------


_MCP_URL_AKTIV = [MCP_URL]        # von main() per --mcp-url gesetzt


def _mcp_url() -> str:
    return _MCP_URL_AKTIV[0]


async def stelle_frage(fall: dict, modell: str, timeout_s: int, passwort: str = "",
                       nur_retrieval: bool = False, experiment: dict | None = None) -> dict:
    """Eine Frage, eine frische MCP-Session.

    Bewusst keine über den ganzen Lauf wiederverwendete Session: Reißt sie ab,
    wäre sonst der komplette Rest verloren statt nur eines Falls.
    """
    tmo = timedelta(seconds=timeout_s)
    async with streamablehttp_client(_mcp_url(), timeout=tmo) as (r, w, _):
        async with ClientSession(r, w, read_timeout_seconds=tmo) as session:
            await session.initialize()
            res = await session.call_tool(
                "frag_hsh",
                {"frage": fall["question"], "modell": modell,
                 "fakultaet": fall.get("fakultaet", ""), "speichern": False,
                 "mit_chunks": True, "passwort": passwort,
                 "nur_retrieval": nur_retrieval,
                 **({"experiment": experiment} if experiment else {})},
                read_timeout_seconds=tmo,
            )
            if res.isError:
                text = res.content[0].text if res.content else "unbekannter Tool-Fehler"
                return {"fehler": f"MCP-Fehler: {text[:400]}"}
            if not res.content:
                return {"fehler": "leere Antwort vom Server"}
            return json.loads(res.content[0].text)


def schliesse_kette(lauf_dir: Path, bericht_erzeugen: bool, log: logging.Logger) -> None:
    """Auswertung und Bericht direkt im Anschluss an den Lauf.

    Sammeln und Werten bleiben getrennte Werkzeuge — hier werden sie nur
    nacheinander aufgerufen, damit niemand die Lauf-ID von Hand weiterreichen
    muss. Ein Fehler in der Auswertung darf den Lauf nicht entwerten: Die
    Rohdaten liegen bereits vollständig auf der Platte.
    """
    if lauf_dir.parent != LAEUFE:
        melde(log, f"Kette übersprungen: {lauf_dir} liegt nicht unter {LAEUFE}")
        return
    sys.path.insert(0, str(_BASE))
    try:
        import auswertung
        manifest, bewertet = auswertung.lade_lauf(lauf_dir.name)
        auswertung.zeige(manifest, bewertet, detail=False)
        ziel = auswertung.schreibe_zusammenfassung(lauf_dir.name, bewertet)
        melde(log, f"Zusammenfassung: {ziel}")
        if bericht_erzeugen:
            import bericht
            html = bericht.baue(manifest, bewertet, bericht.lade_roh(lauf_dir.name))
            (lauf_dir / "bericht.html").write_text(html, encoding="utf-8")
            melde(log, f"Bericht: {lauf_dir / 'bericht.html'}")
    except Exception as exc:
        log.exception("Auswertung nach dem Lauf fehlgeschlagen")
        melde(log, f"Auswertung fehlgeschlagen ({type(exc).__name__}: {exc}) — "
                   f"Rohdaten sind vollständig, nachholen mit: "
                   f"python auswertung/gold/auswertung.py {lauf_dir.name}")


# ---------------------------------------------------------------------------
# Hauptlauf
# ---------------------------------------------------------------------------


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modell", default=STANDARDMODELL)
    ap.add_argument("--faelle", default="alle",
                    help="alle | chargy_beidawi | nutzer_gold | q06,q11,…")
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Fälle")
    ap.add_argument("--timeout", type=int, default=420, help="Sekunden je Fall")
    ap.add_argument("--versuche", type=int, default=2, help="Versuche je Fall")
    ap.add_argument("--pause", type=float, default=2.0, help="Sekunden zwischen Fällen")
    ap.add_argument("--notiz", default="", help="freie Notiz fürs Manifest")
    ap.add_argument("--mcp-url", default=MCP_URL,
                    help="Endpunkt, z. B. eine Testinstanz auf anderem Port (Standard: Produktion)")
    # Retrieval-only: der Server bricht nach der Kandidatenliste ab, kein LLM.
    # 25 Fälle in unter einer Minute statt fünf; damit wird die Stellschraube,
    # an der die meisten Fälle verlorengehen, zum billigsten Experiment.
    ap.add_argument("--nur-retrieval", action="store_true",
                    help="nur Hybrid-Suche und Kandidatenliste messen, kein LLM")
    # Experiment: überschreibt die Suchkonfiguration je Aufruf; der Server
    # prüft die Werte und meldet zurück, was wirksam war — das steht dann im
    # Manifest, nicht die Modulkonstanten.
    ex = ap.add_argument_group("Experiment (Suchkonfiguration je Aufruf)")
    ex.add_argument("--dense", help="Dense-Arme, kommagetrennt aus text,questions; '' = keiner")
    ex.add_argument("--sparse", help="Sparse-Arme, kommagetrennt aus sparse,sparse_expanded; '' = keiner")
    ex.add_argument("--fusion", choices=("dbsf", "rrf"))
    ex.add_argument("--candidate-limit", type=int)
    ex.add_argument("--dedup-buffer", type=int)
    ex.add_argument("--max-per-url", type=int)
    ex.add_argument("--top-docs", type=int)
    ex.add_argument("--kein-boost", action="store_true", help="Keyword-Boost aus")
    ex.add_argument("--collection", help="Qdrant-Collection, z. B. eine Testkopie (hsh_knowledge_v2test)")
    # Premium-Modelle laufen über OpenRouter und kosten Geld; der Server prüft
    # deshalb ein Passwort. Es steht bewusst nicht hier, sondern kommt aus der
    # Umgebung oder vom Aufruf.
    ap.add_argument("--passwort", default=os.environ.get("HSH_FULL_ACCESS_PASSWORD", ""),
                    help="nur für kostenpflichtige Premium-Modelle nötig")
    ap.add_argument("--fortsetzen", type=Path, help="vorhandenes Laufverzeichnis")
    ap.add_argument("--auswerten", action="store_true",
                    help="nach dem Lauf auswerten und zusammenfassung.json schreiben")
    ap.add_argument("--bericht", action="store_true",
                    help="nach dem Lauf bericht.html erzeugen (schließt --auswerten ein)")
    args = ap.parse_args()
    _MCP_URL_AKTIV[0] = args.mcp_url

    faelle = lade_faelle(args.faelle)
    if args.limit:
        faelle = faelle[:args.limit]
    experiment = baue_experiment(args)
    # Der Vertrag, gegen den gemessen wird — als Hash ins Manifest und als
    # Kopie in den Laufordner. Ohne das ist ein Laufvergleich später Raten:
    # Derselbe Lauf vom 18.08. hat je nach Fallbasis 12, 14, 17 oder 18 Treffer.
    fallbasis_hash = hash_fallbasis(lade_fallbasis(FALLBASIS))   # die ganze Fallbasis
    auswahl_hash   = hash_fallbasis(faelle)                       # die gelaufenen Fälle

    # ── Laufverzeichnis ───────────────────────────────────────────────────
    if args.fortsetzen:
        lauf_dir = args.fortsetzen if args.fortsetzen.is_absolute() else konfig.AUSWERTUNG / args.fortsetzen
        if not lauf_dir.exists():
            raise SystemExit(f"Laufverzeichnis fehlt: {lauf_dir}")
        manifest = json.loads((lauf_dir / "manifest.json").read_text(encoding="utf-8"))
        modell = manifest["modell"]
        nur_retrieval = manifest.get("modus") == "retrieval"
        experiment = manifest.get("experiment") or {}
        alt = (manifest.get("fallbasis") or {}).get("hash")
        if alt and alt != fallbasis_hash:
            raise SystemExit(
                f"Fallbasis hat sich geändert (Lauf {hash_kurz(alt)}, jetzt {hash_kurz(fallbasis_hash)}). "
                f"Ein Lauf darf nur gegen einen Vertrag gemessen werden — neu starten "
                f"oder die Fallbasis aus {lauf_dir.name}/fallbasis.json wiederherstellen.")
    else:
        nur_retrieval = args.nur_retrieval
        kurz = "retrieval" if nur_retrieval else args.modell.split("/")[-1].replace(":", "-")
        if experiment:
            kurz += "_" + experiment_kuerzel(experiment)
        lauf_dir = LAEUFE / f"{datetime.now():%Y-%m-%d_%H%M}_{kurz}"
        (lauf_dir / "faelle").mkdir(parents=True, exist_ok=True)
        modell = "" if nur_retrieval else args.modell
        manifest = {
            "lauf_id":       lauf_dir.name,
            "gestartet":     datetime.now().isoformat(timespec="seconds"),
            "modus":         "retrieval" if nur_retrieval else "voll",
            "modell":        modell,
            "experiment":    experiment,
            "mcp_url":       args.mcp_url,
            "fallauswahl":   args.faelle,
            "fallzahl":      len(faelle),
            "notiz":         args.notiz,
            "pause_s":       0.0 if nur_retrieval else args.pause,
            "premium":       bool(args.passwort),
            "fallbasis":     {"hash": fallbasis_hash, "hash_auswahl": auswahl_hash,
                              "fallzahl": len(faelle),
                              "stand": datetime.fromtimestamp(FALLBASIS.stat().st_mtime)
                                               .isoformat(timespec="seconds")},
            "konfiguration": sammle_konfiguration(),
        }
        (lauf_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Die gelaufenen Fälle mit ihrem vollständigen Vertrag — Beleg wie faelle/.
        (lauf_dir / "fallbasis.json").write_text(
            json.dumps(faelle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (lauf_dir / "faelle").mkdir(parents=True, exist_ok=True)
    log = richte_log_ein(lauf_dir)
    erledigt = {p.stem for p in (lauf_dir / "faelle").glob("*.json")
                if not json.loads(p.read_text(encoding="utf-8")).get("fehler")}
    offen = [f for f in faelle if f["id"] not in erledigt]

    k = manifest["konfiguration"]
    melde(log, f"Lauf {lauf_dir.name}")
    melde(log, f"  Modell : {modell or '— (nur Retrieval, kein LLM)'}")
    melde(log, f"  Arme   : {k.get('DENSE_ARMS')} / {k.get('SPARSE_ARMS')}  Fusion: {k.get('FUSION')}")
    melde(log, f"  Punkte : {k.get('points_count')}")
    melde(log, f"  Vertrag: {hash_kurz(fallbasis_hash)} ({len(faelle)} Fälle)")
    if experiment:
        melde(log, f"  Experiment: {json.dumps(experiment, ensure_ascii=False)}")
    melde(log, f"  Fälle  : {len(offen)} offen von {len(faelle)}\n")

    t_wall = time.perf_counter()
    gemessen = 0.0

    for i, fall in enumerate(offen, start=1):
        t0 = time.perf_counter()
        ergebnis: dict = {}
        for versuch in range(1, args.versuche + 1):
            try:
                ergebnis = await stelle_frage(fall, modell, args.timeout, args.passwort,
                                              nur_retrieval=nur_retrieval,
                                              experiment=experiment)
                if not ergebnis.get("fehler"):
                    break
            except Exception as exc:
                ergebnis = {"fehler": f"{type(exc).__name__}: {str(exc)[:300]}"}
            if versuch < args.versuche:
                await asyncio.sleep(5)
        dauer = time.perf_counter() - t0
        gemessen += dauer

        datensatz = {
            "fall":          fall,
            "modell":        modell,
            "zeitstempel":   datetime.now().isoformat(timespec="seconds"),
            "dauer_s":       round(dauer, 2),
            "fehler":        ergebnis.get("fehler"),
            "antwort":       ergebnis.get("antwort", ""),
            "kandidaten":    ergebnis.get("kandidaten", []),
            "ausgewaehlte_docs": ergebnis.get("ausgewaehlte_docs", []),
            "quellen":       ergebnis.get("quellen", []),
            "chunks":        ergebnis.get("chunks", []),
            "arme":          ergebnis.get("arme", {}),
            "fakultaet_wirksam":  ergebnis.get("fakultaet", ""),
            "fakultaet_erkannt":  ergebnis.get("fakultaet_erkannt", ""),
            "token_budget":  ergebnis.get("token_budget", {}),
            "zeiten":        ergebnis.get("zeiten", {}),
            "auswahl_begruendung": ergebnis.get("auswahl_begruendung", ""),
        }
        (lauf_dir / "faelle" / f"{fall['id']}.json").write_text(
            json.dumps(datensatz, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Die wirksame Konfiguration meldet der Server zurück. Sie ersetzt im
        # Manifest, was sammle_konfiguration() aus den Modulkonstanten geraten
        # hat — nur so kann das Manifest nicht mehr etwas anderes behaupten als
        # das, was gesucht hat (bis 21.09.2026 stand dort ein Reranker, der nie lief).
        if ergebnis.get("konfiguration") and not manifest["konfiguration"].get("_wirksam"):
            manifest["konfiguration"].update(ergebnis["konfiguration"])
            manifest["konfiguration"]["_wirksam"] = True
            manifest["konfiguration"].pop("USE_RERANKER", None)
            (lauf_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        zeichen = "✗" if ergebnis.get("fehler") else "✓"
        melde(log, f"[{i:>2}/{len(offen)}] {zeichen} {dauer:>6.1f}s  {fall['id']}  "
                   f"{fall['question'][:58]}")
        if ergebnis.get("fehler"):
            log.warning("%s: %s", fall["id"], ergebnis["fehler"])

        pause = 0.0 if nur_retrieval else args.pause
        if i < len(offen):
            await asyncio.sleep(pause)

    wall = time.perf_counter() - t_wall
    pausen = (0.0 if nur_retrieval else args.pause) * max(len(offen) - 1, 0)
    # Nur echte Phasen summieren: zeiten["quellen_s"] ist eine kumulative
    # Zwischensumme (t_load − t_start) und würde doppelt zählen.
    manifest["laufzeit"] = {
        "wall_clock_s":  round(wall, 1),
        "gemessen_s":    round(gemessen, 1),
        "geplante_pausen_s": round(pausen, 1),
        "ungetrackt_s":  round(wall - gemessen - pausen, 1),
        "beendet":       datetime.now().isoformat(timespec="seconds"),
    }
    (lauf_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    melde(log, f"\nFertig in {wall/60:.1f} min "
               f"(Arbeit {gemessen/60:.1f} · Pausen {pausen/60:.1f} · "
               f"ungetrackt {(wall-gemessen-pausen)/60:.1f})")
    melde(log, f"Rohdaten: {lauf_dir}")

    if args.auswerten or args.bericht:
        schliesse_kette(lauf_dir, bericht_erzeugen=args.bericht, log=log)
    else:
        melde(log, f"Auswertung: python auswertung/gold/auswertung.py {lauf_dir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
