"""Gemeinsame Bausteine der drei HTML-Berichte.

Drei Berichte lesen dieselben Rohdaten und zeigen sie aus drei Richtungen:
`bericht.py` einen Volllauf (Antwort und Quellen), `bericht_retrieval.py`
einen Retrieval-Lauf (Ränge und Arme), `bericht_vergleich.py` zwei Läufe
gegeneinander. Was sie teilen, steht hier — Seitengerüst, Konfigurationsband,
Escaping, die Panels für Trefferliste, Kandidaten und Quellen — damit ein
Chunk in allen drei Berichten gleich aussieht und eine Korrektur nur einmal
nötig ist.

Zahlen kommen ausschließlich aus auswertung.py (bewerte_fall, trichter,
quadranten, kennzahlen). Hier wird nichts gerechnet, nur gezeigt.
"""
from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))

from auswertung import LAEUFE, URSACHEN  # noqa: E402
from raghsh.crawl.crawler_utils import normalize_url  # noqa: E402  (Pfad kommt aus auswertung)
from vertrag import kurz as hash_kurz  # noqa: E402

STILDATEI = _BASE / "bericht.css"

# Wie viele Zeilen der Trefferliste je Fall in die Seite kommen. Ein Volllauf
# hat 64 × 30 + 11 × 120 Chunks; alle einzubetten machte den Bericht 5 MB
# schwer. Die ersten zehn zeigen, was oben stand; Goldtreffer darunter werden
# zusätzlich gezeigt, damit „Gold auf Rang 61" belegt bleibt.
CHUNKS_IM_BERICHT = 10

# Feldwerte aus der Fallbasis in lesbare Sprache. Die englischen Schlüssel
# stammen aus der Fallbasis von Charghy und El Beidawi und bleiben dort
# unverändert, damit ein Vergleich mit ihrem Lauf möglich bleibt.
HERKUNFT = {
    "chargy_beidawi": "Fallbasis Charghy / El Beidawi",
    "nutzer_gold": "eigener Goldstandard",
    "eval_gold_50": "Fragenmenge eval/fragen.csv",
}
HERKUNFT_SATZ = {
    "chargy_beidawi": "aus der Fallbasis von Charghy und El Beidawi",
    "nutzer_gold": "aus dem eigenen Goldstandard",
    "eval_gold_50": "aus der Fragenmenge eval/fragen.csv (ohne Pflichtbegriffe, "
                    "nur über die Quellenlage bewertbar)",
}
_HERKUNFT_REIHE = ("chargy_beidawi", "nutzer_gold", "eval_gold_50")


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def _e(text) -> str:
    # U+FFFD stammt aus PDFs, deren Textebene beim Crawlen unlesbare Stellen
    # hatte, und steht so auch im Index. Im Bericht wird daraus ein sichtbares
    # Kästchen: Als Ersatzzeichen belassen wäre es beim Ausliefern der Seite ein
    # Kodierungsfehler, stillschweigend entfernt würde die Lücke unsichtbar.
    return html.escape(str(text if text is not None else "")).replace("�", "▯")


def _zahl(n) -> str:
    return f"{n:,}".replace(",", ".") if isinstance(n, int) else str(n)


def _kurz(text: str, n: int) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _liste(werte, leer: str = "—") -> str:
    werte = [w for w in (werte or []) if w]
    if not werte:
        return f'<span class="pfad-zusatz">{_e(leer)}</span>'
    return "<ul>" + "".join(f"<li>{_e(w)}</li>" for w in werte) + "</ul>"


def herkunft_satz(bewertet: list[dict]) -> str:
    """„19 aus der Fallbasis …, 6 aus dem eigenen Goldstandard, 50 aus …"."""
    from collections import Counter
    zaehler = Counter(b["herkunft"] for b in bewertet)
    reihe = [h for h in _HERKUNFT_REIHE if h in zaehler] + \
            [h for h in zaehler if h not in _HERKUNFT_REIHE]
    return ", ".join(f"{zaehler[h]} {HERKUNFT_SATZ.get(h, h)}" for h in reihe)


# ---------------------------------------------------------------------------
# Seite
# ---------------------------------------------------------------------------
def lade_stil() -> str:
    """Das Stylesheet liegt als CSS-Datei daneben, nicht als Python-String.

    Vorher waren es 332 Zeilen in einem Literal: ohne Syntaxhervorhebung, ohne
    Editorhilfe und mit der ständigen Gefahr, dass jemand die Datei in einen
    f-String verwandelt und an jeder geschweiften Klammer scheitert.
    """
    if not STILDATEI.exists():
        raise SystemExit(f"Stylesheet fehlt: {STILDATEI} — gehört neben bericht_teile.py.")
    return STILDATEI.read_text(encoding="utf-8")


def seite(titel: str, koerper: str) -> str:
    """Vollständiges HTML-Dokument. Die Zeichensatzangabe ist Pflicht: Ohne sie
    rät der Browser bei einer lokal geöffneten Datei und zeigt Umlaute je nach
    System falsch an."""
    return (f'<!DOCTYPE html>\n<html lang="de">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{_e(titel)}</title>\n<style>{lade_stil()}</style>\n</head>\n'
            f'<body>\n<div class="blatt">\n{koerper}\n</div>\n</body>\n</html>\n')


def fuss(quelle: str, saetze: list[str] = ()) -> str:
    absaetze = "".join(f"<p>{s}</p>" for s in saetze)
    return (f'<div class="fuss">{absaetze}'
            f'<p>Erzeugt {datetime.now():%d.%m.%Y %H:%M} aus '
            f'<span class="mono">{_e(quelle)}</span>.</p></div>')


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
def konfig_felder(manifest: dict) -> list[tuple[str, str]]:
    """Was den Lauf von einem anderen unterscheidet — als (Name, Wert)-Paare.

    Der Reranker wird seit 21.09.2026 vom Server als `reranker_aktiv`
    gemeldet. Ältere Manifeste tragen nur die Konstante USE_RERANKER, die der
    Server nie ausgewertet hat; das steht dann auch so da.
    """
    k = manifest.get("konfiguration") or {}
    lz = manifest.get("laufzeit") or {}
    fb = manifest.get("fallbasis") or {}
    if k.get("reranker_aktiv") is None:
        reranker = "unbekannt (Konstante " + ("an" if k.get("USE_RERANKER") else "aus") + \
                   ", vom Server nie aufgerufen)"
    else:
        reranker = "an" if k.get("reranker_aktiv") else "aus"
    kandidaten = f"Top {k.get('TOP_DOCS') or 8} Dokumente"
    if k.get("FAKULTAET_FAKTOR"):
        kandidaten += f" · bei Fakultätsfrage ×{k['FAKULTAET_FAKTOR']}"
    vertrag = hash_kurz(fb.get("hash") or fb.get("hash_auswahl"))
    if fb.get("rekonstruiert"):
        vertrag += " (rekonstruiert)"
    if manifest.get("vertrag"):
        vertrag = "⚠ " + manifest["vertrag"]
    felder = [
        ("Modell", manifest.get("modell") or "— (nur Retrieval)"),
        ("Dense-Arme", ", ".join(k.get("DENSE_ARMS") or []) or "—"),
        ("Sparse-Arme", ", ".join(k.get("SPARSE_ARMS") or []) or "—"),
        ("Fusion", (k.get("FUSION") or "—").upper()),
        ("Kandidaten", kandidaten),
        ("Stichwort-Boost", "aus" if k.get("keyword_boost") is False else "an"),
        ("Reranker", reranker),
        ("Vertrag", vertrag),
        ("Index", _zahl(k.get("points_count")) + " Chunks"),
        ("Laufzeit", f'{lz.get("wall_clock_s", 0) / 60:.1f} min' if lz else "—"),
    ]
    return felder


def konfig_band(manifest: dict) -> str:
    felder = "".join(f"<div><dt>{_e(t)}</dt><dd>{_e(w)}</dd></div>"
                     for t, w in konfig_felder(manifest))
    notiz = f' · {_e(manifest["notiz"])}' if manifest.get("notiz") else ""
    experiment = ""
    if manifest.get("experiment"):
        experiment = (f'<p class="merkmal">Experiment gegenüber den Modulkonstanten: '
                      f'<span class="mono">{_e(json.dumps(manifest["experiment"], ensure_ascii=False))}'
                      f'</span></p>')
    return (f'<dl class="konfig">{felder}</dl>'
            f'<p class="merkmal">Lauf <span class="mono">{_e(manifest["lauf_id"])}</span>{notiz}</p>'
            f'{experiment}')


# ---------------------------------------------------------------------------
# Rohdaten und Goldprüfung
# ---------------------------------------------------------------------------
def lade_roh(lauf_id: str) -> dict[str, dict]:
    """Die unveränderten Falldateien des Laufs, nach Fall-ID."""
    ordner = LAEUFE / lauf_id / "faelle"
    return {d["fall"]["id"]: d
            for d in (json.loads(p.read_text(encoding="utf-8"))
                      for p in sorted(ordner.glob("*.json")))}


def ziel_urls(fall: dict) -> set[str]:
    """Die URLs, die für diesen Fall als Goldtreffer zählen — wie in bewerte_fall,
    einschließlich der Beleg-URLs aus der Chunk-Vertretung."""
    ziele = [fall.get("source_url", "")]
    if fall.get("source_policy") == "accepted_sources":
        ziele += list(fall.get("accepted_source_urls") or [])
    ziele += list(fall.get("beleg_urls") or [])
    return {normalize_url(z) for z in ziele if z}


def ziel_ids(fall: dict) -> set[str]:
    return set(fall.get("beleg_punkte") or []) | set(fall.get("gold_punkte") or [])


def ist_gold(url: str, ziele: set[str]) -> bool:
    return bool(url) and normalize_url(url) in ziele


def ist_gold_chunk(chunk: dict, fall: dict, ziele: set[str]) -> bool:
    """Punktgenau wie bewerte_fall._rang_chunk: ID oder URL."""
    return str(chunk.get("id", "")) in ziel_ids(fall) or ist_gold(chunk.get("source_url", ""), ziele)


GOLD_MARKE = '<span class="chip chip--ja">Gold</span> '
LEER_HINWEIS = '<p class="panel-hinweis">keine</p>'


def panel(titel: str, hinweis: str, inhalt: str) -> str:
    return (f'<div class="panel"><p class="panel-titel">{_e(titel)}</p>'
            f'<p class="panel-hinweis">{_e(hinweis)}</p>{inhalt}</div>')


def marken(faelle: list[dict]) -> str:
    return "".join(
        f'<span class="marke" title="{_e(b["frage"])}">{_e(b["id"])}</span>'
        for b in sorted(faelle, key=lambda x: x["id"]))


def weg(b: dict) -> str:
    """„Chunk 61 → Kandidat 13 → Quelle 1" als HTML-Band."""
    teile = [("Chunk", b["rang_chunk"]), ("Kandidat", b["rang_kandidat"])]
    if b.get("antwort_status") != "entfaellt":
        teile.append(("Quelle", b["rang_quelle"]))
    return '<span class="wegband">' + ' <span class="pfeil">→</span> '.join(
        f'{name} <span class="rang mono">{r if r else "—"}</span>' for name, r in teile) + "</span>"


# ---------------------------------------------------------------------------
# Trichter — gemeinsam für Voll- und Retrieval-Bericht
# ---------------------------------------------------------------------------
def trichter_html(stufen: list[dict]) -> str:
    gesamt = stufen[0]["anzahl"]
    zeilen = []
    for s in stufen:
        anteil = s["anzahl"] / max(gesamt, 1) * 100
        zeilen.append(
            f'<div class="stufe">'
            f'  <div class="stufe-name">{_e(s["label"])}</div>'
            f'  <div class="spur"><div class="spur-voll" style="width:{anteil:.2f}%"></div></div>'
            f'  <div class="stufe-wert mono">{s["anzahl"]}</div>'
            f'</div>')
        if s["verloren"]:
            _, grund, _f = URSACHEN[s["ursache"]]
            ids = " ".join(s["verlust_ids"])
            zeilen.append(
                f'<div class="verlust-zeile"><div></div><div class="verlust-text">'
                f'<b>−{s["verloren"]}</b> {_e(grund)} '
                f'<span class="verlust-ids mono">{_e(ids)}</span></div></div>')
    return "".join(zeilen)


# ---------------------------------------------------------------------------
# Panels der Pipeline-Stufen
# ---------------------------------------------------------------------------
def panel_chunks(d: dict, ziele: set[str], fusion: str = "DBSF") -> str:
    """Trefferliste der Suche — die ersten CHUNKS_IM_BERICHT Zeilen und jeder
    Goldtreffer darüber hinaus."""
    fall = d["fall"]
    chunks = d.get("chunks", [])
    zeilen, ausgelassen = [], 0
    for c in chunks:
        gold = ist_gold_chunk(c, fall, ziele)
        if c.get("rang", 0) > CHUNKS_IM_BERICHT and not gold:
            ausgelassen += 1
            continue
        if ausgelassen and gold:
            zeilen.append(f'<tr><td colspan="5" class="pfad-zusatz">… {ausgelassen} Zeilen '
                          f'ohne Gold ausgelassen …</td></tr>')
            ausgelassen = 0
        voll = c.get("text_vollstaendig")
        zeilen.append(
            f'<tr class="{"gold" if gold else ""}">'
            f'<td class="z mono">{_e(c.get("rang"))}</td>'
            f'<td class="z mono">{c.get("score", 0):.3f}</td>'
            f'<td>{GOLD_MARKE if gold else ""}<span class="dok-titel">{_e(_kurz(c.get("title_anzeige"), 110))}</span>'
            f'<span class="dok-auszug">{_e(_kurz(c.get("section_heading"), 90))} — '
            f'{_e(_kurz(c.get("auszug"), 180))}</span>'
            f'<a class="dok-url" href="{_e(c.get("source_url"))}">{_e(_kurz(c.get("source_url"), 100))}</a></td>'
            f'<td class="z mono">{_e(c.get("chunk_index"))}</td>'
            f'<td>{"ganz" if voll else "Fragment"}</td></tr>')
    if ausgelassen:
        zeilen.append(f'<tr><td colspan="5" class="pfad-zusatz">… {ausgelassen} weitere '
                      f'Zeilen ohne Gold …</td></tr>')
    tabelle = (f'<div class="tabellenhuelle"><table class="daten"><thead><tr>'
               f'<th class="z">Rang</th><th class="z">Score</th><th>Dokument, Abschnitt, Auszug</th>'
               f'<th class="z">Chunk-Nr.</th><th>Umfang</th></tr></thead>'
               f'<tbody>{"".join(zeilen)}</tbody></table></div>')
    fak = d.get("fakultaet_wirksam") or d.get("fakultaet_erkannt")
    fak_hinweis = (f' Die Frage nennt die Fakultät {_e(fak)}: Es liefen zwei Suchen — eine auf '
                   f'die Fakultät gefiltert, eine über zentrale Seiten — und die Liste ist '
                   f'die Vereinigung beider.' if fak else "")
    return (f'<details class="unterpanel"><summary>Trefferliste der Suche — '
            f'{len(chunks)} Chunks, die ersten {min(CHUNKS_IM_BERICHT, len(chunks))} '
            f'und jeder Goldtreffer</summary>'
            f'<p class="panel-hinweis">Ein Chunk ist ein Textstück eines Dokuments, '
            f'nicht das Dokument selbst. Der Score kommt aus der {_e(fusion)}-Fusion der '
            f'Arme und ist nur innerhalb dieser Liste vergleichbar. '
            f'„Fragment" heißt: Der Chunk gehört zu einem Dokument, von dem der Index '
            f'nicht alle Teile hat.{fak_hinweis}</p>{tabelle}</details>')


def panel_kandidaten(d: dict, ziele: set[str], top_docs: int) -> str:
    platz = {doc.get("nr"): i for i, doc in enumerate(d.get("ausgewaehlte_docs", []), start=1)}
    zeilen = []
    for k in d.get("kandidaten", []):
        gold = ist_gold(k.get("url", ""), ziele)
        p = platz.get(k.get("nr"))
        zeilen.append(
            f'<tr class="{"gold" if gold else ""}">'
            f'<td class="z mono">{_e(k.get("nr"))}</td>'
            f'<td>{GOLD_MARKE if gold else ""}'
            f'<span class="dok-titel">{_e(_kurz(k.get("titel"), 110))}</span>'
            f'<span class="dok-auszug">{_e(_kurz(k.get("struktur"), 200))}</span>'
            f'<a class="dok-url" href="{_e(k.get("url"))}">{_e(_kurz(k.get("url"), 100))}</a></td>'
            f'<td class="mono">{_e(k.get("fakultaet") or "—")}</td>'
            f'<td class="z mono">{p if p else "—"}</td></tr>')
    tabelle = (f'<div class="tabellenhuelle"><table class="daten"><thead><tr>'
               f'<th class="z">Nr.</th><th>Dokument mit Gliederung</th>'
               f'<th>Fakultät</th><th class="z">Auswahlplatz</th></tr></thead>'
               f'<tbody>{"".join(zeilen)}</tbody></table></div>')
    anzahl = len(d.get("kandidaten", []))
    fak = d.get("fakultaet_wirksam") or d.get("fakultaet_erkannt")
    grenze = top_docs
    erklaerung = ""
    if fak and anzahl > top_docs:
        grenze = anzahl
        erklaerung = (f' Bei einer Fakultätsfrage ist die Liste länger: je zur Hälfte aus der '
                      f'Fakultätssuche und aus zentralen Seiten.')
    elif anzahl > top_docs:
        erklaerung = (f' Über {top_docs} hinaus stehen hier {anzahl - top_docs} Dokumente, die der '
                      f'Stichwort-Boost direkt nachgeladen hat — Eigennamen und Modulkürzel aus der '
                      f'Frage, unabhängig vom Suchergebnis.')
    auswahl = ""
    if d.get("auswahl_begruendung"):
        try:
            roh = json.dumps(json.loads(d["auswahl_begruendung"]), ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            roh = str(d["auswahl_begruendung"])
        auswahl = (f'<p class="panel-hinweis">Rohantwort der Auswahl: '
                   f'<span class="mono">{_e(_kurz(roh, 600))}</span></p>')
    return (f'<details class="unterpanel"><summary>Kandidatenliste — '
            f'{anzahl} Dokumente zur Auswahl</summary>'
            f'<p class="panel-hinweis">Aus den Chunks werden Dokumente: Von den Treffern '
            f'oben bleiben die ersten {grenze} <em>verschiedenen</em> Quellen übrig, je '
            f'Dokument mit seiner Gliederung als Entscheidungsgrundlage.{erklaerung} '
            f'Diese Liste geht an die LLM-Auswahl der ersten Runde; sie antwortet mit '
            f'den Nummern, die sie behalten will. Der Auswahlplatz gibt ihre Reihenfolge '
            f'wieder, nicht den Suchrang.</p>{tabelle}{auswahl}</details>')


def panel_quellen(d: dict, ziele: set[str]) -> str:
    karten = []
    for q in d.get("quellen", []):
        gold = ist_gold(q.get("url", ""), ziele)
        schnipsel = "".join(f'<p class="schnipsel">{_e(_kurz(s, 700))}</p>'
                            for s in (q.get("snippets") or []))
        fremd = q.get("fremde_fundstellen") or []
        fremd_html = ""
        if fremd:
            fremd_html = (f'<p class="panel-hinweis">Der Text stand wortgleich auch in '
                          f'{len(fremd)} weiteren Dokumenten — '
                          f'{_e(_kurz(", ".join(fremd), 260))}</p>')
        marke = GOLD_MARKE.replace(">Gold<", ">Goldquelle<") if gold else ""
        karten.append(
            f'<div class="quelle-karte"><div class="quelle-kopf">'
            f'{marke}'
            f'<span class="quelle-titel">{_e(q.get("titel"))}</span>'
            f'<span class="quelle-meta">gecrawlt {_e(q.get("crawl_date") or "—")}</span>'
            f'</div>'
            f'<a class="dok-url" href="{_e(q.get("url"))}">{_e(q.get("url"))}</a>'
            f'<p class="quelle-meta">Fundstelle: {_e(q.get("abschnitt") or "—")}</p>'
            f'{schnipsel}{fremd_html}</div>')
    return (f'<details class="unterpanel"><summary>Quellen im Kontext — '
            f'{len(d.get("quellen", []))} Dokumente</summary>'
            f'<p class="panel-hinweis">Diese Dokumente lagen der zweiten Runde '
            f'vollständig vor, nicht nur als Chunk. „Fundstelle" ist die Überschriftenkette '
            f'des Abschnitts, über den das Dokument gefunden wurde; der Ausschnitt darunter '
            f'ist die Stelle selbst.</p>{"".join(karten) or LEER_HINWEIS}'
            f'</details>')


def panel_messwerte(d: dict) -> str:
    z = d.get("zeiten") or {}
    phasen = [("Suche", "hybrid_suche_s"), ("Arm-Diagnose", "arme_s"), ("Kandidaten", "kandidaten_s"),
              ("LLM-Auswahl", "llm_auswahl_s"), ("Laden", "docs_laden_s"),
              ("Antwort", "antwort_llm_s")]
    werte = [(name, float(z.get(k) or 0)) for name, k in phasen if z.get(k)]
    summe = sum(w for _n, w in werte) or 1.0
    spur = "".join(f'<span style="width:{w / summe * 100:.2f}%" title="{_e(n)} {w:.2f} s"></span>'
                   for n, w in werte if w > 0)
    legende = "".join(f'<li>{_e(n)} <span class="mono">{w:.2f} s</span></li>' for n, w in werte)

    tb = d.get("token_budget") or {}
    verwendet, maximum = tb.get("verwendet") or 0, tb.get("maximum") or 0
    kacheln = [("Gesamtdauer", f'{d.get("dauer_s") or 0:.2f} s')]
    if maximum:
        kacheln += [("Kontext", _zahl(verwendet) + " Token"),
                    ("Budget genutzt", f"{verwendet / maximum * 100:.0f} %")]
    kacheln.append(("Gelaufen", (d.get("zeitstempel") or "—").replace("T", " ")))
    kachel_html = "".join(f'<div class="mess"><div class="mess-titel">{_e(t)}</div>'
                          f'<div class="mess-wert mono">{_e(w)}</div></div>' for t, w in kacheln)
    return (f'<details class="unterpanel"><summary>Messwerte des Durchlaufs</summary>'
            f'<p class="panel-hinweis">Der Balken zeigt, wohin die Zeit ging. '
            f'Zwischensummen aus der Rohdatei sind nicht noch einmal aufgeführt — '
            f'sonst zählte die Zeit doppelt.</p>'
            f'<div class="zeitspur">{spur}</div>'
            f'<ul class="zeitlegende">{legende}</ul>'
            f'<div class="messwerte" style="margin-top:.7rem">{kachel_html}</div></details>')


def panel_arme(d: dict, ziele: set[str], b: dict) -> str:
    """Je Arm die ersten fünf Treffer und der Goldrang — nur Retrieval-Läufe
    tragen `arme`. Titel gibt es dort nicht; wo die URL auch in der
    Trefferliste steht, wird ihr Titel geliehen."""
    arme = d.get("arme") or {}
    if not arme:
        return ""
    titel_von = {c.get("source_url"): c.get("title_anzeige") for c in d.get("chunks", [])}
    titel_von.update({k.get("url"): k.get("titel") for k in d.get("kandidaten", [])})
    spalten = []
    for arm, treffer in arme.items():
        rang = (b.get("rang_arm") or {}).get(arm)
        zeilen = []
        for t in treffer[:5]:
            gold = ist_gold(t.get("source_url", ""), ziele)
            url = t.get("source_url", "")
            zeilen.append(
                f'<li class="{"gold" if gold else ""}"><span class="mono">{t.get("rang")}</span> '
                f'{GOLD_MARKE if gold else ""}{_e(_kurz(titel_von.get(url) or url, 70))}</li>')
        stand = (f'Gold auf Rang <span class="mono">{rang}</span>' if rang
                 else 'Gold nicht in den Top-50')
        spalten.append(
            f'<div class="arm"><div class="pfad-titel">{_e(arm)}</div>'
            f'<div class="pfad-wert">{stand}</div>'
            f'<ol class="armliste">{"".join(zeilen)}</ol></div>')
    return (f'<details class="unterpanel" open><summary>Die Arme einzeln — Top-5 je Arm, '
            f'Goldrang in den Top-50</summary>'
            f'<p class="panel-hinweis">Jeder Arm wurde für die Diagnose zusätzlich allein '
            f'gefragt. Steht das Gold hier oben, aber nicht in der Trefferliste, hat die '
            f'Fusion es verloren; steht es nirgends, findet es kein Arm.</p>'
            f'<div class="arme">{"".join(spalten)}</div></details>')
