"""
Gesamtbericht: alle Fragen eines Messstands auf einer Seite — eval_gold (Volllauf),
eigene PO-Testmenge und Holdout, je Fall mit Frage, Erwartung, Antwort, Begriffen
und Quellen.

Aufruf über bericht.py:
    python auswertung/gold/bericht.py --gesamt
    python auswertung/gold/bericht.py --gesamt --eval <lauf> --po <lauf> --holdout <lauf>

Ohne Angaben nimmt er je Menge den **Stand**: den jüngsten vollständigen Luna-Lauf
(eval_gold ab 50, PO ab 20 Fällen) plus alle späteren Luna-Teilläufe derselben Menge
(Nachmessungen einzelner Fälle; der jüngste Lauf gilt je Fall). Gewertet wird nach dem
**aktuellen** Vertrag (fallbasis.json bzw. pruefung.json der Agenten-Instanz), mit
denselben Funktionen wie auswertung.py bzw. lauf_po.py. Den Holdout nimmt er als
jüngsten `*_voll`-Lauf (bewertung_neu.json, falls vorhanden).
Mit --eval/--po lassen sich Läufe kommagetrennt vorgeben (spätere überschreiben frühere).

Weil die Seite die Holdout-Fälle enthält, liegt sie immer unter
/home/christian/RAGHSH_holdout/berichte/ — nie im Agenten-Repo, im Dashboard oder in
/var/log/raghsh-agent.

Die Rohdaten der Läufe werden nicht verändert; die Neuwertung entsteht nur für diese Seite.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

from bericht_po import (EXTRA_STIL, ERGEBNIS, HOLDOUT, KATEGORIEN, PO_LAEUFE, _fallkarte,  # noqa: E402
                        _json, ergebnis, finde_lauf, lade_holdout, po_eintrag)
from bericht_teile import _e, _kurz, fuss, panel, panel_messwerte, panel_quellen, seite  # noqa: E402
from vergleich import anzeige, enthaelt, enthaelt_begriff  # noqa: E402

AGENT_EVAL = Path("/home/raghsh-agent/RAGHSH_agent/hsh_scraper/eval_gold")
EVAL_LAEUFE = AGENT_EVAL / "laeufe"
PO_AGENT = AGENT_EVAL / "po" / "laeufe"


def _agent_modul(name: str, pfad: Path):
    """Modul der Agenten-Instanz laden (auswertung.py mit Belegprüfung, lauf_po.py) —
    unter eigenem Namen, damit es die gleichnamigen Module hier nicht verdrängt."""
    spec = importlib.util.spec_from_file_location(name, pfad)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul
ABLAGE = HOLDOUT / "berichte"

# Diagnose aus auswertung.py → Ergebnis dieser Seite
DIAGNOSE_ERGEBNIS = {
    "treffer": "richtig", "treffer_andere_quelle": "richtig",
    "unbestimmt": "teilweise",
    "antwort_unvollstaendig": "falsch", "auswahl_fehler": "falsch", "antwort_verweigert": "falsch",
    "verbotene_aussage": "falsch", "retrieval_fehler": "falsch", "korpus_luecke": "falsch",
    "lauffehler": "fehler",
}
DIAGNOSE_NAME = {
    "treffer": "Treffer", "treffer_andere_quelle": "Treffer aus anderer Quelle",
    "unbestimmt": "nicht prüfbar", "antwort_unvollstaendig": "Antwortproblem",
    "auswahl_fehler": "Antwortproblem (Auswahl)", "antwort_verweigert": "keine Antwort trotz Gold",
    "verbotene_aussage": "verbotene Aussage", "retrieval_fehler": "Retrievalproblem",
    "korpus_luecke": "Korpuslücke", "lauffehler": "Lauffehler",
}


# ---------------------------------------------------------------------------
# Läufe finden
# ---------------------------------------------------------------------------
def _stand(ort: Path, mindestens: int) -> list[Path]:
    """Jüngster Luna-Lauf mit mindestens `mindestens` Fällen plus alle späteren Luna-Läufe."""
    laeufe = sorted(d for d in ort.iterdir() if d.is_dir() and "luna" in d.name
                    and "_retrieval" not in d.name and (d / "faelle").is_dir())
    voll = [d for d in laeufe if len(list((d / "faelle").glob("*.json"))) >= mindestens]
    if not voll:
        raise SystemExit(f"Kein vollständiger Luna-Lauf unter {ort}")
    return [d for d in laeufe if d.name >= voll[-1].name]


def juengster_eval() -> list[Path]:
    return _stand(EVAL_LAEUFE, 50)


def juengster_po() -> list[Path]:
    return _stand(PO_AGENT, 20)


def juengster_holdout() -> Path:
    for d in sorted((HOLDOUT / "laeufe").iterdir(), reverse=True):
        if d.is_dir() and d.name.endswith("_voll") and (d / "bewertung.json").exists():
            return d
    raise SystemExit("Kein Holdout-Volllauf gefunden")


def _liste(angabe: str | None, ort: Path, standard) -> list[Path]:
    if not angabe:
        return standard()
    aus = []
    for a in angabe.split(","):
        p = Path(a.strip())
        p = p if p.is_dir() else ort / a.strip()
        if not p.is_dir():
            raise SystemExit(f"Lauf nicht gefunden: {a}")
        aus.append(p.resolve())
    return aus


# ---------------------------------------------------------------------------
# eval_gold je Fall
# ---------------------------------------------------------------------------
def lade_eval(laeufe: list[Path]) -> tuple[list[dict], dict, dict]:
    """Nach aktuellem Vertrag gewertet, je Fall der jüngste Lauf. Braucht quelltexte.json
    in jedem Lauf (entsteht bei der ersten Auswertung als Agent)."""
    A = _agent_modul("auswertung_agent", AGENT_EVAL / "auswertung.py")
    aktuell = {f["id"]: f for f in json.loads((AGENT_EVAL / "fallbasis.json").read_text(encoding="utf-8"))}
    je_fall: dict[str, dict] = {}
    for lauf in laeufe:
        _, bewertet = A.lade_lauf(lauf.name, vertrag_aktuell=True)
        for b in bewertet:
            d = _json(lauf / "faelle" / f"{b['id']}.json")
            je_fall[b["id"]] = {"b": b, "roh": d, "fall": aktuell.get(b["id"], d.get("fall") or {}),
                                "lauf": lauf.name}
    faelle = [je_fall[i] for i in sorted(je_fall)]
    manifest = _json(laeufe[0] / "manifest.json") if (laeufe[0] / "manifest.json").exists() else {}
    return faelle, A.kennzahlen([e["b"] for e in faelle]), manifest


def lade_po_aktuell(laeufe: list[Path]) -> list[dict]:
    """PO-Fälle nach aktueller pruefung.json gewertet (lauf_po.werte_fall), je Fall der jüngste Lauf."""
    LP = _agent_modul("lauf_po_agent", AGENT_EVAL / "po" / "lauf_po.py")
    aktuell = {f["id"]: f for f in
               json.loads((AGENT_EVAL / "po" / "pruefung.json").read_text(encoding="utf-8"))["faelle"]}
    je_fall: dict[str, dict] = {}
    for lauf in laeufe:
        for datei in sorted((lauf / "faelle").glob("*.json")):
            d = _json(datei)
            fid = (d.get("fall") or {}).get("id")
            if fid not in aktuell:
                continue
            d["fall"] = aktuell[fid]
            je_fall[fid] = po_eintrag(aktuell[fid], d, LP.werte_fall(d, "voll"))
    return [je_fall[i] for i in sorted(je_fall)]


def eval_ergebnis(e: dict) -> tuple[str, str]:
    b = e["b"]
    schluessel = DIAGNOSE_ERGEBNIS.get(b["diagnose"], "falsch")
    grund = DIAGNOSE_NAME.get(b["diagnose"], b["diagnose"])
    if schluessel == "richtig" and b.get("erfundene_fakten"):
        return "teilweise", grund + ", aber erfundene Fakten"
    return schluessel, grund


def _eval_karte(e: dict) -> str:
    b, d, fall = e["b"], e["roh"], e["fall"]
    schluessel, grund = eval_ergebnis(e)
    name, farbe = ERGEBNIS[schluessel]
    antwort = d.get("antwort") or ""
    gold = {fall.get("source_url", ""), *(fall.get("accepted_source_urls") or [])} - {""}
    kopf = (f'<summary><span class="fall-id">{_e(b["id"])}</span>'
            f'<span class="fall-frage">{_e(fall.get("question") or b.get("frage", ""))}</span>'
            f'<span class="ergebnis">{_e(name)}</span></summary>')
    vertrag = (f'<dl class="begriffe"><dt>Referenzantwort</dt><dd>{_e(fall.get("reference_answer", ""))}</dd>'
               f'<dt>Goldquelle</dt><dd><a class="dok-url" href="{_e(fall.get("source_url", ""))}">'
               f'{_e(fall.get("source_url", ""))}</a>'
               + (f' (+{len(fall.get("accepted_source_urls") or [])} akzeptierte)'
                  if fall.get("accepted_source_urls") else "")
               + f'</dd><dt>Herkunft</dt><dd>{_e(fall.get("herkunft", ""))}</dd></dl>')
    chips = []
    for p in fall.get("pflicht_begriffe") or []:
        ok = enthaelt_begriff(antwort, p)
        chips.append(f'<span class="chip {"chip--ja" if ok else "chip--nein"}">{"✓" if ok else "✗"} '
                     f'{_e(anzeige(p))}</span>')
    begriffe = (f'<div class="chipreihe">{"".join(chips)}</div>' if chips else
                '<p class="panel-hinweis">Keine Pflichtbegriffe im Vertrag.</p>')
    befunde = ""
    for art, titel in (("erfundene_fakten", "⚠ Nicht belegt (erfunden oder falsche Quellenmarke)"),
                       ("zitate_umgeformt", "~ Zitat nur grammatisch angepasst")):
        liste = b.get(art) or []
        if liste:
            befunde += (f'<p class="merkmal"><b>{_e(titel)}:</b></p><ul>'
                        + "".join(f"<li>„{_e(x['text'])}“ (Quelle {_e(x['quellen'] or 'alle')})</li>"
                                  for x in liste) + "</ul>")
    weg = (f'Chunk {b.get("rang_chunk") or "—"} → Kandidat {b.get("rang_kandidat") or "—"} '
           f'→ Quelle {b.get("rang_quelle") or "—"}')
    antwort_html = (f'<p class="antworttext">{_e(antwort)}</p>' if antwort else
                    '<p class="antworttext antworttext--leer">keine Antwort</p>')
    return (f'<details class="fall po" style="--r:{farbe}">{kopf}<div class="fall-koerper">'
            f'<p class="merkmal"><b>{_e(name)}:</b> {_e(grund)} · Weg der Goldquelle: {_e(weg)}'
            f'{" · aus Lauf " + _e(e["lauf"]) if e.get("lauf") else ""}</p>'
            f'{panel("Vertrag", "Was richtig wäre.", vertrag)}'
            f'{panel("Antwort", "Antwort des Systems im Wortlaut; Pflichtbegriffe geprüft mit vergleich.py.", antwort_html + begriffe + befunde)}'
            f'<div class="panel"><p class="panel-titel">Rohdaten</p>'
            f'{panel_quellen(d, gold)}{panel_messwerte(d)}</div>'
            f'</div></details>')


# ---------------------------------------------------------------------------
# Seite
# ---------------------------------------------------------------------------
def _zaehl_zeile(name: str, anker: str, schluessel: list[str]) -> str:
    zaehl = {k: schluessel.count(k) for k in ERGEBNIS}
    return (f"<tr><td><a href='#{anker}'>{_e(name)}</a></td><td class='z'>{len(schluessel)}</td>"
            + "".join(f"<td class='z'>{zaehl[k]}</td>" for k in ERGEBNIS) + "</tr>")


def baue_gesamt(eval_laeufe: list[Path], po_laeufe: list[Path], ho_lauf: Path) -> str:
    rang = {"fehler": 0, "falsch": 1, "teilweise": 2, "richtig": 3}
    zeilen, abschnitte = [], []

    # eval_gold
    ev, k, man = lade_eval(eval_laeufe)
    ev_schl = [eval_ergebnis(e)[0] for e in ev]
    zeilen.append(_zaehl_zeile("eval_gold (Fallbasis)", "eval", ev_schl))
    karten = "".join(_eval_karte(e) for e in sorted(ev, key=lambda e: (rang[eval_ergebnis(e)[0]], e["b"]["id"])))
    kopf_ev = (f"Läufe <span class='mono'>{_e(', '.join(x.name for x in eval_laeufe))}</span>, "
               f"gewertet nach aktuellem Vertrag · {_e(man.get('modell', ''))} · "
               f"Treffer {k.get('treffer')}/{k.get('faelle_gesamt')}, erfundene Fakten {k.get('erfundene_fakten')}, "
               f"Zitate nicht wörtlich {k.get('zitate_umgeformt')}")
    abschnitte.append(f"<h2 id='eval'>eval_gold <span class='panel-hinweis'>({len(ev)} Fälle)</span></h2>"
                      f"<p class='panel-hinweis'>{kopf_ev}. Fehler zuerst.</p>{karten}")

    # PO und Holdout: je Kategorie
    for art, titel, anker in (("po", "PO-Testmenge", "po"), ("holdout", "Holdout (verdeckt)", "holdout")):
        if art == "po":
            faelle, meta = lade_po_aktuell(po_laeufe), {}
            herkunft = (f"Läufe {', '.join(x.name for x in po_laeufe)}, gewertet nach aktueller pruefung.json")
        else:
            faelle, _summen, meta = lade_holdout(ho_lauf)
            herkunft = f"Lauf {ho_lauf.name}"
        zeilen.append(_zaehl_zeile(titel, anker, [ergebnis(f)[0] for f in faelle]))
        teile = [f"<h2 id='{anker}'>{_e(titel)} <span class='panel-hinweis'>({len(faelle)} Fälle)</span></h2>"
                 f"<p class='panel-hinweis'><span class='mono'>{_e(herkunft)}</span>"
                 + (f" · {_e(meta.get('hinweis'))}" if meta.get("hinweis") else "") + "</p>"]
        for kat, name, text in KATEGORIEN:
            fs = [f for f in faelle if f["kategorie"] == kat]
            if not fs:
                continue
            teile.append(f"<h3>{_e(name)} ({len(fs)})</h3><p class='panel-hinweis'>{_e(text)}</p>"
                         + "".join(_fallkarte(f) for f in sorted(fs, key=lambda f: (rang[ergebnis(f)[0]], f["id"]))))
        abschnitte.append("".join(teile))

    tabelle = ("<table class='po-kurz'><tr><th>Menge</th><th>Fälle</th>"
               + "".join(f"<th>{_e(ERGEBNIS[x][0])}</th>" for x in ERGEBNIS) + "</tr>"
               + "".join(zeilen) + "</table>")
    titel = f"Gesamtbericht — {datetime.now():%d.%m.%Y %H:%M}"
    kopf = (f"<h1>{_e(titel)}</h1>"
            "<div class='warnband'><b>Enthält den Holdout.</b> Diese Seite bleibt unter "
            "RAGHSH_holdout/berichte/. Nicht in Verzeichnisse kopieren, die die Agenten lesen können "
            "(Agenten-Repo, /var/log/raghsh-agent, Dashboard) und nicht veröffentlichen.</div>"
            "<p class='panel-hinweis'>Alle Fragen des Messstands: eval_gold, eigene PO-Testmenge und "
            "Holdout. Ein Fall ist „teilweise“, wenn eine Rückfrage fehlt, die Antwort aber nach PO "
            "unterscheidet, oder wenn ein Treffer nicht belegte Zitate/Zahlen enthält.</p>" + tabelle)
    quellen = " · ".join([*(str(x) for x in eval_laeufe), *(str(x) for x in po_laeufe), str(ho_lauf)])
    return seite(titel, EXTRA_STIL + kopf + "".join(abschnitte) + fuss(quellen))


def schreibe_gesamt(eval_angabe: str | None, po_angabe: str | None, ho_angabe: str | None,
                    ziel: Path | None = None) -> Path:
    eval_laeufe = _liste(eval_angabe, EVAL_LAEUFE, juengster_eval)
    po_laeufe = _liste(po_angabe, PO_AGENT, juengster_po)
    ho_lauf = finde_lauf(ho_angabe, "holdout") if ho_angabe else juengster_holdout()
    if ziel is None:
        ABLAGE.mkdir(parents=True, exist_ok=True)
        ziel = ABLAGE / f"gesamt_{datetime.now():%Y-%m-%d_%H%M}.html"
    elif not str(ziel.resolve()).startswith(str(HOLDOUT)):
        raise SystemExit(f"Der Gesamtbericht enthält den Holdout und gehört unter {HOLDOUT}, nicht nach {ziel}.")
    ziel.write_text(baue_gesamt(eval_laeufe, po_laeufe, ho_lauf), encoding="utf-8")
    return ziel
