"""
Bericht je Fall für die PO-Testmenge (`auswertung/po/`) und den Holdout.

Der Bericht aus bericht.py kennt nur die Fallbasis von eval_gold. PO-Läufe
(`lauf_po.py`) und Holdout-Läufe (`hsh-holdout voll`) haben eigene Fälle mit
eigener Wertung: Rückfrage ja oder nein, Antwort aus der richtigen Fassung,
Pflicht- und Verbotsbegriffe. Bisher gab es davon nur Summen. Diese Seite zeigt
je Fall Frage, erwartetes Verhalten, Antwort, Rückfrage, erkannte Fassungen,
Begriffe und Quellen.

Aufruf über bericht.py:
    python auswertung/gold/bericht.py --po <lauf-id|pfad> [--ziel DATEI]
    python auswertung/gold/bericht.py --holdout <lauf-id|pfad> [--ziel DATEI]

Die Wertung wird nicht neu gerechnet: Sie kommt aus `auswertung.json`
(lauf_po.py) bzw. `bewertung.json` (bewerte.py) des Laufs, damit Seite und
Summen nicht auseinanderlaufen. Nur die Markierung der einzelnen Begriffe
rechnet die Seite selbst, mit demselben vergleich.py.

Holdout: Die Seite enthält Fragen und Antworten des Holdouts. Sie wird deshalb
im Holdout-Verzeichnis abgelegt und darf nicht in Verzeichnisse kopiert werden,
die die Agenten lesen können (Agenten-Repo, /var/log/raghsh-agent, Dashboard).
"""
from __future__ import annotations

import json
from pathlib import Path

from bericht_teile import (_e, _kurz, fuss, panel, panel_messwerte, panel_quellen,  # noqa: E402
                           seite)
from raghsh.crawl.crawler_utils import normalize_url  # noqa: E402
from vergleich import anzeige, enthaelt, enthaelt_begriff  # noqa: E402

_BASE = Path(__file__).resolve().parent
from raghsh import konfig  # noqa: E402
PO_LAEUFE = [konfig.AUSWERTUNG / "po" / "laeufe",
             Path("/home/raghsh-agent/RAGHSH_agent/hsh_scraper/eval_gold/po/laeufe")]
HOLDOUT = Path("/home/christian/RAGHSH_holdout")
ABLAGE = konfig.DASHBOARD      # gitignored, falls der Laufordner nicht schreibbar ist

KATEGORIEN = [
    ("po_rueckfrage", "Rückfrage nötig",
     "Die Fassungen antworten verschieden, die Frage nennt keine. Richtig ist eine Rückfrage; "
     "eine Antwort, die die Werte je Fassung nennt, zählt als teilweise richtig."),
    ("po_gegeben", "PO genannt",
     "Die Frage nennt Studiengang und Fassung. Richtig ist eine Antwort aus genau dieser Fassung."),
    ("po_keine_rueckfrage", "Keine Rückfrage nötig",
     "Alle Fassungen antworten gleich, oder der Kontext macht die aktuelle eindeutig. "
     "Eine Rückfrage wäre hier ein Fehler."),
    ("allgemein", "Allgemein", "Keine PO-Frage."),
]
ERGEBNIS = {
    "richtig": ("richtig", "var(--treffer)"),
    "teilweise": ("teilweise", "var(--antwort)"),
    "falsch": ("falsch", "var(--retrieval)"),
    "fehler": ("Fehler", "var(--tinte-3)"),
}

EXTRA_STIL = """
<style>
details.fall.po { --rolle: var(--r); }
.ergebnis { font-size: .74rem; font-weight: 600; color: var(--r); white-space: nowrap; }
.warnband { border: 1px solid var(--retrieval); color: var(--retrieval); padding: .6rem .9rem;
            border-radius: 6px; margin: 1rem 0; font-size: .9rem; }
pre.summen { font-size: .8rem; overflow-x: auto; background: var(--flaeche); padding: .8rem;
             border: 1px solid var(--linie); border-radius: 6px; }
table.po-kurz { border-collapse: collapse; width: 100%; font-size: .88rem; margin: .5rem 0 1rem; }
table.po-kurz th, table.po-kurz td { text-align: left; padding: .35rem .6rem;
             border-bottom: 1px solid var(--linie); }
table.po-kurz td.z { text-align: right; font-variant-numeric: tabular-nums; }
.antworttext { white-space: pre-wrap; }
ol.kandidaten { margin: .3rem 0 0 1.2rem; padding: 0; font-size: .82rem; }
ol.kandidaten li { margin: .15rem 0; }
</style>
"""


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------
def _json(pfad: Path):
    return json.loads(pfad.read_text(encoding="utf-8"))


def finde_lauf(angabe: str, art: str) -> Path:
    p = Path(angabe)
    if p.is_dir():
        return p.resolve()
    orte = PO_LAEUFE if art == "po" else [HOLDOUT / "laeufe"]
    for ort in orte:
        if (ort / angabe).is_dir():
            return ort / angabe
    raise SystemExit(f"Lauf nicht gefunden: {angabe} (gesucht in {', '.join(map(str, orte))})")


def lade_po(lauf: Path) -> tuple[list[dict], str, dict]:
    """PO-Lauf: Fall steht in der Rohdatei unter `fall`, Wertung in auswertung.json."""
    bew = {w["id"]: w for w in _json(lauf / "auswertung.json")}
    faelle = []
    for datei in sorted((lauf / "faelle").glob("*.json")):
        d = _json(datei)
        fall = d.get("fall") or {}
        faelle.append(po_eintrag(fall, d, bew.get(fall.get("id"), {})))
    summen = (lauf / "summen.txt").read_text(encoding="utf-8") if (lauf / "summen.txt").exists() else ""
    manifest = _json(lauf / "manifest.json") if (lauf / "manifest.json").exists() else {}
    return faelle, summen, manifest


def po_eintrag(fall: dict, d: dict, w: dict) -> dict:
    """Ein PO-Fall für die Seite: Vertrag `fall`, Rohdaten `d`, Wertung `w` (lauf_po.werte_fall)."""
    gold_f = fall.get("gold_fassung")
    fassungen = fall.get("fassungen") or []
    gold = {u for f in fassungen if gold_f is None or f.get("fassung") == gold_f for u in f.get("urls", [])}
    falsch = ({u for f in fassungen if f.get("fassung") != gold_f for u in f.get("urls", [])}
              if gold_f is not None else set())
    if fall.get("source_url"):
        gold.add(fall["source_url"])
    return {
        "id": fall.get("id"), "kategorie": fall.get("kategorie"),
        "frage": fall.get("question", ""), "referenz": fall.get("reference_answer", ""),
        "begruendung": fall.get("begruendung", ""),
        "po": fall.get("po"), "kontext_fassung": fall.get("kontext_fassung"),
        "fassungen": fassungen, "gold_fassung": gold_f,
        "pflicht": fall.get("pflicht_begriffe") or [], "verboten": fall.get("verboten") or [],
        "gold": gold, "falsch": falsch, "roh": d,
        "fehler": bool(w.get("fehler")) or bool(d.get("fehler")),
        "rueckfrage": bool(w.get("rueckfrage")), "korrekt": w.get("korrekt"),
        "differenziert": w.get("nach_po_unterschieden"),
        "gold_rang": w.get("gold_rang"), "falsche_vor_gold": w.get("falsche_fassung_vor_gold"),
    }


def lade_holdout(lauf: Path) -> tuple[list[dict], str, dict]:
    """Holdout-Lauf: Fälle aus faelle.json (Stand heute), Wertung in bewertung.json."""
    defs = {f["id"]: f for f in _json(HOLDOUT / "faelle.json")}
    # Nach einer Korrektur des Vertrags liegt die Neuwertung daneben
    # (bewertung_neu.json / summe_neu.txt); die ursprüngliche bleibt als Beleg.
    neu = (lauf / "bewertung_neu.json").exists()
    bew = {w["id"]: w for w in _json(lauf / ("bewertung_neu.json" if neu else "bewertung.json"))}
    faelle = []
    for fid, w in bew.items():
        fall = defs.get(fid, {})
        roh_pfad = lauf / f"{fid}.json"
        d = _json(roh_pfad) if roh_pfad.exists() else {}
        kat = w.get("kategorie") or fall.get("kategorie")
        pflicht_ok = w.get("pflicht_ok")
        korrekt = (bool(pflicht_ok or pflicht_ok is None) and not w.get("verboten")
                   and not w.get("rueckfrage"))
        faelle.append({
            "id": fid, "kategorie": kat,
            "frage": fall.get("question", ""), "referenz": fall.get("reference_answer", ""),
            "begruendung": "", "po": ({"studiengang": fall.get("po_studiengang"),
                                       "fassung": fall.get("po_fassung")}
                                      if fall.get("po_fassung") else None),
            "kontext_fassung": None,
            "fassungen": [{"fassung": f} for f in fall.get("po_optionen", [])], "gold_fassung": fall.get("po_fassung"),
            "pflicht": fall.get("pflicht_begriffe") or [],
            "verboten": fall.get("verbotene_begriffe") or [],
            "differenziert_wenn": fall.get("differenziert_wenn"),
            "gold": {fall.get("source_url", ""), *fall.get("accepted_source_urls", [])} - {""},
            "falsch": set(fall.get("nicht_erwartet_urls", [])), "roh": d,
            "fehler": bool(w.get("fehler")),
            "rueckfrage": bool(w.get("rueckfrage")), "korrekt": korrekt,
            "differenziert": w.get("differenziert"),
            "gold_rang": w.get("gold_rang"), "falsche_vor_gold": bool(
                w.get("falsche_fassung_rang") and (not w.get("gold_rang")
                                                   or w["falsche_fassung_rang"] < w["gold_rang"])),
        })
    datei = lauf / ("summe_neu.txt" if neu else "summe.txt")
    summen = datei.read_text(encoding="utf-8") if datei.exists() else ""
    hinweis = ("Gewertet nach dem heutigen Vertrag (bewertung_neu.json); die Wertung zur Laufzeit "
               "steht in bewertung.json." if neu else "")
    return faelle, summen, {"hinweis": hinweis}


# ---------------------------------------------------------------------------
# Wertung und Darstellung je Fall
# ---------------------------------------------------------------------------
def ergebnis(f: dict) -> tuple[str, str]:
    """(Schlüssel in ERGEBNIS, Kurzbegründung) — aus der gespeicherten Wertung."""
    if f["fehler"]:
        return "fehler", "Fehler oder Zeitüberschreitung"
    if f["kategorie"] == "po_rueckfrage":
        if f["rueckfrage"]:
            return "richtig", "Rückfrage gestellt"
        if f["differenziert"]:
            return "teilweise", "keine Rückfrage, aber nach PO unterschieden"
        return "falsch", "keine Rückfrage, nicht nach PO unterschieden"
    if f["rueckfrage"]:
        return "falsch", "unnötige Rückfrage"
    return ("richtig", "Antwort korrekt") if f["korrekt"] else ("falsch", "Antwort nicht korrekt")


def _ist(url: str, menge: set[str]) -> bool:
    n = normalize_url(url or "")
    return any(n == normalize_url(u) for u in menge)


def _erwartet(f: dict) -> str:
    kat = f["kategorie"]
    if kat == "po_rueckfrage":
        jahre = ", ".join(str(x.get("fassung")) for x in f["fassungen"] if x.get("fassung"))
        return f"Rückfrage nach der Fassung ({jahre or 'mehrere Fassungen'})"
    if kat == "po_gegeben" and f["po"]:
        return f"Direkte Antwort aus {f['po'].get('studiengang') or ''} PO {f['po'].get('fassung')}".replace("  ", " ")
    if kat == "po_keine_rueckfrage":
        if f.get("kontext_fassung"):
            return f"Direkte Antwort aus der aktuellen Fassung {f['kontext_fassung']}, keine Rückfrage"
        return "Direkte Antwort, keine Rückfrage"
    return "Direkte Antwort"


def _begriffe(f: dict, antwort: str) -> str:
    zeilen = []
    for b in f["pflicht"]:
        ok = enthaelt_begriff(antwort, b)
        zeilen.append(f'<span class="chip {"chip--ja" if ok else "chip--nein"}">'
                      f'{"✓" if ok else "✗"} {_e(anzeige(b))}</span>')
    for b in f["verboten"]:
        treffer = enthaelt(antwort, b)
        zeilen.append(f'<span class="chip {"chip--nein" if treffer else ""}">'
                      f'{"✗ enthalten" if treffer else "nicht enthalten"}: {_e(b)}</span>')
    if f.get("differenziert_wenn"):
        teile = [enthaelt_begriff(antwort, t) for t in f["differenziert_wenn"]]
        zeilen.append(f'<span class="chip {"chip--ja" if all(teile) else ""}">nach PO unterschieden, '
                      f'wenn alle genannt: {_e(" · ".join(anzeige(t) for t in f["differenziert_wenn"]))}</span>')
    if not zeilen:
        return '<p class="panel-hinweis">Keine Begriffe im Vertrag.</p>'
    return ('<p class="panel-hinweis">Pflichtbegriffe (✓ gefunden) und verbotene Begriffe '
            '(dürfen nicht vorkommen), geprüft mit vergleich.py.</p>'
            f'<div class="chipreihe">{"".join(zeilen)}</div>')


def _po_info(d: dict) -> str:
    teile = []
    if d.get("po_erkannt"):
        teile.append(f"PO aus der Frage erkannt: <b>{_e(d['po_erkannt'])}</b>")
    v = d.get("po_vielfalt")
    if v:
        je = v.get("je_fassung") or {}
        raenge = "; ".join(f"{k}: Rang {', '.join(map(str, r))}" for k, r in je.items())
        teile.append(f"Mehrere Fassungen in den Kandidaten: <b>{_e(v.get('studiengang'))}</b> "
                     f"{_e(', '.join(map(str, v.get('fassungen', []))))}"
                     + (f" <span class='mono'>({_e(raenge)})</span>" if raenge else ""))
    e = d.get("po_entscheidung")
    if e:
        grund = f", Grund {e['grund']}" if e.get("grund") else ""
        teile.append(f"Vergleich der Fassungen: Urteil <b>{_e(e.get('urteil'))}</b>"
                     f" (verglichen {_e(e.get('verglichen'))}, {e.get('dauer_s', '—')} s{_e(grund)})")
    if not teile:
        return ""
    return panel("PO-Erkennung", "Was der Server über Fassungen erkannt und entschieden hat.",
                 "".join(f"<p class='merkmal'>{t}</p>" for t in teile))


def _rueckfrage(d: dict) -> str:
    r = d.get("rueckfrage")
    if not r:
        return ""
    optionen = "".join(
        f"<li>{_e(o.get('studiengang'))} {_e(o.get('fassung'))}"
        f"{' (aktuell)' if o.get('aktuell') else ''}</li>" for o in r.get("optionen", []))
    return panel("Rückfrage des Systems", "Statt einer Antwort fragt das System nach der Fassung.",
                 f"<p class='antworttext'>{_e(r.get('frage'))}</p><ul>{optionen}</ul>")


def _kandidaten(d: dict, f: dict) -> str:
    zeilen = []
    for i, k in enumerate((d.get("kandidaten") or [])[:15], start=1):
        url = k.get("url", "")
        marke = ('<span class="chip chip--ja">Gold</span> ' if _ist(url, f["gold"]) else
                 '<span class="chip chip--nein">falsche Fassung</span> ' if _ist(url, f["falsch"]) else "")
        zeilen.append(f"<li>{marke}{_e(_kurz(k.get('titel') or url, 120))}</li>")
    rang = f"Gold auf Rang {f['gold_rang']}" if f.get("gold_rang") else "Gold nicht unter den Kandidaten"
    if f.get("falsche_vor_gold"):
        rang += ", eine falsche Fassung steht davor"
    return (f'<details class="unterpanel"><summary>Kandidaten — {_e(rang)}</summary>'
            f'<ol class="kandidaten">{"".join(zeilen) or "<li>keine</li>"}</ol></details>')


def _fallkarte(f: dict) -> str:
    schluessel, grund = ergebnis(f)
    name, farbe = ERGEBNIS[schluessel]
    d = f["roh"] or {}
    antwort = d.get("antwort") or ""
    kopf = (f'<summary><span class="fall-id">{_e(f["id"])}</span>'
            f'<span class="fall-frage">{_e(f["frage"])}</span>'
            f'<span class="ergebnis">{_e(name)}</span></summary>')
    fehler = (f'<p class="merkmal"><b>Fehler:</b> {_e(_kurz(str(d.get("fehler")), 300))}</p>'
              if d.get("fehler") else "")
    antwort_html = (f'<p class="antworttext">{_e(antwort)}</p>' if antwort else
                    '<p class="antworttext antworttext--leer">keine Antwort</p>')
    vertrag = (f'<dl class="begriffe"><dt>Erwartet</dt><dd>{_e(_erwartet(f))}</dd>'
               f'<dt>Referenzantwort</dt><dd>{_e(f["referenz"])}</dd>'
               + (f'<dt>Beleg</dt><dd>{_e(_kurz(f["begruendung"], 400))}</dd>' if f["begruendung"] else "")
               + '</dl>')
    return (f'<details class="fall po" style="--r:{farbe}">{kopf}<div class="fall-koerper">'
            f'<p class="merkmal"><b>{_e(name)}:</b> {_e(grund)}</p>{fehler}'
            f'{panel("Vertrag", "Was richtig wäre.", vertrag)}'
            f'{_rueckfrage(d)}'
            f'{panel("Antwort", "Antwort des Systems im Wortlaut.", antwort_html + _begriffe(f, antwort))}'
            f'{_po_info(d)}'
            f'<div class="panel"><p class="panel-titel">Rohdaten</p>'
            f'{_kandidaten(d, f)}{panel_quellen(d, f["gold"])}{panel_messwerte(d)}</div>'
            f'</div></details>')


# ---------------------------------------------------------------------------
# Seite
# ---------------------------------------------------------------------------
def baue_po_bericht(lauf: Path, art: str) -> str:
    faelle, summen, manifest = lade_po(lauf) if art == "po" else lade_holdout(lauf)
    titel = ("PO-Testmenge" if art == "po" else "Holdout") + f" — {lauf.name}"
    modell = manifest.get("modell") or ""

    zeilen, abschnitte = [], []
    for kat, name, text in KATEGORIEN:
        fs = [f for f in faelle if f["kategorie"] == kat]
        if not fs:
            continue
        zaehl = {k: sum(1 for f in fs if ergebnis(f)[0] == k) for k in ERGEBNIS}
        zeilen.append(f"<tr><td>{_e(name)}</td><td class='z'>{len(fs)}</td>"
                      + "".join(f"<td class='z'>{zaehl[k]}</td>" for k in ERGEBNIS) + "</tr>")
        rang = {"fehler": 0, "falsch": 1, "teilweise": 2, "richtig": 3}
        karten = "".join(_fallkarte(f) for f in sorted(fs, key=lambda f: (rang[ergebnis(f)[0]], f["id"])))
        abschnitte.append(f"<h2>{_e(name)} <span class='panel-hinweis'>({len(fs)} Fälle)</span></h2>"
                          f"<p class='panel-hinweis'>{_e(text)}</p>{karten}")

    tabelle = ("<table class='po-kurz'><tr><th>Kategorie</th><th>Fälle</th>"
               + "".join(f"<th>{_e(ERGEBNIS[k][0])}</th>" for k in ERGEBNIS) + "</tr>"
               + "".join(zeilen) + "</table>")
    if manifest.get("hinweis"):
        modell = (modell + " · " if modell else "") + manifest["hinweis"]
    warnung = ("" if art == "po" else
               "<div class='warnband'><b>Holdout.</b> Diese Seite enthält die verdeckten Fälle. "
               "Nicht in Verzeichnisse kopieren, die die Agenten lesen können "
               "(Agenten-Repo, /var/log/raghsh-agent, Dashboard).</div>")
    kopf = (f"<h1>{_e(titel)}</h1>"
            f"<p class='panel-hinweis'>{_e(modell)} · {len(faelle)} Fälle. Innerhalb jeder "
            f"Kategorie stehen Fehler zuerst. Fall anklicken für Antwort, Begriffe und Quellen.</p>"
            f"{warnung}{tabelle}"
            + (f"<details class='unterpanel'><summary>Summen des Laufs (wie im Protokoll)</summary>"
               f"<pre class='summen'>{_e(summen)}</pre></details>" if summen else ""))
    koerper = EXTRA_STIL + kopf + "".join(abschnitte) + fuss(str(lauf))
    return seite(titel, koerper)


def ziel_fuer(lauf: Path, art: str) -> Path:
    """Holdout-Bericht immer in den Holdout-Lauf; PO-Bericht in den Laufordner,
    wenn er schreibbar ist (eigener Lauf), sonst nach agenten_dashboard/."""
    if art == "holdout":
        return lauf / "bericht.html"
    try:
        probe = lauf / ".schreibprobe"
        probe.touch()
        probe.unlink()
        return lauf / "bericht.html"
    except OSError:
        ABLAGE.mkdir(exist_ok=True)
        return ABLAGE / f"bericht_po_{lauf.name}.html"
