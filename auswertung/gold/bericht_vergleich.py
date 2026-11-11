"""HTML-Vergleich zweier Läufe: Was hat sich geändert, und bei welchen Fällen?

Die Seite, die man braucht, um eine Umstellung zu beurteilen — etwa ob
`DENSE_ARMS = ("text", "questions")` in die Produktion soll. Das Terminal
(`auswertung.py --vergleich A B`) nennt dieselben Zahlen; hier stehen dazu
die Fälle, die den Unterschied ausmachen, mit beiden Antworten nebeneinander,
damit ein Mensch nachlesen kann, ob ein „Gewinn" wirklich einer ist.

    python auswertung/gold/bericht.py --vergleich <lauf-a> <lauf-b>

B wird gegen A gelesen: „+" heißt, B hat den Fall, A nicht.

Was die Seite nicht behauptet: Signifikanz. Bei 73 Fällen ist das 95-%-
Intervall einer Quote rund ±10 Punkte breit, und die LLM-Auswahl eines
Volllaufs streut zwischen zwei identischen Läufen um ein bis zwei Fälle (V1
gegen V2, 21.09.2026). Beides steht dabei; die Schlussfolgerung bleibt beim
Leser.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))

from auswertung import ist_treffer, kennzahlen, lade_lauf, trichter, wilson  # noqa: E402
from bericht_teile import (_e, _kurz, fuss, ist_gold, konfig_felder, lade_roh,  # noqa: E402
                           marken, seite, weg, ziel_urls)
from vertrag import kurz as hash_kurz  # noqa: E402


def _delta(a, b, hoch_gut: bool = True) -> str:
    if a is None or b is None:
        return '<span class="delta delta--null">—</span>'
    d = b - a
    if d == 0:
        return '<span class="delta delta--null">±0</span>'
    gut = (d > 0) == hoch_gut
    return f'<span class="delta delta--{"plus" if gut else "minus"}">{f"{d:+g}".replace("-", "−")}</span>'


def _quote(k: int, n: int) -> str:
    if not n:
        return ""
    lo, hi = wilson(k, n)
    return f'<span class="wilson">{k / n * 100:.0f} % ({lo * 100:.0f}–{hi * 100:.0f})</span>'


def _antwort_block(d: dict | None, b: dict, ziele: set[str]) -> str:
    if d is None:
        return '<p class="merkmal">keine Rohdatei</p>'
    quellen = "".join(
        f'<li class="{"gold" if ist_gold(q.get("url", ""), ziele) else ""}">'
        f'{"<b>Gold</b> " if ist_gold(q.get("url", ""), ziele) else ""}'
        f'{_e(_kurz(q.get("titel") or q.get("url"), 80))}</li>'
        for q in d.get("quellen", []))
    antwort = (d.get("antwort") or "").strip()
    fehlend = (f'<p class="merkmal"><b>Fehlende Pflichtbegriffe:</b> {_e(", ".join(b["begriffe_fehlend"]))}</p>'
               if b["begriffe_fehlend"] else "")
    return (f'<p class="merkmal">{weg(b)} · Diagnose <b>{_e(b["diagnose"].replace("_", " "))}</b></p>'
            f'{fehlend}'
            f'<p class="panel-titel">Quellen im Kontext</p><ul class="armliste">{quellen or "<li>keine</li>"}</ul>'
            f'<p class="panel-titel">Antwort</p>'
            f'<p class="antworttext{"" if antwort else " antworttext--leer"}">{_e(antwort) or "(keine Antwort)"}</p>')


def _kandidaten_block(d: dict | None, b: dict, ziele: set[str]) -> str:
    if d is None:
        return '<p class="merkmal">keine Rohdatei</p>'
    zeilen = "".join(
        f'<li class="{"gold" if ist_gold(k.get("url", ""), ziele) else ""}">'
        f'<span class="mono">{k.get("nr")}</span> {"<b>Gold</b> " if ist_gold(k.get("url", ""), ziele) else ""}'
        f'{_e(_kurz(k.get("titel") or k.get("url"), 80))}</li>'
        for k in d.get("kandidaten", [])[:12])
    arme = ""
    if b.get("rang_arm"):
        arme = ('<p class="merkmal">Goldrang je Arm: ' + " · ".join(
            f'{_e(a)} <span class="mono">{r or "—"}</span>' for a, r in b["rang_arm"].items()) + "</p>")
    return (f'<p class="merkmal">{weg(b)} · Diagnose <b>{_e(b["diagnose"].replace("_", " "))}</b></p>{arme}'
            f'<p class="panel-titel">Kandidatenliste (erste 12)</p><ol class="armliste">{zeilen}</ol>')


def baue_vergleich(a: str, b: str, vertrag_aktuell: bool = False) -> str:
    ma, ba = lade_lauf(a, vertrag_aktuell=vertrag_aktuell)
    mb, bb = lade_lauf(b, vertrag_aktuell=vertrag_aktuell)
    modus_a, modus_b = ma.get("modus", "voll"), mb.get("modus", "voll")
    if modus_a != modus_b:
        raise SystemExit(f"Verschiedene Modi: {a} = {modus_a}, {b} = {modus_b} — "
                         f"ein Retrieval-Lauf hat keine Antworten, ein Volllauf misst "
                         f"dieselben Kandidaten mit LLM dahinter. Nicht vergleichbar.")
    nur_retrieval = modus_a == "retrieval"
    roh_a, roh_b = lade_roh(a), lade_roh(b)
    da = {x["id"]: x for x in ba}
    db = {x["id"]: x for x in bb}
    gemeinsam = [i for i in da if i in db]

    # ── Vertrag ───────────────────────────────────────────────────────────
    nur_a = [i for i in da if i not in db]
    nur_b = [i for i in db if i not in da]
    anders = [i for i in gemeinsam if da[i]["vertrag_fall"] != db[i]["vertrag_fall"]]
    fa, fbm = ma.get("fallbasis") or {}, mb.get("fallbasis") or {}
    ka = hash_kurz(fa.get("hash") or fa.get("hash_auswahl"))
    kb = hash_kurz(fbm.get("hash") or fbm.get("hash_auswahl"))
    if nur_a or nur_b or anders:
        teile = []
        if nur_a:
            teile.append(f"nur in A: {', '.join(nur_a)}")
        if nur_b:
            teile.append(f"nur in B: {', '.join(nur_b)}")
        if anders:
            teile.append(f"anderer Fallvertrag bei {len(anders)} Fällen: {', '.join(anders)}")
        vertrag_html = (f'<p class="merkmal" style="color:var(--retrieval)"><b>⚠ Verschiedene Verträge</b> '
                        f'(A {_e(ka)}, B {_e(kb)}) — {_e("; ".join(teile))}. Die Zahlen unten '
                        f'vergleichen Läufe gegen verschiedene Messlatten; verglichen werden nur die '
                        f'{len(gemeinsam)} gemeinsamen Fälle.</p>')
    else:
        vertrag_html = (f'<p class="merkmal">Gleicher Vertrag <span class="mono">{_e(ka)}</span> in '
                        f'beiden Läufen, {len(gemeinsam)} gemeinsame Fälle.</p>')

    # ── Konfiguration nebeneinander ───────────────────────────────────────
    fa_k, fb_k = dict(konfig_felder(ma)), dict(konfig_felder(mb))
    konf_zeilen = "".join(
        f'<tr><td>{_e(name)}</td>'
        f'<td class="{"anders" if fa_k[name] != fb_k.get(name) else ""}">{_e(fa_k[name])}</td>'
        f'<td class="{"anders" if fa_k[name] != fb_k.get(name) else ""}">{_e(fb_k.get(name, "—"))}</td></tr>'
        for name in fa_k)
    unterschiede = [name for name in fa_k if fa_k[name] != fb_k.get(name) and name != "Laufzeit"]

    # ── Kennzahlen ────────────────────────────────────────────────────────
    la, lb = [da[i] for i in gemeinsam], [db[i] for i in gemeinsam]
    za, zb = kennzahlen(la), kennzahlen(lb)
    n = za["faelle_moeglich"]
    zeilen = [("Gold in Kandidaten", "gold_in_kandidaten", True, True),
              ("davon auf Rang 1", "gold_kandidat_rang1", True, True)]
    if not nur_retrieval:
        zeilen += [("Gold in Quellen", "gold_in_quellen", True, True),
                   ("Treffer (prüfbar)", "treffer", True, False),
                   ("nicht prüfbar", "unbestimmt", False, False),
                   ("Widerspruch gemeldet", "widersprueche", False, False),
                   ("Fragment im Kontext", "fragmente", False, False)]
    kz = []
    for label, feld, hoch_gut, mit_quote in zeilen:
        kz.append(f'<tr><td>{_e(label)}</td>'
                  f'<td class="z mono">{za[feld]} {_quote(za[feld], n) if mit_quote else ""}</td>'
                  f'<td class="z mono">{zb[feld]} {_quote(zb[feld], n) if mit_quote else ""}</td>'
                  f'<td class="z">{_delta(za[feld], zb[feld], hoch_gut)}</td></tr>')
    ma_med, mb_med = za["median_rang_kandidat"], zb["median_rang_kandidat"]
    kz.append(f'<tr><td>Median Rang Kandidat</td><td class="z mono">{f"{ma_med:g}" if ma_med is not None else "—"}</td>'
              f'<td class="z mono">{f"{mb_med:g}" if mb_med is not None else "—"}</td>'
              f'<td class="z">{_delta(ma_med, mb_med, hoch_gut=False)}</td></tr>')
    lza, lzb = (ma.get("laufzeit") or {}).get("wall_clock_s"), (mb.get("laufzeit") or {}).get("wall_clock_s")
    kz.append(f'<tr><td>Laufzeit</td><td class="z mono">{lza / 60:.1f} min</td>'
              f'<td class="z mono">{lzb / 60:.1f} min</td><td class="z"></td></tr>'
              if lza and lzb else "")

    # ── Trichter doppelt ──────────────────────────────────────────────────
    sa, sb = trichter(la), trichter(lb)
    if nur_retrieval:
        sa, sb = sa[:4], sb[:4]
    gesamt = sa[0]["anzahl"]
    doppel = []
    for xa, xb in zip(sa, sb):
        pa, pb = xa["anzahl"] / max(gesamt, 1) * 100, xb["anzahl"] / max(gesamt, 1) * 100
        doppel.append(
            f'<div class="doppel"><div class="stufe-name">{_e(xa["label"])}'
            f'<small>{_delta(xa["anzahl"], xb["anzahl"])}</small></div>'
            f'<div><div class="spur"><div class="spur-voll" style="width:{pa:.2f}%"></div></div>'
            f'<div class="spur"><div class="spur-voll spur-voll--b" style="width:{pb:.2f}%"></div></div></div>'
            f'<div class="stufe-wert mono">{xa["anzahl"]}<br>{xb["anzahl"]}</div></div>')

    # ── Bewegte Fälle ─────────────────────────────────────────────────────
    erfolg = (lambda x: x["gold_in_kandidaten"]) if nur_retrieval else ist_treffer
    was = "Gold in Kandidaten" if nur_retrieval else "Treffer"

    def richtung(i: str) -> str:
        ea, eb = erfolg(da[i]), erfolg(db[i])
        if ea != eb:
            return "plus" if eb else "minus"
        if not nur_retrieval and da[i]["gold_in_quellen"] != db[i]["gold_in_quellen"]:
            return "plus" if db[i]["gold_in_quellen"] else "minus"
        if da[i]["gold_in_kandidaten"] != db[i]["gold_in_kandidaten"]:
            return "plus" if db[i]["gold_in_kandidaten"] else "minus"
        return "gleich"

    def bewegt(i: str) -> bool:
        x, y = da[i], db[i]
        return (erfolg(x) != erfolg(y) or x["diagnose"] != y["diagnose"]
                or x["gold_in_kandidaten"] != y["gold_in_kandidaten"]
                or x["gold_in_quellen"] != y["gold_in_quellen"]
                or x["rang_kandidat"] != y["rang_kandidat"])

    gewonnen = [i for i in gemeinsam if not erfolg(da[i]) and erfolg(db[i])]
    verloren = [i for i in gemeinsam if erfolg(da[i]) and not erfolg(db[i])]
    bewegte = [i for i in gemeinsam if bewegt(i)]
    ruhig = [db[i] for i in gemeinsam if not bewegt(i)]
    reihen = {"minus": 0, "plus": 1, "gleich": 2}
    bewegte.sort(key=lambda i: (reihen[richtung(i)], i))

    def karte(i: str) -> str:
        r = richtung(i)
        symbol = {"plus": "+", "minus": "−", "gleich": "="}[r]
        x, y = da[i], db[i]
        ziele = ziel_urls((roh_b.get(i) or roh_a.get(i))["fall"])
        block = _kandidaten_block if nur_retrieval else _antwort_block
        titel = {"plus": f"{was} neu in B", "minus": f"{was} in B verloren",
                 "gleich": "anderer Weg, gleicher Ausgang"}[r]
        if r != "gleich" and erfolg(x) == erfolg(y):
            titel = ("Gold in Quellen neu in B" if r == "plus" else "Gold in Quellen in B verloren") \
                if not nur_retrieval else titel
        rolle = {"plus": "treffer", "minus": "retrieval", "gleich": "unbestimmt"}[r]
        return (f'<details class="fall zelle--{rolle}"><summary>'
                f'<span class="fall-id richtung richtung--{r}">{symbol}</span>'
                f'<span class="fall-id">{_e(i)}</span>'
                f'<span class="fall-frage">{_e(x["frage"])}</span>'
                f'<span class="fall-rolle">{_e(titel)}</span></summary>'
                f'<div class="fall-koerper"><div class="zwei-spalten">'
                f'<div class="panel"><p class="panel-titel">A · {_e(_kurz(a, 40))}</p>{block(roh_a.get(i), x, ziele)}</div>'
                f'<div class="panel"><p class="panel-titel">B · {_e(_kurz(b, 40))}</p>{block(roh_b.get(i), y, ziele)}</div>'
                f'</div></div></details>')

    karten = "".join(karte(i) for i in bewegte if richtung(i) != "gleich")

    # Gleicher Ausgang, anderer Weg: nur Rang oder Diagnose haben sich bewegt.
    # Eine Zeile je Fall genügt — die Antworten wären hier Ballast.
    still = [i for i in bewegte if richtung(i) == "gleich"]
    still_html = ""
    if still:
        zeilen_still = "".join(
            f'<tr><td class="mono">{_e(i)}</td><td>{_e(_kurz(da[i]["frage"], 70))}</td>'
            f'<td>{weg(da[i])}<br><span class="pfad-zusatz">{_e(da[i]["diagnose"].replace("_", " "))}</span></td>'
            f'<td>{weg(db[i])}<br><span class="pfad-zusatz">{_e(db[i]["diagnose"].replace("_", " "))}</span></td></tr>'
            for i in still)
        still_html = (f'<h3>Gleicher Ausgang, anderer Weg ({len(still)})</h3>'
                      f'<p class="merkmal">Rang oder Diagnose haben sich geändert, {was} nicht.</p>'
                      f'<div class="tabellenhuelle"><table class="bewegt"><thead><tr><th>Fall</th>'
                      f'<th>Frage</th><th>A</th><th>B</th></tr></thead><tbody>{zeilen_still}</tbody></table></div>')

    netto = len(gewonnen) - len(verloren)
    rausch_satz = ("" if nur_retrieval else
                   " Zwischen zwei identischen Vollläufen streut die LLM-Auswahl um ein bis zwei "
                   "Fälle (V1 gegen V2 am 21.09.2026: 64 identische Fälle, je einer gewonnen und "
                   "verloren) — ein Netto von ±2 ist damit nicht von Rauschen zu unterscheiden.")
    fazit = (f'B gegenüber A: <b>{len(gewonnen)} gewonnen, {len(verloren)} verloren</b> '
             f'({was}), netto {netto:+d} bei {len(gemeinsam)} gemeinsamen Fällen; '
             f'weitere {len(bewegte) - len(gewonnen) - len(verloren)} Fälle haben Rang oder '
             f'Diagnose gewechselt, {len(ruhig)} sind identisch geblieben.{rausch_satz}')

    koerper = f"""
<h1>Was sich geändert hat</h1>
<p class="unterzeile">Zwei Läufe, gelesen als B gegen A. Erst die Konfiguration, dann
die Zahlen mit ihrer Unschärfe, dann die Fälle, die den Unterschied ausmachen —
mit beiden {"Kandidatenlisten" if nur_retrieval else "Antworten"} nebeneinander.</p>

<dl class="konfig">
<div><dt>A</dt><dd class="mono">{_e(a)}</dd></div>
<div><dt>B</dt><dd class="mono">{_e(b)}</dd></div>
</dl>
{f'<p class="merkmal">A: {_e(ma["notiz"])}</p>' if ma.get("notiz") else ""}
{f'<p class="merkmal">B: {_e(mb["notiz"])}</p>' if mb.get("notiz") else ""}
{vertrag_html}

<h2>Konfiguration</h2>
<p class="unterzeile">{"Unterschiede: " + _e(", ".join(unterschiede)) + "." if unterschiede
 else "Beide Läufe sind gleich konfiguriert — Unterschiede sind Rauschen der LLM-Auswahl."}</p>
<div class="tabellenhuelle"><table class="vgl"><thead><tr><th></th><th>A</th><th>B</th></tr></thead>
<tbody>{konf_zeilen}</tbody></table></div>

<h2>Kennzahlen</h2>
<p class="unterzeile">Über die {len(gemeinsam)} gemeinsamen Fälle, davon {n} mit Goldquelle im
Korpus. In Klammern das 95-%-Wilson-Intervall der Quote — bei n = {n} rund ±10 Punkte.</p>
<div class="tabellenhuelle"><table><thead><tr><th></th><th class="z">A</th><th class="z">B</th>
<th class="z">Δ</th></tr></thead><tbody>{"".join(kz)}</tbody></table></div>

<h2>Der Weg durch die Pipeline, doppelt</h2>
<p class="legende-ab"><span></span>A <span class="b"></span>B</p>
{"".join(doppel)}

<h2>Die Fälle, die den Unterschied machen</h2>
<p class="unterzeile">{fazit}</p>
<p class="merkmal">Gewonnen: <span class="mono">{_e(" ".join(gewonnen)) or "—"}</span> ·
Verloren: <span class="mono">{_e(" ".join(verloren)) or "—"}</span></p>
{karten or '<p class="merkmal">Kein Fall hat den Ausgang gewechselt.</p>'}
{still_html}

<h2>Unverändert</h2>
<p class="unterzeile">{len(ruhig)} Fälle mit gleichem Ausgang, gleicher Diagnose und gleichem
Kandidatenrang in beiden Läufen.</p>
<div class="marken">{marken(ruhig)}</div>

{fuss(f"auswertung/laeufe/{a} und {b}", [
    'Das Wilson-Intervall beschreibt die Unsicherheit einer einzelnen Quote bei dieser '
    'Fallzahl. Für den Unterschied zweier Läufe auf denselben Fällen zählt die Liste der '
    'gewechselten Fälle mehr als die Differenz der Quoten: Sie zeigt, ob ein Gewinn ein '
    'anderes Dokument oder nur eine andere Laune der Auswahl ist.',
])}
"""
    return seite("Was sich geändert hat", koerper)
