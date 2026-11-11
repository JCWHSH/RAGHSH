# -*- coding: utf-8 -*-
"""Misst, ob die neuen doc2query-Fragen naeher an den echten Nutzerfragen
liegen als die alten.

Fuer jeden Fall aus goldfragen.json wird der Chunk des Golddokuments gesucht,
der die Belegstelle enthaelt, und die Kosinus-Aehnlichkeit zwischen der
Nutzerfrage und der zusammengefuegten Fragenmenge dieses Chunks berechnet —
mit demselben Modell und denselben Task-Adaptern wie im Index
(jina-embeddings-v3, beide Seiten 'retrieval.query', Fragen mit ' ' verbunden;
siehe hpc_vectorizer.py, Zweiter Dense-Vektor aus den Fragen).

Zwei Zuordnungen fuer die ALTE Seite, weil id = hash(url, chunk_index) und
Chunking v2 die Indizes verschiebt:

  Variante A  Belegstelle im v1-Chunking suchen, dort die alten Fragen holen.
              Beide Seiten inhaltlich korrekt zugeordnet.
  Variante B  Belegstelle im v2-Chunking suchen, deren ID in der alten Datei
              nachschlagen (woertliche Auftragsfassung). Trifft dort teils
              einen anderen Textausschnitt.

Die neue Seite ist in beiden Varianten dieselbe.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, re, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fastembed import TextEmbedding


def log(*a):
    print(*a, file=sys.stderr, flush=True)

TASK_QUERY = 2          # jina-v3 Task-ID fuer 'retrieval.query'
MODELL     = "jinaai/jina-embeddings-v3"


def lade_modul(pfad: str, name: str):
    """chunking.py zweimal laden — v1 und v2 unterscheiden sich im Schnitt."""
    spec = importlib.util.spec_from_file_location(name, pfad)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def sammle(mod, ordner: Path, urls: set[str]) -> dict[str, dict]:
    md = mod.list_markdown(ordner)
    out: dict[str, dict] = {}
    for _p, chunks in mod.iter_chunks(md):
        for c in chunks:
            if c["source_url"] in urls:
                out[mod.make_point_id(c["source_url"], c["chunk_index"])] = c
    return out


def finde_beleg(pool: dict[str, dict], url: str, beleg: str):
    """Chunk mit der Belegstelle; sonst der mit der groessten Wortueberschneidung."""
    b = norm(beleg)
    kand = [(i, c) for i, c in pool.items() if c["source_url"] == url]
    if not kand:
        return None, None
    for i, c in kand:
        if b and b in norm(c["text"]):
            return i, c
    bw = set(b.split())
    if not bw:
        return kand[0]
    return max(kand, key=lambda ic: len(bw & set(norm(ic[1]["text"]).split())) / len(bw))


def fragen_map(pfad: Path) -> dict[str, list[str]]:
    t = pq.read_table(pfad, columns=["id", "questions"])
    return {i: (q or []) for i, q in zip(t.column("id").to_pylist(),
                                         t.column("questions").to_pylist())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("goldfragen.json"))
    ap.add_argument("--urls", type=Path, default=Path("urls.txt"))
    ap.add_argument("--ingested", type=Path, default=Path("ingested"))
    ap.add_argument("--alt", type=Path, default=Path("../enrichment_v2_wdh10.parquet"))
    ap.add_argument("--neu", type=Path, default=Path("enrichment_gold_v2.parquet"))
    ap.add_argument("--chunking-v1", default="../chunking.py")
    ap.add_argument("--tsv", type=Path, default=Path("fragen_v2_messung.tsv"))
    args = ap.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    urls = {z.strip() for z in args.urls.read_text(encoding="utf-8").splitlines() if z.strip()}

    import pickle
    cache = Path("chunkmaps.pkl")
    if cache.exists():
        A, N = pickle.loads(cache.read_bytes())
        log(f"Chunk-Maps aus {cache}")
    else:
        log("Chunking v2 …")
        sys.path.insert(0, ".")
        v2 = lade_modul("chunking.py", "chunking")
        N = sammle(v2, args.ingested, urls)
        for k in ("chunking", "titel"):
            sys.modules.pop(k, None)
        log("Chunking v1 …")
        sys.path.insert(0, str(Path(args.chunking_v1).parent))
        v1 = lade_modul(args.chunking_v1, "chunking_v1")
        A = sammle(v1, args.ingested, urls)
        cache.write_bytes(pickle.dumps((A, N)))
        log(f"Chunk-Maps nach {cache} gesichert")
    log(f"Chunks der {len(urls)} Dokumente — v1: {len(A)}  v2: {len(N)}")

    f_alt, f_neu = fragen_map(args.alt), fragen_map(args.neu)

    zeilen = []
    for g in gold:
        if g["source_url"] not in urls:
            zeilen.append({"id": g["id"], "frage": g["frage"], "status": "Dokument nicht in den 108"})
            continue
        i1, c1 = finde_beleg(A, g["source_url"], g["belegstelle"])
        i2, c2 = finde_beleg(N, g["source_url"], g["belegstelle"])
        zeilen.append({
            "id": g["id"], "frage": g["frage"], "status": "",
            "alt_a": f_alt.get(i1, []),          # Variante A
            "alt_b": f_alt.get(i2, []),          # Variante B (woertlich)
            "neu":   f_neu.get(i2, []),
            "ci_v1": c1["chunk_index"] if c1 else None,
            "ci_v2": c2["chunk_index"] if c2 else None,
        })

    # ── Einbetten: Nutzerfragen, Fragenmengen und jede Einzelfrage ──────────
    texte: list[str] = []
    def merke(s: str) -> int:
        texte.append(s)
        return len(texte) - 1
    for z in zeilen:
        if z["status"]:
            continue
        z["_q"] = merke(z["frage"])
        for feld in ("alt_a", "alt_b", "neu"):
            z[f"_{feld}"] = merke(" ".join(z[feld])) if z[feld] else None
            z[f"_{feld}_einzel"] = [merke(f) for f in z[feld]]

    # GPU, wenn vorhanden: rund 1.000 Texte auf der Login-CPU dauern zu lange.
    import onnxruntime as ort
    prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"])
    log(f"Einbetten: {len(texte)} Text(e), Provider {prov[0]}")
    emb = TextEmbedding(MODELL, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
                        providers=prov)
    V = np.array([v for v in emb.embed(texte, batch_size=16, task_id=TASK_QUERY)],
                 dtype=np.float32)
    log("Einbettung fertig.")
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12

    def sim(a: int | None, b: int | None) -> float | None:
        return None if a is None or b is None else float(V[a] @ V[b])

    for z in zeilen:
        if z["status"]:
            continue
        for feld in ("alt_a", "alt_b", "neu"):
            z[f"sim_{feld}"] = sim(z["_q"], z[f"_{feld}"])
            beste, bestw = "", None
            for f, idx in zip(z[feld], z[f"_{feld}_einzel"]):
                w = sim(z["_q"], idx)
                if bestw is None or w > bestw:
                    beste, bestw = f, w
            z[f"best_{feld}"], z[f"bestw_{feld}"] = beste, bestw

    # ── Ausgabe ────────────────────────────────────────────────────────────
    spalten = ["id", "sim_alt_a", "sim_alt_b", "sim_neu", "diff_a", "diff_b",
               "n_alt_a", "n_alt_b", "n_neu", "frage", "best_alt_a", "best_neu", "status"]
    with args.tsv.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(spalten) + "\n")
        for z in zeilen:
            if z["status"]:
                fh.write(f"{z['id']}\t\t\t\t\t\t\t\t\t{z['frage']}\t\t\t{z['status']}\n")
                continue
            d_a = (z["sim_neu"] - z["sim_alt_a"]) if (z["sim_neu"] is not None and z["sim_alt_a"] is not None) else None
            d_b = (z["sim_neu"] - z["sim_alt_b"]) if (z["sim_neu"] is not None and z["sim_alt_b"] is not None) else None
            z["diff_a"], z["diff_b"] = d_a, d_b
            def f(x): return "" if x is None else f"{x:.4f}"
            fh.write("\t".join([z["id"], f(z["sim_alt_a"]), f(z["sim_alt_b"]), f(z["sim_neu"]),
                                f(d_a), f(d_b), str(len(z["alt_a"])), str(len(z["alt_b"])),
                                str(len(z["neu"])), z["frage"].replace("\t", " "),
                                (z["best_alt_a"] or "").replace("\t", " "),
                                (z["best_neu"] or "").replace("\t", " "), ""]) + "\n")

    gemessen = [z for z in zeilen if not z["status"] and z["sim_neu"] is not None]
    def stat(feld: str, diff: str):
        paare = [z for z in gemessen if z[f"sim_{feld}"] is not None]
        if not paare:
            return
        d = np.array([z["sim_neu"] - z[f"sim_{feld}"] for z in paare])
        alt = np.array([z[f"sim_{feld}"] for z in paare])
        neu = np.array([z["sim_neu"] for z in paare])
        print(f"\n=== Variante {feld[-1].upper()} — alte Fragen aus "
              f"{'dem v1-Belegstellenchunk' if feld.endswith('a') else 'dem ID-Join auf den v2-Chunk'}")
        print(f"  Faelle mit beiden Seiten: {len(paare)}")
        print(f"  Aehnlichkeit alt  Median {np.median(alt):.4f}  Mittel {alt.mean():.4f}")
        print(f"  Aehnlichkeit neu  Median {np.median(neu):.4f}  Mittel {neu.mean():.4f}")
        print(f"  Differenz         Median {np.median(d):+.4f}  Mittel {d.mean():+.4f}")
        print(f"  besser: {int((d > 0).sum())}   schlechter: {int((d < 0).sum())}   gleich: {int((d == 0).sum())}")
        ohne_alt = [z["id"] for z in gemessen if not z[feld]]
        if ohne_alt:
            print(f"  ohne alte Fragen (zaehlen nicht mit): {', '.join(ohne_alt)}")
    stat("alt_a", "diff_a")
    stat("alt_b", "diff_b")

    print("\n=== Die im Auftrag hervorgehobenen Faelle (Variante A) ===")
    print(f"{'Fall':5} {'alt':>7} {'neu':>7} {'Diff':>8}  {'n_alt':>5} {'n_neu':>5}")
    for fid in ("g16", "g23", "g27", "g28", "g30", "g34", "g39", "q04", "q19"):
        z = next((x for x in zeilen if x["id"] == fid), None)
        if not z or z["status"]:
            print(f"{fid:5}  {'—':>7}"); continue
        f = lambda x: "—" if x is None else f"{x:.4f}"
        d = "" if z.get("diff_a") is None else f"{z['diff_a']:+.4f}"
        print(f"{fid:5} {f(z['sim_alt_a']):>7} {f(z['sim_neu']):>7} {d:>8}  "
              f"{len(z['alt_a']):>5} {len(z['neu']):>5}")
    print(f"\nTabelle geschrieben: {args.tsv}")
    json.dump([{k: v for k, v in z.items() if not k.startswith("_")} for z in zeilen],
              open("fragen_v2_messung.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
