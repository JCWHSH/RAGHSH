"""Misst die beiden Schwaechen aus Abschnitt 9 des Berichts an echter Ausgabe."""
import re, sys
from collections import Counter
import pyarrow.parquet as pq

path = sys.argv[1] if len(sys.argv) > 1 else "enrichment_vllm.parquet"
rows = pq.read_table(path).to_pylist()
rows = [r for r in rows if r["questions"]]
print("Datei:", path)
print("Chunks mit Fragen:", len(rows))
print("=" * 66)

norm = lambda s: re.sub(r"[^\wäöüß ]", "", s.lower()).strip()
STOP = {"wie", "was", "wer", "wo", "wann", "welche", "welcher", "welches",
        "warum", "wozu", "woran", "an", "wen", "ich", "der", "die", "das",
        "den", "dem", "des", "ein", "eine", "einen", "einem", "eines", "und",
        "fuer", "für", "von", "im", "in", "am", "es", "sich", "bei", "mit",
        "zu", "zum", "zur", "ist", "sind", "wird", "werden", "man", "auf"}
content = lambda s: {w for w in norm(s).split() if w not in STOP and len(w) > 2}

# --- 1. Paraphrasen innerhalb eines Chunks -------------------------------
sims, near = [], 0
for r in rows:
    qs = r["questions"]
    pairs = [(a, b) for i, a in enumerate(qs) for b in qs[i + 1:]]
    if not pairs:
        continue
    best = 0.0
    for a, b in pairs:
        ca, cb = content(a), content(b)
        if not ca or not cb:
            continue
        j = len(ca & cb) / len(ca | cb)
        best = max(best, j)
    sims.append(best)
    if best >= 0.6:
        near += 1

if sims:
    print("1. PARAPHRASEN (Jaccard der Inhaltswoerter je Chunk, hoechstes Paar)")
    print("   Mittelwert          : {:.2f}".format(sum(sims) / len(sims)))
    print("   Chunks mit >= 0.60  : {} ({:.1f} %)".format(near, near / len(sims) * 100))
print()

# --- 2. Wortgleiche Fragen ueber verschiedene Chunks ----------------------
c = Counter()
where = {}
for r in rows:
    for q in r["questions"]:
        k = norm(q)
        c[k] += 1
        where.setdefault(k, []).append(r["source_url"])

total_q = sum(c.values())
dup_q = total_q - len(c)
multi = {k: n for k, n in c.items() if n > 1}
# nur solche, die auf VERSCHIEDENEN Seiten auftauchen
cross = {k: n for k, n in multi.items() if len(set(where[k])) > 1}

print("2. WORTGLEICHE FRAGEN")
print("   Fragen gesamt              :", total_q)
print("   eindeutige Formulierungen  :", len(c))
print("   Wiederholungen             : {} ({:.1f} %)".format(dup_q, dup_q / total_q * 100))
print("   davon auf verschiedenen URLs: {} Formulierungen".format(len(cross)))
print()
print("   Haeufigste seitenuebergreifende Wiederholungen:")
for k, n in sorted(cross.items(), key=lambda x: -x[1])[:8]:
    print("     {:4d}x auf {:3d} Seiten : {}".format(n, len(set(where[k])), k[:70]))
print()

# --- 3. Generische Fragen ohne Eigennamen --------------------------------
VAGUE = ("das modul", "die veranstaltung", "der studiengang", "die stelle",
         "das projekt", "die seite", "dem abschnitt", "der abschnitt",
         "diese seite", "das angebot", "die einrichtung")
vague_n = sum(1 for r in rows for q in r["questions"]
              if any(v in q.lower() for v in VAGUE))
print("3. VAGE REFERENZ ('das Modul' o. ae. statt Name): {} von {} ({:.1f} %)".format(
    vague_n, total_q, vague_n / total_q * 100))

# --- 4. Ja/Nein trotz Regel ----------------------------------------------
YN = {"ist", "sind", "kann", "koennen", "können", "muss", "müssen", "darf",
      "gibt", "hat", "haben", "bietet", "wird", "werden", "soll", "besteht"}
yn = sum(1 for r in rows for q in r["questions"] if norm(q).split()[:1] and norm(q).split()[0] in YN)
print("4. JA/NEIN-FRAGEN trotz Regel: {} ({:.1f} %)".format(yn, yn / total_q * 100))
