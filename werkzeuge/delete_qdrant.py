"""
Qdrant-Collection zurücksetzen — alle Vektoren löschen.

WARNUNG: Dieses Skript löscht unwiderruflich alle Vektoren und Metadaten
aus der Collection. Danach muss local_importer.py erneut vollständig
durchgeführt werden, um die Datenbank wieder zu befüllen.

Anwendungsfall:
    - Nach einem vollständigen Neucrawl, wenn alle Inhalte aktualisiert wurden
    - Wenn die Collection durch fehlerhafte Daten korrumpiert wurde
    - Zum Zurücksetzen auf einen leeren Zustand für Testzwecke
    - Nach Änderungen am Vektorschema (z.B. neue Sparse Vectors)

Warum hier nur gelöscht und nicht neu angelegt wird
---------------------------------------------------
Das Neuanlegen (Vektorschema + Payload-Indizes) übernimmt bewusst
local_importer.ensure_collection(). Legte dieses Skript die leere Collection
selbst an, sähe der Importer sie als bereits existierend an und liefe im
Upsert-Modus: er löscht dann vor dem Schreiben jedes Dokument einzeln per
delete-by-filter (purge_documents) — bei ~5.000 Dokumenten tausende
überflüssige Requests gegen eine nachweislich leere Collection.

Fehlt die Collection dagegen, erkennt der Importer das (collection_is_new),
überspringt das Aufräumen komplett und schreibt direkt. Schema und Indizes
sind identisch — sie sind in ensure_collection() ohnehin die maßgebliche
Definition.

Aufruf:
    python delete_qdrant.py
    python -m raghsh.index.local_importer    # legt Collection + Indizes an und befüllt sie
"""

import sys

from qdrant_client import QdrantClient

# Name der Collection — muss mit COLLECTION_NAME in local_importer.py übereinstimmen
COLLECTION_NAME = "hsh_knowledge"

# Verbindung zum lokalen Qdrant-Docker-Container
try:
    client = QdrantClient(host="localhost", port=6333, timeout=30)
    existing = {c.name for c in client.get_collections().collections}
except Exception as exc:
    print(f"Verbindung zu Qdrant fehlgeschlagen: {exc}")
    print("Docker-Container starten:  docker compose up -d")
    sys.exit(1)

# ── Bestehende Collection vollständig löschen ────────────────────────────────
# Entfernt alle Vektoren, Payloads und Indizes dieser Collection.
if COLLECTION_NAME not in existing:
    print(f"Collection '{COLLECTION_NAME}' existiert nicht — nichts zu löschen.")
else:
    info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}': {info.points_count} Punkte werden gelöscht.")
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' wurde gelöscht.")

print()
print("Nächster Schritt: python -m raghsh.index.local_importer")
print("  → legt die Collection samt Dense-/Sparse-Schema und Payload-Indizes")
print("    neu an und befüllt sie ohne überflüssigen Purge-Durchlauf.")
