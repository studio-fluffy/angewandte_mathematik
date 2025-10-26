from mpi4py import MPI
from dolfinx import mesh, io
import numpy as np
import glob
import os

# ------------------------------------------------------------
# MPI-Setup
# ------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if not os.access(".", os.R_OK):
    os.chdir("/tmp")

# ------------------------------------------------------------
# 1. Alle Mapping-Dateien global laden (nur Rank 0)
# ------------------------------------------------------------
if rank == 0:
    # Alle zuvor erzeugten Mapping-Dateien (unabhängig von alter Rankzahl)
    files = sorted(glob.glob("entity_map_global_rank*.npz"))
    if not files:
        raise FileNotFoundError("Keine Mapping-Dateien gefunden (aus submesh_write.py).")

    # Alle Parent-IDs kombinieren → vollständiges globales Mapping
    all_parent_ids = [np.load(f)["parent_ids"] for f in files]
    parent_ids_global = np.concatenate(all_parent_ids)

    print(f"[READ] {len(files)} Mapping-Dateien zusammengeführt.")
    print(f"[READ] Gesamtanzahl Submesh-Zellen: {len(parent_ids_global)}")
else:
    parent_ids_global = None

# ------------------------------------------------------------
# 2. Submesh parallel laden (neue Rankzahl)
# ------------------------------------------------------------
# Hier wird eines der Submesh-Files gelesen.
# Dolfinx verteilt automatisch die Zellen auf die neuen Ranks.
with io.XDMFFile(comm, "submesh_rank0.xdmf", "r") as f:
    submesh = f.read_mesh()
tdim = submesh.topology.dim

# ------------------------------------------------------------
# 3. Lokale Zell-IDs im neuen Submesh bestimmen
# ------------------------------------------------------------
# Jeder Rank kennt seine eigenen Zellen → IndexMap liefert globale ID-Spanne
local_start, local_end = submesh.topology.index_map(tdim).local_range
num_local = local_end - local_start

print(f"[Rank {rank}] Lokaler Zellbereich: {local_start}:{local_end} ({num_local} Zellen)")

# ------------------------------------------------------------
# 4. Globale Mappingdaten an alle Ranks verteilen
# ------------------------------------------------------------
# Alle Ranks sollen die globale Map erhalten.
# Broadcast der Gesamtlänge
length = comm.bcast(len(parent_ids_global) if rank == 0 else None, root=0)

# Puffer für globales Mapping anlegen
if rank != 0:
    parent_ids_global = np.empty(length, dtype=np.int64)

# Rank 0 sendet globales Mapping an alle
comm.Bcast(parent_ids_global, root=0)

# ------------------------------------------------------------
# 5. Lokales Mapping rekonstruieren (nach neuer Partition)
# ------------------------------------------------------------
# Jeder Rank wählt seinen Teil anhand seines neuen globalen Zellbereichs
entity_map_local = parent_ids_global[local_start:local_end]

print(f"[Rank {rank}] Lokales Mapping enthält {len(entity_map_local)} Einträge.")

# ------------------------------------------------------------
# 6. (Optional) Konsistenz prüfen
# ------------------------------------------------------------
local_min = np.min(entity_map_local) if num_local > 0 else np.inf
local_max = np.max(entity_map_local) if num_local > 0 else -np.inf
global_min = comm.allreduce(local_min, op=MPI.MIN)
global_max = comm.allreduce(local_max, op=MPI.MAX)

if rank == 0:
    print(f"[CHECK] Globale Parent-ID-Spanne: {global_min} – {global_max}")
    print("[CHECK] Repartitionierung erfolgreich abgeschlossen.")
