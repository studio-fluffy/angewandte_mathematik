from mpi4py import MPI
from dolfinx import mesh, io
import numpy as np
import glob
import os

comm = MPI.COMM_WORLD
rank = comm.rank

output_dir = "output_submesh"
if not os.path.isdir(output_dir):
    raise FileNotFoundError(f"Ordner {output_dir} nicht gefunden. Bitte zuerst submesh_write_parallel.py ausführen.")

# ------------------------------------------------------------
# 1. Mapping-Dateien lesen (nur Rank 0)
# ------------------------------------------------------------
if rank == 0:
    files = sorted(glob.glob(os.path.join(output_dir, "entity_map_global_rank*.npz")))
    if not files:
        raise FileNotFoundError("Keine Mapping-Dateien gefunden.")
    all_parent_ids = [np.load(f)["parent_ids"] for f in files]
    parent_ids_global = np.concatenate(all_parent_ids)
    print(f"[READ] {len(files)} Mapping-Dateien kombiniert → {len(parent_ids_global)} Einträge.")
else:
    parent_ids_global = None

# ------------------------------------------------------------
# 2. Submesh parallel lesen
# ------------------------------------------------------------
xdmf_path = os.path.join(output_dir, "submesh_parallel.xdmf")
with io.XDMFFile(comm, xdmf_path, "r") as f:
    submesh = f.read_mesh()

tdim = submesh.topology.dim
local_start, local_end = submesh.topology.index_map(tdim).local_range
num_local = local_end - local_start
print(f"[Rank {rank}] Zellbereich: {local_start}:{local_end} ({num_local} Zellen)")

# ------------------------------------------------------------
# 3. Mapping broadcasten
# ------------------------------------------------------------
length = comm.bcast(len(parent_ids_global) if rank == 0 else None, root=0)
if rank != 0:
    parent_ids_global = np.empty(length, dtype=np.int64)
comm.Bcast(parent_ids_global, root=0)

# ------------------------------------------------------------
# 4. Lokales Mapping rekonstruieren
# ------------------------------------------------------------
entity_map_local = parent_ids_global[local_start:local_end]
print(f"[Rank {rank}] Lokales Mapping: {len(entity_map_local)} Einträge")

# ------------------------------------------------------------
# 5. Konsistenzprüfung
# ------------------------------------------------------------
local_min = np.min(entity_map_local) if num_local > 0 else np.inf
local_max = np.max(entity_map_local) if num_local > 0 else -np.inf
global_min = comm.allreduce(local_min, op=MPI.MIN)
global_max = comm.allreduce(local_max, op=MPI.MAX)

if rank == 0:
    print(f"[CHECK] Globale Parent-ID-Spanne: {global_min} – {global_max}")
    print(f"[CHECK] Submesh '{xdmf_path}' erfolgreich parallel gelesen und neu verteilt.")
