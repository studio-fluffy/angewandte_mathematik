from mpi4py import MPI
from dolfinx import mesh, io
import numpy as np
import os

# ------------------------------------------------------------
# MPI Setup
# ------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

output_dir = "output_submesh"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------
# 1. Parent-Mesh erzeugen
# ------------------------------------------------------------
domain = mesh.create_unit_cube(comm, 8, 8, 8)
tdim = domain.topology.dim

# Meshtags: jede zweite Zelle = Tag 1
cells = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
vals = np.where(cells % 2 == 0, 1, 0).astype(np.int32)
tags = mesh.meshtags(domain, tdim, cells, vals)

# ------------------------------------------------------------
# 2. Parent-Mesh und Meshtags speichern
# ------------------------------------------------------------
domain_path = os.path.join(output_dir, "domain_with_tags.xdmf")
with io.XDMFFile(comm, domain_path, "w") as f:
    f.write_mesh(domain)
    # Dolfinx ≥ 0.7: Geometrie + XPath angeben
    f.write_meshtags(tags, domain.geometry, "/Xdmf/Domain/Grid")

if rank == 0:
    print(f"[WRITE] Parent-Mesh + Meshtags → '{domain_path}'")

# ------------------------------------------------------------
# 3. Submesh erzeugen (nur Zellen mit Tag == 1)
# ------------------------------------------------------------
marked_local = np.nonzero(vals == 1)[0]

# Dolfinx-Versionen liefern 2–4 Rückgabewerte
result = mesh.create_submesh(domain, tdim, marked_local)
if isinstance(result, tuple):
    n = len(result)
    if n == 2:
        submesh, entity_map = result
    elif n == 3:
        submesh, entity_map, _ = result
    elif n == 4:
        submesh, entity_map, _, _ = result
    else:
        raise RuntimeError(f"Unerwartete Rückgabeanzahl von create_submesh(): {n}")
else:
    submesh = result
    entity_map = submesh.topology.original_cell_index

# ------------------------------------------------------------
# 4. Entity-Map speichern (global)
# ------------------------------------------------------------
parent_index_map = domain.topology.index_map(tdim)
start, end = parent_index_map.local_range  # Property, keine Funktion
parent_global_ids = np.arange(start, end, dtype=np.int64)
entity_map_global = parent_global_ids[entity_map]

np.save(os.path.join(output_dir, f"entity_map_global_rank{rank}.npy"), entity_map_global)

# ------------------------------------------------------------
# 5. Owner-Map (Partitionierung) speichern
# ------------------------------------------------------------
owners = parent_index_map.owners  # Property, kein Aufruf
np.save(os.path.join(output_dir, f"parent_cell_owners_rank{rank}.npy"), owners)

# ------------------------------------------------------------
# 6. Submesh speichern
# ------------------------------------------------------------
submesh_path = os.path.join(output_dir, "submesh_parallel.xdmf")
with io.XDMFFile(comm, submesh_path, "w") as f:
    f.write_mesh(submesh)

comm.Barrier()
if rank == 0:
    print(f"[DONE] Submesh + Partitionierung + Entity-Maps gespeichert ({size} Ranks).")
