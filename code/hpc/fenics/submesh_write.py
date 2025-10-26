from mpi4py import MPI
from dolfinx import mesh, io
import numpy as np
import os

# ------------------------------------------------------------
# MPI-Setup: Initialisiert parallele Umgebung
# ------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Verzeichnis, in dem alle Dateien abgelegt werden
output_dir = "output_submesh"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------
# 1. Parent-Mesh erzeugen
# ------------------------------------------------------------
# Erzeugt einen gleichmäßigen 8x8x8-Würfel (512 Hexaeder-Zellen)
domain = mesh.create_unit_cube(comm, 8, 8, 8)
tdim = domain.topology.dim  # Topologische Dimension = 3

# ------------------------------------------------------------
# 2. Meshtags erstellen (jede zweite Zelle markiert)
# ------------------------------------------------------------
# Lokale Zellindices für diesen Rank
cells = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
# Markierung: gerade Zellen = Tag 1, ungerade Zellen = Tag 0
vals = np.where(cells % 2 == 0, 1, 0).astype(np.int32)
# Meshtags-Objekt erzeugen
tags = mesh.meshtags(domain, tdim, cells, vals)

# ------------------------------------------------------------
# 3. Parent-Mesh + Meshtags speichern
# ------------------------------------------------------------
# XDMF/HDF5 speichern Geometrie, Topologie und Tag-Informationen parallel
domain_path = os.path.join(output_dir, "domain_with_tags.xdmf")
with io.XDMFFile(comm, domain_path, "w") as f:
    f.write_mesh(domain)
    # Achtung: write_meshtags() erwartet Geometrie und Pfad im XDMF-Baum
    f.write_meshtags(tags, domain.geometry, "/Xdmf/Domain/Grid")

if rank == 0:
    print(f"[WRITE] Parent-Mesh + Meshtags → '{domain_path}'")

# ------------------------------------------------------------
# 4. Submesh erzeugen (nur Zellen mit Tag == 1)
# ------------------------------------------------------------
# Wählt die markierten Zellen lokal aus
marked_local = np.nonzero(vals == 1)[0]

# create_submesh kann 2–4 Rückgaben liefern, je nach Dolfinx-Version
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
        raise RuntimeError(f"Unerwartete Rückgabeanzahl: {n}")
else:
    submesh = result
    entity_map = submesh.topology.original_cell_index

# ------------------------------------------------------------
# 5. Entity-Map speichern (Parent → Submesh)
# ------------------------------------------------------------
# Globale Zell-IDs des Parent-Mesh berechnen
parent_index_map = domain.topology.index_map(tdim)
start, end = parent_index_map.local_range  # Property (kein Funktionsaufruf)
parent_global_ids = np.arange(start, end, dtype=np.int64)
# Globale IDs der Zellen, die ins Submesh übernommen wurden
entity_map_global = parent_global_ids[entity_map]
# pro Rank speichern
np.save(os.path.join(output_dir, f"entity_map_global_rank{rank}.npy"), entity_map_global)

# ------------------------------------------------------------
# 6. Partitionierungsinformationen speichern
# ------------------------------------------------------------
# owners: gibt an, welcher Rank welche Zellen "besitzt"
owners = parent_index_map.owners  # Property, kein Aufruf
np.save(os.path.join(output_dir, f"parent_cell_owners_rank{rank}.npy"), owners)

# ------------------------------------------------------------
# 7. Submesh speichern (XDMF/HDF5)
# ------------------------------------------------------------
submesh_path = os.path.join(output_dir, "submesh_parallel.xdmf")
with io.XDMFFile(comm, submesh_path, "w") as f:
    f.write_mesh(submesh)

comm.Barrier()
if rank == 0:
    print(f"[DONE] Submesh + Partitionierung + Entity-Maps gespeichert ({size} Ranks).")
