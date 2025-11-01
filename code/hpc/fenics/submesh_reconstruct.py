'''
┌─────────────────────┐
│ submesh_write.py    │
│---------------------│
│ 1. Parent-Mesh      │
│ 2. Meshtags         │
│ 3. Submesh          │
│ 4. Entity-Map       │
│ 5. Owner-Map        │
│ 6. Speichern (XDMF) │
└────────┬────────────┘
         │  (Dateien)
         ▼
┌─────────────────────┐
│ submesh_reconstruct │
│---------------------│
│ 1. Owner-Map laden  │
│ 2. Parent-Mesh lesen│
│ 3. Entity-Map laden │
│ 4. Submesh neu bauen│
│ 5. Meshtags kopieren│
│ 6. Speichern (XDMF) │
└─────────────────────┘
'''
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
output_dir = "output_submesh"

# ------------------------------------------------------------
# 1. Partitionierungsinformationen laden
# ------------------------------------------------------------
owner_files = sorted(glob.glob(os.path.join(output_dir, "parent_cell_owners_rank*.npy")))
if not owner_files:
    raise FileNotFoundError("Keine gespeicherten Owner-Dateien gefunden.")
owners_local = np.load(owner_files[rank])
comm.Barrier()

# ------------------------------------------------------------
# 2. Parent-Mesh laden (inkl. Meshtags)
# ------------------------------------------------------------
domain_path = os.path.join(output_dir, "domain_with_tags.xdmf")
if not os.path.exists(domain_path):
    raise FileNotFoundError(f"{domain_path} fehlt.")

with io.XDMFFile(comm, domain_path, "r") as f:
    domain = f.read_mesh()
    try:
        domain_tags = f.read_meshtags(domain, "mesh")
        if rank == 0:
            print(f"[READ] Meshtags aus Parent geladen: {len(domain_tags.values)} markierte Zellen.")
    except Exception:
        domain_tags = None
        if rank == 0:
            print("[WARN] Keine Meshtags im Parent gefunden.")

tdim = domain.topology.dim
parent_index_map = domain.topology.index_map(tdim)
start, end = parent_index_map.local_range  # Property, kein Aufruf
parent_global_ids = np.arange(start, end, dtype=np.int64)

# ------------------------------------------------------------
# 3. Entity-Maps kombinieren (Parent-Zellen des Submesh)
# ------------------------------------------------------------
map_files = sorted(glob.glob(os.path.join(output_dir, "entity_map_global_rank*.npy")))
if not map_files:
    raise FileNotFoundError("Keine entity_map_global_rank*.npy-Dateien gefunden.")

if rank == 0:
    entity_map_global = np.concatenate([np.load(f) for f in map_files])
else:
    entity_map_global = None
# Verteile vollständige Entity-Map an alle Ranks
entity_map_global = comm.bcast(entity_map_global, root=0)

# ------------------------------------------------------------
# 4. Konsistenzprüfung: Rankzahl muss gleich bleiben
# ------------------------------------------------------------
if len(owner_files) != size:
    if rank == 0:
        raise RuntimeError(
            f"Ursprüngliche Partition hatte {len(owner_files)} Ranks, "
            f"aktuelle Simulation läuft mit {size}. "
            f"Für konsistente Rekonstruktion muss die Rankzahl identisch sein."
        )
comm.Barrier()

# ------------------------------------------------------------
# 5. Lokale Zellen des Submesh bestimmen (mit Owner-Map Konsistenz)
# ------------------------------------------------------------
# Finde Parent-Zellen, die zum Submesh gehören UND zu diesem Rank gehören sollten
mask_submesh = np.isin(parent_global_ids, entity_map_global)  # Zellen im Submesh
mask_owned = np.isin(parent_global_ids, owners_local)         # Zellen die zu diesem Rank gehören

# Nur Zellen nehmen, die sowohl im Submesh sind als auch zu diesem Rank gehören
mask_local = mask_submesh & mask_owned
marked_local = np.nonzero(mask_local)[0]

if rank == 0:
    print(f"[INFO] Partitionierungs-Konsistenz mit Owner-Maps:")
    
# Debug-Info: Wie viele Zellen hat jeder Rank
submesh_cells_count = len(marked_local)
all_counts = comm.gather(submesh_cells_count, root=0)

if rank == 0:
    for r, count in enumerate(all_counts):
        print(f"  Rank {r}: {count} Submesh-Zellen (nach Owner-Filterung)")
    print(f"  Gesamt: {sum(all_counts)} Submesh-Zellen über alle Ranks")

# Validierung: Prüfen ob wir Zellen haben
if len(marked_local) == 0:
    print(f"[WARN Rank {rank}] Keine Submesh-Zellen nach Owner-Filterung.")
    print(f"       Das kann normal sein, wenn Submesh nicht über alle Ranks verteilt ist.")

# Submesh wieder erzeugen mit owner-konsistenter Partitionierung
result = mesh.create_submesh(domain, tdim, marked_local)
if isinstance(result, tuple):
    n = len(result)
    if n == 2:
        submesh_reconstructed, cell_map_new = result
    elif n == 3:
        submesh_reconstructed, cell_map_new, vertex_map_new = result
    elif n == 4:
        submesh_reconstructed, cell_map_new, vertex_map_new, geom_map_new = result
    else:
        raise RuntimeError(f"Unerwartete Rückgabeanzahl: {n}")
else:
    submesh_reconstructed = result
    cell_map_new = None

print(f"[Rank {rank}] Submesh rekonstruiert mit "
      f"{submesh_reconstructed.topology.index_map(tdim).size_local} Zellen "
      f"(owner-konsistent).")

# Zusätzliche Validierung: Prüfen ob die Zellen wirklich zu diesem Rank gehören
if cell_map_new is not None and len(cell_map_new) > 0:
    reconstructed_parent_cells = parent_global_ids[cell_map_new]
    not_owned = ~np.isin(reconstructed_parent_cells, owners_local)
    if np.any(not_owned):
        print(f"[ERROR Rank {rank}] {np.sum(not_owned)} Zellen gehören nicht zu diesem Rank!")
        print(f"  Fehlerhafte Zellen: {reconstructed_parent_cells[not_owned][:5]}...")
    else:
        print(f"[OK Rank {rank}] Alle rekonstruierten Zellen gehören korrekt zu diesem Rank.")

# ------------------------------------------------------------
# 6. Optional: Meshtags auf Submesh übertragen
# ------------------------------------------------------------
submesh_path = os.path.join(output_dir, "submesh_reconstructed.xdmf")

if domain_tags is not None:
    parent_values = domain_tags.values
    parent_indices = domain_tags.indices
    # Filter nur die Zellen, die im Submesh liegen
    mask = np.isin(parent_indices, marked_local)
    submesh_cell_tags = parent_values[mask]
    submesh_indices = np.arange(len(submesh_cell_tags), dtype=np.int32)

    submesh_tags = mesh.meshtags(
        submesh_reconstructed,
        tdim,
        submesh_indices,
        submesh_cell_tags,
    )

    # Schreiben in XDMF mit richtiger API
    with io.XDMFFile(comm, submesh_path, "w") as f:
        f.write_mesh(submesh_reconstructed)
        f.write_meshtags(
            submesh_tags,
            submesh_reconstructed.geometry,
            "/Xdmf/Domain/Grid",
        )
    if rank == 0:
        print(f"[WRITE] Rekonstruiertes Submesh mit Meshtags → '{submesh_path}'")
else:
    with io.XDMFFile(comm, submesh_path, "w") as f:
        f.write_mesh(submesh_reconstructed)
    if rank == 0:
        print(f"[WRITE] Rekonstruiertes Submesh ohne Meshtags → '{submesh_path}'")

comm.Barrier()
if rank == 0:
    print("[DONE] Rekonstruktion mit gleicher Partition abgeschlossen.")
