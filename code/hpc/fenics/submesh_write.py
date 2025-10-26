from mpi4py import MPI
from dolfinx import mesh, io
import numpy as np
import os

# ------------------------------------------------------------
# MPI-Initialisierung
# ------------------------------------------------------------
comm = MPI.COMM_WORLD        # globaler MPI-Kommunikator
rank = comm.rank             # Prozess-ID (0,1,2,...)
size = comm.size             # Gesamtanzahl der Prozesse

# Optional: Sicherstellen, dass das Arbeitsverzeichnis beschreibbar ist.
# In vielen HPC-/Containerumgebungen ist /tmp universell zugänglich.
if not os.access(".", os.W_OK):
    os.chdir("/tmp")

# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def get_global_indices(index_map):
    """
    Liefert globale IDs der Entities eines gegebenen IndexMap-Objekts.

    Dolfinx verwaltet pro Rank nur lokale Indizes; die globale Spanne
    (start, end) ist über index_map.local_range verfügbar.
    Diese Funktion konstruiert daraus die globalen IDs.
    """
    lr = index_map.local_range
    start, end = lr() if callable(lr) else lr
    return np.arange(start, end, dtype=np.int64)


def create_submesh_safe(domain, dim, entities):
    """
    Erstellt ein Submesh aus ausgewählten Entities (z. B. Zellen mit bestimmtem Tag)
    und gibt zusätzlich das lokale Mapping (Submesh → Parent) zurück.

    Unterschiedliche Dolfinx-Versionen liefern 2–4 Rückgabewerte;
    diese Funktion vereinheitlicht das Verhalten.
    """
    result = mesh.create_submesh(domain, dim, entities)

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
        # Neuere Dolfinx-Version: nur Mesh-Objekt, Mapping als Attribut
        submesh = result
        entity_map = submesh.topology.original_cell_index

    # entity_map wird als NumPy-Array weiterverwendet
    return submesh, np.array(entity_map, dtype=np.int64)


# ------------------------------------------------------------
# 1. Ursprungs-Mesh erzeugen
# ------------------------------------------------------------

# 3D-Würfel [0,1]^3, hexaedrisch unterteilt in 8×8×8 Zellen
domain = mesh.create_box(
    comm,
    [[0, 0, 0], [1, 1, 1]],
    [8, 8, 8],
    cell_type=mesh.CellType.hexahedron
)

tdim = domain.topology.dim   # topologische Dimension (3 für Volumen)

# ------------------------------------------------------------
# 2. Zellen markieren (Tags)
# ------------------------------------------------------------

# Jede lokale Partition kennt ihre eigenen Zellen
num_cells = domain.topology.index_map(tdim).size_local
local_cells = np.arange(num_cells, dtype=np.int32)

# Obere Hälfte der Zellen mit Tag=1 markieren
mask = np.zeros(num_cells, dtype=np.int32)
mask[num_cells // 2:] = 1
tags = mesh.meshtags(domain, tdim, local_cells, mask)

# ------------------------------------------------------------
# 3. Submesh erzeugen (alle Zellen mit Tag==1)
# ------------------------------------------------------------
marked_cells = tags.find(1)
submesh, entity_map = create_submesh_safe(domain, tdim, marked_cells)

# ------------------------------------------------------------
# 4. Globale Zuordnung bestimmen
# ------------------------------------------------------------

# Lokales IndexMap-Objekt → globale ID-Spanne
parent_index_map = domain.topology.index_map(tdim)
parent_global_ids = get_global_indices(parent_index_map)

# Lokale Parent-Indizes durch globale IDs ersetzen
entity_map_global = parent_global_ids[entity_map]

# ------------------------------------------------------------
# 5. Ergebnisse schreiben
# ------------------------------------------------------------

# (a) Mapping-Datei: enthält pro Rank die globalen Parent-IDs
np.savez(f"entity_map_global_rank{rank}.npz",
         parent_ids=entity_map_global)

# (b) Submesh-Datei: Geometrie/Topologie im XDMF/HDF5-Format
# MPI.COMM_SELF = serielles Schreiben, kein paralleles HDF5 nötig
with io.XDMFFile(MPI.COMM_SELF, f"submesh_rank{rank}.xdmf", "w") as f:
    f.write_mesh(submesh)

# Alle Ranks synchronisieren, um sicherzustellen, dass alles geschrieben ist
comm.Barrier()

if rank == 0:
    print(f"[WRITE] {size} Ranks abgeschlossen. "
          f"XDMF- und Mapping-Dateien erfolgreich erzeugt.")
