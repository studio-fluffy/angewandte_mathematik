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

# In sicheres, lesbares Verzeichnis wechseln (z. B. im Container)
if not os.access(".", os.R_OK):
    os.chdir("/tmp")

# ------------------------------------------------------------
# 1. Mapping-Dateien einlesen und prüfen
# ------------------------------------------------------------
if rank == 0:
    # Alle pro-Rank Mapping-Dateien einsammeln
    files = sorted(glob.glob("entity_map_global_rank*.npz"))
    if not files:
        raise FileNotFoundError(
            "Keine Mapping-Dateien gefunden. Bitte zuerst 'submesh_write.py' ausführen."
        )

    all_parent_ids = [np.load(f)["parent_ids"] for f in files]
    parent_ids_global = np.concatenate(all_parent_ids)

    print(f"[READ] {len(files)} Mapping-Dateien gefunden.")
    print(f"[READ] Gesamtanzahl Submesh-Zellen: {len(parent_ids_global)}")
    print(f"[READ] Erste 10 globale Parent-IDs: {parent_ids_global[:10]}")
else:
    parent_ids_global = None

comm.Barrier()

# ------------------------------------------------------------
# 2. Submesh-Dateien einlesen (zur Kontrolle)
# ------------------------------------------------------------
if rank == 0:
    xdmf_files = sorted(glob.glob("submesh_rank*.xdmf"))
    if not xdmf_files:
        raise FileNotFoundError(
            "Keine XDMF-Dateien gefunden. Bitte zuerst 'submesh_write.py' ausführen."
        )

    print(f"[READ] {len(xdmf_files)} XDMF-Dateien gefunden.")

    # Beispiel: erste Datei öffnen und Mesh lesen
    with io.XDMFFile(MPI.COMM_SELF, xdmf_files[0], "r") as f:
        submesh = f.read_mesh()  # automatisches Lesen des ersten <Grid>-Eintrags
        num_cells = submesh.topology.index_map(submesh.topology.dim).size_global

    print(f"[READ] Beispiel-Mesh geladen: {xdmf_files[0]} mit {num_cells} Zellen.")

# ------------------------------------------------------------
# 3. Konsistenzprüfung
# ------------------------------------------------------------
if rank == 0:
    unique_ids = np.unique(parent_ids_global)
    if len(unique_ids) == len(parent_ids_global):
        print("[CHECK] Alle globalen Parent-IDs sind eindeutig – Mapping konsistent.")
    else:
        duplicates = len(parent_ids_global) - len(unique_ids)
        print(f"[CHECK] Warnung: {duplicates} doppelte IDs gefunden – prüfen empfohlen.")

comm.Barrier()
if rank == 0:
    print("[READ] Verifikation abgeschlossen.")
