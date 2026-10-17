from dolfinx import default_scalar_type
from dolfinx.fem import (Constant, dirichletbc, Function, functionspace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import create_unit_square, locate_entities, create_submesh
from dolfinx.plot import vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, inner, Measure)

from mpi4py import MPI
from dolfinx.mesh import meshtags

import meshio
import gmsh
import numpy as np
import os
# Matplotlib für headless Nutzung konfigurieren (kein X/GLX nötig)
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")  # Offscreen Backend
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def Omega_0(x):
    return x[1] <= 0.5


def Omega_1(x):
    return x[1] >= 0.5

mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
Q = functionspace(mesh, ("DG", 0))

kappa = Function(Q)
cells_0 = locate_entities(mesh, mesh.topology.dim, Omega_0)
cells_1 = locate_entities(mesh, mesh.topology.dim, Omega_1)

kappa.x.array[cells_0] = np.full_like(cells_0, 1, dtype=default_scalar_type)
kappa.x.array[cells_1] = np.full_like(cells_1, 0.1, dtype=default_scalar_type)

V = functionspace(mesh, ("Lagrange", 1))
u, v = TrialFunction(V), TestFunction(V)
a = inner(kappa * grad(u), grad(v)) * dx
x = SpatialCoordinate(mesh)
L = Constant(mesh, default_scalar_type(1)) * v * dx
dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
bcs = [dirichletbc(default_scalar_type(1), dofs, V)]


problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Filter out ghosted cells
tdim = mesh.topology.dim
num_cells_local = mesh.topology.index_map(tdim).size_local
marker = np.zeros(num_cells_local, dtype=np.int32)
cells_0 = cells_0[cells_0 < num_cells_local]
cells_1 = cells_1[cells_1 < num_cells_local]
marker[cells_0] = 1
marker[cells_1] = 2
mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, x = vtk_mesh(mesh, tdim, np.arange(num_cells_local, dtype=np.int32))


tdim = mesh.topology.dim
# Sicherstellen, dass die Konnektivitäten gebaut sind (für Meshtags oft nötig)
mesh.topology.create_connectivity(tdim, 0)

# Marker-Array auf voller (lokaler + ghost) Zellanzahl vorbereiten
num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
cell_values = np.zeros(num_cells, dtype=np.int32)
cell_values[cells_0] = 1
cell_values[cells_1] = 2

# Meshtags erzeugen (nur wirklich belegte IDs verwenden)
cell_indices = np.arange(num_cells, dtype=np.int32)
ctags = meshtags(mesh, tdim, cell_indices, cell_values)

# Angepasstes Maß
dx_sub = Measure("dx", domain=mesh, subdomain_data=ctags)

# Integrale auf den Teilgebieten
I0 = assemble_scalar(form(kappa * dx_sub(1)))
I1 = assemble_scalar(form(kappa * dx_sub(2)))

# Globale Reduktion (assemble_scalar gibt schon den globalen Wert zurück)
if mesh.comm.rank == 0:
    print("Integral über Omega_0:", I0)
    print("Integral über Omega_1:", I1)

#############################################
# Submesh-Erstellung für Omega_0 (cells_0)  #
#############################################

# create_submesh liefert:
#  submesh0:        neues Mesh
#  cell_map0:       Mapping Submesh-Zelle -> Eltern-Zelle (lokal)
#  vertex_map0:     Mapping Submesh-Vertex -> Eltern-Vertex (lokal)
#  geom_map0:       Mapping Submesh-Geometrie-Node -> Eltern-Geometrie-Node

submesh0, cell_map0, vertex_map0, geom_map0 = create_submesh(mesh, tdim, cells_0)

# Funktionsraum DG0 auf Submesh und Transfer von kappa (ein DoF pro Zelle)
Q_sub0 = functionspace(submesh0, ("DG", 0))
kappa_sub0 = Function(Q_sub0)
kappa_sub0.name = "kappa_sub0"
kappa_sub0.x.array[:] = kappa.x.array[cell_map0]

# Integrale zur Verifikation auf Submesh direkt
I0_sub = assemble_scalar(form(kappa_sub0 * dx(domain=submesh0)))

# Analyse-Ausgabe nur auf Rank 0
if mesh.comm.rank == 0:
    print("--- Submesh Omega_0 Analyse ---")
    print(f"Zellen (Submesh lokal): {submesh0.topology.index_map(tdim).size_local}")
    print(f"Vertices (Submesh lokal): {submesh0.topology.index_map(0).size_local}")
    print("Erste 10 cell_map0:", cell_map0[:min(10, len(cell_map0))])
    print("Erste 10 vertex_map0:", vertex_map0[:min(10, len(vertex_map0))])
    print("Erste 10 geom_map0:", geom_map0[:min(10, len(geom_map0))])
    print("Min/Max kappa_sub0:", float(kappa_sub0.x.array.min()), float(kappa_sub0.x.array.max()))
    print("Integral kappa über Submesh (Omega_0):", I0_sub)
    print("(Referenz Integral Omega_0 aus Gesamtmesh):", I0)
    print("-------------------------------------------")

# Optionaler Export des Submeshs und kappa_sub0
if mesh.comm.rank == 0:
    with XDMFFile(mesh.comm, "submesh0.xdmf", "w") as xdmf:
        xdmf.write_mesh(submesh0)
        xdmf.write_function(kappa_sub0)