import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx.io import XDMFFile
import ufl
from petsc4py import PETSc

"""
-laplace(u)=f
u = u_D
u_D = 1 + x² + 2y²
f = -6
"""

solver_deg = 2

#mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 8, 8, dolfinx.cpp.mesh.CellType.quadrilateral)
mesh = dolfinx.IntervalMesh(MPI.COMM_WORLD, 100, [-1, 1])

V = dolfinx.FunctionSpace(mesh, ("CG", solver_deg))

uD = dolfinx.Function(V)
uD.interpolate(lambda x: np.sin(np.pi * x[0]))
uD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# Create facet to cell connectivity required to determine boundary facets
fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)
boundary_facets = np.where(np.array(dolfinx.cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]

boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = dolfinx.DirichletBC(uD, boundary_dofs)

f = dolfinx.Function(V)
f.interpolate(lambda x: np.pi ** 2 * np.sin(np.pi * x[0]))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = dolfinx.fem.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

with XDMFFile(MPI.COMM_WORLD, 'sol_poisson.xdmf', 'w') as file:
    file.write_mesh(mesh)
    file.write_function(uh)
