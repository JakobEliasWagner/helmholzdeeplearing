import numpy as np
import time
import matplotlib
from mpi4py import MPI
import dolfinx
from dolfinx.io import XDMFFile
from dolfinx.cpp.mesh import CellType
import ufl
from petsc4py import PETSc
import matplotlib.pyplot as plt

k = 20
c = 343.
omega = k / c
rho_0 = 1.2041
wave_len = 2 * np.pi / k

solver_deg = 2
absorber_deg = 2

mesh = dolfinx.IntervalMesh(MPI.COMM_WORLD, round(2 / wave_len) * 10, [-1, 1])
V = dolfinx.FunctionSpace(mesh, ("CG", solver_deg))

#
k0 = dolfinx.Constant(V, k)


def eval_absorber(x):
    rt = 1e-6
    sigma_0 = -(absorber_deg + 1) * np.log(rt) / (2.0 * 1)

    in_absorber_x = x[0] >= 0
    sigma_x = sigma_0 * (np.abs(x[0]) - 0) ** absorber_deg
    x_layer = in_absorber_x * (2j * sigma_x * k - sigma_x ** 2)
    return x_layer


k_absorb = dolfinx.Function(V)
k_absorb.interpolate(eval_absorber)

#
ui = dolfinx.Function(V)
ui.interpolate(lambda x: np.sign(-(x[0] - 1)))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
    - k ** 2 * ufl.inner(u, v) * ufl.dx \
    - k_absorb * ufl.inner(u, v) * ufl.dx

L = -1j * omega * rho_0 * ufl.inner(ui, v) * ufl.ds

# Assemble matrix and vector and set up direct solver
A = dolfinx.fem.assemble_matrix(a)
A.assemble()
b = dolfinx.fem.assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

solver = PETSc.KSP().create(mesh.mpi_comm())
opts = PETSc.Options()
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
solver.setFromOptions()
solver.setOperators(A)

# Solve linear system
u = dolfinx.Function(V)
solver.solve(b, u.vector)

with XDMFFile(MPI.COMM_WORLD, 'sol.xdmf', 'w') as file:
    file.write_mesh(mesh)
    file.write_function(u)
