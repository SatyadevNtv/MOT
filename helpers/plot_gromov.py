"""
==========================
Gromov-Wasserstein example
==========================

This example is designed to show how to use the Gromov-Wassertsein distance
computation in POT.
"""

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import matplotlib.pyplot as plt

import scipy as sp
import numpy as np
import ot

np.random.seed(200)

#############################################################################
#
# Sample two Gaussian distributions (2D and 3D)
# ---------------------------------------------
#
# The Gromov-Wasserstein distance allows to compute distances with samples that
# do not belong to the same metric space. For demonstration purpose, we sample
# two Gaussian distributions in 2- and 3-dimensional spaces.


n_samples = 1000  # nb samples

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4, 4])
cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s)
P = sp.linalg.sqrtm(cov_t)
xt = np.random.randn(n_samples, 3).dot(P) + mu_t

#############################################################################
#
# Compute distance kernels, normalize them and then display
# ---------------------------------------------------------


C1 = sp.spatial.distance.cdist(xs, xs)
C2 = sp.spatial.distance.cdist(xt, xt)

C1 /= C1.max()
C2 /= C2.max()

#############################################################################
#
# Compute Gromov-Wasserstein plans and distance
# ---------------------------------------------

#p = ot.unif(n_samples)
#q = ot.unif(n_samples)

p = np.random.rand(n_samples)
q = np.random.rand(n_samples)
p = p/np.sum(p)
q = q/np.sum(q)

from pymanopt.manifolds.doublystochastic import DoublyStochastic, SKnopp
from pymanopt.solvers import ConjugateGradient, TrustRegions
from pymanopt.solvers.linesearch import LineSearchBackTracking
from pymanopt import Problem
from pymanopt import function

import os
gpu_opt = os.environ.get('USE_GPU_OPT', False)
if gpu_opt:
    import cupy as np
    C1 = np.array(C1)
    C2 = np.array(C2)
    p = np.array(p)
    q = np.array(q)


def _gwcost(x):
    return (
        np.sum(p * (C1**2).dot(p)) +
        np.sum(q * (C2**2).dot(q)) -
        2 * np.trace(x.T.dot(C1.T.dot(x.dot(C2))))
    )

MOTENTROPYREG = 0
MOTFROBREG = 1e-2
def _cost(x):
    x = x[0]
    return _gwcost(x) + MOTENTROPYREG * np.sum(sp.special.xlogy(x, x)) + MOTFROBREG * np.sum(x**2)


def _egrad(x):
    x = x[0]
    xmask = np.zeros(x.shape)
    xmask[x>0] = 1
    return - 2 * (
        C1.T.dot(x.dot(C2)) +
        C1.dot(x.dot(C2.T))
    )[np.newaxis, :, :] + MOTENTROPYREG * (1 + sp.special.xlogy(xmask, x))[np.newaxis, :, :] + 2*MOTFROBREG*x[np.newaxis, :, :]


def _ehess(x, u):
    x = x[0]
    u = u[0]

    return -2 * (
        C1.T.dot(u.dot(C2)) +
        C1.dot(u.dot(C2.T))
    )[np.newaxis, :, :]



manf = DoublyStochastic(n_samples, n_samples, [p], [q])
solver = ConjugateGradient(maxiter=20, maxtime=100000)
prblm = Problem(manifold=manf, cost=lambda x: _cost(x), egrad=lambda x: _egrad(x), ehess=lambda x, u: _ehess(x, u), verbosity=0)
Tinit = manf.rand()

gw0, gwtimings = ot.gromov.entropic_gromov_wasserstein(
    C1, C2, p, q, Tinit[0], 'square_loss', 1e-3, max_iter=30, verbose=False, log=False, callback=_gwcost)

xopt, mottimings = solver.solve(prblm, x=Tinit, callback=_gwcost)

plt.plot([x[0] for x in gwtimings], [x[1] for x in gwtimings], 'b.--', label=f'GW ({len(gwtimings)} iters)')
plt.plot([x[0] for x in mottimings], [x[1] for x in mottimings], 'r.--', label=f'MOT ({len(mottimings)} iters)')
plt.legend(loc='upper right')
plt.show()
print(f"MOT Cost: {_gwcost(xopt[0])}, GW Cost: {_gwcost(gw0)}")
print(f"""
Constraints

MOT:
w.r.t p: {np.linalg.norm(xopt[0].sum(axis=1) - p)}
w.r.t q: {np.linalg.norm(xopt[0].sum(axis=0) - q)}


GW:
w.r.t p: {np.linalg.norm(gw0.sum(axis=1) - p)}
w.r.t q: {np.linalg.norm(gw0.sum(axis=0) - q)}
""")
