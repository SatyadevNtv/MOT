import warnings

from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, cg
# Workaround for SciPy bug: https://github.com/scipy/scipy/pull/8082
try:
    from scipy.linalg import solve_continuous_lyapunov as lyap
except ImportError:
    from scipy.linalg import solve_lyapunov as lyap

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multilog, multiprod, multisym, multitransp

import numpy as np
import time

import os
gpu_opt = os.environ.get('USE_GPU_OPT', False)

if not gpu_opt:
    import numpy as proc
    from numpy import random as rnd
else:
    import cupy as proc
    from cupy import random as rnd

def SKnopp(A, p, q, maxiters=None, checkperiod=None):
    # TODO: Modify and optimize for "same marginals" case

    A = proc.array(A)
    p = proc.array(p)
    q = proc.array(q)

    #tol = proc.finfo(float).eps
    tol = 1e-9
    if maxiters is None:
        maxiters = A.shape[0]*A.shape[1]

    if checkperiod is None:
        checkperiod = 10

    if p.ndim < 2 and q.ndim < 2:
        p = p[proc.newaxis, :]
        q = q[proc.newaxis, :]

    C = A

    # TODO: Maybe improve this if-else by looking
    # for other broadcasting techniques
    if C.ndim < 3:
        d1 = q / proc.sum(C, axis=0)[proc.newaxis, :]
    else:
        d1 = q / proc.sum(C, axis=1)

    if C.ndim < 3:
        d2 = p / d1.dot(C.T)
    else:
        d2 = p / proc.sum(C * d1[:, proc.newaxis, :], axis=2)

    gap = proc.inf

    iters = 0
    while iters < maxiters:
        if C.ndim < 3:
            row = d2.dot(C)
        else:
            row = proc.sum(C * d2[:, :, proc.newaxis], axis=1)

        if iters % checkperiod == 0:
            gap = proc.max(proc.absolute(row * d1 - q))
            if proc.any(proc.isnan(gap)) or gap <= tol:
                break
        iters += 1

        d1_prev = d1
        d2_prev = d2
        d1 = q / row
        if C.ndim < 3:
            d2 = p / d1.dot(C.T)
        else:
            d2 = p / proc.sum(C * d1[:, proc.newaxis, :], axis=2)

        if proc.any(proc.isnan(d1)) or proc.any(proc.isinf(d1)) or proc.any(proc.isnan(d2)) or proc.any(proc.isinf(d2)):
            warnings.warn("""SKnopp: NanInfEncountered
                    Nan or Inf occured at iter {:d} \n""".format(iters))
            d1 = d1_prev
            d2 = d2_prev
            break

    result = C * (proc.einsum('bn,bm->bnm', d2, d1, dtype='float'))
    return convert2numpy(result)


class DoublyStochastic(Manifold):
    """Manifold of `k` (n x m) positive matrices

    Implementation is based on multinomialdoublystochasticgeneralfactory.m
    """

    def __init__(self, n, m, p=None, q=None, maxSKnoppIters=None, checkperiod=None):
        self._n = n
        self._m = m
        self._p = proc.array(p)
        self._q = proc.array(q)
        self._maxSKnoppIters = maxSKnoppIters
        self._checkperiod = checkperiod

        # Assuming that the problem is on single manifold.
        if p is None:
            self._p = proc.repeat(1/n, n)
        if q is None:
            self._q = proc.repeat(1/m, m)

        if self._p.ndim < 2 and self._q.ndim < 2:
            self._p = self._p[proc.newaxis, :]
            self._q = self._q[proc.newaxis, :]

        if maxSKnoppIters is None:
            self._maxSKnoppIters = min(2000, 100 + m + n)
        if checkperiod is None:
            self._checkperiod = 10

        # `k` doublystochastic manifolds
        self._k = self._p.shape[0]

        self._name = ("{:d} {:d}X{:d} matrices with positive entries such that row sum is p and column sum is q respectively.".format(len(self._p), n, m))

        self._dim = self._k * (self._n - 1)*(self._m - 1)
        self._e1 = proc.ones(n)
        self._e2 = proc.ones(m)


    def __str__(self):
        return self._name


    @property
    def dim(self):
        return self._dim


    @property
    def typicaldist(self):
        return proc.sqrt(self._k) * (self._m + self._n)


    def inner(self, x, u, v):
        x = proc.array(x)
        u = proc.array(u)
        v = proc.array(v)

        return convert2numpy(proc.sum(u * v/ x))


    def norm(self, x, u):
        return np.sqrt(self.inner(x, u, u))


    def rand(self):
        Z = proc.absolute(rnd.randn(self._n, self._m))
        return SKnopp(Z, self._p, self._q, self._maxSKnoppIters, self._checkperiod)


    def randvec(self, x):
        raise RuntimeError
        Z = rnd.randn(self._n, self._m)
        Zproj = self.proj(x, Z[proc.newaxis, :, :])
        return Zproj / self.norm(x, Zproj)


    def _matvec(self, v):
        self._k = int(self.X.shape[0])
        v = v.reshape(self._k, int(v.shape[0]/self._k))
        vtop = proc.array(v[:, :self._n])
        vbottom = proc.array(v[:, self._n:])
        Avtop = (vtop * self._p) + proc.sum(self.X * vbottom[:, proc.newaxis, :], axis=2)
        Avbottom = proc.sum(self.X * vtop[:, :, proc.newaxis], axis=1) + (vbottom * self._q)
        Av = proc.hstack((Avtop, Avbottom))
        return convert2numpy(Av.ravel())


    def _lsolve(self, x, b):
        self.X = x.copy()
        _dim = self._k * (self._n + self._m)
        shape = (_dim, _dim)
        sol, _iters = cg(LinearOperator(shape, matvec=self._matvec), convert2numpy(b), tol=1e-6, maxiter=100)
        sol = sol.reshape(self._k, int(sol.shape[0]/self._k))
        del self.X
        alpha, beta = sol[:, :self._n], sol[:, self._n:]
        return proc.array(alpha), proc.array(beta)



    def proj(self, x, v):
        assert v.ndim == 3
        b = proc.hstack((proc.sum(v, axis=2), proc.sum(v, axis=1)))
        alpha, beta = self._lsolve(x, b.ravel())
        result = v - (proc.einsum('bn,m->bnm', alpha, self._e2, dtype='float') + proc.einsum('n,bm->bnm', self._e1, beta, dtype='float'))*x
        return result


    def dist(self, x, y):
        raise NotImplementedError


    def egrad2rgrad(self, x, u):
        x = proc.array(x)
        u = proc.array(u)
        mu = x * u
        return convert2numpy(self.proj(x, mu))


    def ehess2rhess(self, x, egrad, ehess, u):
        x = proc.array(x)
        egrad = proc.array(egrad)
        ehess = proc.array(u)

        gamma = egrad * x
        gamma_dot = (ehess * x) + (egrad * u)

        assert gamma.ndim == 3 and gamma_dot.ndim == 3
        b = proc.hstack((proc.sum(gamma, axis=2), proc.sum(gamma, axis=1)))
        b_dot = proc.hstack((proc.sum(gamma_dot, axis=2), proc.sum(gamma_dot, axis=1)))

        alpha, beta = self._lsolve(x, b.ravel())
        alpha_dot, beta_dot = self._lsolve(
            x,
            b_dot.ravel() - proc.hstack((
                proc.einsum('bnm,bm->bn', u, beta, dtype='float'),
                proc.einsum('bnm,bn->bm', u, alpha, dtype='float')
            )).ravel()
        )

        S = proc.einsum('bn,m->bnm', alpha, self._e2, dtype='float') + proc.einsum('n,bm->bnm', self._e1, beta, dtype='float')
        S_dot = proc.einsum('bn,m->bnm', alpha_dot, self._e2, dtype='float') + proc.einsum('n,bm->bnm', self._e1, beta_dot, dtype='float')
        delta_dot = gamma_dot - (S_dot*x) - (S*u)

        delta = gamma - (S*x)

        nabla = delta_dot - (0.5 * (delta * u)/x)

        return convert2numpy(self.proj(x, nabla))


    def retr(self, x, u):
        x = proc.array(x)
        u = proc.array(u)

        Y = x * proc.exp(u/x)
        Y = proc.maximum(Y, 1e-16)
        Y = proc.minimum(Y, 1e16)
        return SKnopp(Y, self._p, self._q, self._maxSKnoppIters, self._checkperiod)


    def zerovec(self, x):
        return convert2numpy(proc.zeros((self._k, self._n, self._m)))


    def transp(self, x1, x2, d):
        x2 = proc.array(x2)
        d = proc.array(d)
        return convert2numpy(self.proj(x2, d))


def convert2numpy(val):
    if gpu_opt:
        return val.get()
    return val
