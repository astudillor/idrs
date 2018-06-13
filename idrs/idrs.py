#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Reinaldo Astudillo and Martin B. van Gijzen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import division, absolute_import

import numpy as np
from scipy.linalg import get_blas_funcs
from scipy.sparse.linalg.isolve.utils import make_system

__all__ = ['idrs']


def idrs(A, b, x0=None, tol=1e-5, s=4, maxiter=None,
         M=None, callback=None):
    """
    Use Induced Dimension Reduction method [IDR(s)] to solve A x = b

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0  : {array, matrix}
        Starting guess for the solution.
    tol : float, optional
        Tolerance to achieve. The algorithm terminates when either the relative
        or the absolute residual is below `tol`.
    s : integer, optional
        specifies the dimension of the shadow space. Normally, a higher
        s gives faster convergence, but also makes the method more expensive.
        Default is 4.
    maxiter : integer, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, dense matrix, LinearOperator}, optional
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.

    Returns
    -------
    x : array or matrix
        The converged solution.
    info : integer
        Provides convergence information:
            - 0  : successful exit
            - >0 : convergence to tolerance not achieved, number of iterations
            - <0 : illegal input or breakdown

    References
    ----------

    .. [1] P. Sonneveld and M. B. van Gijzen
             SIAM J. Sci. Comput. Vol. 31, No. 2, pp. 1035--1062, (2008).
    .. [2] M. B. van Gijzen and P. Sonneveld
             ACM Trans. Math. Software,, Vol. 38, No. 1, pp. 5:1-5:19, (2011).
    .. [3] This file is a translation of the following MATLAB implementation:
            http://ta.twi.tudelft.nl/nw/users/gijzen/idrs.m

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    matvec = A.matvec
    psolve = M.matvec
    xtype = x.dtype

    axpy, dot = get_blas_funcs(['axpy', 'dot'], dtype=xtype)

    np.random.seed(0)
    P = np.random.randn(s, n)
    bnrm = np.linalg.norm(b)
    info = 0

    if callback is not None:
        callback(x)

    # Check for zero rhs:
    if bnrm == 0.0:
        # Solution is null-vector
        return postprocess(np.zeros(n, dtype=xtype)), 0
    # Compute initial residual:
    r = b - matvec(x)
    rnrm = np.linalg.norm(r)
    # Relative tolerance
    tolb = tol*np.linalg.norm(b)
    if rnrm < tolb:
        # Initial guess is a good enough solution
        return postprocess(x), 0

    # Initialization
    angle = 0.7
    G = np.zeros((n, s), dtype=xtype)
    U = np.zeros((n, s), dtype=xtype)
    Ms = np.eye(s, dtype=xtype)
    om = 1.0
    iter_ = 0

    # Main iteration loop, build G-spaces:
    while rnrm >= tolb and iter_ < maxiter:
        # New right-hand size for small system:
        f = P.dot(r)
        for k in range(0, s):
            # Solve small system and make v orthogonal to P:
            c = np.linalg.solve(Ms[k:s, k:s], f[k:s])
            v = r - G[:, k:s].dot(c)
            # Preconditioning:
            v = psolve(v)
            # Compute new U(:,k) and G(:,k), G(:,k) is in space G_j
            U[:, k] = axpy(v, U[:, k:s].dot(c), None, om)
            # Matrix-vector product
            G[:, k] = matvec(U[:, k])
            # Bi-Orthogonalize the new basis vectors:
            for i in range(0, k):
                alpha = dot(P[i, :], G[:, k]) / Ms[i, i]
                G[:, k] = axpy(G[:, i], G[:, k], None, -alpha)
                U[:, k] = axpy(U[:, i], U[:, k], None, -alpha)

            # New column of M = P'*G  (first k-1 entries are zero)
            for i in range(k, s):
                Ms[i, k] = dot(P[i, :], G[:, k])

            if Ms[k, k] == 0.0:
                # Breakdown
                return postprocess(x), -1

            # Make r orthogonal to g_i, i = 1..k
            beta = f[k] / Ms[k, k]
            x = axpy(U[:, k], x, None, beta)
            r = axpy(G[:, k], r, None, -beta)
            rnrm = np.linalg.norm(r)
            if callback is not None:
                callback(x)
            iter_ += 1
            if rnrm < tolb or iter_ >= maxiter:
                break
            # New f = P'*r (first k  components are zero)
            if k < s - 1:
                f[k + 1:s] = axpy(Ms[k + 1:s, k], f[k + 1:s], None, -beta)

        # Now we have sufficient vectors in G_j to compute residual in G_j+1
        # Note: r is already perpendicular to P so v = r
        if rnrm < tolb or iter_ >= maxiter:
            break
        # Preconditioning:
        v = psolve(r)
        # Matrix-vector product
        t = matvec(v)
        # Computation of a new omega
        nr = np.linalg.norm(r)
        nt = np.linalg.norm(t)
        ts = dot(t, r)
        rho = abs(ts / (nt * nr))
        om = ts / (nt * nt)
        if rho < angle:
            om = om * angle / rho
        # New vector in G_j+1
        x = axpy(v, x, None, om)
        r = axpy(t, r, None, -om)
        rnrm = np.linalg.norm(r)
        if callback is not None:
            callback(x)
        iter_ += 1

    if rnrm >= tolb:
        info = iter_
    return postprocess(x), info
