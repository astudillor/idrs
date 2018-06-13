# IDR(s) Solver in Python
The Induced Dimension Reduction method (IDR(s)) [1] is a short-recurrences Krylov method that
solves the system of linear equation,
                                      Ax = b.
This Python implementation is based on [2]. The interface of the idrs function is compatible
with the Krylov methods implemented in Scipy.

      idrs(A, b, x0=None, tol=1e-5, s=4, maxit=None, M=None, callback=None)

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

# License

This software is distributed under the [MIT License](http://opensource.org/licenses/MIT).
