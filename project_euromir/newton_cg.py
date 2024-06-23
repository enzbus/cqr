"""Newton-CG implementation from scipy.optimize."""

import math
import warnings

import numpy as np
from numpy import asarray, sqrt, zeros
from scipy._lib._util import _call_callback_maybe_halt, _RichResult
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._differentiable_functions import FD_METHODS, ScalarFunction
from scipy.optimize._linesearch import (LineSearchWarning, line_search_wolfe1,
                                        line_search_wolfe2)
from scipy.sparse.linalg import LinearOperator

_epsilon = sqrt(np.finfo(float).eps)

class _LineSearchError(RuntimeError):
    pass

class OptimizeWarning(UserWarning):
    pass


# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}


class OptimizeResult(_RichResult):
    """Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    Depending on the specific solver being used, `OptimizeResult` may
    not have all attributes listed here, and they may have additional
    attributes not listed here. Since this class is essentially a
    subclass of dict with attribute accessors, one can see which
    attributes are available using the `OptimizeResult.keys` method.
    """
    pass

def _prepare_scalar_function(fun, x0, jac=None, args=(), bounds=None,
                             epsilon=None, finite_diff_rel_step=None,
                             hess=None):
    """Creates a ScalarFunction object for use with scalar minimizers
    (BFGS/LBFGSB/SLSQP/TNC/CG/etc).

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    jac : {callable,  '2-point', '3-point', 'cs', None}, optional
        Method for computing the gradient vector. If it is a callable, it
        should be a function that returns the gradient vector:

            ``jac(x, *args) -> array_like, shape (n,)``

        If one of `{'2-point', '3-point', 'cs'}` is selected then the gradient
        is calculated with a relative step for finite differences. If `None`,
        then two-point finite differences with an absolute step is used.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` functions).
    bounds : sequence, optional
        Bounds on variables. 'new-style' bounds are required.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``jac='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    hess : {callable,  '2-point', '3-point', 'cs', None}
        Computes the Hessian matrix. If it is callable, it should return the
        Hessian matrix:

            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``

        Alternatively, the keywords {'2-point', '3-point', 'cs'} select a
        finite difference scheme for numerical estimation.
        Whenever the gradient is estimated via finite-differences, the Hessian
        cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
        to be estimated using one of the quasi-Newton strategies.

    Returns
    -------
    sf : ScalarFunction
    """
    if callable(jac):
        grad = jac
    elif jac in FD_METHODS:
        # epsilon is set to None so that ScalarFunction is made to use
        # rel_step
        epsilon = None
        grad = jac
    else:
        # default (jac is None) is to do 2-point finite differences with
        # absolute step size. ScalarFunction has to be provided an
        # epsilon value that is not None to use absolute steps. This is
        # normally the case from most _minimize* methods.
        grad = '2-point'
        epsilon = epsilon

    if hess is None:
        # ScalarFunction requires something for hess, so we give a dummy
        # implementation here if nothing is provided, return a value of None
        # so that downstream minimisers halt. The results of `fun.hess`
        # should not be used.
        def hess(x, *args):
            return None

    if bounds is None:
        bounds = (-np.inf, np.inf)

    # ScalarFunction caches. Reuse of fun(x) during grad
    # calculation reduces overall function evaluations.
    sf = ScalarFunction(fun, x0, args, grad, hess,
                        finite_diff_rel_step, bounds, epsilon=epsilon)

    return sf

def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found
    """

    extra_condition = kwargs.pop('extra_condition', None)

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is not None and extra_condition is not None:
        xp1 = xk + ret[0] * pk
        if not extra_condition(ret[0], xp1, ret[3], ret[5]):
            # Reject step if extra_condition fails
            ret = (None,)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            kwargs2 = {}
            for key in ('c1', 'c2', 'amax'):
                if key in kwargs:
                    kwargs2[key] = kwargs[key]
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval,
                                     extra_condition=extra_condition,
                                     **kwargs2)

    if ret[0] is None:
        raise _LineSearchError()

    return ret

def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in SciPy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, stacklevel=4)


def _print_success_message_or_warn(warnflag, message, warntype=None):
    if not warnflag:
        print(message)
    else:
        warnings.warn(message, warntype or OptimizeWarning, stacklevel=3)


def approx_fhess_p(x0, p, fprime, epsilon, *args):
    # calculate fprime(x0) first, as this may be cached by ScalarFunction
    f1 = fprime(*((x0,) + args))
    f2 = fprime(*((x0 + epsilon*p,) + args))
    return (f2 - f1) / epsilon

def _minimize_newtoncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                       callback=None, xtol=1e-5, eps=_epsilon, maxiter=None,
                       disp=False, return_all=False, c1=1e-4, c2=0.9,
                       **unknown_options):
    """Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Note that the `jac` parameter (Jacobian) is required.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `hessp` is approximated, use this value for the step size.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    c1 : float, default: 1e-4
        Parameter for Armijo condition rule.
    c2 : float, default: 0.9
        Parameter for curvature condition rule.

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.
    """
    _check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    fhess_p = hessp
    fhess = hess
    avextol = xtol
    epsilon = eps
    retall = return_all

    x0 = asarray(x0).flatten()
    # TODO: add hessp (callable or FD) to ScalarFunction?
    sf = _prepare_scalar_function(
        fun, x0, jac, args=args, epsilon=eps, hess=hess
    )
    f = sf.fun
    fprime = sf.grad
    _h = sf.hess(x0)

    # Logic for hess/hessp
    # - If a callable(hess) is provided, then use that
    # - If hess is a FD_METHOD, or the output from hess(x) is a LinearOperator
    #   then create a hessp function using those.
    # - If hess is None but you have callable(hessp) then use the hessp.
    # - If hess and hessp are None then approximate hessp using the grad/jac.

    if (hess in FD_METHODS or isinstance(_h, LinearOperator)):
        fhess = None

        def _hessp(x, p, *args):
            return sf.hess(x).dot(p)

        fhess_p = _hessp

    def terminate(warnflag, msg):
        if disp:
            _print_success_message_or_warn(warnflag, msg)
            print("         Current function value: %f" % old_fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % sf.nfev)
            print("         Gradient evaluations: %d" % sf.ngev)
            print("         Hessian evaluations: %d" % hcalls)
        fval = old_fval
        result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                                njev=sf.ngev, nhev=hcalls, status=warnflag,
                                success=(warnflag == 0), message=msg, x=xk,
                                nit=k)
        if retall:
            result['allvecs'] = allvecs
        return result

    hcalls = 0
    if maxiter is None:
        maxiter = len(x0)*200
    cg_maxiter = 20*len(x0)

    xtol = len(x0) * avextol
    # Make sure we enter the while loop.
    update_l1norm = np.finfo(float).max
    xk = np.copy(x0)
    if retall:
        allvecs = [xk]
    k = 0
    gfk = None
    old_fval = f(x0)
    old_old_fval = None
    float64eps = np.finfo(np.float64).eps
    while update_l1norm > xtol:
        if k >= maxiter:
            msg = "Warning: " + _status_message['maxiter']
            return terminate(1, msg)
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - grad f(xk) starting from 0.
        b = -fprime(xk)
        maggrad = np.linalg.norm(b, ord=1)
        eta = min(0.5, math.sqrt(maggrad))
        termcond = eta * maggrad
        xsupi = zeros(len(x0), dtype=x0.dtype)
        ri = -b
        psupi = -ri
        i = 0
        dri0 = np.dot(ri, ri)

        if fhess is not None:             # you want to compute hessian once.
            A = sf.hess(xk)
            hcalls += 1

        for k2 in range(cg_maxiter):
            if np.add.reduce(np.abs(ri)) <= termcond:
                print(f'iter {k}, breaking CG loop at cgiter {k2} with termcond {termcond:.2e}')
                # breakpoint()
                break
            if fhess is None:
                if fhess_p is None:
                    Ap = approx_fhess_p(xk, psupi, fprime, epsilon)
                else:
                    Ap = fhess_p(xk, psupi, *args)
                    hcalls += 1
            else:
                # hess was supplied as a callable or hessian update strategy, so
                # A is a dense numpy array or sparse matrix
                Ap = A.dot(psupi)
            # check curvature
            Ap = asarray(Ap).squeeze()  # get rid of matrices...
            curv = np.dot(psupi, Ap)
            if 0 <= curv <= 0. * float64eps:
                print(f'iter {k}, breaking CG loop at cgiter {k2} with curv {curv:.2e}')
                break
            elif curv < 0:
                if (i > 0):
                    break
                else:
                    # fall back to steepest descent direction
                    xsupi = dri0 / (-curv) * b
                    break
            alphai = dri0 / curv
            xsupi += alphai * psupi
            ri += alphai * Ap
            dri1 = np.dot(ri, ri)
            betai = dri1 / dri0
            psupi = -ri + betai * psupi
            i += 1
            dri0 = dri1          # update np.dot(ri,ri) for next time.
        else:
            # curvature keeps increasing, bail out
            msg = ("Warning: CG iterations didn't converge. The Hessian is not "
                   "positive definite.")
            return terminate(3, msg)

        pk = xsupi  # search direction is solution to system.
        gfk = -b    # gradient at xk

        try:
            alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, fprime, xk, pk, gfk,
                                          old_fval, old_old_fval, c1=c1, c2=c2)
        except _LineSearchError:
            # Line search failed to find a better solution.
            msg = "Warning: " + _status_message['pr_loss']
            return terminate(2, msg)

        update = alphak * pk
        xk += update        # upcast if necessary
        if retall:
            allvecs.append(xk)
        k += 1
        intermediate_result = OptimizeResult(x=xk, fun=old_fval)
        if _call_callback_maybe_halt(callback, intermediate_result):
            return terminate(5, "")
        update_l1norm = np.linalg.norm(update, ord=1)

        print(f'update_l1norm, {update_l1norm:.2e}')

    else:
        if np.isnan(old_fval) or np.isnan(update_l1norm):
            return terminate(3, _status_message['nan'])

        msg = _status_message['success']
        return terminate(0, msg)

if __name__ == '__main__':
    import scipy as sp
    from scipy.optimize import fmin_ncg as fmin_ncg_orig

    from .loss import (common_computation_main, create_workspace_main,
                       gradient, hessian, loss)

    np.random.seed(0)
    m = 20
    n = 10
    zero = 5
    nonneg = 15
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    c = np.random.randn(n)
    Q = sp.sparse.bmat([
        [None, A.T, c.reshape(n, 1)],
        [-A, None, b.reshape(m, 1)],
        [-c.reshape(1, n), -b.reshape(1, m), None],
        ]).tocsc()

    workspace = create_workspace_main(n, zero, nonneg)
    u = np.random.randn(m+n+1)
    common_computation_main(u, Q, n, zero, nonneg, workspace)

    def my_loss(u):
        common_computation_main(u, Q, n, zero, nonneg, workspace)
        return loss(u, Q, n, zero, nonneg, workspace)

    def my_grad(u):
        common_computation_main(u, Q, n, zero, nonneg, workspace)
        return np.copy(gradient(u, Q, n, zero, nonneg, workspace))

    def my_hessian(u):
        common_computation_main(u, Q, n, zero, nonneg, workspace)
        return hessian(u, Q, n, zero, nonneg, workspace)

    print('ORIGINAL')

    # original; defaults copied from scipy docs page
    u_0 = np.zeros(m+n+1)
    u_0[-1] = 1.
    result_orig = fmin_ncg_orig(
        f = my_loss,
        x0 = u_0,
        fprime = my_grad,
        fhess_p=None,
        fhess=my_hessian,
        args=(),
        avextol=1e-05,
        epsilon=1.4901161193847656e-08,
        maxiter=None,
        full_output=0,
        disp=1,
        retall=0,
        callback=None,
        c1=0.0001,
        c2=0.9)

    u = result_orig
    v = Q @ u
    print(f'kappa {u[-1]:.2e}')
    print(f'tau {v[-1]:.2e}')
    print(f'loss {my_loss(u):.2e}')

    print('OURS')

    # ours
    u_0 = np.zeros(m+n+1)
    u_0[-1] = 1.
    result_ours = _minimize_newtoncg(
        fun=my_loss,
        x0=u_0,
        args=(),
        jac=my_grad,
        hess=my_hessian,
        hessp=None,
        callback=None,
        xtol=0., #1e-5,
        eps=_epsilon,
        maxiter=None,
        disp=1,
        return_all=False,
        c1=1e-4, c2=0.9)

    u = result_ours['x']
    v = Q @ u
    print(f'kappa {u[-1]:.2e}')
    print(f'tau {v[-1]:.2e}')
    print(f'loss {my_loss(u):.2e}')
