import numpy as np
import scipy as sp

from .equilibrate import equilibrate

class Solver:

    def __init__(
        self,
        A: sp.sparse.csc_matrix,
        b: np.array,
        c: np.array,
        P: sp.sparse.csc_matrix,
        zero: int,
        nonneg: int,
    ):

        self.A = sp.sparse.csc_matrix(A)
        self.m, self.n = A.shape
        assert len(b) == self.m
        self.b = np.array(b)
        assert len(c) == self.n
        self.c = np.array(c)
        P = sp.sparse.csc_matrix(P)
        assert P.shape == (self.n, self.n)
        assert (P != P.T).nnz == 0
        self.P = P
        assert zero >= 0
        assert nonneg >= 0
        assert (zero + nonneg) == self.m
        self.zero = zero
        self.nonneg = nonneg

        self.equilibrate()

    def equilibrate(self):
        self.e, self.d, self.sigma, _ = equilibrate(
            A=self.A, b=self.b, c=self.c, P=self.P,
            lnorm=2, niter=25)

        self.A_eq = \
            sp.sparse.diags(self.d) @ self.A @ sp.sparse.diags(self.e)
        self.b_eq = self.sigma * self.b * self.d
        self.c_eq = self.sigma * self.c * self.e
        self.P_eq = \
            sp.sparse.diags(self.e) @ self.P @ sp.sparse.diags(self.e)

    def residuals(
        self,
        x: np.array,
        y: np.array,):

        s = self.b_eq - self.A_eq @ x
        primal_cone_residual = s
        primal_cone_residual[self.zero:] = np.minimum(
            primal_cone_residual[self.zero:], 0.
        )
        dual_cone_residual = np.minimum(
            y[self.zero:], 0.
        )
        dual_residual = \
            self.P_eq @ x + self.A_eq.T @ y + self.c_eq
        gap_residual = \
            self.c_eq @ x + self.b_eq @ y + x.T @ self.P_eq @ x
        return \
            primal_cone_residual, dual_cone_residual, dual_residual, \
            gap_residual

    def residual(
        self,
        x: np.array,
        y: np.array,):
        primal_cone_residual, dual_cone_residual, dual_residual, \
            gap_residual = self.residuals(x=x, y=y)

        return np.concatenate(
            [primal_cone_residual,
            dual_cone_residual,
            dual_residual,
            [gap_residual]]
        )

    def gradient(
        self,
        x: np.array,
        primal_cone_residual: np.array,
        dual_cone_residual: np.array,
        dual_residual: np.array,
        gap_residual: float
    ):
        grad_x = \
            -self.A_eq.T @ primal_cone_residual + self.P_eq @ dual_residual \
            + self.c_eq * gap_residual + 2 * gap_residual * self.P_eq.T @ x
        grad_y = self.A_eq @ dual_residual + self.b_eq * gap_residual
        grad_y[self.zero:] += dual_cone_residual

        return grad_x, grad_y

    def compact_gradient(
        self,
        x: np.array,
        y: np.array,
    ):
        primal_cone_residual, dual_cone_residual, dual_residual, \
            gap_residual = self.residuals(x, y)
        grad_x, grad_y = self.gradient(
            x=x,
            primal_cone_residual=primal_cone_residual,
            dual_cone_residual=dual_cone_residual,
            dual_residual=dual_residual,
            gap_residual=gap_residual)
        return np.concatenate([grad_x, grad_y])

    def loss(self,
        x: np.array,
        y: np.array):
            return sum(np.sum(el**2) for el in self.residuals(x,y)) / 2.0

    def toy_solve(self):

        return sp.optimize.minimize(
            lambda var: self.loss(var[:self.n], var[self.n:]),
            np.zeros(self.m + self.n),
            tol=1e-12,
        )

    def toy_solve_gradient(self):

        return sp.optimize.minimize(
            fun=lambda var: self.loss(var[:self.n], var[self.n:]),
            x0=np.zeros(self.m + self.n),
            jac=lambda var: self.compact_gradient(var[:self.n], var[self.n:]),
            tol=1e-12,
            # method='L-BFGS-B',
        )

    def gradient_descent(self):

        var = np.zeros(self.m + self.n)
        vars = []
        grads = []

        for i in range(2000):
            grad = self.compact_gradient(var[:self.n], var[self.n:])
            grads.append(grad)
            vars.append(np.copy(var))
            print(self.loss(var[:self.n], var[self.n:]))
            if i < 5:
                var -= grad/10
            else:
                var = self.get_new_var(vars=vars, grads=grads)
        
        # vars = np.array(vars)
        # S = np.diff(vars, axis=0)
        # grads = np.array(grads)
        # Y = np.diff(grads, axis=0)

        # #breakpoint()
        # import cvxpy as cp
        # MEM = 5
        # H = cp.Variable((self.n + self.m, self.n + self.m), PSD=True)
        # used_Y = Y[-MEM:].T
        # used_S = S[-MEM:].T
        # normer = MEM / np.linalg.norm(used_S)
        # used_Y *= normer
        # used_S *= normer
        # #breakpoint()
        # gamma = (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
        # print('gamma',gamma)
        # objective = cp.Minimize(
        #     1.0 * cp.sum_squares(H - gamma*np.eye(self.n + self.m))
        #      +
        #     ((self.n + self.m ) / MEM) * cp.sum_squares(H @ used_Y - used_S) * (gamma**2)
        # )
        # cp.Problem(objective).solve(solver='CLARABEL')
        # print('obj val 1', cp.sum_squares(H - np.eye(self.n + self.m)).value)
        # print('obj val 2', cp.sum_squares(H @ used_Y - used_S).value)
        # print('eivals', np.linalg.eigh(H.value)[0])

        # newvar = vars[-1] - H.value @ grads[-1]
        # print('newloss', self.loss(newvar[:self.n], newvar[self.n:]))
        
        # raise Exception

    def get_new_var(self, vars, grads):

        vars = np.array(vars)
        S = np.diff(vars, axis=0)
        grads = np.array(grads)
        Y = np.diff(grads, axis=0)

        #breakpoint()
        import cvxpy as cp
        MEM = 5
        H = cp.Variable((self.n + self.m, self.n + self.m), PSD=True)
        used_Y = Y[-MEM:].T
        used_S = S[-MEM:].T
        normer = MEM / np.linalg.norm(used_S)
        used_Y *= normer
        used_S *= normer
        #breakpoint()
        gamma = (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
        print('gamma',gamma)
        objective = cp.Minimize(
            0.0000001 * cp.sum_squares(H - gamma*np.eye(self.n + self.m))
             +
            ((self.n + self.m ) / MEM) * cp.sum_squares(H @ used_Y - used_S) * (gamma**2)
        )
        cp.Problem(objective).solve(solver='SCS', eps=1e-12, max_iters=1000)
        print('obj val 1', cp.sum_squares(H - np.eye(self.n + self.m)).value)
        print('obj val 2', cp.sum_squares(H @ used_Y - used_S).value)
        print('eivals', np.linalg.eigh(H.value)[0])

        newvar = np.copy(vars[-1] - H.value @ grads[-1])
        print('newloss', self.loss(newvar[:self.n], newvar[self.n:]))
        print('grad norm', np.linalg.norm(grads[-1]))
        
        return newvar


if __name__ == '__main__':

    # create data
    n = 20
    nonneg = 40
    zero = 0
    m = nonneg + zero

    A = sp.sparse.rand(m,n, density=.3)
    b = np.random.randn(m)
    c = np.random.randn(n)
    P = sp.sparse.rand(n,n, density=.15)
    P += P.T

    # initialize solver
    solver = Solver(
        A=A,
        b=b,
        c=c,
        P=P,
        zero=zero,
        nonneg=nonneg
    )

    # test GD
    solver.gradient_descent()
    
    # test toy solve, no gradient

    x0 = np.zeros(n)
    y0 = np.zeros(m)

    solver.residual(x=x0, y=y0)

    sol = solver.toy_solve()
    print(sol)

    # run gradient code

    primal_cone_residual0, dual_cone_residual0, dual_residual0, \
        gap_residual0 = solver.residuals(x=x0, y=y0)
    grad_x0, grad_y0 = solver.gradient(
        x=x0,
        primal_cone_residual=primal_cone_residual0,
        dual_cone_residual=dual_cone_residual0,
        dual_residual=dual_residual0,
        gap_residual=gap_residual0)

    # check gradient
    def loss_test(var):
        x, y = var[:solver.n], var[solver.n:]
        return solver.loss(x, y)
    def grad_test(var):
        x, y = var[:solver.n], var[solver.n:]
        return solver.compact_gradient(x=x, y=y)
    
    for i in range(10):
        print('check grad', i)
        print(sp.optimize.check_grad(
            loss_test, grad_test, np.random.randn(solver.m+solver.n)))
        
    # test toy solve with gradient
    sol = solver.toy_solve_gradient()
    print(sol)

