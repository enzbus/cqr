# Copyright 2024 Enzo Busseti
#
# This file is part of CQR, the Conic QR Solver.
#
# CQR is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CQR is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# CQR. If not, see <https://www.gnu.org/licenses/>.
"""Work out original EuroMir model."""


import numpy as np
import scipy as sp

from ..base_solver import BaseSolver
from .new_cqr import l2_ruiz

class BaseEuroMir(BaseSolver):
    """EuroMir model."""

    # class constants
    epsilon_convergence = 1e-12
    max_iterations = 10_000

    used_hsde = "hsde_q"

    def prepare_loop(self):
        """Define anything we need to re-use."""
        assert len(self.soc) == 0
        self.u = np.zeros(self.n+self.m+1)
        self.u[-1] = 1.

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        self.x = self.u[:self.n] / self.u[-1]
        self.y = self.u[self.n:-1] / self.u[-1]

    def residual(self, u):
        """Residual map."""
        # v = getattr(self, self.used_hsde) @ u
        # u_cone_proj = self.project_u(u)
        # v_cone_proj = self.project_v(v)
        # result1 = np.concatenate([
        #     self.project_u(u)[self.n+self.zero:] - u[self.n+self.zero:],
        #     self.project_v(v) - v
        #     ])
        result2 = np.concatenate([
            self.project_v(-u)[self.n+self.zero:],
            self.project_u(-getattr(self, self.used_hsde) @ u)
            ])
        # breakpoint()
        # assert np.allclose(result1, result2)
        return result2

    def jacobian_residual(self, u):
        """For simplicity we first do sparse jacobian matrix."""

        diag_top = (u[self.n+self.zero:] < 0.) * 1.
        top_part = sp.sparse.csc_array((self.m+1-self.zero, self.m+self.n+1))
        top_part[:, -len(diag_top):] = sp.sparse.diags(diag_top, format='csc')

        v = getattr(self, self.used_hsde) @ u
        diag_bottom = np.ones(self.m + self.n + 1)
        diag_bottom[self.n+self.zero:] = (v[self.n+self.zero:] < 0.) * 1.
        bottom_part = sp.sparse.diags(diag_bottom, format='csc') @ getattr(self, self.used_hsde)
        result = sp.sparse.vstack([top_part, bottom_part])
        return -result

    def iterate(self):
        """Do one iteration."""
        cur_residual = self.residual(self.u)
        cur_jacobian = self.jacobian_residual(self.u)
        # print(np.linalg.norm(cur_residual))
        # breakpoint()
        # step = np.linalg.lstsq(cur_jacobian.todense(), -cur_residual)[0]
        step = sp.sparse.linalg.lsqr(
            cur_jacobian, -cur_residual,
            #damp=1,
            atol=0., btol=0., iter_lim=1,
            )[0]
        self.u += step

    def pi_u(self, u, result):
        """Projection u cone."""
        result[:self.n + self.zero] = u[:self.n + self.zero]
        result[self.n+self.zero:] = u[
            self.n+self.zero:] * (u[self.n+self.zero:] > 0.)

    def dpi_u(self, u, du, result):
        """Derivative projection u cone."""
        result[:self.n + self.zero] = du[:self.n + self.zero]
        result[self.n+self.zero:] = du[
            self.n+self.zero:] * (u[self.n+self.zero:] > 0.)

    def multiply_jacobian_residual(self, u, du):
        """Multiply by Jacobian of residual map."""
        dr = np.zeros(2 * self.m + self.n + 2 - self.zero)
        dr[:self.m + 1 - self.zero] = (u[self.n + self.zero:] < 0.) * du[self.n + self.zero:]
        dv = getattr(self, self.used_hsde) @ du
        self.dpi_u(u=u, du=-dv, result=dr[-(self.m + self.n + 1):])
        dr[-(self.m + self.n + 1):] = -dr[-(self.m + self.n + 1):]

    def multiply_jacobian_residual_transpose(self, u, dr):
        """Multiply by Jacobian of residual map transpose."""
        du = np.zeros(self.m + self.n + 1)

class BaseBroydenEuroMir(BaseEuroMir):
    """Add logic to save du's and dresiduals's."""

    memory = 10
    verbose = False

    def prepare_loop(self):
        """Create storage arrays."""
        super().prepare_loop()
        lenu = len(self.u)
        lenres = lenu + self.m + 1 - self.zero
        self.dus = np.empty((self.memory, lenu), dtype=float)
        self.dresiduals = np.empty((self.memory, lenres), dtype=float)
        self.old_u = np.empty(lenu, dtype=float)
        self.old_residual = np.empty(lenres, dtype=float)

    def iterate(self):
        """Simple Douglas Rachford iteration with Broyden update to override.
        """

        self.dus[
            len(self.solution_qualities) % self.memory] = self.u - self.old_u
        self.old_u[:] = self.u

        cur_residual = self.residual(self.u)
        if self.verbose:
            print(np.linalg.norm(cur_residual))
        self.dresiduals[
            len(self.solution_qualities) % self.memory] = cur_residual - self.old_residual
        self.old_residual[:] = cur_residual

        if len(self.solution_qualities) > self.memory + 2:
            newstep = self.compute_broyden_step(cur_residual)
            self.u[:] = self.u + newstep
        else:
            cur_jacobian = self.jacobian_residual(self.u)
            # print(np.linalg.norm(cur_residual))
            # breakpoint()
            # step = np.linalg.lstsq(cur_jacobian.todense(), -cur_residual)[0]
            step = sp.sparse.linalg.lsqr(
                cur_jacobian, -cur_residual,
                #damp=1,
                atol=0., btol=0., iter_lim=1,
                )[0]
            self.u[:] = self.u + step

    def compute_broyden_step(self, cur_residual):
        """Base method to compute a Broyden-style approximate Newton step."""
        # breakpoint()
        cur_jacobian = self.jacobian_residual(self.u)
        step = sp.sparse.linalg.lsqr(
            cur_jacobian, -cur_residual,
            #damp=1,
            atol=0., btol=0., iter_lim=1,
            )[0]
        return step

class SparseBroydenEuroMir(BaseBroydenEuroMir):
    """Test with sparse update like done for CQR."""
    memory = 10
    lsqr_iters = 1
    damp = 1e-8
    max_iterations = 100_000 // (2 * lsqr_iters + 2)
    acceleration_cap = 10

    def lsqr(self, mycur_residual):
        """To test post-eq below."""
        # final correction
        cur_jacobian = self.jacobian_residual(self.u)
        # if len(self.solution_qualities) > 20000:
        #     import matplotlib.pyplot as plt
        #     plt.plot(np.linalg.norm(cur_jacobian.todense(), axis=1))
        #     plt.plot(np.linalg.norm(cur_jacobian.todense(), axis=0))
        #     plt.show()
        #     breakpoint()

        lsqr_result = sp.sparse.linalg.lsqr(
            cur_jacobian, -mycur_residual,
            # x0=result,
            damp=self.damp,
            atol=0., btol=0., iter_lim=self.lsqr_iters,
            )[0]
        return lsqr_result

    def compute_broyden_step(self, cur_residual):
        """1-Memory sparse update."""

        mycur_residual = np.copy(cur_residual)
        result = np.zeros_like(self.u)

        for back_index in range(self.memory):
            current_index = (len(
                self.solution_qualities) - back_index) % self.memory

            # correction by current index
            dres = self.dresiduals[current_index, :]
            dres_norm = np.linalg.norm(dres)
            dres_normed = dres / dres_norm
            du = self.dus[current_index, :]
            du_norm = np.linalg.norm(du)
            du_resnormed = du / dres_norm

            acceleration = du_norm / dres_norm

            # we cap the acceleration
            if acceleration > self.acceleration_cap:
                reduction_factor = acceleration / self.acceleration_cap
                # print(f"HIT CAP, ITER {len(self.solution_qualities)}, update {back_index}")
            else:
                reduction_factor = 1.

            # I had to adjust the signs, not sure why
            dres_component = -mycur_residual @ dres_normed / reduction_factor
            mycur_residual += dres_normed * dres_component
            result += du_resnormed * dres_component
            # breakpoint()
        lsqr_result = self.lsqr(mycur_residual)
        # final correction
        result += lsqr_result
        return result


class EquilibratedEuroMir(BaseEuroMir):
    """With Ruiz Eq."""

    used_hsde = "eq_hsde"
    ruiz_rounds = 100

    def prepare_loop(self):
        """Do Ruiz equilibration."""
        # if len(self.soc) > 0:
        #     raise ValueError()
        matrix = self.matrix.todense()
        concatenated = np.block(
            [[matrix, self.b.reshape(self.m, 1)],
            [self.c.reshape(1, self.n), np.zeros((1, 1))]]).A
        work_matrix = np.copy(concatenated)

        def norm_cols(concatenated):
            return np.max(np.abs(concatenated), axis=0)

        def norm_rows(concatenated):
            return np.max(np.abs(concatenated), axis=1)

        m, n = matrix.shape

        d_and_rho = np.ones(m+1)
        e_and_sigma = np.ones(n+1)

        for i in range(self.ruiz_rounds):

            nr = norm_rows(work_matrix)
            nc = norm_cols(work_matrix)

            # equalize nr for SOCs
            cur = self.zero + self.nonneg
            for soc_dim in self.soc:
                nr[cur:cur+soc_dim] = np.max(nr[cur:cur+soc_dim])
                cur += soc_dim
            # breakpoint()

            r1 = max(nr[nr > 0]) / min(nr[nr > 0])
            r2 = max(nc[nc > 0]) / min(nc[nc > 0])
            print(r1, r2)
            if (r1-1 < 1e-5) and (r2-1 < 1e-5):
                # logger.info('Equilibration converged.')
                break

            # print(r1, r2)

            d_and_rho[nr > 0] *= nr[nr > 0]**(-0.5)
            e_and_sigma[nc > 0] *= ((m+1)/(n+1))**(0.25) * nc[nc > 0]**(-0.5)

            work_matrix = ((concatenated * e_and_sigma).T * d_and_rho).T

        self.equil_e = e_and_sigma[:-1]
        self.equil_d = d_and_rho[:-1]
        self.equil_sigma = e_and_sigma[-1]
        self.equil_rho = d_and_rho[-1]

        self.eq_matrix = sp.sparse.csc_matrix(work_matrix[:-1, :-1])
        self.eq_b = work_matrix[:-1, -1]
        self.eq_c = work_matrix[-1, :-1]
        self.eq_hsde = self._build_custom_q(self.eq_matrix, self.eq_b, self.eq_c)

        super().prepare_loop()

    def obtain_x_and_y(self):
        """Redefine if/as needed."""
        super().obtain_x_and_y()

        self.x = (self.equil_e * self.x) / self.equil_sigma
        self.y = (self.equil_d * self.y) / self.equil_rho

class SparseBroydenEquilibratedEuroMir(EquilibratedEuroMir, SparseBroydenEuroMir):
    """With equilibration."""
    ruiz_rounds = 5
    memory = 10
    lsqr_iters = 1
    damp = 1e-8
    max_iterations = 100_000 // (2 * lsqr_iters + 2)
    acceleration_cap = 10
    verbose = True

class SparseBroydenEquilibratedPostEqEuroMir(EquilibratedEuroMir, SparseBroydenEuroMir):
    """With equilibration."""
    ruiz_rounds = 25
    memory = 10
    lsqr_iters = 1
    damp = 1e-8
    max_iterations = 100_000 // (2 * lsqr_iters + 2)
    acceleration_cap = 10
    verbose = True

    # def lsqr(self, mycur_residual):
    #     """To test post-eq below."""
    #     # final correction
    #     cur_jacobian = self.jacobian_residual(self.u)
    #     # if len(self.solution_qualities) > 20000:
    #     #     import matplotlib.pyplot as plt
    #     #     plt.plot(np.linalg.norm(cur_jacobian.todense(), axis=1))
    #     #     plt.plot(np.linalg.norm(cur_jacobian.todense(), axis=0))
    #     #     plt.show()
    #     #     breakpoint()

    #     lsqr_result = sp.sparse.linalg.lsqr(
    #         cur_jacobian, -mycur_residual,
    #         # x0=result,
    #         damp=self.damp,
    #         atol=0., btol=0., iter_lim=self.lsqr_iters,
    #         )[0]
    #     return lsqr_result

    def prepare_loop(self):
        """Skip SOCs."""
        assert len(self.soc) == 0
        super().prepare_loop()
        self.post_d = np.ones_like(self.old_residual)
        self.post_e = np.ones_like(self.old_u)

    def lsqr(self, mycur_residual):
        """To test post-eq below."""
        # final correction
        cur_jacobian = self.jacobian_residual(self.u)

        m = cur_jacobian.todense()
        # breakpoint()

        # something wrong, if I warmstart it explodes
        self.post_d[:], self.post_e[:] = l2_ruiz(m)#, d=self.post_d, e=self.post_e)

        d = self.post_d
        e = self.post_e
        # print(d)
        # print(e)
        internal_mat = sp.sparse.diags(d) @ cur_jacobian @ sp.sparse.diags(e)

        # breakpoint()

        lsqr_result = sp.sparse.linalg.lsqr(
            internal_mat, -sp.sparse.diags(d) @ mycur_residual,
            # x0=result,
            damp=self.damp,
            atol=0., btol=0., iter_lim=self.lsqr_iters,
            )[0]
        return sp.sparse.diags(e) @ lsqr_result

        # self.y[:] = self.cone_project(self.z)
        # step = self.linspace_project(2 * self.y - self.z) - self.y
        # # print(np.linalg.norm(step))

        # result = sp.sparse.linalg.lsqr(
        #             internal_mat, -sp.sparse.diags(d) @ step,
        #             x0=sp.sparse.diags(1./e) @ step,
        #             damp=0., # might make sense to change this?
        #             atol=0., btol=0., # might make sense to change this
        #             iter_lim=self.lsqr_iters)
        # # breakpoint()
        # # print(result[1:-1])
        # self.z[:] = self.z + sp.sparse.diags(e) @ result[0]

        # # import matplotlib.pyplot as plt
        # # breakpoint()
        # # plt.plot(np.linalg.norm(m, axis=0))
        # # plt.plot(np.linalg.norm(m, axis=1))
        # # plt.show()

        # # # breakpoint()

        # super().iterate()
