"""Diameter-based point generation for Quadratic Constraints.

"""
import numpy as np
import torch
import cvxpy as cp
from typing import Literal, Tuple
from ..nn import NNOptSolver, LinearConstraintsOptProblem, get_rng
from ...problems.quadratic_constraints import QuadraticConstraintsOptProblem, QuadraticConstraints
from .diameter import *


def qc_find_interior_point_quadprog(P, q, b):
    n_features = q.shape[1]
    x = cp.Variable(n_features)
    loss = 0.0
    for i in range(P.shape[0]):
        loss = loss + 0.5 * cp.quad_form(x, P[i], assume_PSD=True) + q[i] @ x
    constraints = [
        0.5 * cp.quad_form(x, P[i], assume_PSD=True) + q[i] @ x <= b[i]
        for i in range(P.shape[0])
    ]
    problem = cp.Problem(
        cp.Minimize(loss),
        constraints
    )
    problem.solve()
    feasible_point = x.value
    alpha = cp.Variable()
    best = (0.0, 0)
    # for d in (-1., 1.):
    #     for i in range(n_features):
    #         ray = np.zeros(n_features, dtype=np.float64)
    #         ray[i] = d
    #         second_problem = cp.Problem(
    #             cp.Maximize(alpha),
    #             constraints + [
    #                 x == feasible_point + alpha * ray,
    #                 alpha >= 0
    #             ]
    #         )
    #         second_problem.solve()
    #         if alpha.value >= best[0]:
    #             best = (alpha.value, ray)
    return feasible_point + 0.5 * best[0] * best[1]


def tensor_quad_form(P, r):
    """
    Args:
        P: Tensor of shape (n_constraints, n_features, n_features).
        r: Matrix of shape (n_samples, n_features).

    Returns:
        Matrix of shape (n_samples, n_constraints).

    """
    rPr = torch.einsum('scf,sf->sc', torch.einsum('sf,ckf->sck', r, P), r)
    return rPr


def tensor_mixed_form(t, P, r):
    """
    Args:
        t: Matrix of shape (n_samples, n_features).
        P: Tensor of shape (n_constraints, n_features, n_features).
        r: Matrix of shape (n_samples, n_features).

    Returns:
        Matrix of shape (n_samples, n_constraints).

    """
    tPr = torch.einsum('scf,sf->sc', torch.einsum('sf,ckf->sck', t, P), r)
    return tPr




class QCCenterDiameterBasedNN(torch.nn.Module):
    """

    Input dimension: (n_features + 1).
                     - one ray (n_features), will be normalized;
                     - one ray length scaling coefficient.

    x = p + l(z) * r(z),
    where:
        - p -- point on the diameter;
        - alpha(z) -- shift coefficient in [0, 1];
        - l(z) = beta(z) * max_length(p, r(z))  -- ray length scaling coefficient;
        - r(z) = r0(z) / norm(r0(z))  -- normalized ray;
        - beta(z) -- scaling coefficient in [0, 1];

    """
    def __init__(self, P: torch.Tensor, q: torch.Tensor, b: torch.Tensor,
                 point: torch.Tensor,
                 nonnegative_mode: Literal['sqr', 'abs'] = 'sqr',
                 normalize_rays: bool = True,
                 to_01_mode: Literal['sigmoid', 'relu1'] = 'sigmoid'):
        super().__init__()
        self.P = P
        self.q = q
        self.b = b
        self.point = point
        self.normalize_rays = normalize_rays

        self.to_nonnegative = make_nonnegative_op(nonnegative_mode)
        if to_01_mode == 'sigmoid':
            self.sigmoid = torch.nn.Sigmoid()
        elif to_01_mode == 'relu1':
            self.sigmoid = lambda x: torch.clamp(x, 0.0, 1.0)
        else:
            raise ValueError(f'Wrong {to_01_mode=!r}')

    def get_max_length(self, points: torch.Tensor,
                       rays: torch.Tensor,
                       eps: float = 0.0):
        # mu = (self.b - eps - self.A @ point) / (self.A @ ray)
        # P shape: (n_constraints, n_features, n_features)
        # q shape: (n_constraints, n_features)
        # b shape: (n_constraints)
        # ray shape: (n_samples, n_features)
        # points shape: (n_samples, n_features)

        # r^T P r for each constraint
        # rPr = torch.einsum('scf,sf->sc', torch.einsum('sf,cff->scf', rays, self.P), rays)
        rPr = tensor_quad_form(self.P, rays)
        # rPr >= 0
        # rPr shape: (n_samples, n_constraints)

        tPr_add_qr = tensor_mixed_form(points, self.P, rays) + torch.einsum('cf,sf->sc', self.q, rays)
        tPt = tensor_quad_form(self.P, points)
        qt_add_half_tPt_sub_b = torch.einsum('cf,sf->sc', self.q, points) + 0.5 * tPt - self.b.unsqueeze(0)

        # 1. Suppose equation is quadratic
        d4 = torch.square(tPr_add_qr) - rPr * (2 * qt_add_half_tPt_sub_b)
        # Here we are using `ZeroNaNGradientsFn` for both arguments of division,
        # because it produces NaNs when denominator is zero.
        alpha_quad_num = ZeroNaNGradientsFn.apply(-tPr_add_qr + torch.sqrt(d4))
        alpha_quad_denom = ZeroNaNGradientsFn.apply(rPr)
        alpha_quadratic = alpha_quad_num / alpha_quad_denom

        # 2. Suppose equation is linear
        alpha_lin_num = ZeroNaNGradientsFn.apply(qt_add_half_tPt_sub_b)
        alpha_lin_denom = ZeroNaNGradientsFn.apply(tPr_add_qr)
        alpha_linear = -alpha_lin_num / alpha_lin_denom

        EPS = 1.e-12
        mu = torch.where(
            torch.abs(rPr.detach()) <= EPS,
            alpha_linear,
            alpha_quadratic
        )
        # mu shape: (n_samples, n_constraints)

        mu_pos = torch.where(mu >= 0.0, mu, torch.inf)
        max_scale = torch.min(mu_pos, dim=1)[0]
        return max_scale

    def forward(self, zs):
        """Find point inside the domain.

        Args:
            zs: Latent parameters of shape (n_samples, n_latent),
                where `n_latent = len(vertices) + len(rays)`.

        """
        EPS = 1.e-12
        length_scale = zs[:, 0]
        rays = zs[:, 1:]

        if self.normalize_rays:
            norm_rays = rays / torch.clamp_min(torch.norm(rays, dim=1).unsqueeze(1), EPS)
        else:
            norm_rays = rays

        # ps = self.point.unsqueeze(0)
        ps = self.point.tile((zs.shape[0], 1))

        betas = self.sigmoid(length_scale)
        max_lengths = self.get_max_length(ps, norm_rays)
        ls = betas * max_lengths
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps


class QCCenterDiameterBasedOptSolver(NNOptSolver):
    """Center on diameter is fixed, only ray and its relative length is optimized.
    """
    def make_projection_nn(self, problem: QuadraticConstraintsOptProblem):
        rng = get_rng(self)
        # find diameter
        # ray_id, (span, min_c, max_c, point) = find_diameter(problem.constraints)
        # point_on_diameter = torch.tensor(
        #     make_point_on_segment(
        #         point,
        #         -problem.constraints.A[ray_id],  # important: ray is a negative normal
        #         float(min_c),
        #         float(max_c)
        #     ),
        #     dtype=torch.double
        # )

        # TODO: find interior point by solving quadratic optimization problem
        interior_point = torch.zeros(problem.constraints.n_features, dtype=torch.double)

        projection_nn = QCCenterDiameterBasedNN(
            torch.tensor(problem.constraints.P, dtype=torch.double),
            torch.tensor(problem.constraints.q, dtype=torch.double),
            torch.tensor(problem.constraints.b, dtype=torch.double),
            interior_point,
            nonnegative_mode=self.nonnegative_mode,
        ).double()
        n_features = problem.constraints.n_features
        latent_shape = n_features + 1
        init_solutions = torch.tensor(rng.normal(size=(self.n_solutions, latent_shape)))
        init_solutions[:, :1] = 0.0
        return projection_nn, init_solutions

    def solve(self, problem: QuadraticConstraintsOptProblem) -> np.ndarray:
        rng = get_rng(self)
        torch_seed = rng.randint(0, np.iinfo(np.int32).max)
        torch.random.manual_seed(torch_seed)

        projection_nn, init_solutions = self.make_projection_nn(problem)
        solutions = self.optimize(problem, projection_nn, init_solutions)
        losses = problem.loss(solutions)
        best_solution_id = losses.argmin()
        return solutions[best_solution_id].detach().numpy().copy()





class QCProjectionCenterDiameterNN(torch.nn.Module):
    """

    Input dimension: (n_features + 1).
                     - one ray (n_features), will be normalized;
                     - one ray length scaling coefficient.

    x = p + l(z) * r(z),
    where:
        - p -- point on the diameter;
        - alpha(z) -- shift coefficient in [0, 1];
        - l(z) = beta(z) * max_length(p, r(z))  -- ray length scaling coefficient;
        - r(z) = r0(z) / norm(r0(z))  -- normalized ray;
        - beta(z) -- scaling coefficient in [0, 1];

    """
    def __init__(self, P: torch.Tensor, q: torch.Tensor, b: torch.Tensor,
                 point: torch.Tensor,
                 nonnegative_mode: str = 'ignored'):
        super().__init__()
        self.P = P
        self.q = q
        self.b = b
        self.point = point

    def get_max_length(self, points: torch.Tensor,
                       rays: torch.Tensor,
                       eps: float = 0.0):
        # mu = (self.b - eps - self.A @ point) / (self.A @ ray)
        # P shape: (n_constraints, n_features, n_features)
        # q shape: (n_constraints, n_features)
        # b shape: (n_constraints)
        # ray shape: (n_samples, n_features)
        # points shape: (n_samples, n_features)

        # r^T P r for each constraint
        # rPr = torch.einsum('scf,sf->sc', torch.einsum('sf,cff->scf', rays, self.P), rays)
        rPr = tensor_quad_form(self.P, rays)
        # rPr >= 0
        # rPr shape: (n_samples, n_constraints)

        tPr_add_qr = tensor_mixed_form(points, self.P, rays) + torch.einsum('cf,sf->sc', self.q, rays)
        tPt = tensor_quad_form(self.P, points)
        qt_add_half_tPt_sub_b = torch.einsum('cf,sf->sc', self.q, points) + 0.5 * tPt - self.b.unsqueeze(0)

        # 1. Suppose equation is quadratic
        d4 = torch.square(tPr_add_qr) - rPr * (2 * qt_add_half_tPt_sub_b)
        # Here we are using `ZeroNaNGradientsFn` for both arguments of division,
        # because it produces NaNs when denominator is zero.
        alpha_quad_num_plus = ZeroNaNGradientsFn.apply(-tPr_add_qr + torch.sqrt(d4))
        # alpha_quad_num_minus = ZeroNaNGradientsFn.apply(-tPr_add_qr - torch.sqrt(d4))
        # alpha_quad_num = torch.where(
        #     alpha_quad_num_minus >= 0.0,
        #     alpha_quad_num_minus,
        #     alpha_quad_num_plus
        # )
        alpha_quad_num = alpha_quad_num_plus
        alpha_quad_denom = ZeroNaNGradientsFn.apply(rPr)
        alpha_quadratic = alpha_quad_num / alpha_quad_denom

        # 2. Suppose equation is linear
        alpha_lin_num = ZeroNaNGradientsFn.apply(qt_add_half_tPt_sub_b)
        alpha_lin_denom = ZeroNaNGradientsFn.apply(tPr_add_qr)
        alpha_linear = -alpha_lin_num / alpha_lin_denom

        EPS = 1.e-12
        mu = torch.where(
            torch.abs(rPr.detach()) <= EPS,
            alpha_linear,
            alpha_quadratic
        )
        # mu shape: (n_samples, n_constraints)

        mu_pos = torch.where(mu >= 0.0, mu, torch.inf)
        max_scale = torch.min(mu_pos, dim=1)[0]
        return max_scale

    def forward(self, zs):
        """Find point inside the domain.

        Args:
            zs: Latent parameters of shape (n_samples, n_latent),
                where `n_latent = n_features + 1`.

        """
        EPS = 1.e-12

        rays = zs - self.point.unsqueeze(0)

        lengths = torch.clamp_min(torch.norm(rays, dim=1, keepdim=True), EPS)
        norm_rays = rays / lengths

        # ps = self.point.unsqueeze(0)
        ps = self.point.unsqueeze(0).repeat(zs.shape[0], 1)

        max_lengths = self.get_max_length(ps, norm_rays)
        ls = torch.minimum(lengths.squeeze(1), max_lengths)
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps


class QCProjectionCenterDiameterOptSolver(NNOptSolver):
    """Center on diameter is fixed, only ray and its relative length is optimized.
    """
    def make_projection_nn(self, problem: QuadraticConstraintsOptProblem):
        rng = get_rng(self)

        # TODO: find interior point by solving quadratic optimization problem
        interior_point = torch.zeros(problem.constraints.n_features, dtype=torch.double)

        projection_nn = QCProjectionCenterDiameterNN(
            torch.tensor(problem.constraints.P, dtype=torch.double),
            torch.tensor(problem.constraints.q, dtype=torch.double),
            torch.tensor(problem.constraints.b, dtype=torch.double),
            interior_point,
            nonnegative_mode=self.nonnegative_mode,
        ).double()
        n_features = problem.constraints.n_features
        init_solutions = torch.tensor(rng.normal(size=(self.n_solutions, n_features)))
        return projection_nn, init_solutions

    def solve(self, problem: QuadraticConstraintsOptProblem) -> np.ndarray:
        rng = get_rng(self)
        torch_seed = rng.randint(0, np.iinfo(np.int32).max)
        torch.random.manual_seed(torch_seed)

        projection_nn, init_solutions = self.make_projection_nn(problem)
        solutions = self.optimize(problem, projection_nn, init_solutions)
        losses = problem.loss(solutions)
        best_solution_id = losses.argmin()
        return solutions[best_solution_id].detach().numpy().copy()




class QCEdgeCenterDiameterNN(QCProjectionCenterDiameterNN):
    """Model that generates points on the surface of QC.

    Input dimension: (n_features) - one ray, will be normalized;

    x = p + l(z) * r(z),
    where:
        - p -- point on the diameter;
        - alpha(z) -- shift coefficient in [0, 1];
        - l(z) = beta(z) * max_length(p, r(z))  -- ray length scaling coefficient;
        - r(z) = r0(z) / norm(r0(z))  -- normalized ray;
        - beta(z) -- scaling coefficient in [0, 1];

    """
    def forward(self, zs):
        """Find point inside the domain.

        Args:
            zs: Latent parameters of shape (n_samples, n_features).

        """
        EPS = 1.e-12
        rays = zs - self.point.unsqueeze(0)

        lengths = torch.clamp_min(torch.norm(rays, dim=1).unsqueeze(1), EPS)
        norm_rays = rays / lengths

        # ps = self.point.unsqueeze(0)
        ps = self.point.tile((zs.shape[0], 1))

        max_lengths = self.get_max_length(ps, norm_rays)
        # ls = torch.minimum(lengths.squeeze(1), max_lengths)
        ls = max_lengths
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps
