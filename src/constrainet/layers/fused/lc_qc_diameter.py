import numpy as np
import torch
import cvxpy as cp
from typing import Literal
from .qc_diameter import tensor_mixed_form, tensor_quad_form, ZeroNaNGradientsFn


def lc_qc_find_interior_point_quadprog(A, b_lin, P, q, b_quad):
    n_features = q.shape[1]
    x = cp.Variable(n_features)
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    loss = cp.sum(A_norm @ x)
    for i in range(P.shape[0]):
        loss = loss + 0.5 * cp.quad_form(x, P[i], assume_PSD=True) + q[i] @ x
    constraints = [
            0.5 * cp.quad_form(x, P[i], assume_PSD=True) + q[i] @ x <= b_quad[i]
            for i in range(P.shape[0])
    ] + [A @ x <= b_lin]
    problem = cp.Problem(
        cp.Minimize(loss),
        constraints
    )
    problem.solve()
    feasible_point = x.value
    # return feasible_point
    alpha = cp.Variable()
    best = (0.0, 0)
    for d in (-1., 1.):
        for i in range(n_features):
            ray = np.zeros(n_features, dtype=np.float64)
            ray[i] = d
            second_problem = cp.Problem(
                cp.Maximize(alpha),
                constraints + [
                    x == feasible_point + alpha * ray,
                    alpha >= 0
                ]
            )
            second_problem.solve()
            if alpha.value >= best[0]:
                best = (alpha.value, ray)
    return feasible_point + 0.5 * best[0] * best[1]


class LQRayShiftNN(torch.nn.Module):
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
    def __init__(self, A: torch.Tensor, b_lin: torch.Tensor,
                 P: torch.Tensor, q: torch.Tensor, b_quad: torch.Tensor,
                 point: torch.Tensor,
                 normalize_rays: bool = True,
                 to_01_mode: Literal['sigmoid', 'relu1'] = 'sigmoid'):
        super().__init__()
        self.A = A
        self.b_lin = b_lin
        self.P = P
        self.q = q
        self.b_quad = b_quad
        self.point = point
        self.normalize_rays = normalize_rays

        if to_01_mode == 'sigmoid':
            self.sigmoid = torch.nn.Sigmoid()
        elif to_01_mode == 'relu1':
            self.sigmoid = lambda x: torch.clamp(x, 0.0, 1.0)
        else:
            raise ValueError(f'Wrong {to_01_mode=!r}')

    def get_max_length_quadratic(self, points: torch.Tensor,
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
        qt_add_half_tPt_sub_b = torch.einsum('cf,sf->sc', self.q, points) + 0.5 * tPt - self.b_quad.unsqueeze(0)

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

    def get_max_length_linear(self, points: torch.Tensor,
                              rays: torch.Tensor,
                              eps: float = 0.0):
        # mu = (self.b - eps - self.A @ point) / (self.A @ ray)
        # A shape: (n_constraints, n_features)
        # b shape: (n_constraints)
        # ray shape: (n_samples, n_features)
        # points shape: (n_samples, n_features)

        numerator = (self.b_lin.unsqueeze(0) - eps - torch.einsum('cf,sf->sc', self.A, points))
        dots = torch.einsum('cf,sf->sc', self.A, rays)
        mu = numerator / dots
        # mu shape: (n_samples, n_constraints)
        # mu = torch.nan_to_num(mu, neginf=torch.inf)

        mu_pos = torch.where(dots >= 0.0, mu, torch.inf)
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
        max_lengths = torch.minimum(
            self.get_max_length_linear(ps, norm_rays),
            self.get_max_length_quadratic(ps, norm_rays),
        )
        ls = betas * max_lengths
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps


class LQCenterProjectionNN(LQRayShiftNN):
    def forward(self, zs):
        """Find point inside the domain.

        Args:
            zs: Latent parameters of shape (n_samples, n_latent),
                where `n_latent = len(vertices) + len(rays)`.

        """
        EPS = 1.e-12
        rays = zs - self.point.unsqueeze(0)

        lengths = torch.clamp_min(torch.norm(rays, dim=1).unsqueeze(1), EPS)
        norm_rays = rays / lengths

        # ps = self.point.unsqueeze(0)
        ps = self.point.tile((zs.shape[0], 1))

        max_lengths = torch.minimum(
            self.get_max_length_linear(ps, norm_rays),
            self.get_max_length_quadratic(ps, norm_rays),
        )
        ls = torch.minimum(lengths.squeeze(1), max_lengths)
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps
