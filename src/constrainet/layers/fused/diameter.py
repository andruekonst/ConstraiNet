"""Diameter-based point generation.

"""
import numpy as np
import torch
import cvxpy as cp
from typing import Literal, Tuple
from ..nn import NNOptSolver, LinearConstraintsOptProblem, get_rng
from ...problems.linear_constraints import LinearConstraints


def make_nonnegative_op(mode: Literal['sqr', 'abs'] = 'sqr'):
    """Make operator that maps (-inf, +inf) to [0, +inf).
    """
    if mode == 'sqr':
        return torch.square
    elif mode == 'abs':
        return torch.abs
    else:
        raise ValueError(f'Wrong {mode=!r}')



class DiameterBasedNN(torch.nn.Module):
    """

    Input dimension: (n_features + 2).
                     - one non-negative shift (along diameter) coefficient;
                     - one ray (n_features), will be normalized;
                     - one ray length scaling coefficient.

    x = p(z) + l(z) * r(z),
    where:
        - p(z) = s - c(z) * d  -- point on the diameter;
        - s -- static point on the constraint;
        - c(z) = alpha(z) * c_min + (1 - alpha(z)) * c_max  -- shift along diameter;
        - d -- diameter ray;
        - alpha(z) -- shift coefficient in [0, 1];
        - l(z) = beta(z) * max_length(p, r(z))  -- ray length scaling coefficient;
        - r(z) = r0(z) / norm(r0(z))  -- normalized ray;
        - beta(z) -- scaling coefficient in [0, 1];

    """
    def __init__(self, A: torch.Tensor, b: torch.Tensor,
                 source: torch.Tensor,
                 diameter_ray: torch.Tensor,
                 c_min: float,
                 c_max: float,
                 orthogonalize_rays: bool = False,
                 nonnegative_mode: Literal['sqr', 'abs'] = 'sqr'):
        super().__init__()
        self.A = A
        self.b = b
        self.source = source
        self.diameter_ray = diameter_ray
        self.diam_sqr_norm = diameter_ray @ diameter_ray
        self.c_min = c_min
        self.c_max = c_max
        self.orthogonalize_rays = orthogonalize_rays

        self.to_nonnegative = make_nonnegative_op(nonnegative_mode)

        self.sigmoid = torch.nn.Sigmoid()

    def get_max_length(self, points: torch.Tensor,
                       rays: torch.Tensor,
                       eps: float = 0.0):
        # mu = (self.b - eps - self.A @ point) / (self.A @ ray)
        # A shape: (n_constraints, n_features)
        # b shape: (n_constraints)
        # ray shape: (n_samples, n_features)
        # points shape: (n_samples, n_features)

        numerator = (self.b.unsqueeze(0) - eps - torch.einsum('cf,sf->sc', self.A, points))
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
                where `n_latent = n_features + 1`.

        """
        EPS = 1.e-12
        diam_shift = zs[:, 0]
        length_scale = zs[:, 1]
        rays = zs[:, 2:]

        if self.orthogonalize_rays:
            # project rays into orthogonal plane (to diameter_ray)
            projs = (rays @ self.diameter_ray) / self.diam_sqr_norm
            rays = rays - projs.unsqueeze(1) * self.diameter_ray.unsqueeze(0)

        norm_rays = rays / torch.clamp_min(torch.norm(rays, dim=1).unsqueeze(1), EPS)

        if np.isfinite(self.c_max):
            alphas = self.sigmoid(diam_shift)
            cs = alphas * self.c_min + (1 - alphas) * self.c_max
        else:
            alphas = self.to_nonnegative(diam_shift)
            cs = self.c_min + alphas
        ps = self.source.unsqueeze(0) - self.diameter_ray.unsqueeze(0) * cs.unsqueeze(1)

        betas = self.sigmoid(length_scale)
        max_lengths = self.get_max_length(ps, norm_rays)
        ls = betas * max_lengths
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps


def project_onto_ray(constraints, ray):
    """
        A x <= b
        Let x = s - c * r

        Optimization problem:
            minimize or maximize c
            s.t.
            -c * A r <= b - A s

    """
    A = constraints.A
    b = constraints.b
    Ar = A @ ray
    c_max = cp.Variable()
    c_min = cp.Variable()
    s = cp.Variable(A.shape[1])
    prob = cp.Problem(
        cp.Maximize(c_max - c_min),
        [
            -c_max * Ar + A @ s <= b,
            -c_min * Ar + A @ s <= b,
            A @ s <= b,
            c_max >= 0,
            c_min >= 0,
        ]
    )
    prob.solve()
    point = s.value
    min_coef = c_min.value
    max_coef = c_max.value

    if np.isfinite(prob.value):
        return prob.value, min_coef, max_coef, point
    elif prob.status == 'unbounded':
        new_prob = cp.Problem(
        cp.Minimize(c_min),
            [
                -c_min * Ar + A @ s <= b,
                A @ s <= b,
                c_min >= 0,
            ]
        )
        new_prob.solve()
        point = s.value
        min_coef = c_min.value
        max_coef = np.inf
        return np.inf, min_coef, max_coef, point
    else:
        raise RuntimeError(f'Wrong solver status: {prob.status=!r}')


def find_diameter(constraints):
    A = constraints.A
    b = constraints.b
    best = (-np.inf,)
    best_i = None
    for i in range(A.shape[0]):
        proj_i = project_onto_ray(constraints, A[i])
        if best[0] < proj_i[0]:
            best = proj_i
            best_i = i
    return best_i, best


# def find_backprojection_matrix(a, b):
#     s = b * a.T / np.linalg.norm(a) ** 2
#     u, sigma, v = np.linalg.svd(a[np.newaxis])
#     R = v[1:].T
#     return R, s
#
#
# def find_intersection_coefs(point: np.ndarray,
#                             ray: np.ndarray,
#                             constraints: LinearConstraints,
#                             eps: float = 1.e-12):
#     mu = (constraints.b - eps - constraints.A @ point) / (constraints.A @ ray)
#     mu[np.isinf(mu)] = np.nan
#     mu_pos = mu[mu >= 0.0]
#     alpha = np.nanmin(mu_pos) if len(mu_pos) > 0 else np.inf
#     nu = -mu
#     inv_ray = -ray
#     nu_pos = nu[nu >= 0.0]
#     beta = np.nanmin(nu_pos) if len(nu_pos) > 0 else np.inf
#     return alpha, -beta


class DiameterBasedOptSolver(NNOptSolver):
    def make_projection_nn(self, problem: LinearConstraintsOptProblem):
        rng = get_rng(self)
        # find diameter
        ray_id, (span, min_c, max_c, point) = find_diameter(problem.constraints)
        diameter_ray = torch.tensor(problem.constraints.A[ray_id], dtype=torch.double)
        source = torch.tensor(point, dtype=torch.double)

        projection_nn = DiameterBasedNN(
            torch.tensor(problem.constraints.A, dtype=torch.double),
            torch.tensor(problem.constraints.b, dtype=torch.double),
            source,
            diameter_ray,
            float(min_c),
            float(max_c),
            orthogonalize_rays=self.orthogonalize_rays,
            nonnegative_mode=self.nonnegative_mode,
        ).double()
        n_features = problem.constraints.A.shape[1]
        latent_shape = n_features + 2
        init_solutions = torch.tensor(rng.normal(size=(self.n_solutions, latent_shape)))
        init_solutions[:, :2] = 0.0
        return projection_nn, init_solutions

    def solve(self, problem: LinearConstraintsOptProblem) -> np.ndarray:
        rng = get_rng(self)
        torch_seed = rng.randint(0, np.iinfo(np.int32).max)
        torch.random.manual_seed(torch_seed)

        projection_nn, init_solutions = self.make_projection_nn(problem)
        solutions = self.optimize(problem, projection_nn, init_solutions)
        losses = problem.loss(solutions)
        best_solution_id = losses.argmin()
        return solutions[best_solution_id].detach().numpy().copy()





class CenterDiameterBasedNN(torch.nn.Module):
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
    def __init__(self, A: torch.Tensor, b: torch.Tensor,
                 point: torch.Tensor,
                 nonnegative_mode: Literal['sqr', 'abs'] = 'sqr',
                 normalize_rays: bool = True):
        super().__init__()
        self.A = A
        self.b = b
        self.point = point
        self.normalize_rays = normalize_rays

        self.to_nonnegative = make_nonnegative_op(nonnegative_mode)
        self.sigmoid = torch.nn.Sigmoid()

    def get_max_length(self, points: torch.Tensor,
                       rays: torch.Tensor,
                       eps: float = 0.0):
        # mu = (self.b - eps - self.A @ point) / (self.A @ ray)
        # A shape: (n_constraints, n_features)
        # b shape: (n_constraints)
        # ray shape: (n_samples, n_features)
        # points shape: (n_samples, n_features)

        numerator = (self.b.unsqueeze(0) - eps - torch.einsum('cf,sf->sc', self.A, points))
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
                where `n_latent = n_features + 1`.

        """
        EPS = 1.e-12
        length_scale = zs[:, 0]
        rays = zs[:, 1:]

        if self.normalize_rays:
            norm_rays = rays / torch.clamp_min(torch.norm(rays, dim=1).unsqueeze(1), EPS)
        else:
            norm_rays = rays

        if self.point.ndim == 1:
            ps = self.point.unsqueeze(0)
        else:
            ps = self.point

        betas = self.sigmoid(length_scale)
        max_lengths = self.get_max_length(ps, norm_rays)
        ls = betas * max_lengths
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps


def make_point_on_segment(source: np.ndarray,
                          ray: np.ndarray,
                          min_c: float,
                          max_c: float) -> np.ndarray:
    """Make point that belongs to the segment specified by:
        - source -- point on segment;
        - ray -- direction of segment;
        - min_c -- min scale of ray that belongs to the segment;
        - max_c -- max scale of ray that belong to the segment;

    I.e. x = source + (c * min_c + (1 - c) * max_c) * ray   belongs to the segment
    for any c in [0, 1].

    Additionally, min_c or max_c may be infinite.

    """
    if np.isfinite(max_c):
        c = 0.5
        lin = c * min_c + (1.0 - c) * max_c
    else:
        shift = 1.0
        lin = min_c + shift
    return source + lin * ray


class CenterDiameterBasedOptSolver(NNOptSolver):
    """Center on diameter is fixed, only ray and its relative length is optimized.
    """
    def make_projection_nn(self, problem: LinearConstraintsOptProblem):
        rng = get_rng(self)
        # find diameter
        ray_id, (span, min_c, max_c, point) = find_diameter(problem.constraints)
        # diameter_ray = torch.tensor(problem.constraints.A[ray_id], dtype=torch.double)
        # source = torch.tensor(point, dtype=torch.double)
        point_on_diameter = torch.tensor(
            make_point_on_segment(
                point,
                -problem.constraints.A[ray_id],  # important: ray is a negative normal
                float(min_c),
                float(max_c)
            ),
            dtype=torch.double
        )

        projection_nn = CenterDiameterBasedNN(
            torch.tensor(problem.constraints.A, dtype=torch.double),
            torch.tensor(problem.constraints.b, dtype=torch.double),
            point_on_diameter,
            nonnegative_mode=self.nonnegative_mode,
        ).double()
        n_features = problem.constraints.A.shape[1]
        latent_shape = n_features + 1
        init_solutions = torch.tensor(rng.normal(size=(self.n_solutions, latent_shape)))
        init_solutions[:, :1] = 0.0
        return projection_nn, init_solutions

    def solve(self, problem: LinearConstraintsOptProblem) -> np.ndarray:
        rng = get_rng(self)
        torch_seed = rng.randint(0, np.iinfo(np.int32).max)
        torch.random.manual_seed(torch_seed)

        projection_nn, init_solutions = self.make_projection_nn(problem)
        solutions = self.optimize(problem, projection_nn, init_solutions)
        losses = problem.loss(solutions)
        best_solution_id = losses.argmin()
        return solutions[best_solution_id].detach().numpy().copy()






class TwoSidedCenterDiameterBasedNN(torch.nn.Module):
    """The save as CenterDiameterBasedNN, but it estimates two intersections
    with constraints (with ray and with negative ray), allowing to easily generate
    point close to the point on the diameter.

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
    def __init__(self, A: torch.Tensor, b: torch.Tensor,
                 point: torch.Tensor,
                 nonnegative_mode: Literal['sqr', 'abs'] = 'sqr',
                 normalize_rays: bool = True):
        super().__init__()
        self.A = A
        self.b = b
        self.point = point
        self.normalize_rays = normalize_rays

        self.to_nonnegative = make_nonnegative_op(nonnegative_mode)
        self.sigmoid = torch.nn.Sigmoid()

    def get_min_max_length(self, points: torch.Tensor,
                       rays: torch.Tensor,
                       eps: float = 0.0):
        # mu = (self.b - eps - self.A @ point) / (self.A @ ray)
        # A shape: (n_constraints, n_features)
        # b shape: (n_constraints)
        # ray shape: (n_samples, n_features)
        # points shape: (n_samples, n_features)

        numerator = (self.b.unsqueeze(0) - eps - torch.einsum('cf,sf->sc', self.A, points))
        dots = torch.einsum('cf,sf->sc', self.A, rays)
        mu = numerator / dots
        # mu shape: (n_samples, n_constraints)
        # mu = torch.nan_to_num(mu, neginf=torch.inf)

        mu_pos = torch.where(dots >= 0.0, mu, torch.inf)
        max_scale = torch.min(mu_pos, dim=1)[0]

        min_dots = -dots
        min_mu = -mu
        min_mu_pos = torch.where(min_dots >= 0, min_mu, torch.inf)
        min_scale = torch.min(min_mu_pos, dim=1)[0]

        return -min_scale, max_scale

    def forward(self, zs):
        """Find point inside the domain.

        Args:
            zs: Latent parameters of shape (n_samples, n_latent),
                where `n_latent = n_features + 1`.

        """
        EPS = 1.e-12
        length_scale = zs[:, 0]
        rays = zs[:, 1:]

        if self.normalize_rays:
            norm_rays = rays / torch.clamp_min(torch.norm(rays, dim=1).unsqueeze(1), EPS)
        else:
            norm_rays = rays

        ps = self.point.unsqueeze(0)

        betas = self.sigmoid(length_scale)
        min_lengths, max_lengths = self.get_min_max_length(ps, norm_rays)
        ls = (1 - betas) * min_lengths + betas * max_lengths
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps


class DCCenterDiameterBasedNN(torch.nn.Module):
    """Dynamically-constrained Center Diameter Based NN.

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
    def __init__(self, nonnegative_mode: Literal['sqr', 'abs'] = 'sqr',
                 normalize_rays: bool = True):
        super().__init__()
        self.normalize_rays = normalize_rays

        self.to_nonnegative = make_nonnegative_op(nonnegative_mode)
        self.sigmoid = torch.nn.Sigmoid()

    def get_max_length(self, A: torch.Tensor,
                       b: torch.Tensor,
                       points: torch.Tensor,
                       rays: torch.Tensor,
                       eps: float = 0.0):
        # mu = (self.b - eps - self.A @ point) / (self.A @ ray)
        # A shape: (n_samples, n_constraints, n_features)
        # b shape: (n_samples, n_constraints)
        # ray shape: (n_samples, n_features)
        # points shape: (n_samples, n_features)

        numerator = (b.unsqueeze(1) - eps - torch.einsum('scf,sf->sc', A, points))
        dots = torch.einsum('scf,sf->sc', A, rays)
        mu = numerator / dots
        # mu shape: (n_samples, n_constraints)
        # mu = torch.nan_to_num(mu, neginf=torch.inf)

        mu_pos = torch.where(dots >= 0.0, mu, torch.inf)
        max_scale = torch.min(mu_pos, dim=1)[0]
        return max_scale

    def forward(self, zs, A, b, points):
        """Find point inside the domain.

        Args:
            zs: Latent parameters of shape (n_samples, n_latent),
                where `n_latent = n_features + 1`.
            A: System matrix A of shape (n_samples, n_constraints, n_features).
            b: System vector b of shape (n_samples, n_constraints).
            points: Interior points for each sample, of shape (n_samples, n_features).

        """
        EPS = 1.e-12
        length_scale = zs[:, 0]
        rays = zs[:, 1:]

        if self.normalize_rays:
            norm_rays = rays / torch.clamp_min(torch.norm(rays, dim=1).unsqueeze(1), EPS)
        else:
            norm_rays = rays

        # ps = self.point.unsqueeze(0)
        ps = points

        betas = self.sigmoid(length_scale)
        max_lengths = self.get_max_length(A, b, ps, norm_rays)
        ls = betas * max_lengths
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps






class ProjectionCenterDiameterNN(torch.nn.Module):
    """

    Input dimension: (n_features) - one ray, will be normalized.

    x = p + l(z) * r(z),
    where:
        - p -- point on the diameter;
        - alpha(z) -- shift coefficient in [0, 1];
        - l(z) = beta(z) * max_length(p, r(z))  -- ray length scaling coefficient;
        - r(z) = r0(z) / norm(r0(z))  -- normalized ray;
        - beta(z) -- scaling coefficient in [0, 1];

    """
    def __init__(self, A: torch.Tensor, b: torch.Tensor,
                 point: torch.Tensor):
        super().__init__()
        self.A = A
        self.b = b
        self.point = point

    def get_max_length(self, points: torch.Tensor,
                       rays: torch.Tensor,
                       eps: float = 0.0):
        # mu = (self.b - eps - self.A @ point) / (self.A @ ray)
        # A shape: (n_constraints, n_features)
        # b shape: (n_constraints)
        # ray shape: (n_samples, n_features)
        # points shape: (n_samples, n_features)

        numerator = (self.b.unsqueeze(0) - eps - torch.einsum('cf,sf->sc', self.A, points))
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
            zs: Latent parameters of shape (n_samples, n_features).

        """
        EPS = 1.e-12
        if self.point.ndim == 1:
            ps = self.point.unsqueeze(0)
        else:
            ps = self.point

        rays = zs - ps

        lengths = torch.clamp_min(torch.norm(rays, dim=1).unsqueeze(1), EPS)
        norm_rays = rays / lengths

        max_lengths = self.get_max_length(ps, norm_rays)
        ls = torch.minimum(lengths.squeeze(1), max_lengths)
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps


class ProjectionCenterDiameterOptSolver(NNOptSolver):
    """Center on diameter is fixed, only ray and its relative length is optimized.
    """
    def make_projection_nn(self, problem: LinearConstraintsOptProblem):
        rng = get_rng(self)
        # find diameter
        ray_id, (span, min_c, max_c, point) = find_diameter(problem.constraints)
        # diameter_ray = torch.tensor(problem.constraints.A[ray_id], dtype=torch.double)
        # source = torch.tensor(point, dtype=torch.double)
        point_on_diameter = torch.tensor(
            make_point_on_segment(
                point,
                -problem.constraints.A[ray_id],  # important: ray is a negative normal
                float(min_c),
                float(max_c)
            ),
            dtype=torch.double
        )

        projection_nn = ProjectionCenterDiameterNN(
            torch.tensor(problem.constraints.A, dtype=torch.double),
            torch.tensor(problem.constraints.b, dtype=torch.double),
            point_on_diameter,
        ).double()
        n_features = problem.constraints.A.shape[1]
        init_solutions = torch.tensor(rng.normal(size=(self.n_solutions, n_features)))
        return projection_nn, init_solutions

    def solve(self, problem: LinearConstraintsOptProblem) -> np.ndarray:
        rng = get_rng(self)
        torch_seed = rng.randint(0, np.iinfo(np.int32).max)
        torch.random.manual_seed(torch_seed)

        projection_nn, init_solutions = self.make_projection_nn(problem)
        solutions = self.optimize(problem, projection_nn, init_solutions)
        losses = problem.loss(solutions)
        best_solution_id = losses.argmin()
        return solutions[best_solution_id].detach().numpy().copy()


class ZeroNaNGradientsFn(torch.autograd.Function):
    """Function that replace NaN gradients with zeros.

    It is intended to apply this function in cope with `torch.where`.
    """
    def forward(ctx, xs):
        return xs

    def backward(ctx, xs_grad):
        return torch.nan_to_num(xs_grad, nan=0.0)


class DynConstraintsCenterBasedNN(torch.nn.Module):
    """Center NN with dynamic constraints.

    `point` should be a feasible point for all input constraints,
    i.e. it determines a feasible set for constraints.

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
    def __init__(self, point: torch.Tensor,
                 normalize_rays: bool = True,
                 scale_upper_bound: float = 1.e9):
        super().__init__()
        self.point = point
        self.normalize_rays = normalize_rays
        self.scale_upper_bound = scale_upper_bound

        self.sigmoid = torch.nn.Sigmoid()

    def get_max_length(self, A, b,
                       points: torch.Tensor,
                       rays: torch.Tensor,
                       eps: float = 0.0):
        # mu = (self.b - eps - self.A @ point) / (self.A @ ray)
        # A shape: (n_samples, n_constraints, n_features)
        # b shape: (n_samples, n_constraints)
        # ray shape: (n_samples, n_features)
        # points shape: (n_samples, n_features)

        numerator = ZeroNaNGradientsFn.apply(b - eps - torch.einsum('scf,sf->sc', A, points))
        dots = ZeroNaNGradientsFn.apply(torch.einsum('scf,sf->sc', A, rays))

        mu = numerator / dots
        # mu shape: (n_samples, n_constraints)
        # mu = torch.nan_to_num(mu, neginf=torch.inf)

        mu_pos = torch.where(dots >= 0.0, mu, torch.inf)
        max_scale = torch.min(mu_pos, dim=1)[0]
        return torch.clamp_max(max_scale, self.scale_upper_bound)

    def forward(self, A, b, zs):
        """Find point inside the domain.

        Args:
            zs: Latent parameters of shape (n_samples, n_latent),
                where `n_latent = n_features + 1`.

        """
        EPS = 1.e-12
        length_scale = zs[:, 0]
        rays = zs[:, 1:]

        if self.normalize_rays:
            norm_rays = rays / torch.clamp_min(torch.norm(rays, dim=1).unsqueeze(1), EPS)
        else:
            norm_rays = rays

        ps = self.point.repeat(zs.shape[0], 1)

        betas = self.sigmoid(length_scale)
        max_lengths = self.get_max_length(A, b, ps, norm_rays)
        ls = betas * max_lengths
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps





class EdgeCenterDiameterNN(torch.nn.Module):
    """Projects points to the edge.

    Made from ProjectionCenterDiameterNN.

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
    def __init__(self, A: torch.Tensor, b: torch.Tensor,
                 point: torch.Tensor):
        super().__init__()
        self.A = A
        self.b = b
        self.point = point

    def get_max_length(self, points: torch.Tensor,
                       rays: torch.Tensor,
                       eps: float = 0.0):
        # mu = (self.b - eps - self.A @ point) / (self.A @ ray)
        # A shape: (n_constraints, n_features)
        # b shape: (n_constraints)
        # ray shape: (n_samples, n_features)
        # points shape: (n_samples, n_features)

        numerator = (self.b.unsqueeze(0) - eps - torch.einsum('cf,sf->sc', self.A, points))
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
            zs: Latent parameters of shape (n_samples, n_features).

        """
        EPS = 1.e-12
        rays = zs - self.point.unsqueeze(0)

        lengths = torch.clamp_min(torch.norm(rays, dim=1).unsqueeze(1), EPS)
        norm_rays = rays / lengths

        ps = self.point.unsqueeze(0)

        max_lengths = self.get_max_length(ps, norm_rays)
        # ls = torch.minimum(lengths.squeeze(1), max_lengths)
        ls = max_lengths
        xs = ps + ls.unsqueeze(1) * norm_rays
        return xs  # , norm_rays, max_lengths, ps
