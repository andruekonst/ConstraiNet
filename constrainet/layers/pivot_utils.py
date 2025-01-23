from typing import Literal, Optional, Tuple, NamedTuple
import numpy as np
import torch
from scipy.optimize import nnls
from scipy.optimize import linprog

from ..problems.linear_constraints import LinearConstraints, LinearConstraintsOptProblem
from ..utils import get_rng


class BBox(NamedTuple):
    left: np.ndarray
    right: np.ndarray


def find_feasible_sol(A, b):
    """Find a feasible solution to inequality system `A x <= b`.
    """
    m = A.shape[1]
    s_comps, residual = nnls(np.concatenate([A, -A, np.eye(A.shape[0])], axis=1), b)
    s = s_comps[:m] - s_comps[m:2 * m]
    return s, residual


def find_polytope_bbox(constraints: LinearConstraints) -> BBox:
    n_features = constraints.n_features
    ones = np.eye(n_features)
    left_point = np.full(n_features, -np.inf, dtype=np.float64)
    right_point = np.full(n_features, np.inf, dtype=np.float64)
    bounds = [(None, None)] * n_features
    for i in range(n_features):
        min_res = linprog(ones[i], A_ub=constraints.A, b_ub=constraints.b, bounds=bounds)
        if min_res.status == 0:
            left_point[i] = min_res.x[i]
        max_res = linprog(-ones[i], A_ub=constraints.A, b_ub=constraints.b, bounds=bounds)
        if max_res.status == 0:
            right_point[i] = max_res.x[i]
    return BBox(left=left_point, right=right_point)


def generate_in_interval(a: float, b:float, number: int, rng: np.random.RandomState) -> np.ndarray:
    a_fin = np.isfinite(a)
    b_fin = np.isfinite(b)
    if a_fin and b_fin:
        return rng.uniform(a, b, number)
    if a_fin:
        return a + rng.exponential(size=number)
    if b_fin:
        return b - rng.exponential(size=number)
    return rng.normal(size=number)


def generate_in_bbox(bbox: BBox, number: int, rng: np.random.RandomState) -> np.ndarray:
    return np.stack([
        generate_in_interval(bbox.left[i], bbox.right[i], number=number, rng=rng)
        for i in range(bbox.left.shape[0])
    ], axis=1)


def intersect_constraints(point: np.ndarray, ray: np.ndarray, constraints: LinearConstraints,
                          eps: float = 1.e-12):
    mu = (constraints.b - eps - constraints.A @ point) / (constraints.A @ ray)
    mu[np.isinf(mu)] = np.nan
    mu_pos = mu[mu >= 0.0]
    alpha = np.nanmin(mu_pos) if len(mu_pos) > 0 else 0.0
    nu = -mu
    inv_ray = -ray
    nu_pos = nu[nu >= 0.0]
    beta = np.nanmin(nu_pos) if len(nu_pos) > 0 else 0.0
    return np.stack([(point + alpha * ray), (point + beta * inv_ray)], axis=0)


def find_central_pivot(initial_pivot: np.ndarray,
                      constraints: LinearConstraints,
                      n_iterations: int = 5,
                      eps: float = 1.e-12) -> Tuple[np.ndarray, np.ndarray]:
    center_approx = initial_pivot
    n_features = constraints.A.shape[1]
    directions = np.eye(n_features)
    # print("center is ok", np.all(constraints.A @ center_approx - constraints.b <= 1.e-9))
    for i in range(n_iterations):
        corners_approx = np.concatenate([
            intersect_constraints(center_approx, directions[j], constraints, eps)
            for j in range(n_features)
        ], axis=0)
        err = (corners_approx @ constraints.A.T - constraints.b[np.newaxis])
        constraints_ok = np.all(err <= 1.e-9)
        if not constraints_ok:
            corners_approx = corners_approx[np.all(err <= 1.e-9, axis=1)]
        # print(
        #     "Constraints ok:",
        #     np.all(corners_approx @ constraints.A.T - constraints.b[np.newaxis] <= 1.e-9)
        # )
        # err = corners_approx @ constraints.A.T - constraints.b[np.newaxis]
        # argmax = np.argmax(err)
        # max_idx = np.unravel_index(argmax, err.shape)
        # print("max:", err[max_idx], "at:", max_idx)
        # print(err[max_idx[0]])
        # print(constraints.A @ corners_approx[max_idx[0]] - constraints.b)
        # print("center is ok", np.all(constraints.A @ center_approx - constraints.b <= 1.e-9))
        # exit(1)
        center_approx = corners_approx.mean(axis=0)
    return center_approx, corners_approx


def generate_in_polytope_dirichlet(initial_pivot: np.ndarray,
                                   bbox: BBox,
                                   constraints: LinearConstraints,
                                   number: int,
                                   rng: np.random.RandomState,
                                   n_iterations: int = 5) -> np.ndarray:
    # touched_constraints = (constraints.A @ initial_pivot - constraints.b <= EPS)
    center_approx, corners_approx = find_central_pivot(
        initial_pivot,
        constraints,
        n_iterations=n_iterations
    )
    weights = rng.dirichlet(
        np.ones(corners_approx.shape[0]),
        size=(number,)
    )
    points = weights @ corners_approx
    return points


def make_init_solutions(constraints, bbox, number: int, rng: np.random.RandomState):
    candidates = generate_in_bbox(bbox, number=number, rng=rng)
    # candidates_ind = np.all(candidates @ A.T - b[np.newaxis] <= EPS, axis=1)
    init_pivot = find_feasible_sol(constraints.A, constraints.b)[0].copy()
    candidates = generate_in_polytope_dirichlet(
        init_pivot,
        bbox,
        constraints,
        number,
        rng,
        n_iterations=20,
    )
    return candidates


def make_pivot(constraints: LinearConstraints, bbox: BBox, rng: np.random.RandomState,
               n_candidates: Optional[int] = 1000) -> Tuple[np.ndarray]:
    EPS = 1.e-9
    init_pivot = find_feasible_sol(constraints.A, constraints.b)[0].copy()
    if n_candidates is None:
        pivot = init_pivot
    else:
        # pivot_candidates = generate_in_bbox(bbox, number=n_candidates, rng=rng)
        pivot_candidates = generate_in_polytope_dirichlet(
            init_pivot,
            bbox,
            constraints,
            number=n_candidates,
            rng=rng,
            n_iterations=20,
        )

        # print("Init pivot ok:", np.all(constraints.A @ init_pivot - constraints.b <= EPS))
        # new_pivot, pivot_candidates = find_central_pivot(
        #     init_pivot,
        #     constraints=constraints,
        #     n_iterations=10
        # )
        candidates_ind = np.all(
            pivot_candidates @ constraints.A.T - constraints.b[np.newaxis] <= EPS,
            axis=1
        )
        # print("shape:", (pivot_candidates @ constraints.A.T - constraints.b[np.newaxis]).shape)
        # print(np.max(pivot_candidates @ constraints.A.T - constraints.b[np.newaxis], axis=1))
        pivot_candidates = pivot_candidates[candidates_ind]
        # print((pivot_candidates @ constraints.A.T).shape)
        # print(pivot_candidates.shape, candidates_ind.shape)
        # assert np.all(pivot_candidates @ constraints.A.T - constraints.b[np.newaxis] <= EPS)
        # print(constraints.A @ pivot_candidates.mean(axis=0) - constraints.b)
        # raise ValueError(f'{len(pivot_candidates)}: {pivot_candidates.shape}')
        if len(pivot_candidates) > 0:
            pivot = pivot_candidates.mean(axis=0)
            # print(constraints.A @ pivot - constraints.b)
            assert np.all(constraints.A @ pivot - constraints.b <= EPS)
        else:
            # print("Init pivot ok:", np.all(constraints.A @ init_pivot - constraints.b <= EPS))
            # print("New pivot ok:", np.all(constraints.A @ new_pivot - constraints.b <= EPS))
            # print("New pivot err:", np.max(constraints.A @ new_pivot - constraints.b))
            raise ValueError(f'{n_candidates}; {init_pivot}; {bbox}')
            pivot = init_pivot
    return pivot
