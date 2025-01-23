import torch
from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Literal, Optional, Tuple
from functools import cached_property
from .base import AbstractConstraints, AbstractOptProblem, AbstractConstraintsGenerator
from ..utils import check_system, get_rng


@dataclass
class LinearEqualityConstraints(AbstractConstraints):
    """A x = b.
    """
    A: np.ndarray
    b: np.ndarray

    def __post_init__(self):
        assert self.A.ndim == 2
        assert self.b.ndim == 1

    @property
    def n_features(self) -> int:
        return self.A.shape[1]

    @property
    def n_constraints(self) -> int:
        return self.A.shape[0]

    def copy(self) -> 'LinearEqualityConstraints':
        return LinearEqualityConstraints(self.A.copy(), self.b.copy())

    def append(self, other: 'LinearEqualityConstraints') -> 'LinearEqualityConstraints':
        assert self.n_features == other.n_features, \
               'Constraints dimensions (n_features) should match'
        return LinearConstraints(
            np.concatenate((self.A, other.A), axis=0),
            np.concatenate((self.b, other.b), axis=0),
        )

    def is_inside(self, xs: np.ndarray, eps: float = 1.e-9) -> bool:
        b = self.b
        if xs.ndim == 1:
            xs = xs[np.newaxsis]
        b = b[:, np.newaxsis]
        return np.all(np.abs(self.A @ xs.T - b) <= eps, axis=0)

    @cached_property
    def null_space(self) -> Tuple[np.ndarray, np.ndarray]:
        u, sigma, v = np.linalg.svd(self.A)
        R = v[len(sigma):].T
        # p, *_  = np.linalg.lstsq(A, b, rcond=None)
        s_inv = np.zeros((v.shape[0], u.shape[0]))
        np.fill_diagonal(s_inv, 1 / sigma)
        p = v.T @ s_inv @ u.T @ self.b
        return R, p


@dataclass
class LinearConstraints(AbstractConstraints):
    """A x <= b.
    """
    A: np.ndarray
    b: np.ndarray

    @property
    def n_features(self) -> int:
        return self.A.shape[1]

    @property
    def n_constraints(self) -> int:
        return self.A.shape[0]

    def copy(self) -> 'LinearConstraints':
        return LinearConstraints(self.A.copy(), self.b.copy())

    def append(self, other: 'LinearConstraints') -> 'LinearConstraints':
        assert self.n_features == other.n_features, \
               'Constraints dimensions (n_features) should match'
        return LinearConstraints(
            np.concatenate((self.A, other.A), axis=0),
            np.concatenate((self.b, other.b), axis=0),
        )

    def is_inside(self, xs: np.ndarray, eps: float = 1.e-9) -> np.ndarray:
        b = self.b
        if xs.ndim == 1:
            xs = xs[np.newaxis]
        b = b[:, np.newaxis]
        return np.all(self.A @ xs.T - b <= eps, axis=0)


class LinearConstraintsGenerator(AbstractConstraintsGenerator):
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def check_system(self, A, b, tol=1.e-7):
        """Check if `A x <= b` has at least one feasible point.
        """
        return check_system(A, b, tol=tol)

    def generate(self, n_inequalities: int, n_features: int, n_attempts: int = 1000,
                 method: Literal['bruteforce', 'iterative_bounded'] = 'bruteforce',
                 zero_center: bool = False,
                 check_system: bool = True) -> LinearConstraints:
        """Generate correct system `A x <= b`.
        """
        rng = get_rng(self)
        shape = (n_inequalities, n_features)

        if method == 'bruteforce':
            for i in range(n_attempts):
                A = rng.uniform(size=shape)
                b = rng.uniform(size=n_inequalities)
                if self.check_system(A, b):
                    return LinearConstraints(A=A, b=b)
            raise Exception(f"Cannot generate at least one good system of shape {shape}")
        elif method == 'iterative_bounded':
            center = rng.normal(size=(n_features,))
            if zero_center:
                center *= 0.0
            directions = rng.normal(size=shape)
            lengths = np.linalg.norm(directions, axis=1)
            A = directions / lengths[:, np.newaxis]
            # scales = rng.exponential(size=(n_inequalities,))
            # points = center[np.newaxis] + directions * (1.0 + scales[:, np.newaxis])
            points = center[np.newaxis] + directions
            b = np.einsum('if,if->i', A, points)
            # print(A.shape, b.shape)
            if check_system:
                assert self.check_system(A, b)
            return LinearConstraints(A=A, b=b)


class LinearConstraintsOptProblem(AbstractOptProblem):
    @property
    @abstractmethod
    def constraints(self) -> LinearConstraints:
        ...

    def check_solution(self, solutions: torch.Tensor):
        x = np.asarray(solutions)
        b = self.constraints.b
        if x.ndim == 1:
            x = x[np.newaxis]
        b = b[:, np.newaxis]
        EPS = 1.e-9
        return np.all(self.constraints.A @ x.T - b <= EPS, axis=0)
