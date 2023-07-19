import numpy as np
import torch
from dataclasses import dataclass
from typing import Literal, Optional
from abc import ABC, abstractmethod

from sklearn.datasets import make_spd_matrix
from .base import AbstractConstraints, AbstractConstraintsGenerator, AbstractOptProblem, OptProblemGenerator
from ..utils import check_system, get_rng


@dataclass
class QuadraticConstraints(AbstractConstraints):
    """1/2 x^T P_i x + q_i^t x <= b_i.

    Attributes:
        P: Tensor of shape (n_constraints, n_features, n_features).
        q: Matrix of shape (n_constraints, n_features).
        b: Vector of shape (n_constraints,).
    """
    P: np.ndarray
    q: np.ndarray
    b: np.ndarray

    @property
    def n_features(self) -> int:
        assert self.P.shape[1] == self.P.shape[2]
        assert self.P.shape[1] == self.q.shape[1]
        return self.P.shape[1]

    def n_constraints(self) -> int:
        assert self.P.shape[0] == self.q.shape[0]
        assert self.P.shape[0] == self.b.shape[0]
        return self.P.shape[0]

    def copy(self) -> 'QuadraticConstraints':
        return QuadraticConstraints(self.P.copy(), self.q.copy(), self.b.copy())

    def append(self, other: 'QuadraticConstraints') -> 'QuadraticConstraints':
        assert self.n_features == other.n_features, \
               'Constraints dimensions (n_features) should match'
        return QuadraticConstraints(
            np.concatenate((self.P, other.P), axis=0),
            np.concatenate((self.q, other.q), axis=0),
            np.concatenate((self.b, other.b), axis=0),
        )

    def is_inside(self, xs: np.ndarray, eps: float = 1.e-9) -> bool:
        b = self.b
        if xs.ndim == 1:
            xs = xs[np.newaxis]
        b = b[np.newaxis]
        return np.all(quad_forms(self.P, self.q, xs) - b <= eps, axis=1)


def quad_forms(P, q, xs):
    """
    Args:
        P: Tensor of shape (n_constraints, n_features, n_features).
        q: Matrix of shape (n_constraints, n_features).
        xs: Matrix of shape (n_samples, n_features).

    Return:
        Matrix of shape (n_samples, n_constraints).

    """
    return 0.5 * np.einsum('scf,sf->sc', np.einsum('cfk,sf->sck', P, xs), xs) + np.einsum('cf,sf->sc', q, xs)


class QuadraticConstraintsGenerator(AbstractConstraintsGenerator):
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def check_system(self, A, b, tol=1.e-7):
        """Check if the quadratic system has at least one feasible point.
        """
        raise NotImplementedError()

    def generate(self, n_inequalities: int, n_features: int, n_attempts: int = 1000,
                 method: Literal['bruteforce', 'iterative_bounded'] = 'iterative_bounded',
                 weibull_k: float = 4.0) -> QuadraticConstraints:
        assert method == 'iterative_bounded'
        rng = get_rng(self)
        shape = (n_inequalities, n_features)

        # center = rng.normal(size=(n_features,))
        center = np.zeros((n_features,))
        P = np.stack([
            make_spd_matrix(n_features, random_state=rng)
            for _ in range(n_inequalities)
        ])
        q = rng.normal(size=shape)
        eps = rng.weibull(weibull_k, size=(n_inequalities))
        b = quad_forms(P, q, center[np.newaxis])[0] + eps
        return QuadraticConstraints(P=P, q=q, b=b)


class QuadraticConstraintsOptProblem(AbstractOptProblem):
    @property
    @abstractmethod
    def constraints(self) -> QuadraticConstraints:
        ...

    def check_solution(self, solutions: torch.Tensor):
        x = np.asarray(solutions)
        b = self.constraints.b
        if x.ndim == 1:
            x = x[np.newaxis]
        b = b[np.newaxis]
        EPS = 1.e-9
        return np.all(quad_forms(self.constraints.P, self.constraints.q, x) - b <= EPS, axis=0)


class QCConvexOptProblem(QuadraticConstraintsOptProblem):
    has_exact_solution: bool = True
    ...
