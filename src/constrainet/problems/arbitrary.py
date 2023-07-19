from functools import cached_property
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
from scipy.optimize import minimize, LinearConstraint as ScipyLinearConstraint
from scipy.optimize import NonlinearConstraint as ScipyNonlinearConstraint

from .lin_quad import CannotGenerateProblemError
from .base import OptProblemGenerator
from .linear_constraints import LinearConstraints, LinearConstraintsOptProblem
from .quadratic_constraints import QuadraticConstraints, QuadraticConstraintsOptProblem, quad_forms
from ..utils import get_rng


class LCNonConvexOptProblem(LinearConstraintsOptProblem):
    has_exact_solution: bool = False
    ...


class QCNonConvexOptProblem(QuadraticConstraintsOptProblem):
    has_exact_solution: bool = False
    ...


class LCRosenbrockOptProblem(LCNonConvexOptProblem):
    """Convex problem.
    """
    constraints = None

    def __init__(self, constraints: LinearConstraints):
        self.constraints = constraints

    @cached_property
    def loss(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def _loss(x):
            """
            Args:
                x: Of shape (batch_size, n_features).

            """
            p = x[:, :-1]
            n = x[:, 1:]
            vals = 100 * (n - p ** 2) ** 2 + (1 - p) ** 2
            return torch.sum(vals, dim=1)
        return _loss

    @cached_property
    def exact_solution(self) -> Optional[np.ndarray]:
        A = self.constraints.A
        b = self.constraints.b
        def point_loss(x):
            p = x[:-1]
            n = x[1:]
            vals = 100 * (n - p ** 2) ** 2 + (1 - p) ** 2
            return np.sum(vals)

        n_features = self.constraints.n_features
        res = minimize(
            point_loss,
            np.zeros(n_features),
            method='trust-constr',
            # jac=point_loss_gradient,
            constraints=[ScipyLinearConstraint(A, ub=b)],
        )
        return res.x


class LCRosenbrockOptProblemGenerator(OptProblemGenerator):
    def generate(self, constraints: LinearConstraints,
                 bounded: bool = True,
                 positive_definite: bool = True,
                 n_attempts: int = 1000) -> LCRosenbrockOptProblem:
        n = constraints.n_features
        rng = get_rng(self)
        problem = LCRosenbrockOptProblem(constraints)
        if bounded == (problem.exact_solution is not None):
            return problem
        raise CannotGenerateProblemError(f'Cannot generate {bounded=} problem in {n_attempts=}')

    def load(self, path: Path) -> LCRosenbrockOptProblem:
        data = np.load(path)
        return LCRosenbrockOptProblem(
            constraints=LinearConstraints(data['A'], data['b']),
        )

    def save(self, path: Path, problem: LCRosenbrockOptProblem):
        np.savez(
            path,
            A=problem.constraints.A,
            b=problem.constraints.b,
        )


class QCRosenbrockOptProblem(QCNonConvexOptProblem):
    """Convex problem.
    """
    constraints = None

    def __init__(self, constraints: QuadraticConstraintsOptProblem):
        self.constraints = constraints

    @cached_property
    def loss(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def _loss(x):
            """
            Args:
                x: Of shape (batch_size, n_features).

            """
            p = x[:, :-1]
            n = x[:, 1:]
            vals = 100 * (n - p ** 2) ** 2 + (1 - p) ** 2
            return torch.sum(vals, dim=1)
        return _loss

    @cached_property
    def exact_solution(self) -> Optional[np.ndarray]:
        P = self.constraints.P
        q = self.constraints.q
        b = self.constraints.b
        def point_loss(x):
            p = x[:-1]
            n = x[1:]
            vals = 100 * (n - p ** 2) ** 2 + (1 - p) ** 2
            return np.sum(vals)

        def quad_constraint_lhs(x):
            return (quad_forms(P, q, x[np.newaxis]) - b[np.newaxis])[0]

        n_features = self.constraints.n_features
        res = minimize(
            point_loss,
            np.zeros(n_features),
            method='trust-constr',
            # jac=point_loss_gradient,
            constraints=[ScipyNonlinearConstraint(quad_constraint_lhs, lb=-np.inf, ub=0)],
        )
        return res.x


class QCRosenbrockOptProblemGenerator(OptProblemGenerator):
    def generate(self, constraints: QuadraticConstraints,
                 bounded: bool = True,
                 positive_definite: bool = True,
                 n_attempts: int = 1000) -> QCRosenbrockOptProblem:
        n = constraints.n_features
        rng = get_rng(self)
        problem = QCRosenbrockOptProblem(constraints)
        if bounded == (problem.exact_solution is not None):
            return problem
        raise CannotGenerateProblemError(f'Cannot generate {bounded=} problem in {n_attempts=}')

    def load(self, path: Path) -> QCRosenbrockOptProblem:
        data = np.load(path)
        return QCRosenbrockOptProblem(
            constraints=QuadraticConstraints(data['P'], data['q'], data['b']),
        )

    def save(self, path: Path, problem: QCRosenbrockOptProblem):
        np.savez(
            path,
            P=problem.constraints.P,
            q=problem.constraints.q,
            b=problem.constraints.b,
        )



class LCRastriginOptProblem(LCNonConvexOptProblem):
    constraints = None

    def __init__(self, constraints: LinearConstraints, init_solution: np.ndarray):
        self.constraints = constraints
        self.init_solution = init_solution

    @cached_property
    def loss(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def _loss(x):
            """
            Args:
                x: Of shape (batch_size, n_features).

            """
            n_features = x.shape[1]
            a = 10
            return a * n_features + torch.sum((x ** 2 - a * torch.cos(2 * np.pi * x)), dim=1)
        return _loss

    @cached_property
    def exact_solution(self) -> Optional[np.ndarray]:
        A = self.constraints.A
        b = self.constraints.b
        n_features = self.constraints.n_features
        a = 10
        def point_loss(x):
            return a * n_features + np.sum(x ** 2 - a * np.cos(2 * np.pi * x))

        def point_loss_grad(x):
            return 2 * x + a * 2 * np.pi * np.sin(2 * np.pi * x)

        res = minimize(
            point_loss,
            self.init_solution.copy(),
            method='trust-constr',
            jac=point_loss_grad,
            constraints=[ScipyLinearConstraint(A, ub=b)],
        )
        return res.x


class LCRastriginOptProblemGenerator(OptProblemGenerator):
    def generate(self, constraints: LinearConstraints,
                 bounded: bool = True,
                 positive_definite: bool = True,
                 n_attempts: int = 1000) -> LCRastriginOptProblem:
        n = constraints.n_features
        rng = get_rng(self)
        # generate initial solution for ground truth solver
        init_solution = rng.normal(size=(n,))
        problem = LCRastriginOptProblem(constraints, init_solution=init_solution)
        if bounded == (problem.exact_solution is not None):
            return problem
        raise CannotGenerateProblemError(f'Cannot generate {bounded=} problem in {n_attempts=}')

    def load(self, path: Path) -> LCRastriginOptProblem:
        data = np.load(path)
        return LCRastriginOptProblem(
            constraints=LinearConstraints(data['A'], data['b']),
            init_solution=data['init_solution'],
        )

    def save(self, path: Path, problem: LCRastriginOptProblem):
        np.savez(
            path,
            A=problem.constraints.A,
            b=problem.constraints.b,
            init_solution=problem.init_solution
        )
