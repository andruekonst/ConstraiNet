"""Linear and quadratic optimization problems.
"""
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
import cvxpy as cp
from cvxpy.error import SolverError
from functools import cached_property
from sklearn.datasets import make_spd_matrix
import logging
from .base import OptProblemGenerator, AbstractOptProblem
from ..utils import get_rng
from .quadratic_constraints import QuadraticConstraints, QCConvexOptProblem
from .linear_constraints import LinearConstraints, LinearConstraintsOptProblem


class CannotGenerateProblemError(Exception):
    pass


class LinearOptProblem(LinearConstraintsOptProblem):
    constraints = None

    def __init__(self, constraints: LinearConstraints, cost: np.ndarray,
                 exact_solution: Optional[np.ndarray] = None):
        self.constraints = constraints
        self.cost = torch.tensor(cost, dtype=torch.double)
        if exact_solution is not None:
            self.exact_solution = exact_solution

    @cached_property
    def loss(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def _loss(x):
            """
            Args:
                x: Of shape (batch_size, n_features).

            """
            return x @ self.cost
        return _loss

    @cached_property
    def exact_solution(self) -> Optional[np.ndarray]:
        A = self.constraints.A
        b = self.constraints.b
        c = self.cost.numpy()
        n = self.constraints.n_features
        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(c.T @ x),
            [
                A @ x <= b,
            ]
        )
        try:
            prob.solve()
        except SolverError:
            return None
        return x.value


class LinearOptProblemGenerator(OptProblemGenerator):
    def generate(self, constraints: LinearConstraints,
                 bounded: bool = True,
                 n_attempts: int = 1000,
                 finally_use_normal: bool = False) -> LinearOptProblem:
        n = constraints.n_features
        rng = get_rng(self)
        for i in range(n_attempts):
            # generate a cost vector in 1-ball
            cost = rng.uniform(-1.0, 1.0, size=(n,))
            cost /= np.linalg.norm(cost)
            problem = LinearOptProblem(constraints, cost)
            if bounded == (problem.exact_solution is not None):
                return problem
        if finally_use_normal:
            constraint_id = rng.randint(0, constraints.A.shape[0])
            cost = -constraints.A[constraint_id].copy()
            problem = LinearOptProblem(constraints, cost)
            logging.info("  Random problem generation failed. Choosing normal from constraints")
            if bounded == (problem.exact_solution is not None):
                logging.info("  Generated problem is ok")
                return problem
        raise CannotGenerateProblemError(f'Cannot generate {bounded=} problem in {n_attempts=}')

    def load(self, path: Path) -> LinearOptProblem:
        data = np.load(path)
        return LinearOptProblem(
            constraints=LinearConstraints(data['A'], data['b']),
            cost=data['cost'],
            exact_solution=data.get('exact_solution', None),
        )

    def save(self, path: Path, problem: LinearOptProblem):
        np.savez(
            path,
            A=problem.constraints.A,
            b=problem.constraints.b,
            cost=problem.cost.numpy(),
            exact_solution=problem.exact_solution,
        )


class QuadraticOptProblem(AbstractOptProblem):
    constraints = None

    def __init__(self, constraints: LinearConstraints, P: np.ndarray, q: np.ndarray,
                 exact_solution: Optional[np.ndarray] = None):
        self.constraints = constraints
        self.P = torch.tensor(P)
        self.q = torch.tensor(q)
        if exact_solution is not None:
            self.exact_solution = exact_solution

    @cached_property
    def loss(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def _loss(x):
            """
            Args:
                x: Of shape (batch_size, n_features).

            """
            quad_form = torch.einsum('bi,ij,bj->b', x, self.P, x) * 0.5
            return quad_form + x @ self.q
        return _loss

    @cached_property
    def exact_solution(self) -> Optional[np.ndarray]:
        A = self.constraints.A
        b = self.constraints.b
        n = A.shape[1]
        x = cp.Variable(n)
        P = self.P.numpy()
        q = self.q.numpy()
        prob = cp.Problem(
            cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
            [
                A @ x <= b,
            ]
        )
        try:
            prob.solve()
        except SolverError:
            return None
        return x.value


class QuadraticOptProblemGenerator(OptProblemGenerator):
    def generate(self, constraints: LinearConstraints,
                 bounded: bool = True,
                 positive_definite: bool = True,
                 n_attempts: int = 1000) -> QuadraticOptProblem:
        n = constraints.n_features
        rng = get_rng(self)
        for i in range(n_attempts):
            if positive_definite:
                # if np.any(np.linalg.eigvals(P) <= 0):  # not positive definite
                #     continue
                P = make_spd_matrix(n, random_state=rng)
            else:
                P = rng.uniform(-1.0, 1.0, size=(n, n))
                # making matrix symmetric
                P = np.maximum(P, P.T)

            # generate a cost vector in 1-ball
            q = rng.uniform(-1.0, 1.0, size=(n,))
            q /= np.linalg.norm(q)
            problem = QuadraticOptProblem(constraints, P, q)
            if bounded == (problem.exact_solution is not None):
                return problem
        raise CannotGenerateProblemError(f'Cannot generate {bounded=} problem in {n_attempts=}')

    def load(self, path: Path) -> QuadraticOptProblem:
        data = np.load(path)
        return QuadraticOptProblem(
            constraints=LinearConstraints(data['A'], data['b']),
            P=data['P'],
            q=data['q'],
            exact_solution=data.get('exact_solution', None),
        )

    def save(self, path: Path, problem: QuadraticOptProblem):
        np.savez(
            path,
            A=problem.constraints.A,
            b=problem.constraints.b,
            P=problem.P.numpy(),
            q=problem.q.numpy(),
            exact_solution=problem.exact_solution,
        )


class QCLinearOptProblem(QCConvexOptProblem):
    """Linear Opt Problem with Quadratic Constraints.
    """
    constraints = None

    def __init__(self, constraints: QuadraticConstraints, cost: np.ndarray,
                 exact_solution: Optional[np.ndarray] = None):
        assert isinstance(constraints, QuadraticConstraints)
        self.constraints = constraints
        self.cost = torch.tensor(cost, dtype=torch.double)
        if exact_solution is not None:
            self.exact_solution = exact_solution

    @cached_property
    def loss(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def _loss(x):
            """
            Args:
                x: Of shape (batch_size, n_features).

            """
            return x @ self.cost
        return _loss

    @cached_property
    def exact_solution(self) -> Optional[np.ndarray]:
        P = self.constraints.P
        q = self.constraints.q
        b = self.constraints.b
        c = self.cost.numpy()
        n = self.constraints.n_features
        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(c.T @ x),
            [
                0.5 * cp.quad_form(x, P[i], assume_PSD=True) + q[i] @ x <= b[i]
                for i in range(P.shape[0])
            ]
        )
        try:
            prob.solve()
        except SolverError:
            return None
        return x.value


class QCLinearOptProblemGenerator(OptProblemGenerator):
    def generate(self, constraints: QuadraticConstraints,
                 bounded: bool = True,
                 n_attempts: int = 1000) -> QCLinearOptProblem:
        assert isinstance(constraints, QuadraticConstraints)
        n = constraints.n_features
        rng = get_rng(self)
        for i in range(n_attempts):
            # generate a cost vector in 1-ball
            cost = rng.uniform(-1.0, 1.0, size=(n,))
            cost /= np.linalg.norm(cost)
            problem = QCLinearOptProblem(constraints, cost)
            if bounded == (problem.exact_solution is not None):
                return problem
        raise CannotGenerateProblemError(f'Cannot generate {bounded=} problem in {n_attempts=}')

    def load(self, path: Path) -> QCLinearOptProblem:
        data = np.load(path)
        return QCLinearOptProblem(
            constraints=QuadraticConstraints(data['P'], data['q'], data['b']),
            cost=data['cost'],
            exact_solution=data.get('exact_solution', None),
        )

    def save(self, path: Path, problem: QCLinearOptProblem):
        np.savez(
            path,
            P=problem.constraints.P,
            q=problem.constraints.q,
            b=problem.constraints.b,
            cost=problem.cost.numpy(),
            exact_solution=problem.exact_solution,
        )


class QCQuadraticOptProblem(QCConvexOptProblem):
    """Quadratic Opt Problem with Quadratic Constraints.

    Problem with cost function: 1/2 x^T M x + f^T x.
    """
    constraints = None

    def __init__(self, constraints: QuadraticConstraints, M: np.ndarray, f: np.ndarray,
                 exact_solution: Optional[np.ndarray] = None):
        assert isinstance(constraints, QuadraticConstraints)
        self.constraints = constraints
        self.M = torch.tensor(M, dtype=torch.double)
        self.f = torch.tensor(f, dtype=torch.double)
        if exact_solution is not None:
            self.exact_solution = exact_solution

    @cached_property
    def loss(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def _loss(x):
            """
            Args:
                x: Of shape (batch_size, n_features).

            """
            quad_form = torch.einsum('bi,ij,bj->b', x, self.M, x) * 0.5
            return quad_form + x @ self.f
        return _loss

    @cached_property
    def exact_solution(self) -> Optional[np.ndarray]:
        P = self.constraints.P
        q = self.constraints.q
        b = self.constraints.b
        n = self.constraints.n_features
        x = cp.Variable(n)
        M = self.M.numpy()
        f = self.f.numpy()
        prob = cp.Problem(
            cp.Minimize((1/2)*cp.quad_form(x, M) + f.T @ x),
            [
                0.5 * cp.quad_form(x, P[i], assume_PSD=True) + q[i] @ x <= b[i]
                for i in range(P.shape[0])
            ]
        )
        try:
            prob.solve()
        except SolverError:
            return None
        return x.value


class QCQuadraticOptProblemGenerator(OptProblemGenerator):
    def generate(self, constraints: QuadraticConstraints,
                 bounded: bool = True,
                 n_attempts: int = 1000,
                 positive_definite: bool = True) -> QCQuadraticOptProblem:
        assert isinstance(constraints, QuadraticConstraints)
        n = constraints.n_features
        rng = get_rng(self)
        for i in range(n_attempts):
            if positive_definite:
                # if np.any(np.linalg.eigvals(P) <= 0):  # not positive definite
                #     continue
                M = make_spd_matrix(n, random_state=rng)
            else:
                M = rng.uniform(-1.0, 1.0, size=(n, n))
                # making matrix symmetric
                M = np.maximum(M, M.T)

            # generate a cost vector in 1-ball
            f = rng.uniform(-1.0, 1.0, size=(n,))
            f /= np.linalg.norm(f)
            problem = QCQuadraticOptProblem(constraints, M, f)
            if bounded == (problem.exact_solution is not None):
                return problem

    def load(self, path: Path) -> QCQuadraticOptProblem:
        data = np.load(path)
        return QCQuadraticOptProblem(
            constraints=QuadraticConstraints(data['P'], data['q'], data['b']),
            M=data['M'],
            f=data['f'],
            exact_solution=data.get('exact_solution', None),
        )

    def save(self, path: Path, problem: QCQuadraticOptProblem):
        np.savez(
            path,
            P=problem.constraints.P,
            q=problem.constraints.q,
            b=problem.constraints.b,
            M=problem.M.numpy(),
            f=problem.f.numpy(),
            exact_solution=problem.exact_solution,
        )
