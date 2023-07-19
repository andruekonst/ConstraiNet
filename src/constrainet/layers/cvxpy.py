import numpy as np
import torch
from typing import Literal, Tuple
import cvxpy as cp
from .nn import NNOptSolver, LinearConstraintsOptProblem, get_rng
from ..problems.linear_constraints import LinearConstraints
from ..problems.lin_quad import (
    LinearOptProblem,
    QCLinearOptProblem,
    QCQuadraticOptProblem,
    QuadraticOptProblem,
)
from ..problems.quadratic_constraints import QuadraticConstraintsOptProblem


def make_objective(problem, x):
    print(type(problem))
    if isinstance(problem, LinearOptProblem) or isinstance(problem, QCLinearOptProblem):
        cost = problem.cost.numpy()
        objective = cp.Minimize(cost.T @ x)
    elif isinstance(problem, QuadraticOptProblem):
        P = problem.P.numpy()
        q = problem.q.numpy()
        objective = cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x)
    elif isinstance(problem, QCQuadraticOptProblem):
        M = problem.M.numpy()
        f = problem.f.numpy()
        objective = cp.Minimize((1/2)*cp.quad_form(x, M) + f.T @ x)
    return objective



class LCCVXPYOptSolver(NNOptSolver):
    """CVXPY opt solver for problems with linear constraints.

    """
    def make_projection_nn(self):
        pass

    def solve(self, problem: LinearConstraintsOptProblem) -> np.ndarray:
        n_features = problem.constraints.n_features
        x = cp.Variable(n_features)
        objective = make_objective(problem, x)
        opt_problem = cp.Problem(objective, [
            problem.constraints.A @ x <= problem.constraints.b,
        ])
        opt_problem.solve()
        return x.value.copy()


class QCCVXPYOptSolver(NNOptSolver):
    """CVXPY opt solver for problems with quadratic constraints.

    """
    def make_projection_nn(self):
        pass

    def solve(self, problem: QuadraticConstraintsOptProblem) -> np.ndarray:
        n_features = problem.constraints.n_features
        x = cp.Variable(n_features)
        objective = make_objective(problem, x)
        P = problem.constraints.P
        q = problem.constraints.q
        b = problem.constraints.b
        opt_problem = cp.Problem(objective, [
            0.5 * cp.quad_form(x, P[i], assume_PSD=True) + q[i] @ x <= b[i]
            for i in range(P.shape[0])
        ])
        opt_problem.solve()
        return x.value.copy()
