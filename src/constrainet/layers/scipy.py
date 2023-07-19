import numpy as np
import torch
from typing import Literal, Tuple
from scipy.optimize import minimize, LinearConstraint as ScipyLinearConstraint
from scipy.optimize import NonlinearConstraint as ScipyNonlinearConstraint
from .nn import NNOptSolver, LinearConstraintsOptProblem, get_rng
from ..problems.linear_constraints import LinearConstraints
from ..problems.lin_quad import LinearOptProblem, QCLinearOptProblem, QCQuadraticOptProblem, QuadraticOptProblem
from ..problems.quadratic_constraints import QuadraticConstraintsOptProblem, quad_forms


def make_point_loss(problem):
    def point_loss(x):
        x_tensor = torch.tensor(x, dtype=torch.double).unsqueeze(0)
        return problem.loss(x_tensor).sum().item()
    return point_loss


class LCScipyOptSolver(NNOptSolver):
    """SciPy opt solver for problems with linear constraints.

    """
    def make_projection_nn(self):
        pass

    def solve(self, problem: LinearConstraintsOptProblem) -> np.ndarray:
        n_features = self.constraints.n_features
        res = minimize(
            make_point_loss(problem),
            np.zeros(n_features),
            method='trust-constr',
            # jac=point_loss_gradient,
            constraints=[ScipyLinearConstraint(problem.constraints.A, ub=problem.constraints.b)],
        )
        return res.x if res.x is not None else np.full(n_features, np.nan, dtype=np.float64)


class QCSciPyOptSolver(NNOptSolver):
    """SciPy opt solver for problems with quadratic constraints.

    """
    def make_projection_nn(self):
        pass

    def solve(self, problem: QuadraticConstraintsOptProblem) -> np.ndarray:
        P = problem.constraints.P
        q = problem.constraints.q
        b = problem.constraints.b

        def quad_constraint_lhs(x):
            return (quad_forms(P, q, x[np.newaxis]) - b[np.newaxis])[0]

        n_features = problem.constraints.n_features
        res = minimize(
            make_point_loss(problem),
            np.zeros(n_features),
            method='trust-constr',
            # jac=point_loss_gradient,
            constraints=[ScipyNonlinearConstraint(quad_constraint_lhs, lb=-np.inf, ub=0)],
        )
        return res.x if res.x is not None else np.full(n_features, np.nan, dtype=np.float64)
