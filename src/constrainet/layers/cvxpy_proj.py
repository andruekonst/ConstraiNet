import numpy as np
import torch
from typing import Literal, Tuple

from .nn import NNOptSolver, LinearConstraintsOptProblem, get_rng
from ..problems.linear_constraints import LinearConstraints
from ..problems.quadratic_constraints import QuadraticConstraintsOptProblem
import cvxpy as cp
import warnings
try:
    from cvxpylayers.torch import CvxpyLayer
except ImportError:
    warnings.warn("Cannot import CvxpyLayer")


class CvxpyProjNN(torch.nn.Module):
    """NN Layer that uses CvxpyLayer to project an input vector to the domain.
    """
    def __init__(self, A: np.ndarray, b: np.ndarray):
        super().__init__()
        self.A = A
        self.b = b
        self.__init_projection_layer()

    def __init_projection_layer(self):
        n_features = self.A.shape[1]
        x = cp.Variable(n_features)
        z = cp.Parameter(n_features)
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(x - z)),
            [
                self.A @ x <= self.b,
            ]
        )
        self.layer = CvxpyLayer(problem, [z], [x])

    def forward(self, zs):
        """Project point to satisfy the constraint set.

        Args:
            zs: Latent parameters of shape (n_samples, n_features).

        """
        xs, = self.layer(zs[0])
        return xs.unsqueeze(0)


class CvxpyProjOptSolver(NNOptSolver):
    """CvxpyProj solver that project point to the domain.

    """
    def make_projection_nn(self, problem: LinearConstraintsOptProblem):
        rng = get_rng(self)
        projection_nn = CvxpyProjNN(
            problem.constraints.A,
            problem.constraints.b,
        ).double()
        # self.n_solutions
        init_solutions = torch.tensor(rng.normal(size=(1, problem.constraints.n_features)))
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


class QCCvxpyProjNN(torch.nn.Module):
    def __init__(self, P: torch.Tensor, q: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.P = P
        self.q = q
        self.b = b
        self.__init_projection_layer()

    def __init_projection_layer(self):
        n_features = self.q.shape[1]
        x = cp.Variable(n_features)
        z = cp.Parameter(n_features)
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(x - z)),
            [
                0.5 * cp.quad_form(x, self.P[i], assume_PSD=True) + self.q[i] @ x <= self.b[i]
                for i in range(self.P.shape[0])
            ]
        )
        self.layer = CvxpyLayer(problem, [z], [x])

    def forward(self, zs):
        """Project point to satisfy the constraint set.

        Args:
            zs: Latent parameters of shape (n_samples, n_features).

        """
        xs, = self.layer(zs[0])
        return xs.unsqueeze(0)


class QCCvxpyProjOptSolver(NNOptSolver):
    """QCCvxpyProj solver that project point to the domain.

    """
    def make_projection_nn(self, problem: QuadraticConstraintsOptProblem):
        rng = get_rng(self)
        projection_nn = QCCvxpyProjNN(
            problem.constraints.P,
            problem.constraints.q,
            problem.constraints.b,
        ).double()
        # self.n_solutions
        init_solutions = torch.tensor(rng.normal(size=(1, problem.constraints.n_features)))
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
