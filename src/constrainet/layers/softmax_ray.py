import numpy as np
import torch
from typing import Literal, Tuple
from .nn import NNOptSolver, LinearConstraintsOptProblem, get_rng
from ..problems.linear_constraints import LinearConstraints
from constrainet.polyhedron.convert import h_to_v_representation


def constraints_to_vertex_repr(constraints: LinearConstraints) -> Tuple[torch.Tensor, torch.Tensor]:
    vertices, rays = h_to_v_representation(-constraints.A, -constraints.b)
    return torch.tensor(vertices).double(), torch.tensor(rays).double()


class SoftmaxRayNN(torch.nn.Module):
    def __init__(self, vertices: torch.Tensor, rays: torch.Tensor,
                 nonnegative_mode: Literal['sqr', 'abs'] = 'sqr'):
        super().__init__()
        self.vertices = vertices
        self.rays = rays
        if nonnegative_mode == 'sqr':
            self.to_nonnegative = torch.square
        elif nonnegative_mode == 'abs':
            self.to_nonnegative = torch.abs
        else:
            raise ValueError(f'Wrong {nonnegative_mode=!r}')


    def forward(self, zs):
        """Find point inside the domain.

        Args:
            zs: Latent parameters of shape (n_samples, n_latent),
                where `n_latent = len(vertices) + len(rays)`.

        """
        n_vertices = len(self.vertices)
        n_rays = len(self.rays)
        vertex_coords = zs[:, :n_vertices]
        rays_coords = zs[:, n_vertices:]
        vertex_part = 0.0
        if n_vertices > 0:
            vertex_part = torch.softmax(vertex_coords, dim=1) @ self.vertices
        ray_part = 0.0
        if n_rays > 0:
            ray_part = self.to_nonnegative(rays_coords) @ self.rays
        return vertex_part + ray_part


class SoftmaxRayOptSolver(NNOptSolver):
    """Softmax-based solver that maps multiple parameters to point inside the domain.

    """
    def make_projection_nn(self, problem: LinearConstraintsOptProblem):
        rng = get_rng(self)
        projection_nn = SoftmaxRayNN(
            vertices,
            rays,
            nonnegative_mode=self.nonnegative_mode,
        ).double()
        vertices, rays = constraints_to_vertex_repr(problem.constraints)
        latent_shape = len(vertices) + len(rays)
        init_solutions = torch.tensor(rng.normal(size=(self.n_solutions, latent_shape)))
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
