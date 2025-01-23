import numpy as np
import torch
from .base import (
    make_fused_system_nn,
    NNOptSolver,
    LinearConstraintsOptProblem,
    get_rng,
    find_polytope_bbox,
    make_init_solutions,
)


class ParametricInnerFusedConstrainNetwork(torch.nn.Module):
    def __init__(self, pivot: torch.Tensor, A: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.pivot = pivot
        self.At = A.T
        self.b = b
        self.to_interval = torch.nn.Sigmoid()

    def forward(self, z_and_nu):
        # z_and_nu shape: (n_samples, n_features + 1)
        # z shape: (n_samples, n_features)
        # At shape: (n_features, n_constraints)
        z = z_and_nu[:, :-1]
        nu = self.to_interval(z_and_nu[:, -1])

        sign_arg = z @ self.At - self.b.unsqueeze(0)
        # # when ind == True, no projection is required
        ind = (sign_arg >= 0)
        ray = self.pivot.unsqueeze(0) - z
        alpha = -sign_arg / torch.clamp_max(ray @ self.At, -1.e-9).detach()
        # # alpha shape: (n_samples, n_constraints)
        # # alpha = torch.clamp(alpha, min=0.0, max=1.0)

        # # aggregate alpha values among constraints that are violated
        # # we have to select the maximum value
        common_alpha, max_ids = torch.max(alpha * ind, dim=1)
        common_ind = torch.any(ind, dim=1).detach()

        # final_common_alpha = common_alpha + (1 - common_alpha.detach()) * nu
        # return z + common_ind.unsqueeze(1) * (final_common_alpha.unsqueeze(1) * ray)

        # plane_projection = StepToInteriorFusedConstrainFn.apply(z, self.At, self.b, self.pivot)
        plane_projection = z + common_ind.unsqueeze(1) * (common_alpha.unsqueeze(1) * ray)
        inside_shift = ((1 - common_alpha) * nu).unsqueeze(1) * ray
        return plane_projection + common_ind.unsqueeze(1) * inside_shift


class ParametricInnerFusedNNOptSolver(NNOptSolver):
    def make_projection_nn(self, problem: LinearConstraintsOptProblem):
        rng = get_rng(self)
        bbox = find_polytope_bbox(problem.constraints)
        projection_nn = make_fused_system_nn(
            problem.constraints,
            ParametricInnerFusedConstrainNetwork,
            rng=rng,
            n_candidates=self.n_pivot_candidates,
            return_pivot=False
        ).double()
        init_sol_samples = torch.tensor(make_init_solutions(
            problem.constraints,
            bbox,
            self.n_solutions,
            rng
        ))
        PARAM_INIT_VALUE = 1.e-2  # sigmoid will be applied to it
        init_param = np.full((init_sol_samples.shape[0], 1), PARAM_INIT_VALUE, dtype=np.float64)
        init_solutions = torch.tensor(np.concatenate((init_sol_samples, init_param), axis=1))
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
