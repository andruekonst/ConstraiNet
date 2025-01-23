import torch
from .base import (
    make_fused_system_nn,
    NNOptSolver,
    LinearConstraintsOptProblem,
    get_rng,
    find_polytope_bbox,
    make_init_solutions,
)


class SimpleFusedConstrainNetwork(torch.nn.Module):
    def __init__(self, pivot: torch.Tensor, A: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.pivot = pivot
        self.At = A.T
        self.b = b.unsqueeze(0)

    def forward(self, z):
        # z shape: (n_samples, n_features)
        # At shape: (n_features, n_constraints)
        sign_arg = z @ self.At - self.b
        # when ind == True, no projection is required
        ind = (sign_arg >= 0)
        ray = self.pivot.unsqueeze(0) - z
        alpha = -sign_arg / torch.clamp_max(ray @ self.At, -1.e-9).detach()
        # alpha shape: (n_samples, n_constraints)
        # alpha = torch.clamp(alpha, min=0.0, max=1.0)

        # aggregate alpha values among constraints that are violated
        # we have to select the maximum value
        common_alpha, max_ids = torch.max(alpha * ind, dim=1)
        common_ind = torch.any(ind, dim=1)

        return z + common_ind.unsqueeze(1) * (common_alpha.unsqueeze(1) * ray)



class SimpleFusedNNOptSolver(NNOptSolver):
    def make_projection_nn(self, problem: LinearConstraintsOptProblem):
        rng = get_rng(self)
        bbox = find_polytope_bbox(problem.constraints)
        projection_nn = make_fused_system_nn(
            problem.constraints,
            SimpleFusedConstrainNetwork,
            rng=rng,
            bbox=bbox,
            n_candidates=self.n_pivot_candidates,
            return_pivot=False
        ).double()
        init_solutions = torch.tensor(make_init_solutions(
            problem.constraints,
            bbox,
            self.n_solutions,
            rng
        ))
        return projection_nn, init_solutions
