import torch
from .base import (
    make_fused_system_nn,
    NNOptSolver,
    LinearConstraintsOptProblem,
    get_rng,
    find_polytope_bbox,
    make_init_solutions,
)


class BalancedStepToInteriorFusedConstrainFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, At, b, pivot):
        # z shape: (n_samples, n_features)
        # At shape: (n_features, n_constraints)
        sign_arg = z @ At - b.unsqueeze(0)
        # sign_arg shape: (n_samples, n_constraints)
        # when ind == True, no projection is required
        ind = (sign_arg >= 0)
        # ind shape: (n_samples, n_constraints)
        ray = pivot.unsqueeze(0) - z
        alpha = -sign_arg / torch.clamp_max(ray @ At, -1.e-9)
        # alpha shape: (n_samples, n_constraints)

        # aggregate alpha values among constraints that are violated
        # we have to select the maximum value
        common_alpha, _max_ids = torch.max(alpha * ind, dim=1)
        common_ind = torch.any(ind, dim=1)

        ctx.save_for_backward(At, ind, common_ind, alpha, sign_arg)
        return z + common_ind.unsqueeze(1) * (common_alpha.unsqueeze(1) * ray)

    @staticmethod
    def backward(ctx, grad_output):
        At, ind, common_ind, alpha, sign_arg = ctx.saved_tensors
        # gradients for points that satisfy constraint
        grad_z_inside = (~common_ind).unsqueeze(1) * grad_output

        # gradients for points that violate constraint
        # projection and end point correction
        # beta = 1.0 / torch.clamp(1.0 - alpha, min=1.e-5)
        # TODO: change beta, this is suboptimal

        A_norm_square = torch.norm(At, dim=0) ** 2
        # A_norm_square shape: (n_constraints,)
        mu = (sign_arg) / A_norm_square.unsqueeze(0)
        # mu shape: (n_samples, n_constraints)

        # average directions
        # outside = torch.sum(At.unsqueeze(0) * beta.unsqueeze(1) * ind.unsqueeze(1), dim=2)
        outside = torch.sum(At.unsqueeze(0) * mu.unsqueeze(1) * ind.unsqueeze(1), dim=2)
        outside = outside / torch.clamp_min(torch.sum(ind.unsqueeze(1), dim=2), 1.e-20)
        # grad_z_outside = common_ind.unsqueeze(1) * (grad_output + outside)
        grad_z_outside = common_ind.unsqueeze(1) * outside

        grad_z = grad_z_inside + grad_z_outside
        return grad_z, None, None, None


class BalancedStepToInteriorFusedConstrainNetwork(torch.nn.Module):
    def __init__(self, pivot: torch.Tensor, A: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.pivot = pivot
        self.At = A.T
        self.b = b

    def forward(self, z):
        return BalancedStepToInteriorFusedConstrainFn.apply(z, self.At, self.b, self.pivot)


class BalancedStepToInteriorFusedNNOptSolver(NNOptSolver):
    def make_projection_nn(self, problem: LinearConstraintsOptProblem):
        rng = get_rng(self)
        bbox = find_polytope_bbox(problem.constraints)
        projection_nn = make_fused_system_nn(
            problem.constraints,
            BalancedStepToInteriorFusedConstrainNetwork,
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
