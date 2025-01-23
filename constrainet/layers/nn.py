from typing import Literal, Optional, Tuple
import numpy as np
import torch
from abc import abstractmethod
from scipy.optimize import nnls
from scipy.optimize import linprog

from .base import LinearConstraintsOptSolver
from .layers import ShiftSimpleConstrainLayer, ShiftWithoutMirrorConstrainLayer, StepModifiedGradConstrainLayer
from ..problems.linear_constraints import LinearConstraints, LinearConstraintsOptProblem
from ..utils import get_rng
from .pivot_utils import find_polytope_bbox, make_init_solutions, make_pivot, BBox


class NNOptSolver(LinearConstraintsOptSolver):
    def __init__(self, random_state: Optional[int] = None,
                 n_solutions: int = 10,
                 optim: Literal['Adam', 'AdamW', 'SGD'] = 'AdamW',
                 lr: float = 1.e-3,
                 n_iterations: int = 10000,
                 scheduler: Optional[dict] = None,
                 nonnegative_mode: Literal['sqr', 'abs'] = 'sqr',
                 loss_smoothing: float = 0.0,
                 clip_norm: Optional[float] = None,
                 n_pivot_candidates: Optional[int] = 1000,
                 orthogonalize_rays: bool = False):
        super().__init__(random_state=random_state)
        self.n_solutions = n_solutions
        self.optim = optim
        self.lr = lr
        self.n_iterations = n_iterations
        if scheduler is None:
            scheduler = dict()
        self.scheduler = scheduler
        self.nonnegative_mode = nonnegative_mode
        assert 0.0 <= loss_smoothing <= 1.0
        self.loss_smoothing = loss_smoothing  # required for the scheduler
        self.clip_norm = clip_norm
        self.n_pivot_candidates = n_pivot_candidates
        self.orthogonalize_rays = orthogonalize_rays

    @abstractmethod
    def make_projection_nn(self,
                           problem: LinearConstraintsOptProblem) -> Tuple[torch.nn.Module, torch.Tensor]:
        """
        Returns:
            Tuple (Projection neural network, Initial solutions).

        """
        ...

    def make_optim(self, params):
        if self.optim == 'AdamW':
            return torch.optim.AdamW(params, self.lr)
        if self.optim == 'Adam':
            return torch.optim.Adam(params, self.lr)
        elif self.optim == 'SGD':
            return torch.optim.SGD(params, self.lr)
        raise ValueError(f'Unsupported optim: {self.optim!r}')

    def make_scheduler(self, optim):
        params = dict(
            mode='min',
            factor=0.1,
            patience=100,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-08,
        )
        params.update(self.scheduler)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            verbose=False,
            **params
        )

    def optimize(self, problem: LinearConstraintsOptProblem,
                 projection_nn: torch.nn.Module,
                 init_solutions: torch.Tensor) -> torch.Tensor:
        loss_fn = problem.loss
        zs = init_solutions.double().clone().requires_grad_()
        # print('zs shape:', zs.shape)
        params = [zs]
        optim = self.make_optim(params)
        scheduler = self.make_scheduler(optim)

        smooth_loss = None
        best = (torch.inf, None)  # loss, None
        for i in range(self.n_iterations):
            xs = projection_nn(zs)
            sample_losses = loss_fn(xs)
            loss = torch.sum(sample_losses)

            # save the best solution
            iteration_best_sol_id = sample_losses.argmin(dim=0)
            iteration_best_loss = sample_losses[iteration_best_sol_id].item()
            if iteration_best_loss < best[0]:
                best = (iteration_best_loss, xs[iteration_best_sol_id].detach().clone())

            optim.zero_grad()
            loss.backward()
            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, self.clip_norm)
            optim.step()
            if smooth_loss is None:
                smooth_loss = loss
            else:
                smooth_loss = self.loss_smoothing * smooth_loss + (1.0 - self.loss_smoothing) * loss
            scheduler.step(smooth_loss)

        with torch.no_grad():
            final_result_xs = projection_nn(zs)
        result_xs = torch.concat((final_result_xs, best[1].unsqueeze(0)), dim=0)
        return result_xs

    def solve(self, problem: LinearConstraintsOptProblem) -> np.ndarray:
        rng = get_rng(self)
        torch_seed = rng.randint(0, np.iinfo(np.int32).max)
        torch.random.manual_seed(torch_seed)

        projection_nn, init_solutions = self.make_projection_nn(problem)
        solutions = self.optimize(problem, projection_nn, init_solutions)
        losses = problem.loss(solutions)
        best_solution_id = losses.argmin()
        return solutions[best_solution_id].detach().numpy().copy()


def make_ray_projector_system_nn(constraints,
                                 base_layer,
                                 rng=None,
                                 bbox: Optional[BBox] = None,
                                 n_candidates: int = None,
                                 return_pivot: bool = False):
    assert base_layer is not None
    if bbox is None:
        bbox = find_polytope_bbox(constraints)
    pivot = torch.tensor(make_pivot(constraints, bbox=bbox, rng=rng, n_candidates=n_candidates))

    A_tensor = torch.tensor(constraints.A, dtype=torch.double)
    b_tensor = torch.tensor(constraints.b, dtype=torch.double)
    m = constraints.A.shape[1]
    layers = []
    for i in range(constraints.A.shape[0]):
        layers.append(base_layer(
            pivot=pivot,
            a=A_tensor[i],
            b=b_tensor[i],
        ))
    net = torch.nn.Sequential(*layers).double()
    if return_pivot:
        return net, pivot
    return net


class StepGradToConstraintOptSolver(NNOptSolver):
    """Solver that modifies gradients for solutions outside of the domain,
    such that prototype moves towards the constraint, until it reaches it.

    """
    def make_projection_nn(self, problem: LinearConstraintsOptProblem):
        rng = get_rng(self)
        bbox = find_polytope_bbox(problem.constraints)
        projection_nn = make_ray_projector_system_nn(
            problem.constraints,
            StepModifiedGradConstrainLayer,
            rng=rng,
            bbox=bbox,
            n_candidates=self.n_pivot_candidates,
            return_pivot=False
        ).double()
        init_solutions = torch.tensor(make_init_solutions(problem.constraints, bbox, rng))
        return projection_nn, init_solutions


class ShiftSimpleConstraintOptSolver(NNOptSolver):
    """Solver that projects point into the interior of the domain.

    """
    def make_projection_nn(self, problem: LinearConstraintsOptProblem):
        rng = get_rng(self)
        bbox = find_polytope_bbox(problem.constraints)
        projection_nn = make_ray_projector_system_nn(
            problem.constraints.A,
            problem.constraints.b,
            ShiftSimpleConstrainLayer,
            rng=rng,
            bbox=bbox,
            n_candidates=self.n_pivot_candidates,
            return_pivot=False
        ).double()
        init_solutions = torch.tensor(make_init_solutions(problem.constraints, bbox, rng))
        return projection_nn, init_solutions


class ShiftWithoutMirrorConstraintOptSolver(NNOptSolver):
    """Solver that projects point into the interior of the domain.

    It is intended to avoid mirroring artefact.

    """
    def make_projection_nn(self, problem: LinearConstraintsOptProblem):
        rng = get_rng(self)
        bbox = find_polytope_bbox(problem.constraints)
        projection_nn = make_ray_projector_system_nn(
            problem.constraints.A,
            problem.constraints.b,
            ShiftWithoutMirrorConstrainLayer,
            rng=rng,
            bbox=bbox,
            n_candidates=self.n_pivot_candidates,
            return_pivot=False
        ).double()
        init_solutions = torch.tensor(make_init_solutions(problem.constraints, bbox, rng))
        return projection_nn, init_solutions
