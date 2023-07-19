import numpy as np
import torch
from typing import Literal, Tuple, Optional
from ..nn import NNOptSolver, LinearConstraintsOptProblem, get_rng
from ...problems.linear_constraints import LinearConstraints
from ..pivot_utils import find_polytope_bbox, make_init_solutions, make_pivot, BBox


def make_fused_system_nn(constraints,
                         base_network=None,
                         rng=None,
                         n_candidates: int = None,
                         bbox: Optional[BBox] = None,
                         return_pivot: bool = False):
    if bbox is None:
        bbox = find_polytope_bbox(constraints)
    pivot = torch.tensor(make_pivot(constraints, bbox=bbox, rng=rng, n_candidates=n_candidates))

    A_tensor = torch.tensor(constraints.A, dtype=torch.double)
    b_tensor = torch.tensor(constraints.b, dtype=torch.double)
    net = base_network(pivot, A_tensor, b_tensor).double()
    if return_pivot:
        return net, pivot
    return net
