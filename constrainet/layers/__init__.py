from .base import *
from .nn import *
from .softmax_ray import *
from .fused import *
from .cvxpy import *
from .scipy import *
from .cvxpy_proj import *
from ..utils import prepare_arguments


def make_solver(kind: str, random_state: int, params: dict):
    problem_gen_cls = globals().get(kind, None)
    if problem_gen_cls is None:
        raise ValueError(f'No solver class found: {kind!r}.')
    args = prepare_arguments(problem_gen_cls.__init__, params)
    return problem_gen_cls(random_state=random_state, **args)
