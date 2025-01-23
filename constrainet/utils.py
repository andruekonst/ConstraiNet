import numpy as np
import warnings
from sklearn.utils import check_random_state
from scipy.optimize import nnls
import subprocess


def get_rng(self) -> np.random.RandomState:
    if isinstance(self.random_state, int):
        warnings.warn(
            f'Int random state is passed to {self!r}. '
            'Probably generator was meant to be passed.'
        )
    return check_random_state(self.random_state)

def check_system(A, b, tol=1.e-7):
    """Check if `A x <= b` has at least one feasible point.
    """
    return nnls(np.concatenate([A, -A, np.eye(A.shape[0])], axis=1), b)[1] < tol


def get_method_argnames(method):
    return method.__code__.co_varnames[:method.__code__.co_argcount]


def prepare_arguments(method, params):
    method_argnames = get_method_argnames(method)
    args = {
        argname: params[argname]
        for argname in method_argnames
        if argname in params
    }
    diff = set(params.keys()) - set(args.keys())
    if len(diff) > 0:
        warnings.warn(f'Some parameters cannot be used: {diff}')
    return args


def get_git_revision_hash():
    full_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    full_hash = str(full_hash, "utf-8").strip()
    return full_hash
