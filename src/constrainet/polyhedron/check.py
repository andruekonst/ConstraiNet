import numpy as np
from scipy.optimize import nnls


def check_h_nonempty(A, b, tol=1.e-7):
    """Check that polyhedron in H-representation is not empty.

    Polygon is defines as a set of points x satisfying (Ax >= b).
    """
    return nnls(np.concatenate([A, -A, -np.eye(A.shape[0])], axis=1), b)[1] < tol
