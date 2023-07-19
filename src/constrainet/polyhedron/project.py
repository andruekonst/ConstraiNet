"""Polyhedron projection onto plane.
"""
import numpy as np
import warnings
try:
    from pypoman.projection import project_polyhedron
except:
    warnings.warn("Cannot import pypoman")
from .convert import v_to_h_representation


def project_onto_hyperplane(A, b, q):
    """Project an unequality system onto a plane.

    Find left and right hand side of unequalities set:

        N x >= r,
        that is equivalent to:
        x in D <=> Exists alpha >= 0: A x >= b - alpha q.

    Args:
        A: Left hand side matrix.
        b: Right hand side vector.
        q: Vector at free non-negative coefficient.

    Returns:
        A tuple for projected system.

    """
    # append q, make a new variable `alpha == x[-1]`:
    extended_A = np.concatenate((A, q[:, np.newaxis]), axis=1)
    # append a new row to hold the `alpha >= 0` constraint:
    one_at_end = np.zeros_like(extended_A[0])
    one_at_end[-1] = 1
    extended_A = np.concatenate((extended_A, one_at_end[np.newaxis]), axis=0)
    extended_b = np.append(b, [0])

    # H-representation for A x <= b is [b | -A],
    # ... our system is A_extended x >= b, so the H-representation is [-b | A_extended]
#     h_representation = np.concatenate((-b[:, np.newaxis], A_extended), axis=1)
#     h_mat = cdd.Matrix(h_representation.tolist(), linear=False)
#     polyhedron = cdd.Polyhedron(h_mat)
#     v_representation = np.array(Polyhedron.get_generators()[:])
    x_dim = A.shape[1]
    E = np.eye(x_dim, x_dim + 1)
    vertices, rays = project_polyhedron(
        (E, np.zeros_like(A[0])),
        (-extended_A, -extended_b),
        eq=None,
        canonicalize=False
    )
    # TODO: reimplement `project_polyhedron` to return system
    return v_to_h_representation(np.array(vertices), np.array(rays))
