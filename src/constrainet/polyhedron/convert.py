import numpy as np
import warnings
try:
    import cdd
except:
    warnings.warn("Cannot import cdd")


def v_to_h_representation(vertices, rays):
    """Converts Vertex representation to Halfspace representation.
    """
    if len(rays) == 0:
        v = vertices
    else:
        v = np.concatenate((vertices, rays), axis=0)
    t = np.ones_like(v[:, 0])
    t[len(vertices):] = 0
    v_representation = np.concatenate((t[:, np.newaxis], v), axis=1)
    v_mat = cdd.Matrix(v_representation.tolist(), linear=False)
    v_mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(v_mat)
    h_representation = np.array(poly.get_inequalities()[:])
    new_b = -h_representation[:, 0]
    new_A = h_representation[:, 1:]
    return new_A, new_b


def h_to_v_representation(A, b):
    """Converts Halfspace representation to Vertex representation.
    """
    # A' x <= b' then H = [b' | -A']
    # A' = -A, b' = -b
    h_representation = np.concatenate((-b[:, np.newaxis], A), axis=1)
    h_mat = cdd.Matrix(h_representation.tolist(), linear=False)
    h_mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(h_mat)
    v_representation = np.array(poly.get_generators()[:])
    # [t | V]
    t = v_representation[:, 0]
    V = v_representation[:, 1:]
    return V[t == 1], V[t == 0]
