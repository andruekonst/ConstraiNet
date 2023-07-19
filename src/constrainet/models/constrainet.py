import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, reduce
import torch
from torch.nn import Module
from typing import Callable, List, MutableMapping, Optional, Union, Literal
from ..problems.base import AbstractConstraints
from ..problems.linear_constraints import LinearEqualityConstraints, LinearConstraints
from ..problems.quadratic_constraints import QuadraticConstraints
from ..layers.fused.diameter import (
    CenterDiameterBasedNN,
    ProjectionCenterDiameterNN,
    find_diameter,
    make_point_on_segment,
)
from ..layers.fused.qc_diameter import (
    QCCenterDiameterBasedNN,
    QCProjectionCenterDiameterNN,
    qc_find_interior_point_quadprog,
)
from ..layers.fused.lc_qc_diameter import LQRayShiftNN, LQCenterProjectionNN, lc_qc_find_interior_point_quadprog




@dataclass
class ConstraintSet:
    linear_eq: Optional[LinearEqualityConstraints] = None
    linear: Optional[LinearConstraints] = None
    quadratic: Optional[QuadraticConstraints] = None

    def __post_init__(self):
        """Validate constraint set and update linear and quadratic constraints
        to operate in linear subspace in case when `linear_eq` is specified.
        """
        assert self.linear is not None or self.quadratic is not None or \
               self.linear_eq is not None
        # update linear and quadratic constraints
        if self.linear_eq is not None:
            R, p = self.linear_eq.null_space
            if self.linear is not None:
                self.linear = LinearConstraints(
                    self.linear.A @ R,
                    self.linear.b - self.linear.A @ p,
                )
            if self.quadratic is not None:
                self.quadratic = QuadraticConstraints(
                    P=np.einsum('ab,mbc,cd->mad', R.T, self.quadratic.P, R),  # R^T P_i R
                    q=(
                        np.einsum('n,mnk,kl->ml', p, self.quadratic.P, R) +
                        self.quadratic.q @ R
                    ),
                    b=(
                        self.quadratic.b -
                        0.5 * np.einsum('a,mab,b->m', p, self.quadratic.P, p) -
                        self.quadratic.q @ p
                    )
                )

    @property
    def n_features(self) -> int:
        if self.linear_eq is not None:
            return self.linear_eq.n_features
        elif self.linear is not None:
            return self.linear.n_features
        elif self.quadratic is not None:
            return self.quadratic.n_features
        else:
            raise NotImplementedError()

    def is_linear_eq(self):
        return self.linear_eq is not None and self.linear is None and self.quadratic is None

    def is_linear(self):
        return self.linear is not None and self.quadratic is None

    def is_quadratic(self):
        return self.quadratic is not None and self.linear is None

    @cached_property
    def interior_point(self) -> np.ndarray:
        if self.is_linear():
            # find approximate diameter (may be far from the real diameter)
            ray_id, (span, min_c, max_c, point) = find_diameter(self.linear)
            point_on_diameter = make_point_on_segment(
                point,
                -self.linear.A[ray_id],
                float(min_c),
                float(max_c)
            ),
            return point_on_diameter
        if self.is_quadratic():
            return qc_find_interior_point_quadprog(
                self.quadratic.P, self.quadratic.q, self.quadratic.b
            )
        if self.linear is not None and self.quadratic is not None:
            return lc_qc_find_interior_point_quadprog(
                self.linear.A, self.linear.b,
                self.quadratic.P, self.quadratic.q, self.quadratic.b
            )
        raise NotImplementedError()

    def check(self, points: np.ndarray, eps: float=1.e-6) -> np.ndarray:
        if self.linear_eq is not None:
            R, p = self.linear_eq.null_space
            xs = (points - p[np.newaxis]) @ R
        else:
            xs = points
        mask = True
        if self.linear is not None:
            mask = mask & self.linear.is_inside(xs, eps)
        if self.quadratic is not None:
            mask = mask & self.quadratic.is_inside(xs, eps)
        return mask


ConstraintsOrList = Union[AbstractConstraints, List[AbstractConstraints]]


def merge_constraints(constraints: List[AbstractConstraints]):
    """Reduce constraints with `append` operation.
    """
    iterator = iter(constraints)
    accum = next(iterator).copy()
    for following in iterator:
        accum = accum.append(following)
    return accum


def make_constraint_set(constraints: ConstraintsOrList):
    TYPE_ROUTE = {
        LinearEqualityConstraints: 'linear_eq',
        LinearConstraints: 'linear',
        QuadraticConstraints: 'quadratic',
    }
    if isinstance(constraints, AbstractConstraints):
        key = TYPE_ROUTE.get(type(constraints), None)
        if key is None:
            raise ValueError(f'Wrong constraints type: {type(constraints)!r}')
        kwargs = {key: constraints}
        return ConstraintSet(**kwargs)
    elif isinstance(constraints, List):
        set_by_key: MutableMapping[str, List[AbstractConstraints]] = defaultdict(list)
        for cstr in constraints:
            key = TYPE_ROUTE.get(type(cstr), None)
            if key is None:
                raise ValueError(f'Wrong constraints type: {type(cstr)!r}')
            set_by_key[key].append(cstr)
        return ConstraintSet(**{
            k: merge_constraints(constraints_set)
            for k, constraints_set in set_by_key.items()
        })
    else:
        raise ValueError(f'Wrong constraints type: {type(constraints)!r}')


class ConstraiNetLayer(Module):
    def __init__(self, constraints: Union[ConstraintsOrList, ConstraintSet],
                 mode: Literal['ray_shift', 'center_projection'] = 'ray_shift',
                 onto_edge: bool = False):
        """Neural network layer with constrained outputs.

        Args:
            constraints: List of constraints, one constraint or a constraint set.
            mode: Layer operation mode, default = 'ray_shift' which means that
                  a part of input is considered as a ray (direction) and another
                  part is a scaling parameter.
                  Another mode is 'center_projection', when the layer maps
                  inputs to the domain by central projection.
            onto_edge: Whether projection should be to the domain, or to its edge.
                       By default is False.

        """
        super().__init__()
        if isinstance(constraints, ConstraintSet):
            self.constraint_set = constraints
        else:
            self.constraint_set = make_constraint_set(constraints)
        self.n_features = self.constraint_set.n_features
        self.mode = mode
        self.onto_edge = onto_edge
        self.__init_implementation()

    def __init_implementation(self):
        if self.constraint_set.is_linear():
            MODE_ROUTE = {
                'ray_shift': CenterDiameterBasedNN,
                'center_projection': ProjectionCenterDiameterNN,
            }
            Cls = MODE_ROUTE.get(self.mode, None)
            if Cls is None:
                raise ValueError(f'Wrong mode: {self.mode!r}')
            impl = Cls(
                torch.tensor(self.constraint_set.linear.A, dtype=torch.double),
                torch.tensor(self.constraint_set.linear.b, dtype=torch.double),
                torch.tensor(self.constraint_set.interior_point, dtype=torch.double),
                onto_edge=self.onto_edge,
            )
        elif self.constraint_set.is_quadratic():
            MODE_ROUTE = {
                'ray_shift': QCCenterDiameterBasedNN,
                'center_projection': QCProjectionCenterDiameterNN,
            }
            Cls = MODE_ROUTE.get(self.mode, None)
            if Cls is None:
                raise ValueError(f'Wrong mode: {self.mode!r}')
            impl = Cls(
                torch.tensor(self.constraint_set.quadratic.P, dtype=torch.double),
                torch.tensor(self.constraint_set.quadratic.q, dtype=torch.double),
                torch.tensor(self.constraint_set.quadratic.b, dtype=torch.double),
                torch.tensor(self.constraint_set.interior_point, dtype=torch.double),
                onto_edge=self.onto_edge,
            )
        else:  # both type of constraints
            MODE_ROUTE = {
                'ray_shift': LQRayShiftNN,
                'center_projection': LQCenterProjectionNN,
            }
            Cls = MODE_ROUTE.get(self.mode, None)
            if Cls is None:
                raise ValueError(f'Wrong mode: {self.mode!r}')
            impl = Cls(
                torch.tensor(self.constraint_set.linear.A, dtype=torch.double),
                torch.tensor(self.constraint_set.linear.b, dtype=torch.double),
                torch.tensor(self.constraint_set.quadratic.P, dtype=torch.double),
                torch.tensor(self.constraint_set.quadratic.q, dtype=torch.double),
                torch.tensor(self.constraint_set.quadratic.b, dtype=torch.double),
                torch.tensor(self.constraint_set.interior_point, dtype=torch.double),
                onto_edge=self.onto_edge,
            )
        self.implementation = impl

        if self.constraint_set.linear_eq is not None:
            R, p = self.constraint_set.linear_eq.null_space
            self.backproject_R = torch.tensor(R, dtype=torch.double)
            self.backproject_p = torch.tensor(p, dtype=torch.double)
        else:
            self.backproject_R, self.backproject_p = None, None

    @cached_property
    def n_inputs(self) -> int:
        if self.mode == 'ray_shift':
            return self.n_features + 1
        elif self.mode == 'center_projection':
            return self.n_features
        else:
            raise NotImplementedError()

    def forward(self, zs):
        result = self.implementation(zs)
        if self.backproject_R is None:
            return result
        return torch.einsum('nl,bl->bn', self.backproject_R, result) + self.backproject_p.unsqueeze(0)


def make_feed_forward_network(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int,
                              make_activation=torch.nn.ReLU):
    layers = [torch.nn.Linear(input_dim, hidden_dim, dtype=torch.double)]
    for i in range((n_layers - 1) * 2):
        if i % 2 == 0:
            layers.append(make_activation())
        else:
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.double))
    layers.append(make_activation())
    layers.append(torch.nn.Linear(hidden_dim, output_dim, dtype=torch.double))
    return torch.nn.Sequential(*layers).double()


class DenseConstraiNet(Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 constraints: ConstraintsOrList,
                 mode: Literal['ray_shift', 'center_projection'] = 'ray_shift',
                 onto_edge: bool = False,
                 make_activation: Callable[[], Callable[[torch.Tensor], torch.Tensor]] = torch.nn.ReLU):
        super().__init__()
        self.out_constrained_layer = ConstraiNetLayer(constraints, mode, onto_edge=onto_edge)
        self.dense = make_feed_forward_network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim = self.out_constrained_layer.n_inputs,
            n_layers=n_layers,
            make_activation=make_activation,
        )

    def forward(self, input_features):
        return self.out_constrained_layer(self.dense(input_features))
