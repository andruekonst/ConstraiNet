from typing import Callable, Optional
import numpy as np
import torch
from torch.nn import Module

from ..polyhedron.check import check_h_nonempty
from ..polyhedron.convert import h_to_v_representation


def make_small_encoder(input_dim: int, output_dim: int, n_hidden: int = 16):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, n_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden, output_dim),
    )


class VertexReprNet(Module):
    def __init__(self, A, b, input_dim: int,
                 make_encoder_fn: Optional[Callable[[int, int], Module]] = None,
                 inv_temp: float = 1.0,
                 map_to_nonneg_mode: str = 'abs'):
        super().__init__()
        assert check_h_nonempty(A, b, tol=1.e-7)
        vertices, rays = h_to_v_representation(A, b)
        self.n_vertices = len(vertices)
        self.n_rays = len(rays)
        self.n_gen = self.n_vertices + self.n_rays
        self.vertices = torch.tensor(vertices.astype(np.float64))
        self.rays = torch.tensor(rays.astype(np.float64)) if self.n_rays > 0 else None
        if make_encoder_fn is None:
            make_encoder_fn = make_small_encoder
        self.encoder = make_encoder_fn(input_dim, self.n_gen)
        self.inv_temp = inv_temp

        MAP_TO_NONNEG_FNS = {
            'abs': torch.abs,
            'exp': torch.exp,
        }
        if map_to_nonneg_mode in MAP_TO_NONNEG_FNS:
            self.map_to_nonneg_fn = MAP_TO_NONNEG_FNS[map_to_nonneg_mode]
        else:
            raise ValueError(f'Invalid {map_to_nonneg_mode=!r}')

    def forward(self, x):
        gen = self.encoder(x)
        gen_vert = gen[:, :self.n_vertices]
        gen_vert = torch.softmax(gen_vert * self.inv_temp, dim=1)
        res = gen_vert @ self.vertices
        if self.n_rays > 0:
            gen_ray = gen[:, self.n_vertices:]
            gen_ray = self.map_to_nonneg_fn(gen_ray)
            res = res + gen_ray @ self.rays
        return res
