"""Example of quadratically constrained minimization of Bird function (2d).
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
from constrainet import (
    LinearConstraints,
    LinearEqualityConstraints,
    QuadraticConstraints,
    ConstraiNetLayer,
    DenseConstraiNet
)
# utilities for problem and point generation
from constrainet.problems.linear_constraints import LinearConstraintsGenerator
from constrainet.problems.quadratic_constraints import QuadraticConstraintsGenerator
from constrainet.layers.pivot_utils import BBox, find_polytope_bbox, generate_in_bbox
from constrainet.problems.quadratic_constraints import quad_forms


def optimize_identity(nn: torch.nn.Module,
                      constraints: LinearConstraints,
                      n_epochs: int,
                      batch_size: int = 100,
                      lr: float = 1.e-2,
                      reduce_lr_factor: float = 0.9,
                      reduce_lr_patience: int = 20,
                      reduce_lr_cooldown: int = 20,
                      approx_bbox_scale: float = 5.0):
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.AdamW(nn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        cooldown=reduce_lr_cooldown,
        mode='min',
        min_lr=1.e-5
    )

    bbox = find_approximate_bbox(constraints, scale=approx_bbox_scale)
    print("Bbox:", bbox)
    rng = np.random.RandomState(12345)

    history = []
    for epoch in range(n_epochs):
        gen_points = generate_in_bbox(bbox, batch_size, rng)
        xs = torch.tensor(gen_points)
        predictions = nn(xs)
        loss = loss_fn(predictions, xs)

        optim.zero_grad()
        loss.backward()
        optim.step()
        history.append(loss.item())
        scheduler.step(loss)

    return history


def optimize_loss(loss_fn,
                  inputs,
                  params,
                  nn: torch.nn.Module,
                  constraints: LinearConstraints,
                  n_epochs: int,
                  lr: float = 1.e-2,
                  reduce_lr_factor: float = 0.9,
                  reduce_lr_patience: int = 20,
                  reduce_lr_cooldown: int = 20,
                  iter_callback = (lambda t: None)):
    optim = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        cooldown=reduce_lr_cooldown,
        mode='min',
        min_lr=1.e-5
    )

    history = []
    for epoch in range(n_epochs):
        xs = inputs
        predictions = nn(xs)
        loss = loss_fn(predictions)

        optim.zero_grad()
        loss.backward()
        optim.step()
        history.append(loss.item())
        scheduler.step(loss)
        iter_callback(predictions)

    return history


def get_n_constrain_violations(constraints: QuadraticConstraints, points: np.ndarray,
                               eps: float = 1.e-9,
                               return_correct_mask: bool = False) -> int:
    errs = (quad_forms(constraints.P, constraints.q, points) - constraints.b[np.newaxis])
    status = np.all(errs <= eps, axis=1)
    n_violations = (~status).sum()
    if return_correct_mask:
        return n_violations, status
    return n_violations


def find_approximate_bbox(constraints: QuadraticConstraints, n_probes: int = 100000,
                         scale: float = 5.0,
                         random_state: int = 12345) -> BBox:
    rng = np.random.RandomState(random_state)
    points = rng.normal(loc=0.0, scale=scale, size=(n_probes, constraints.n_features))
    n_violations, correct_mask = get_n_constrain_violations(constraints, points, return_correct_mask=True)
    points = points[correct_mask]
    return BBox(left=points.min(axis=0), right=points.max(axis=0))


Setting = Literal['train_identity', 'projection', 'optimize_in_latent_space']


@dataclass
class ExperimentConfig:
    """Experiment configuration.

    Feel free to change parameters.
    """
    setting: Setting = 'projection'
    random_state: int = 12345


def test_minimize_bird_2d_qc(config: ExperimentConfig):
    torch.random.manual_seed(config.random_state)
    rng = np.random.RandomState(config.random_state)

    constraints = QuadraticConstraints(
        P=2.0 * np.eye(2, dtype=np.float64)[np.newaxis],
        q=10.0 * np.ones((1, 2), dtype=np.float64),
        b=-25.0 * np.ones((1,), dtype=np.float64)
    )

    n_features = constraints.n_features
    if config.setting == 'train_identity':
        nn = DenseConstraiNet(
            input_dim=n_features,
            hidden_dim=100,
            n_layers=5,
            constraints=constraints,
            mode='ray_shift'
        )

        optim_history = optimize_identity(
            nn,
            constraints,
            n_epochs=2000,
            batch_size=200,
            lr=1.e-3,
            approx_bbox_scale=10.0,
        )
        plt.plot(optim_history)
        plt.show()
    elif config.setting == 'projection':
        nn = ConstraiNetLayer(constraints, mode='center_projection')
    elif config.setting == 'optimize_in_latent_space':
        nn = ConstraiNetLayer(constraints, mode='ray_shift')
    else:
        raise ValueError(f'{config.setting=!r}')

    a = 3.0
    zs = torch.tensor([
        [-a, -a],
        [0, -a],
        [a, -a],
        [-a, 0],
        [1.e-5, 1.e-5],
        [a, 0],
        [-a, a],
        [0, a],
        [a, a],
    ], dtype=torch.double)
    zs += torch.tensor([[-5.0, -5.0]], dtype=torch.double)
    zs = zs.requires_grad_()

    if config.setting == 'optimize_in_latent_space':
        # initialize points in latent space
        prev_zs = zs.detach()
        zs = torch.ones((prev_zs.shape[0], prev_zs.shape[1] + 1), dtype=torch.double)
        zs[:, 1:] = prev_zs / torch.norm(prev_zs)
        zs[:, 0] = torch.norm(prev_zs) / np.sqrt(2)
        zs = zs.requires_grad_()
        tmp_opt = torch.optim.Adam([zs], 1.e-2)
        for _epoch in range(5000):
            tmp_opt.zero_grad()
            torch.square(nn(zs) - prev_zs).sum().backward()
            tmp_opt.step()
        optim_params = dict(
            n_epochs=5000,
            lr=1.e-1,
            reduce_lr_patience=100,
            reduce_lr_factor=0.9,
        )
    else:
        optim_params = dict(
            n_epochs=2000,
            lr=1.e-1,
            reduce_lr_patience=100,
            reduce_lr_factor=0.5,
        )

    partial_loss_fn = lambda xs: (
        torch.sin(xs[:, 1]) * torch.exp((1.0 - torch.cos(xs[:, 0])) ** 2) +
        torch.cos(xs[:, 0]) * torch.exp((1.0 - torch.sin(xs[:, 1])) ** 2) +
        (xs[:, 0] - xs[:, 1]) ** 2
    )
    loss_fn = lambda xs: partial_loss_fn(xs).sum()

    tracks = []
    optim_history = optimize_loss(
        loss_fn,
        zs,
        [zs],
        nn,
        constraints,
        iter_callback=lambda preds: tracks.append(preds.detach().numpy()),
        **optim_params
    )

    tracks = np.stack(tracks)
    # loss function contours
    grid = np.stack(list(np.meshgrid(*[
        np.linspace(-10.0, 0.0, 80)
        for j in range(2)
    ]
    )), axis=-1)
    orig_grid_shape = grid.shape[:-1]
    flat_grid = grid.reshape((-1, grid.shape[-1]))
    grid_loss = partial_loss_fn(torch.tensor(flat_grid, dtype=torch.double)).numpy()
    plt.contourf(
        grid[..., 0], grid[..., 1], grid_loss.reshape(orig_grid_shape),
        cmap='plasma',
        alpha=0.85
    )
    plt.colorbar()

    with torch.no_grad():
        solutions = nn(zs)
        comp_losses = partial_loss_fn(solutions).cpu().numpy()
        min_loss_idx = np.argmin(comp_losses)
        print("Best loss: ", comp_losses[min_loss_idx])
        print("Best solution: ", solutions[min_loss_idx])

    top_k = 9
    loss_sort_ids = np.argsort(comp_losses)
    good_track_ids = loss_sort_ids[:top_k]

    colors = pl.cm.jet(np.linspace(0, 1, len(good_track_ids)))
    colors_iter = iter(colors)

    for j in range(tracks.shape[1]):
        if j in good_track_ids:
            plt.scatter(
                tracks[0, j, 0], tracks[0, j, 1], marker='o',
                edgecolor='w',
                label=f'{j}', alpha=1.0, zorder=3, c='k',
            )
            plt.plot(
                tracks[:, j, 0],
                tracks[:, j, 1],
                label=f'{j}',
                alpha=1.0,
                color=next(colors_iter)
            )
            plt.scatter(
                tracks[-1, j, 0], tracks[-1, j, 1], marker='*',
                edgecolor='w',
                label=f'{j}', alpha=1.0, zorder=3
            )
    plt.axis('equal')
    ax = plt.gca()
    ax.add_patch(plt.Circle((-5, -5), 5, color='k', fill=False))
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(optim_history)
    plt.tight_layout()


if __name__ == '__main__':
    test_minimize_bird_2d_qc(ExperimentConfig(
        setting='projection',
        # setting='train_identity',
        # setting='optimize_in_latent_space',
    ))
