from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
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


DistanceType = Literal['l2', 'l1', 'linf']


def optimize_identity(nn: torch.nn.Module,
                      constraints: LinearConstraints,
                      n_epochs: int,
                      batch_size: int = 100,
                      lr: float = 1.e-2,
                      reduce_lr_factor: float = 0.9,
                      reduce_lr_patience: int = 20,
                      reduce_lr_cooldown: int = 20,
                      distance: DistanceType = 'l2'):
    if distance == 'l2':
        loss_fn = torch.nn.MSELoss()
    elif distance == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif distance == 'linf':
        loss_fn = lambda a, b: torch.mean(torch.max(torch.abs(a - b), dim=1)[0])
    else:
        raise ValueError(f'{distance=!r}')

    optim = torch.optim.AdamW(nn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        cooldown=reduce_lr_cooldown,
        mode='min',
        min_lr=1.e-5
    )

    bbox = find_polytope_bbox(constraints)
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


def get_n_constrain_violations(constraints: LinearConstraints, points: np.ndarray,
                               eps: float = 1.e-9,
                               return_correct_mask: bool = False) -> int:
    errs = (points @ constraints.A.T - constraints.b)
    status = np.all(errs <= eps, axis=1)
    n_violations = (~status).sum()
    if return_correct_mask:
        return n_violations, status
    return n_violations


@dataclass
class ExperimentConfig:
    """Experiment configuration.

    Feel free to change parameters.
    """
    mode: Literal['ray_shift', 'center_projection'] = 'center_projection'
    distance: DistanceType = 'l1'
    nn_hidden_dim: int = 100
    nn_n_layers: int = 5
    random_state: int = 12345


def test_gen_edge_identity(config: ExperimentConfig):
    rng = np.random.RandomState(config.random_state)
    torch.random.manual_seed(config.random_state)

    constraints_gen = LinearConstraintsGenerator(random_state=rng)
    constraints = constraints_gen.generate(10, 2, n_attempts=5000, method='iterative_bounded')

    # make a constrained neural network and train it to identity map
    nn = DenseConstraiNet(
        input_dim=2,
        hidden_dim=config.nn_hidden_dim,
        n_layers=config.nn_n_layers,
        constraints=constraints,
        mode=config.mode,
        onto_edge=True,
    )

    opt_params = dict(
        n_epochs=1000,
        batch_size=200,
        lr=1.e-3,
    )

    optim_history = optimize_identity(
        nn,
        constraints,
        distance=config.distance,
        **opt_params
    )
    plt.plot(optim_history)
    plt.show()

    bbox = find_polytope_bbox(constraints)
    gen_points = np.stack(list(np.meshgrid(*[
        np.linspace(bbox.left[j], bbox.right[j], num=40)
        for j in range(len(bbox.left))
    ])), axis=-1).reshape((-1, 2))

    gen_violations, gen_correct_mask = get_n_constrain_violations(
        constraints,
        gen_points,
        return_correct_mask=True
    )
    print("Uniform grid points constraints violations:", gen_violations)

    with torch.no_grad():
        proj_points = nn(torch.tensor(gen_points, dtype=torch.double)).numpy()
    print("Projected points constraints violations:", get_n_constrain_violations(constraints, proj_points))

    sources = gen_points
    diff = proj_points - gen_points
    plt.quiver(sources[:, 0], sources[:, 1], diff[:, 0], diff[:, 1], np.linalg.norm(diff, axis=1),
                angles='xy', scale=1.0, units='xy',
                cmap='jet')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    test_gen_edge_identity(ExperimentConfig(
        mode='center_projection',
        distance='l1',
        nn_hidden_dim=100,
        nn_n_layers=5,
    ))
