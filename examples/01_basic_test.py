from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch
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


def optimize_identity(nn: torch.nn.Module,
                      constraints: LinearConstraints,
                      n_epochs: int,
                      batch_size: int = 100,
                      lr: float = 1.e-2,
                      reduce_lr_factor: float = 0.9,
                      reduce_lr_patience: int = 20,
                      reduce_lr_cooldown: int = 20):
    """Utility function to optimize the neural network parameters
    to represent an identity function.
    """
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

    bbox = find_polytope_bbox(constraints)
    rng = np.random.RandomState(12345)

    history = []
    for epoch in range(n_epochs):
        gen_points = generate_in_bbox(bbox, batch_size, rng)
        mask = torch.tensor(nn.out_constrained_layer.constraint_set.check(gen_points), dtype=torch.bool)
        xs = torch.tensor(gen_points)
        predictions = nn(xs)
        loss = loss_fn(predictions[mask], xs[mask])

        optim.zero_grad()
        loss.backward()
        optim.step()
        history.append(loss.item())
        scheduler.step(loss)

    return history


@dataclass
class ExperimentConfig:
    """Experiment configuration.

    Feel free to change parameters.
    """
    only_layer: bool = False
    mode: Literal['ray_shift', 'center_projection'] = 'ray_shift'
    nn_hidden_dim: int = 100
    nn_n_layers: int = 5
    random_state: int = 12345


def basic_test(config: ExperimentConfig):
    """Basic usage example.
    """
    rng = np.random.RandomState(config.random_state)
    torch.random.manual_seed(config.random_state)
    input_dim = 2

    # make a constarint set
    linear_constraints_gen = LinearConstraintsGenerator(random_state=rng)
    quadratic_constraints_gen = QuadraticConstraintsGenerator(random_state=rng)
    constraints = [
        # generate random linear constraints
        linear_constraints_gen.generate(10, 2, n_attempts=5000, method='iterative_bounded'),
        # generate random quadratic constraints
        quadratic_constraints_gen.generate(2, 2),
        # make a circle constraint
        QuadraticConstraints(
            P=2.0 * np.eye(2, dtype=np.float64)[np.newaxis],
            q=np.zeros((1, 2), dtype=np.float64),
            b=0.75 * np.ones((1,), dtype=np.float64)
        )
    ]

    if config.only_layer:
        # use only one layer with constrained output
        # NOTE: it may not represent a pretty mapping if 'ray_shift' mode is used,
        #       since the 'ray_shift' latent space consists of concatenated
        #       shift parameter and ray.
        nn = ConstraiNetLayer(
            constraints=constraints,
            mode=config.mode,
        )
        constraint_set = nn.constraint_set
    else:
        # use a dense network with a constrained output layer at its tail
        nn = DenseConstraiNet(
            input_dim=input_dim,
            hidden_dim=config.nn_hidden_dim,
            n_layers=config.nn_n_layers,
            constraints=constraints,
            mode=config.mode,
        )
        # optimize neural network weights to represent an identity function
        optimize_identity(nn, constraints[0], n_epochs=2000)
        constraint_set = nn.out_constrained_layer.constraint_set

    # find bbox using only polytope (linear constraints)
    # NOTE: this may be changed to manual bbox construction,
    #       or e.g. stochastic approximation.
    bbox = find_polytope_bbox(constraints[0])
    # bbox = BBox(left=np.array([-1., -1.]), right=np.array([2., 3.]))

    # generate points on a uniform grid
    gen_points = np.stack(list(np.meshgrid(*[
        np.linspace(bbox.left[j], bbox.right[j], num=60)
        for j in range(len(bbox.left))
    ])), axis=-1).reshape((-1, 2))

    # map the generated input points to the domain
    with torch.no_grad():
        proj_points = nn(torch.tensor(gen_points, dtype=torch.double)).numpy()
    # check for constraints violations
    print(
        "Projected points constraints violations:",
        (~constraint_set.check(proj_points)).sum()
    )

    # plot the mapping
    sources = gen_points
    diff = proj_points - gen_points
    plt.quiver(sources[:, 0], sources[:, 1], diff[:, 0], diff[:, 1], np.linalg.norm(diff, axis=1),
                angles='xy', scale=1.0, units='xy',
                cmap='jet')
    plt.axis('equal')
    # highlight constraint violations (there should be no violations at all)
    mask = (~constraint_set.check(proj_points))
    plt.scatter(proj_points[mask][:, 0], proj_points[mask][:, 1], edgecolors='k')
    # plt.savefig('linear_identity_quiver.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    basic_test(ExperimentConfig(
        only_layer=False,
        mode='center_projection',
        random_state=12345,
    ))
