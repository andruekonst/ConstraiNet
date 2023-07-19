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
from constrainet.layers.pivot_utils import BBox, find_polytope_bbox, generate_in_bbox


@dataclass
class ExperimentConfig:
    """Experiment configuration.

    Feel free to change parameters.
    """
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
    circle_radius = 0.75
    constraints = [
        # a triangle
        LinearConstraints(
            A=np.array([[-1., 0.], [0., -1.], [1., 1.]]),
            b=np.array([0., 0., 1.]),
        ),
        # a circle
        QuadraticConstraints(
            P=2.0 * np.eye(2, dtype=np.float64)[np.newaxis],
            q=np.zeros((1, 2), dtype=np.float64),
            b=circle_radius ** 2 * np.ones((1,), dtype=np.float64)
        )
    ]

    # use only one layer with constrained output
    # NOTE: it may not represent a pretty mapping if 'ray_shift' mode is used,
    #       since the 'ray_shift' latent space consists of concatenated
    #       shift parameter and ray.
    nn = ConstraiNetLayer(
        constraints=constraints,
        mode=config.mode,
    )
    constraint_set = nn.constraint_set

    # find bbox using only polytope (linear constraints)
    # NOTE: this may be changed to manual bbox construction,
    #       or e.g. stochastic approximation.

    # bbox = find_polytope_bbox(constraints[0])
    bbox = BBox(left=np.array([-0.1, -0.1]), right=np.array([1., 1.]))

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
    # plot constraints
    ax = plt.gca()
    ax.add_patch(plt.Circle((0, 0), circle_radius, color='k', fill=False))
    plt.plot([0, 1], [1, 0], c='k')
    plt.plot([0, 1], [0, 0], c='k')
    plt.plot([0, 0], [0, 1], c='k')

    ax.set_xlim([bbox.left[0], bbox.right[0]])
    ax.set_ylim([bbox.left[1], bbox.right[1]])
    ax.set_aspect('equal')

    plt.title('Linear and Quadratic Constraints')

    # highlight constraint violations (there should be no violations at all)
    mask = (~constraint_set.check(proj_points))
    plt.scatter(proj_points[mask][:, 0], proj_points[mask][:, 1], edgecolors='k')
    plt.savefig('lin_quad_constraints.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    basic_test(ExperimentConfig(
        mode='center_projection',
        random_state=12345,
    ))
