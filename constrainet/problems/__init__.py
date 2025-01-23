from .lin_quad import (
    LinearOptProblemGenerator,
    QuadraticOptProblemGenerator,
    QCLinearOptProblemGenerator,
    QCQuadraticOptProblemGenerator,
)
from .linear_constraints import LinearConstraintsGenerator
from .quadratic_constraints import QuadraticConstraintsGenerator
from .arbitrary import (
    LCRastriginOptProblemGenerator,
    LCRosenbrockOptProblemGenerator,
    QCRosenbrockOptProblemGenerator,
)


def make_constraints_generator(kind: str, random_state: int):
    kind = kind + 'Generator'
    constraints_gen_cls = globals().get(kind, None)
    if constraints_gen_cls is None:
        raise ValueError(f'No constraints_gen class found: {kind!r}.')
    return constraints_gen_cls(random_state=random_state)


def make_problem_generator(kind: str, random_state: int):
    problem_gen_cls = globals().get(kind, None)
    if problem_gen_cls is None:
        raise ValueError(f'No problem_gen class found: {kind!r}.')
    return problem_gen_cls(random_state=random_state)
