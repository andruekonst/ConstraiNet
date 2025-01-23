from .base import AbstractExperiment
from .solve_opt import *
from .csv_results_writer import CSVResultsWriter


def make_experiment(kind: str, params: dict) -> AbstractExperiment:
    experiment_cls = globals().get(kind, None)
    if experiment_cls is None:
        raise ValueError(f'No experiment class found: {kind!r}.')
    return experiment_cls(**params)
