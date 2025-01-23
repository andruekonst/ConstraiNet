from dataclasses import dataclass
from pathlib import Path
from typing import Union, List
import logging
from .experiments import make_experiment, CSVResultsWriter
from .experiments.params_iterator import to_list_or_expanded_range


@dataclass
class RunConfig:
    name: str
    experiment: str
    seeds: Union[List[int], str]
    params: dict
    description: str = ''

    @property
    def seeds_list(self):
        return to_list_or_expanded_range(self.seeds)


class RunExecutor:
    def __init__(self, config: RunConfig, path: Path):
        self.config = config
        self.path = path

    def run(self):
        experiment = make_experiment(self.config.experiment, self.config.params)
        results_path = self.path / 'results.csv'
        with CSVResultsWriter(results_path, enable_temporary=True) as results_writer:
            for seed in self.config.seeds_list:
                logging.info(f'Random seed = {seed}')
                experiment.run(seed, results_writer)
        logging.info('Done')
