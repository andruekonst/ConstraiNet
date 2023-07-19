from abc import ABC, abstractmethod
from typing import Generator, List, Tuple
from ...layers.base import AbstractOptSolver
from ...problems.base import AbstractOptProblem


class AbstractResultsWriter(ABC):
    @abstractmethod
    def open(self):
        ...

    @abstractmethod
    def close(self):
        ...

    @abstractmethod
    def append(self, params: dict):
        ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AbstractExperiment(ABC):
    def __init__(self, problems: List[dict], solvers: List[dict], timeout: float):
        self.problems = problems
        self.solvers = solvers
        self.timeout = timeout

    @abstractmethod
    def gen_problems(self, random_state: int) -> Generator[Tuple[dict, AbstractOptProblem], None, None]:
        ...

    @abstractmethod
    def gen_solvers(self, random_state: int) -> Generator[Tuple[dict, AbstractOptSolver], None, None]:
        ...

    @abstractmethod
    def run(self, random_state: int, results_writer: AbstractResultsWriter):
        ...

