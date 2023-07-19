from typing import Optional
import numpy as np
from ..problems.base import AbstractOptProblem
from ..problems.linear_constraints import LinearConstraintsOptProblem
from abc import ABC, abstractmethod


class AbstractOptSolver(ABC):
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    @abstractmethod
    def solve(self, problem: AbstractOptProblem) -> np.ndarray:
        ...


class LinearConstraintsOptSolver(AbstractOptSolver):
    @abstractmethod
    def solve(self, problem: LinearConstraintsOptProblem) -> np.ndarray:
        ...
