import numpy as np
import torch
from typing import Callable, List, Literal, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class AbstractConstraints(ABC):
    @property
    @abstractmethod
    def n_features(self) -> int:
        ...

    @property
    @abstractmethod
    def n_constraints(self) -> int:
        ...

    @abstractmethod
    def copy(self) -> 'AbstractConstraints':
        ...

    @abstractmethod
    def append(self, other: 'AbstractConstraints') -> 'AbstractConstraints':
        ...

    @abstractmethod
    def is_inside(self, xs: np.ndarray, eps: float = 1.e-9) -> np.ndarray:
        ...


class AbstractOptProblem(ABC):
    has_exact_solution: bool = False

    @property
    @abstractmethod
    def constraints(self) -> AbstractConstraints:
        ...

    @property
    @abstractmethod
    def loss(self) -> Callable[[torch.Tensor], torch.Tensor]:
        ...

    @property
    @abstractmethod
    def exact_solution(self) -> Optional[np.ndarray]:
        ...

    @abstractmethod
    def check_solution(self, solution: torch.Tensor):
        ...


class OptProblemGenerator(ABC):
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    @abstractmethod
    def generate(self, constraints: AbstractConstraints) -> AbstractOptProblem:
        ...

    @abstractmethod
    def load(self, path: Path) -> AbstractOptProblem:
        ...

    @abstractmethod
    def save(self, path: Path, problem: AbstractOptProblem):
        ...


class AbstractConstraintsGenerator(ABC):
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    @abstractmethod
    def generate(self, n_inequalities: int, n_features: int, n_attempts: int = 1000,
                 method: Literal['bruteforce', 'iterative_bounded'] = 'bruteforce') -> AbstractConstraints:
        ...
