import torch
from torch.nn import Module


class MultiplyLayer(Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * self.alpha


class ShiftLayer(Module):
    def __init__(self, shift: float):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return x + self.shift


class TempSigmoid(Module):
    def __init__(self, inv_temp: float):
        super().__init__()
        self.inv_temp = inv_temp

    def forward(self, x):
        return torch.sigmoid(x * self.inv_temp)
