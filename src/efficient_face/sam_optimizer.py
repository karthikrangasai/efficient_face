from typing import Callable

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    def __init__(self, optimizer: Optimizer, rho: float = 0.05, p: int = 2):
        assert isinstance(optimizer, Optimizer)
        self.base_optimizer: Optimizer = optimizer
        self.rho = rho
        self.p = p
        self.q = p / (p - 1)
        super().__init__(self.base_optimizer.param_groups, {})

    def _grad_norm(self) -> torch.Tensor:
        return torch.linalg.norm(
            torch.stack(
                [
                    param.grad.norm(p=self.q)
                    for param_group in self.param_groups
                    for param in param_group["params"]
                    if param.grad is not None
                ]
            ),
            ord=self.q,
        )

    @torch.no_grad()
    def compute_and_add_e_w(self) -> None:
        grad_norm = self._grad_norm()
        scaled_inverse_grad_norm = self.rho / (grad_norm.pow(1 / self.p) + 1e-12)
        for param_group in self.param_groups:
            param_group["e_w"] = []
            for param in param_group["params"]:
                if param.grad is None:
                    continue
                e_w = scaled_inverse_grad_norm * param.grad.sign() * param.grad.pow(self.q - 1)
                param_group["e_w"].append(e_w)
                param.add_(e_w)

    @torch.no_grad()
    def subtract_e_w_and_step_optimizer(self) -> None:
        for param_group in self.param_groups:
            e_ws = param_group.pop("e_w", [0.0] * len(param_group["params"]))
            for param, e_w in zip(param_group["params"], e_ws):
                if param.grad is None:
                    continue
                param.sub_(e_w)
        self.base_optimizer.step()

    def step(self, closure: Callable = None) -> None:
        assert closure is not None, "SAM Optimizer requires a closure which runs the training step."
        self.compute_and_add_e_w()
        self.zero_grad()
        with torch.enable_grad():
            closure()
        self.subtract_e_w_and_step_optimizer()
