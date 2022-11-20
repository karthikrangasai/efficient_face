# type: ignore
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.optim import Optimizer


class SAM(Optimizer):
    def __init__(self, optimizer: Optimizer, rho: float = 0.05, p: int = 2):
        assert isinstance(optimizer, Optimizer)
        self.base_optimizer: Optimizer = optimizer
        self.rho = rho
        self.p = p
        self.q = p / (p - 1)
        super().__init__(self.base_optimizer.param_groups, self.base_optimizer.defaults)

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
        grad_norm = torch.pow(self._grad_norm(), 1 / self.p)
        scaled_inverse_grad_norm = self.rho / (grad_norm + 1e-12)
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

    def step(self, closure: Callable[[], float]) -> None:
        assert closure is not None, "SAM Optimizer requires a closure which runs the training step."
        self.compute_and_add_e_w()
        self.zero_grad()
        with torch.enable_grad():
            closure()
        self.subtract_e_w_and_step_optimizer()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class ESAM(Optimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        beta: float = 0.5,
        selection_ratio: float = 0.5,
        rho: float = 0.05,
        p: int = 2,
    ):
        assert isinstance(optimizer, Optimizer)
        self.base_optimizer: Optimizer = optimizer
        self.rho = rho
        self.p = p
        self.q = p / (p - 1)
        self.beta = beta
        self.bernouli_sampler = Bernoulli(probs=self.beta)
        self.selection_ratio = selection_ratio
        super().__init__(self.base_optimizer.param_groups, self.base_optimizer.defaults)

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
    def stochastic_weight_perturbation_and_assign_perturbation(self) -> None:
        grad_norm = torch.pow(self._grad_norm(), 1 / self.p)
        scaled_inverse_grad_norm = self.rho / (grad_norm + 1e-12) / (1 - self.beta)
        for param_group in self.param_groups:
            param_group["e_w"] = []
            for param in param_group["params"]:
                if param.grad is None:
                    continue
                e_w = torch.tensor(0.0)
                if self.bernouli_sampler.sample():
                    e_w = scaled_inverse_grad_norm * param.grad.sign() * param.grad.pow(self.q - 1)
                param_group["e_w"].append(e_w)
                param.add_(e_w)

    @torch.no_grad()
    def sharpness_sensitive_data_selection(
        self,
        loss: torch.Tensor,
        new_loss: torch.Tensor,
        num_pairs: int,
        indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> List[int]:
        instance_sharpness: torch.Tensor = new_loss - loss
        position = int(num_pairs * self.selection_ratio)
        _, pair_indices = torch.topk(instance_sharpness, position)
        # cutoff = 0
        # select top k%
        actual_indices = set()
        for _indices in indices:
            actual_indices.update(_indices[pair_indices].tolist())

        return list(actual_indices)

    @torch.no_grad()
    def subtract_e_w_and_step_optimizer(self) -> None:
        for param_group in self.param_groups:
            e_ws = param_group.pop("e_w", [0.0] * len(param_group["params"]))
            for param, e_w in zip(param_group["params"], e_ws):
                if param.grad is None:
                    continue
                param.sub_(e_w)
        self.base_optimizer.step()

    def step(
        self,
        closure: Callable[
            [],
            Tuple[
                torch.Tensor,
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                int,
                Callable[[Optional[List[int]], bool], torch.Tensor],
            ],
        ],
    ) -> None:
        assert closure is not None, "ESAM Optimizer requires a closure which runs the training step."
        # First forward and backward passes have been completed when code reaches here.

        loss, indices, num_pairs, closure_fn = closure()

        self.stochastic_weight_perturbation_and_assign_perturbation()
        self.zero_grad()
        with torch.no_grad():
            new_loss: torch.Tensor = closure_fn(hard_pair_indices=indices)  # type: ignore

        indices = self.sharpness_sensitive_data_selection(loss, new_loss, num_pairs, indices)
        with torch.enable_grad():
            new_loss: torch.Tensor = closure_fn(indices, True)

        self.subtract_e_w_and_step_optimizer()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
