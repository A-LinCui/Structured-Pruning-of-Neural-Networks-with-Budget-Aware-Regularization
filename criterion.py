import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from extorch.nn.utils import net_device

from model.wrapper import BARStructuredWrapper


class DistillationLoss(nn.Module):
    r"""
    Distillation objective.

    Args:
        T (float): The temperature.
        alpha (float): The coefficient to controll the trade-off between distillation loss and origin loss.
    """
    def __init__(self, T: float, alpha: float) -> None:
        self.T = T
        self.alpha = alpha

    def forward(self, output: Tensor, target: Tensor, label: Tensor) -> Tensor:
        r"""
        Args:
            output (Tensor): Output of the network to be trained.
            target (Tensor): Output of the teacher network.
            label (Tensor): Label of the input.

        Returns:
            Tensor: The calculated loss.
        """
        p = F.softmax(target / self.T, dim = 1)
        log_q = F.log_softmax(output / self.T, dim = 1)
        entropy = - torch.sum(p * log_q, dim = 1)
        kl = F.kl_div(log_q, p, reduction = "mean")
        loss = torch.mean(entropy + kl)
        return self.alpha * self.T ** 2 * loss + \
                F.cross_entropy(output, label) * (1 - self.alpha)


class BudgetLoss(nn.Module):
    r"""
    Budget loss for BAR.
    """
    def __init__(self) -> Tensor:
        super(BudgetLoss, self).__init__()

    def forward(self, net: nn.Module) -> Tensor:
        r"""
        Calculate the budget loss.

        Args:
            net (nn.Module): The network to be pruned.

        Returns:
            loss (Tensor): The budget loss.
        """
        loss = torch.zeros(1).to(net_device(net))
        for name, m in net.named_modules():
            if isinstance(m, BARStructuredWrapper):
                # Probability of being alive for each feature map
                alive_probability = torch.sigmoid(
                        m.log_alpha - m.beta * torch.log(-m.gamma / m.zeta))
                loss += torch.sum(alive_probability) * m.computation_overhead
        return loss


class BARStructuredLoss(nn.Module):
    r"""
    Objective of Budget-Aware Regularization Structured Pruning.

    Args:
        budget (float):
        progress_func (str): Type of progress function ("sigmoid" or "exp"). Default: "sigmoid".
        _lambda (float): Coefficient for trade-off of sparsity loss term. Default: 1e-5.
        distillation_temperature (float): Knowledge Distillation temperature. Default: 4.
        distillation_alpha (float): Knowledge Distillation alpha. Default: 0.9.
        tolerance (float):
        margin (float): Parameter a in Eq. 5 of the paper. Default: 0.0001.
        sigmoid_a (float): Slope parameter of sigmoidal progress function. Default: 10.
        finetune_epoch (int): Number of epochs with pruning parameters fixed. Default: 40.
    """
    def __init__(self, budget: float, progress_func: str = "sigmoid", _lambda: float = 1e-5, 
            distillation_temperature: float = 4., distillation_alpha: float = 0.9,
            tolerance: float = 0.01, margin: float = 0.0001, sigmoid_a: float = 10.,
            finetune_epoch: int = 40) -> None:
        super(BARStructuredLoss, self).__init__()
        self.budget = budget

        self.progress_func = progress_func
        self.sigmoid_a = sigmoid_a
        self.tolerance = tolerance
        
        self._lambda = _lambda
        self.margin = margin

        self.classification_criterion = DistillationLoss(distillation_temperature, distillation_alpha)
        self.budget_criterion = BudgetLoss()

    def forward(self, input: Tensor, output: Tensor, target: Tensor, 
            net: nn.Module, teacher: nn.Module, current_epoch_fraction: float) -> Tensor:
        r"""
        Calculate the objective.

        Args:
            input (Tensor): Input image.
            output (Tensor): Output logit.
            target (Tensor): Label of the image,
            net (nn.Module): The network to be updated.
            teacher (nn.Module): Teacher network for distillation.
            current_epoch_fraction (float): Current epoch fraction.

        Returns:
            loss (Tensor): The loss.
        """
        # Step 1: Calculate the cross-entropy loss and distillation loss.
        classification_loss = self.classification_criterion(output, teacher(input), target)

        # Step 2: Calculate the budget loss.
        budget_loss = self.budget_criterion(net)

        # Step 3: Calculate the coefficient of the budget loss.
        current_overhead = self.current_overhead(net)
        origin_overhead = self.origin_overhead(net)
        tolerant_overhead = (1. + self.tolerance) * origin_overhead
        
        p = current_epoch_fraction / self.fine_epoch
        if progress_func == "sigmoid":
            p = self.sigmoid_progress_fn(p, self.sigmoid_a)
        elif progress_func == "exp":
            p = self.exp_progress_fn(p)
        current_budget = (1 - p) * tolerant_overhead + p * self.budget * origin_overhead

        margin = tolerant_overhead * self.margin
        lower_bound = self.budget - self.margin
        budget_respect = (current_overhead - lower_bound) / (current_budget - lower_bound)
        budget_respect = max(budget_respect, 0.)
        upper_bound = 1e10

        lamb_mult = budget_respect ** 2 / (1.0 - budget_respect) \
                if budget_respect < 1. else upper_bound

        # Step 4: Combine the objectives.
        loss = classification_loss + self._lambda / len(input) * lamb_mult * budget_loss
        return loss

    def current_overhead(self, net: nn.Module) -> float:
        r"""
        Calculate the computation overhead after pruning.

        Args:
            net (nn.Module): The network to be calculated.

        Returns:
            overhead (float): The computation overhead after pruning.
        """
        overhead = 0
        for name, m in net.named_modules():
            if isinstance(m, BARStructuredWrapper):
                z = m.cal_mask(stochastic = False)
                overhead += m.computation_overhead * (z > 0.).long().sum().item()
        return overhead

    def origin_overhead(self, net: nn.Module) -> float:
        r"""
        Calculate the origin computation overhead before pruning.

        Args:
            net (nn.Module): The network to be calculated.

        Returns:
            overhead (float): The origin computation overhead before pruning.
        """
        overhead = 0
        for name, m in net.named_modules():
            if isinstance(m, BARStructuredWrapper):
                nchannels = len(m.log_alpha)
                overhead += m.computation_overhead * nchannels
        return overhead

    def sparsity_ratio(self, net: nn.Module) -> float:
        r"""
        Calculate the spartial ratio.
        
        Args:
            net (nn.Module): The network to be calculated.

        Returns:
            sparsity_ratio (float): The spartial ratio.
        """
        current_overhead = self.current_overhead(net)
        origin_overhead = self.origin_overhead(net)
        sparsity_ratio = current_overhead / origin_overhead
        return sparsity_ratio

    @classmethod
    def exp_progress_fn(p: float, a: float = 4.) -> float:
        c = 1. - np.exp(-a)
        exp_progress = 1. - np.exp(-a * p)
        return exp_progress / c

    @classmethod
    def sigmoid_progress_fn(p: float, a: float) -> float:
        b = 1. / (1. + np.exp(a * 0.5))
        sigmoid_progress = 1. / (1. + np.exp(a * (0.5 - p)))
        sigmoid_progress = (sigmoid_progress - b) / (1. - 2. * b)
        return sigmoid_progress
