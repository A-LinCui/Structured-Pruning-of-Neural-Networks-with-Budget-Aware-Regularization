import torch
from torch import Tensor
import torch.nn as nn


class BARStructuredWrapper(nn.Module):
    r""" 
    Module wrapper for structured pruning.
    Implementation for "Structured Pruning of Neural Networks with Budget-Aware Regularization, `Link_`".

    Args:
        module (nn.Module): Base module.
        alpha (float): Default: 0.
        beta (float): Default: 0.667.
        gamma (float): Default: -0.1
        zeta (float): Default: 1.1.

    Examples::
        >>> module = nn.BatchNorm2d(3)
        >>> wrapper = BARStructuredWrapper(module)
        >>> data = torch.randn((2, 3, 3, 4))
        >>> output = wrapper(data)

    .. _Link:
        https://arxiv.org/abs/1811.09332
    """
    def __init__(self, module: nn.Module, alpha: float = 0., beta: float = 0.667, 
            gamma: float = -0.1, zeta: float = 1.1) -> None:
        super(BARStructuredWrapper, self).__init__()
        # the wrapped module
        self.module = module
        
        # pruning hyper-parameters
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.zeta = zeta
        
        # to be calculated in the first feed-forward
        self.log_alpha = None
        self.area = None
        
        self.stochastic = True
        self.to_initialize = True

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        output = self.module(input, *args, **kwargs)
        if self.to_initialize:
            # calculate ``area = H * W``
            self.area = output.size(2) * output.size(3)
            # initialize parameters for gates
            self.log_alpha = nn.Parameter(
                    torch.rand(output.size(1)) * 0.01 + self.alpha).to(input.device)
            self.to_initialize = False
        z = self.cal_mask(self.stochastic).to(input.device)
        output *= z[None, :, None, None]
        return output
            
    def cal_mask(self, stochastic: bool) -> Tensor:
        assert not self.to_initialize, "Please feed-forward for one-step before."
        nchannels = len(self.log_alpha)
        if stochastic:
            u = torch.rand(nchannels).requires_grad_(False)
            s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + self.log_alpha) / self.beta)
        else:
            s = torch.sigmoid(self.log_alpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, min = 0., max = 1.)
        return z

    @property
    def computation_overhead(self) -> float:
        r"""
        Get the computation overhead.

        Returns:
            area (float): The computation overhead is defined as the area in BAR.
        """
        return self.area
