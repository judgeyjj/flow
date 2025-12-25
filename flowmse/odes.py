"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import warnings
import math
import scipy.special as sc
import numpy as np
from flowmse.util.tensors import batch_broadcast
import torch

from flowmse.util.registry import Registry


ODERegistry = Registry("ODE")
class ODE(abc.ABC):
    """ODE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self):        
        super().__init__()
        

    
    @abc.abstractmethod
    def ode(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass


    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass


    @abc.abstractmethod
    def copy(self):
        pass



######################여기 밑에 것이 학습할 대상임##############


@ODERegistry.register("flowmatching")
class FLOWMATCHING(ODE):
    """
    Independent CFM for Audio Super-Resolution (adapted from FLowHigh).
    
    Key changes from standard flow matching:
    - Path: t=0 starts at y (LR), t=1 ends at x (HR)
    - Vector field: directly predicts x0 - y (HR increment)
    - Noise: constant sigma_min (optional small noise for regularization)
    
    Original flow matching: mu_t = (1-t)*x + t*y
    Independent CFM: mu_t = (1-t)*y + t*x  (reversed!)
    """
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma_min", type=float, default=1e-4, help="Constant noise level (small for regularization)")
        parser.add_argument("--sigma_max", type=float, default=1e-4, help="Not used in Independent CFM, kept for compatibility")
        return parser

    def __init__(self, sigma_min=1e-4, sigma_max=1e-4, **ignored_kwargs):
        super().__init__()        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max  # Not used, kept for compatibility
        
    def copy(self):
        return FLOWMATCHING(self.sigma_min, self.sigma_max)

    def ode(self, x, t, *args):
        pass
    
    def _mean(self, x0, t, y):
        """
        Independent CFM interpolation: from y (LR) to x0 (HR)
        t=0: mean = y (low resolution input)
        t=1: mean = x0 (high resolution target)
        """
        t_expanded = t[:, None, None, None]
        return (1 - t_expanded) * y + t_expanded * x0

    def _std(self, t):
        """Constant noise (FLowHigh's independent_cfm_constant style)"""
        return torch.ones_like(t) * self.sigma_min

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        """
        Prior is at t=0, which is y + small noise.
        This is the sampling starting point.
        """
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self.sigma_min
        z = torch.randn_like(y)
        # Start from y (LR) + small noise
        x_0 = y + z * std
        return x_0, z

    def der_mean(self, x0, t, y):
        """
        Vector field target: d(mean)/dt = x0 - y (HR - LR)
        This is the increment the model needs to predict!
        """
        return x0 - y
        
    def der_std(self, t):
        """Constant noise: derivative is 0"""
        return torch.zeros_like(t)
    
