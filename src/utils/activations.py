"""
This file consists of custom activation functions
"""

from typing import Union, List, Tuple

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU, Mish


def get_activation_function(
    activation: str = 'ReLU'
) -> Union[Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU, Mish]:
    """
    This function returns the activation needed to train an architecture

    Args:
        activation (str, optional): The name of the activation to train with. Defaults to 'ReLU'.

    Returns:
        Union[Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU, Mish]: The activation
    """
    if activation == 'Sigmoid':
        return Sigmoid()
    elif activation == 'Tanh':
        return Tanh()
    elif activation == 'ReLU':
        return ReLU()
    elif activation == 'Softplus':
        return Softplus()
    elif activation == 'LeakyReLU':
        return LeakyReLU()
    elif activation == 'PReLU':
        return PReLU()
    elif activation == 'ELU':
        return ELU()
    elif activation == 'SELU':
        return SELU()
    elif activation == 'GELU':
        return GELU()
    elif activation == 'GELU_tanh':
        return GELU(approximate='tanh')
    elif activation == 'Swish':
        return SiLU()
    elif activation == 'Mish':
        return Mish()
    else:
        raise ValueError("The activation name is incorrect")


class NoisyActivation(nn.Module):
    def __init__(
        self, activation, noise='none', mean=0.0, std=1.0, lambda_value=1.0
    ):
        """
        Noisy Activation class to apply different noise formulations during training.
        
        Parameters:
        - activation: the activation function to use.
        - noise: type of noise to apply (e.g., 'gaussian', 'poisson', 'multiplicative' or 'none').
        - mean: mean for Gaussian noise.
        - std: standard deviation for Gaussian or multiplicative noise.
        - lambda_value: rate parameter for Poisson noise.
        """
        super(NoisyActivation, self).__init__()
        self.activation = get_activation_function(activation)
        self.noise = noise
        self.mean = mean
        self.std = std
        self.lambda_value = lambda_value

    def forward(self, x):
        if self.training:
            if self.noise == 'gaussian':
                # Gaussian noise
                return self.activation(x) + torch.randn_like(x) * self.std + self.mean
            elif self.noise == 'poisson':
                # Poisson noise
                return self.activation(x) + torch.poisson(torch.full_like(x, self.lambda_value))
            elif self.noise == 'multiplicative':
                # Multiplicative noise
                return self.activation(x) * (torch.randn_like(x) * self.std + self.mean)
            elif self.noise == 'none':
                # No noise
                return self.activation(x)
        else:
            # No noise during evaluation
            return self.activation(x)
