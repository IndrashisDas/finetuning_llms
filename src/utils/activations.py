"""
This file consists of custom activation functions
"""

from typing import Union, List, Tuple

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU, Mish


@torch.jit.script
def _golu_forward(
    z: torch.Tensor,
    alpha: nn.Parameter,
    beta: nn.Parameter,
    gamma: nn.Parameter
) -> torch.Tensor:

    # Return the forward pass of GoLU Activation
    # nan_to_num uses nan=0.0 by default, so not set explicitly
    # nan_to_num also sets the -inf and +inf to respective extreme values for different data types
    z = torch.nan_to_num(z)
    return torch.nan_to_num(z * alpha * torch.exp(-beta * torch.exp(-gamma * z)))


@torch.jit.script
def _golu_backward(
    grad_output: torch.Tensor,
    z: torch.Tensor,
    alpha: nn.Parameter,
    beta: nn.Parameter,
    gamma: nn.Parameter,
    req_grad: Tuple[bool, bool, bool, bool]
) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:

    # The following two lines are necessary to tackle -inf and +inf in the incoming tensors. Sets to extreme values for respective dtype
    z = torch.nan_to_num(z)
    grad_output = torch.nan_to_num(grad_output)

    # Computes gradients for the backward pass
    if req_grad[0]:
        grad_z = torch.nan_to_num(grad_output * alpha * torch.exp(-beta * torch.exp(-gamma * z)) * (1 + torch.nan_to_num(z * beta * gamma * torch.exp(-gamma * z))))
    else:
        grad_z = None

    if req_grad[1]:
        grad_alpha = torch.nan_to_num(grad_output * z * torch.exp(-beta * torch.exp(-gamma * z)))
    else:
        grad_alpha = None

    if req_grad[2]:
        grad_beta = torch.nan_to_num(grad_output * -z * alpha * torch.exp(-beta * torch.exp(-gamma * z)) * torch.exp(-gamma * z))
    else:
        grad_beta = None

    if req_grad[3]:
        grad_gamma = torch.nan_to_num(grad_output * z * z * alpha * beta * torch.exp(-beta * torch.exp(-gamma * z)) * torch.exp(-gamma * z))
    else:
        grad_gamma = None

    # Return the gradients computed as per the backward pass
    return grad_z, grad_alpha, grad_beta, grad_gamma


class GoLUFunction(Function):

    @staticmethod
    def forward(
        ctx,
        z: torch.Tensor,
        alpha: nn.Parameter,
        beta: nn.Parameter,
        gamma: nn.Parameter
    ) -> torch.Tensor:
        ctx.save_for_backward(z, alpha, beta, gamma)
        return _golu_forward(z=z, alpha=alpha, beta=beta, gamma=gamma)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        z, alpha, beta, gamma = ctx.saved_tensors  # Gives the saved tensors during forward pass
        req_grad = ctx.needs_input_grad
        return _golu_backward(grad_output=grad_output, z=z, alpha=alpha, beta=beta, gamma=gamma, req_grad=req_grad)


class GoLU(nn.Module):
    """
    GoLU Activation Function
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        requires_grad: List = [False, False, False],
        clamp_alpha: List = [],
        clamp_beta: List = [],
        clamp_gamma: List = []
    ) -> None:
        """
        Initialize the GoLU Activation Layer

        Gompertz Linear Unit (GoLU) is an element-wise and self-gated activation function. It uses the 
        Gompertz Function as a gate. The GoLU activation and Gompertz function are defined as follows,

        GoLU(x) = x * gompertz(x), where
        gompertz(x) = alpha * exp(-beta * exp(-gamma * x))

        The stationary activation has the respective alpha, beta and gamma values set to 1.0

        Args:
            alpha (float): Controls the upper asymptote or scale of the gate
            beta (float): Controls the gate displacement along x-axis
            gamma (float): Controls the growth rate of the gate
            requires_grad (List): A list which decides if alpha, beta and gamma need to be trained. You can
            customize training the gate parameters.
            clamp_alpha (List): Clamping thresholds for alpha
            clamp_beta (List): Clamping thresholds for beta
            clamp_gamma (List): Clamping thresholds for gamma
        """
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=requires_grad[0]), requires_grad=requires_grad[0])
        self.beta = nn.Parameter(torch.tensor(beta, requires_grad=requires_grad[1]), requires_grad=requires_grad[1])
        self.gamma = nn.Parameter(torch.tensor(gamma, requires_grad=requires_grad[2]), requires_grad=requires_grad[2])
        self.clamp_alpha = clamp_alpha
        self.clamp_beta = clamp_beta
        self.clamp_gamma = clamp_gamma

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Calls the forward pass of the GoLU Activation Function

        Returns:
            torch.Tensor: The output of the forward method
        """
        self.clamp_params_()
        return self.forward(*args)

    def clamp_params_(self) -> None:
        """
        This function clamps the parameters of the Gompertz Function
        """
        if self.clamp_alpha:
            assert (len(self.clamp_alpha) == 1) or (len(self.clamp_alpha) == 2)
            if len(self.clamp_alpha) == 1:
                self.alpha.data.clamp_(min=self.clamp_alpha[0])
            else:
                self.alpha.data.clamp_(min=self.clamp_alpha[0], max=self.clamp_alpha[1])

        if self.clamp_beta:
            assert (len(self.clamp_beta) == 1) or (len(self.clamp_beta) == 2)
            if len(self.clamp_beta) == 1:
                self.beta.data.clamp_(min=self.clamp_beta[0])
            else:
                self.beta.data.clamp_(min=self.clamp_beta[0], max=self.clamp_beta[1])

        if self.clamp_gamma:
            assert (len(self.clamp_gamma) == 1) or (len(self.clamp_gamma) == 2)
            if len(self.clamp_gamma) == 1:
                self.gamma.data.clamp_(min=self.clamp_gamma[0])
            else:
                self.gamma.data.clamp_(min=self.clamp_gamma[0], max=self.clamp_gamma[1])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GoLU Activation Function

        Args:
            z (torch.Tensor): The input tensor on which the activation is computed

        Returns:
            torch.Tensor: The activated tensor
        """
        return GoLUFunction.apply(z, self.alpha, self.beta, self.gamma)


def get_activation_function(
    activation: str = 'ReLU'
) -> Union[Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU, Mish, GoLU]:
    """
    This function returns the activation needed to train an architecture

    Args:
        activation (str, optional): The name of the activation to train with. Defaults to 'ReLU'.

    Returns:
        Union[Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU, Mish, GoLU]: The activation
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
    elif activation == 'GoLU':
        return GoLU()
    elif activation == 'GoLU_alpha':
        return GoLU(
            requires_grad = [True, False, False], clamp_alpha=[]
        )
    elif activation == 'GoLU_beta':
        return GoLU(
            requires_grad = [False, True, False], clamp_beta=[0.02]
        )
    elif activation == 'GoLU_gamma':
        return GoLU(
            requires_grad = [False, False, True], clamp_gamma=[0.3, 1.0]
        )
    elif activation == 'GoLU_alpha_beta':
        return GoLU(
            requires_grad = [True, True, False], clamp_alpha=[], clamp_beta=[0.02]
        )
    elif activation == 'GoLU_alpha_gamma':
        return GoLU(
            requires_grad = [True, False, True], clamp_alpha=[], clamp_gamma=[0.3, 1.0]
        )
    elif activation == 'GoLU_beta_gamma':
        return GoLU(
            requires_grad = [False, True, True], clamp_beta=[0.02], clamp_gamma=[0.3, 1.0]
        )
    elif activation == 'GoLU_alpha_beta_gamma':
        return GoLU(
            requires_grad = [True, True, True], clamp_alpha=[], clamp_beta=[0.02], clamp_gamma=[0.3, 1.0]
        )
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
