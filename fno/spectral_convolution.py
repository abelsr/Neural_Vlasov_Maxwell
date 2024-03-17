from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSpectralConvolution(nn.Module):
    """
    Adaptive Spectral Convolution module.

    This module performs adaptive spectral convolution on the input tensor.
    It applies a series of complex multiplications and activations to transform the input tensor.

    Args:
        dim (int): Hidden channels.
        size (tuple): Size of the input tensor.
        num_blocks (int, optional): Number of blocks. Defaults to 4.
        activation (function, optional): Activation function. Defaults to F.relu.
        scale (float, optional): Scaling factor for the weights. Defaults to 0.02.
        softshrink (float, optional): Softshrink parameter for activation. Defaults to 0.5.
    """
    def __init__(self, dim: int , size: List[int], num_blocks:int = 4, activation: nn.Module = F.relu, scale: float = 0.02, softshrink: float = 0.5):
        super().__init__()
        self.hidden_size = dim              # Hidden channels
        self.size = size                    # Size of the input (we also infer the dimensionality from this)
        self.dimensions = len(size)         # Number of dimensions
        self.num_blocks = num_blocks        # Number of blocks
        self.block_size = self.hidden_size // self.num_blocks
        
        # Check if the hidden size is divisible by the number of blocks
        assert self.hidden_size % self.num_blocks == 0, "Hidden size must be divisible by the number of blocks"
        
        self.scale = scale                  # Scaling factor for the weights
        
        # Weights and biases
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size))
        
        self.activation = activation
        
        self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        self.softshrink = softshrink
        
    def complex_mul(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Perform complex multiplication between input and weights.

        Args:
            input (torch.Tensor): Input tensor.
            weights (torch.Tensor): Weight tensor.

        Returns:
            torch.Tensor: Result of complex multiplication.
        """
        assert input.shape[-1] == weights.shape[-2], "Input and weight dimensions do not match"
        return torch.einsum('...bd, bdk -> ...bk', input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AdaptiveSpectralConvolution module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Get the batch size, number of channels, and the size of the input
        B, N, C = x.shape 
        # Calculate the bias
        bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        # Reshape the input to [batch, x1_size, x2_size, ..., xN_size, hidden_channels]
        x = x.reshape(B, *self.size, C)
        # Compute the Fourier transform of the input
        x = torch.fft.rfftn(x, dim=[i for i in range(1, self.dimensions+1)], norm='ortho')
        x = x.reshape(B, *x.shape[1:-1], self.num_blocks, self.block_size)
        
        # Perform the convolution
        x_real = self.activation(self.complex_mul(x.real, self.w1[0]) - self.complex_mul(x.imag, self.w1[1]) + self.b1[0], inplace=True)
        x_imag = self.activation(self.complex_mul(x.real, self.w1[1]) - self.complex_mul(x.imag, self.w1[0]) + self.b1[1], inplace=True)
        x_real = self.complex_mul(x_real, self.w2[0]) - self.complex_mul(x_imag, self.w2[1]) + self.b2[0]
        x_imag = self.complex_mul(x_real, self.w2[1]) - self.complex_mul(x_imag, self.w2[0]) + self.b2[1]
        
        # Combine the real and imaginary parts
        x = torch.stack([x_real, x_imag], dim=-1)
        
        # Apply softshrink activation
        x = F.softshrink(x, self.softshrink)
        
        # Compute the inverse Fourier transform
        x = torch.view_as_complex(x)
        x = x.reshape(B, *x.shape[1:-2], self.hidden_size)
        x = torch.fft.irfftn(x, s=self.size, dim=[i for i in range(1, self.dimensions+1)], norm='ortho')
        x = x.reshape(B, N, C)
        x = x + bias
        return x