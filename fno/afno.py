import torch.nn as nn
from .spectral_convolution import AdaptiveSpectralConvolution
from .mlp import MLP
from timm.models.layers import DropPath

class FourierBlock(nn.Module):
    """
    FourierBlock is a module that performs operations on input tensors using Fourier Transform.

    Attributes:
        dim (int): The number of input channels.
        img_size (list[int]): The size of the input image.
        mlp_ratio (float): The ratio of hidden units in the MLP layer to the input channels.
        drop (float): The dropout rate.
        act_layer (torch.nn.Module): The activation function to be used.
        norm_layer (torch.nn.Module): The normalization layer to be used.
        norm1 (torch.nn.Module): The first normalization layer.
        double_skip (bool): Whether to perform double skip connection.
        spectral (AdaptiveSpectralConvolution): The adaptive spectral convolution layer.
        drop_path (torch.nn.Module): The dropout layer for the skip connection.
        norm2 (torch.nn.Module): The second normalization layer.
        mlp (MLP): The MLP layer.
    """

    def __init__(self, dim: int, img_size: list[int], mlp_ratio: float = 4.0, drop: float = 0.0, drop_path: float = 0.0, act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm, double_skip: bool = False):
        """    
        Args:
        --------
        * dim (int): 
            The number of input channels.
        * img_size (list[int]): 
            The size of the input image.
        * mlp_ratio (float, optional):
            The ratio of hidden units in the MLP layer to the input channels. Default is 4.0.
        * drop (float, optional): 
            The dropout rate. Default is 0.0.
        * drop_path (float, optional): 
            The dropout rate for the skip connection. Default is 0.0.
        * act_layer (torch.nn.Module, optional): 
            The activation function to be used. Default is nn.GELU.
        * norm_layer (torch.nn.Module, optional): 
            The normalization layer to be used. Default is nn.LayerNorm.
        * double_skip (bool, optional): 
            Whether to perform double skip connection. Default is False.
        """
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.norm1 = self.norm_layer(self.dim)
        self.double_skip = double_skip
        
        # Layers
        self.spectral = AdaptiveSpectralConvolution(self.dim, self.img_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = self.norm_layer(self.dim)
        self.mlp = MLP(in_features=self.dim, 
                       hidden_size=int(self.dim * self.mlp_ratio), 
                       activation=self.act_layer, 
                       dropout=self.drop)
    
    def forward(self, x):
        """
        Forward pass of the FourierBlock module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        residual = x
        x = self.norm1(x)
        x = self.spectral(x)
        
        if self.double_skip:
            x += residual
            residual = x
        
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x += residual
        return x