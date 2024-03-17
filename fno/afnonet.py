from functools import partial
from collections import OrderedDict
from typing import List, Union
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from timm.models.layers import trunc_normal_
from .patch_embedding import PatchEmbedding
from .afno import FourierBlock

class AdaptiveFourierNeuralOperator(nn.Module):
    """
    Adaptive Fourier Neural Operator (AFNO) model.
    """
    def __init__(self, img_size: list[int], patch_size: Union[int, List[int]] = 4, in_channels: int = 1, out_channels: int = 1, embed_dim: int = 768, num_layers: int = 12, mlp_ratio: float = 4, uniform_drop: bool = False, drop_rate: float = 0, drop_path_rate: float = 0, norm_layer: nn.Module = None, dropcls: int = 0, checkpoints: bool = False):
        """
        Args:
        -----
        * `img_size` (list[int]): 
            Image size. Defaults to None.
        * `patch_size` (int, List[int], optional):
            Patch size. Defaults to 4. If it's higher output size will be smaller.
        * `in_channels` (int, optional): 
            Number of input channels. Defaults to 1.
        * `out_channels` (int, optional): 
            Number of output channels. Defaults to 1.
        * `embed_dim` (int, optional): 
            Embedding dimension. Defaults to 768.
        * `num_layers` (int, optional): 
            Number of layers. Defaults to 12.
        * `mlp_ratio` (float, optional): 
            Ratio of the hidden layer size to the embedding size. Defaults to 4.
        * `uniform_drop` (bool, optional): 
            Whether to use uniform drop rate. Defaults to False.
        * `drop_rate` (float, optional): 
            Dropout rate. Defaults to 0.
        * `drop_path_rate` (float, optional): 
            Drop path rate. Defaults to 0.
        * `norm_layer` (torch.nn.Module, optional): 
            Normalization layer. Defaults to None.
        * `dropcls` (int, optional): 
            Dropout rate for the final layer. Defaults to 0.
        * `checkpoints` (bool, optional): 
            Whether to use checkpoints. Defaults to False.
        """
        super().__init__()
        if not img_size:
            raise ValueError("Image size can't be None, please provide image size")
        self.img_size = img_size
        assert len(self.img_size) in [1, 2, 3], "Image dimensions must be 1, 2 or 3 dimensions. Higher dimensions are not supported yet."
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.checkpoints = checkpoints
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if isinstance(patch_size, int):
            self.patch_size = [patch_size] * len(img_size)
        elif isinstance(patch_size, list):
            self.patch_size = patch_size
        self.size = list(size // patch_size for size, patch_size in zip(img_size, self.patch_size))
        
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(img_size=self.img_size,
                                          patch_size=self.patch_size,
                                          in_channels=self.in_channels,
                                          embed_dim=self.embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Drop
        if uniform_drop:
            self.drop = [drop_path_rate for _ in range(self.num_layers)]
        else:
            self.drop = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]
            
        # Fourier blocks
        self.blocks = nn.ModuleList([FourierBlock(dim=self.embed_dim,
                                                  mlp_ratio=self.mlp_ratio,
                                                  drop=self.drop_rate,
                                                  drop_path=self.drop[i],
                                                  norm_layer=self.norm_layer,
                                                  img_size=self.size) for i in range(self.num_layers)])
        
        # Norm layer
        self.norm = self.norm_layer(self.embed_dim)
        
        # Linear decoder
        if len(self.img_size) == 1:
            self.kernel_size = self.kernel_size[0]
            self.decoder = nn.Sequential(OrderedDict([
                ('conv1', nn.ConvTranspose1d(self.embed_dim, self.out_channels*16, kernel_size=2, stride=2)),
                ('act1', nn.Tanh()),
                ('conv2', nn.ConvTranspose1d(self.out_channels*16, self.out_channels*4, kernel_size=2, stride=2)),
                ('act2', nn.Tanh()),
            ]))
            self.head = nn.ConvTranspose1d(self.out_channels*4, self.out_channels, kernel_size=1, stride=1)
        elif len(self.img_size) == 2:
            self.decoder = nn.Sequential(OrderedDict([
                ('conv1', nn.ConvTranspose2d(self.embed_dim, self.out_channels*16, kernel_size=(2, 2), stride=(2, 2))),
                ('act1', nn.Tanh()),
                ('conv2', nn.ConvTranspose2d(self.out_channels*16, self.out_channels*4, kernel_size=(2, 2), stride=(2, 2))),
                ('act2', nn.Tanh()),
            ]))
            self.head = nn.ConvTranspose2d(self.out_channels*4, self.out_channels, kernel_size=(1, 1), stride=(1, 1))
        elif len(self.img_size) == 3:
            self.decoder = nn.Sequential(OrderedDict([
                ('conv1', nn.ConvTranspose3d(self.embed_dim, self.out_channels*16, kernel_size=(2, 2, 2), stride=(2, 2, 2))),
                ('act1', nn.Tanh()),
                ('conv2', nn.ConvTranspose3d(self.out_channels*16, self.out_channels*4, kernel_size=(2, 2, 2), stride=(2, 2, 2))),
                ('act2', nn.Tanh()),
            ]))
            self.head = nn.ConvTranspose3d(self.out_channels*4, self.out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            
        if dropcls > 0:
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()
            
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """
        Initialize the weights of the neural network layers.

        Args:
            m (nn.Module): The module to initialize the weights for.

        Returns:
            None
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def __repr__(self):
        return f"{self.__class__.__name__}(img_size={self.img_size}, in_channels={self.in_channels}, out_channels={self.out_channels}, embed_dim={self.embed_dim}, num_layers={self.num_layers})"

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Returns a set of parameter names that should not be subject to weight decay during optimization.

        Returns:
            set: A set of parameter names that should not be subject to weight decay.
        """
        return {'pos_embed', 'cls_token'}
    
    def forward_features(self, x):
        """
        Forward pass through the feature extraction layers of the AFNONet model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, x1, ..., xN), where B is the batch size, C is the number of channels, and x1, ..., xN are the dimensions of the input image.

        Returns:
            torch.Tensor: Output tensor of shape (B, embed_dim, x1', ..., xN'), where B is the batch size, embed_dim is the embedding dimension, and x1', ..., xN' are the dimensions of the output image.
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        x += self.pos_embed
        x = self.pos_drop(x)
        
        if not self.checkpoints:
            for blk in self.blocks:
                x = blk(x)
        else:
            x = checkpoint_sequential(self.blocks, 4, x)
        
        x = self.norm(x).transpose(1, 2)
        x = torch.reshape(x, [-1, self.embed_dim, *self.size])
        return x
    
    def forward(self, x):
        """
        Forward pass of the AFNONet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.decoder(x)
        x = self.head(x)
        return x