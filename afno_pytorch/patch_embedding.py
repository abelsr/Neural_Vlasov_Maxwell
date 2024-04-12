import torch
import torch.nn as nn
from typing import List, Union

class PatchEmbedding(nn.Module):
    """
    PatchEmbedding module that converts an input image into a sequence of patches.
    ```python
    Example:
        # Example usage
        x = torch.randn(1, 1, x1, ..., xN)
        patch_embed = PatchEmbedding(img_size=(x1, ..., xN), patch_size=8, in_channels=1, embed_dim=768)
        x = patch_embed(x) # (1, num_patches, embed_dim)
    ```
    """
    def __init__(self, dim: tuple, patch_size: Union[int, List[int]], in_channels: int, embed_dim: int, max_patches: int = 512):
        """
    Args:
        dim (tuple): Number of dimensions.
        patch_size (int or list): Size of each patch. If int, the same patch size is used for all dimensions.
            If list, the patch size can be different for each dimension.
        in_channels (int): Number of input channels.
        embed_dim (int): Dimension of the output patch embeddings.

    Attributes:
        img_size (tuple): Size of the input image.
        patch_size (list): Size of each patch.
        in_channels (int): Number of input channels.
        embed_dim (int): Dimension of the output patch embeddings.
        num_patches (int): Number of patches in the input image.
        proj (nn.Module): Patch embedding layer.
        """
        super().__init__()
        # if img_size is None:
        #     raise ValueError("Image size can't be None, please provide image size")
        # self.img_size = img_size
        if isinstance(patch_size, list):
            assert dim == len(patch_size), "Image and patch sizes must have the same length"
            # assert all(i % j == 0 for i, j in zip(range(1, dim+1), patch_size)), "Image dimensions must be divisible by the patch size"
            self.patch_size = patch_size
        elif isinstance(patch_size, int):
            assert all(list(range(dim))[i] % patch_size == 0 for i in range(dim)), "Image dimensions must be divisible by the patch size"
            self.patch_size = [patch_size] * dim
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate the number of patches
        # self.num_patches = [self.img_size[i] // self.patch_size[i] for i in range(len(self.img_size))]
        # self.num_patches = max_patches or int(torch.prod(torch.tensor(self.num_patches)))
        self.num_patches = max_patches
        
        # Define the patch embedding layer
        if dim == 1:
            self.proj = nn.Conv1d(self.in_channels, self.embed_dim, kernel_size=self.patch_size[0], stride=self.patch_size[0])
        elif dim == 2:
            self.proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        elif dim == 3:
            self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbedding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # batch_size, channels, *data_shape = x.shape
        # assert data_shape == self.img_size, f"Input tensor has wrong shape, expected {self.img_size} but got {data_shape}"
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x