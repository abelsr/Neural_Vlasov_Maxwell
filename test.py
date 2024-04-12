import torch
import torch.nn as nn
from afno_pytorch import AFNO, AdaptiveSpectralConvolution

model = AFNO(3, num_layers=1, patch_size=[2, 2, 2], max_num_patches=5120)
# model = AdaptiveSpectralConvolution(32, [8, 8], 4)
# x = torch.randn(1, 64, 32)
# print(model(x, [8, 8]).shape)
# x = torch.randn(1, 16, 32)
# print(model(x, [8, 8]).shape)
x = torch.randn(1, 1, 64, 64, 10)
print(model(x).shape)
print("*"*50)
# model = AFNO(2, [32, 32], num_layers=1)
# x = torch.randn(1, 1, 32, 32)
# print(model(x).shape)
# print("*"*50)
# x = torch.randn(1, 1, 16, 16)
# print(model(x).shape)