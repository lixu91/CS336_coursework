import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.linear_weights = nn.Parameter(torch.randn(in_features,out_features,device= device,dtype= dtype))

        std_sqr = torch.sqrt(2 / (in_features+out_features))
        nn.init.trunc_normal_(self.linear_weights, mean=0,std=std_sqr,a= -3* std_sqr, b = 3 * std_sqr)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x , self.linear_weights, "... in_features , in_features out_features -> ... out_features")
    