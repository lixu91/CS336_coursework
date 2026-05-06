import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.linear_weights = nn.Parameter(torch.randn(out_features,in_features,device= device,dtype= dtype))

        std_sqr = np.sqrt(2 / (in_features+out_features))
        nn.init.trunc_normal_(self.linear_weights, mean=0,std=std_sqr,a= -3* std_sqr, b = 3 * std_sqr)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x , self.linear_weights, "... d_in , d_out d_in -> ... d_out")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embed_matrix = nn.Parameter(torch.empty(num_embeddings,embedding_dim, device=device,dtype=dtype))
        nn.init.trunc_normal_(self.embed_matrix,mean=0,std= 1, a=-3,b = 3)

    def forward(self, token_ids:torch.Tensor)-> torch.Tensor:
        return self.embed_matrix[token_ids] #高级索引  自动广播成（b,s,d）形状
    
class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super.__init__()
        self.learb_gains = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        self.eps = eps

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)
        x_normalized = (x / rms) * self.learb_gains
        return x_normalized.to(in_dtype)
        
    