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
        super().__init__()
        self.learb_gains = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        self.eps = eps

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)
        x_normalized = (x / rms) * self.learb_gains
        return x_normalized.to(in_dtype)
    

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, device=None, dtype=None, d_ff:int = None):
        super().__init__()
        if not d_ff:
            d_ff = 8* d_model /3
            d_ff = round(d_ff /64) * 64
        self.W1 = nn.Parameter(torch.randn(d_ff, d_model,device=device,dtype = dtype))
        self.W3 = nn.Parameter(torch.randn(d_ff, d_model,device = device,dtype = dtype))
        self.W2 = nn.Parameter(torch.randn(d_model,d_ff,device = device,dtype = dtype))

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        W1_x = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3_x = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        SiLu = W1_x * torch.sigmoid(W1_x)
        element_wise = einsum(SiLu , w3_x , "... d_ff, ... d_ff -> ... d_ff")
        result = einsum(element_wise , self.W2 ,"... d_ff, d_model d_ff -> ... d_model")
        return result
    
        
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        positions = torch.arange(max_seq_len,device = device).float()
        demo = 1.0 / theta** (torch.arange(0,d_k,2,device = device).float() / d_k)
        angles = einsum(positions, demo , " max_seq , d_2 -> max_seq d_2")

        self.register_buffer("cos_cach", torch.cos(angles), persistent= False)
        self.register_buffer("sin_cach", torch.sin(angles), persistent = False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """ x : (..., seq_len, d_k)   token_positions: (..., seq_len)
        """
        cos = self.cos_cach[token_positions].to(device = x.device) #每一个batch 取前seq——len长度的cos
        sin = self.sin_cach[token_positions].to(device = x.device)

        x_pair = rearrange(x , "... seq_len (d two) -> ...  seq_len d two",two = 2)

        x_even = x_pair[...,0]
        x_odd = x_pair[...,1]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even* sin + x_odd * cos

        x_stack = torch.stack([x_rot_even, x_rot_odd],dim = -1)
        rot_x = rearrange(x_stack , 
        "... seq d_half two -> ... seq (d_half two)")

        return rot_x

def Softmax(x: torch.Tensor, dim: int)-> torch.Tensor:
    max_x = x.max(dim=dim, keepdim =  True).values  #保留维度，这样可以之后运算可以自动广播
    shifted = x - max_x

    exp_x = torch.exp(shifted)
    sum_exp = exp_x.sum(dim= dim, keepdim = True)
    return exp_x / sum_exp

def scaled_dot_product_attention(query : torch.Tensor , key: torch.Tensor, value: torch.Tensor, mask_matrix = None)-> torch.Tensor:
    d_k = query.size(-1)
    scaled_q_k = einsum(query,key,"... n d_k, ... m d_k -> ... n m") / np.sqrt(d_k)

    if mask_matrix is not None:
        scaled_q_k = scaled_q_k.masked_fill(~mask_matrix,-float('inf'))
    
    softmax_score = Softmax(scaled_q_k, dim = -1) #对行进行softmax
    attention = einsum(softmax_score , value,"... n m, ... m d_v -> ... n d_v")
    return attention

class Multihead_attention(nn.Module):
    def __init__(self, d_model:int , num_heads: int, max_seq_len:int = None, theta:int = None,device = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.Q_proj = Linear(d_model, d_model,device = device)
        self.K_proj = Linear(d_model, d_model,device = device)
        self.V_proj = Linear(d_model, d_model,device = device)
        self.O_proj = Linear(d_model, d_model,device = device)
        self.theta = theta
        self.max_seq_len = max_seq_len
        if max_seq_len is not None and theta is not None :
            self.rope = RoPE(theta = theta, max_seq_len = max_seq_len, d_k = d_model / num_heads,device = device)
    
    def forward(self, x, token_positions=None):
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)
        Q_split = rearrange(Q, "... seq (num_heads d_head) -> ... num_heads seq d_head", num_heads = self.num_heads)
        K_split = rearrange(K, "... seq (num_heads d_head) -> ... num_heads seq d_head",num_heads = self.num_heads)
        V_split = rearrange(V, "... seq (num_heads d_head) -> ... num_heads seq d_head", num_heads = self.num_heads)
        if self.max_seq_len is not None and self.theta is not None :
            if token_positions is None:
                x_ones = torch.ones(x.size()[:-1],dtype=torch.long,device = x.device)
                ref = torch.arange(x.size(-2),device = x.device)
                token_positions = x_ones * ref
            Q_split = self.rope(Q_split, token_positions)
            K_split = self.rope(K_split, token_positions)
        
        causal_mask = torch.tril(torch.ones(Q.size(-2),Q.size(-2), dtype = torch.bool, device = Q.device))
        multi_head_attention = scaled_dot_product_attention(Q_split, K_split,V_split,causal_mask)
        attention = rearrange(multi_head_attention, "... num_h seq d_head -> ... seq (num_h d_head)")
        return self.O_proj(attention)

class Transformer_block(nn.Module):
    def __init__(self, d_model:int , num_heads:int, d_ff:int, max_seq_len:int, theta:int,device = None):
        super().__init__()
        self.rmsnorm_1 = RMSnorm(d_model,device = device)
        self.rmsnorm_2 = RMSnorm(d_model,device = device)
        self.m_h_a = Multihead_attention(d_model =d_model, num_heads =num_heads,max_seq_len =max_seq_len, theta = theta,device = device )
        self.swiglu = SwiGLU(d_model = d_model,d_ff =d_ff,device = device)
    def forward(self, x:torch.Tensor):
        x_norm_1 = self.rmsnorm_1(x)
        x_attention = self.m_h_a(x_norm_1)
        x = x_attention + x
        x_norm_2 = self.rmsnorm_2(x)
        x_swish = self.swiglu(x_norm_2)
        result = x_swish + x
        return result
        
class Transformer(nn.Module):
    def __init__(self,vocab_size: int,context_length: int, d_model: int,num_layers: int,num_heads: int,d_ff: int,rope_theta: float,device = None):
        super().__init__()
        self.embed = Embedding(num_embeddings= vocab_size, embedding_dim=d_model,device = device)
        self.transformer_blocks = nn.ModuleList([
            Transformer_block(d_model = d_model,num_heads = num_heads,d_ff = d_ff,max_seq_len = context_length,
            theta = rope_theta,device =device)for _ in range(num_layers)])
        self.last_norm = RMSnorm(d_model= d_model, device = device)
        self.Linear = Linear(d_model,vocab_size,device = device)
     
    def forward(self,x:torch.Tensor):
        x= self.embed(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.last_norm(x)
        x = self.Linear(x)
        return x

