import torch
import torch.nn as nn

import torch.nn.functional as F


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        mean = 0
        std = (2 / (in_features + out_features)) ** 0.5

        self.W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    out_features, 
                    in_features,
                    device="cpu" if device is None else device,
                    dtype=torch.float32 if dtype is None else dtype
                ), 
                a=-3*std, 
                b=3*std
            )
        )
    
    def forward(self, x):
        return x @ self.W.transpose(0, 1)
    

class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.embedding = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    num_embeddings, 
                    embedding_dim,
                    device="cpu" if device is None else device,
                    dtype=torch.float32 if dtype is None else dtype
                ),
                a=-3, b=3
            )
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, token_ids):
        one_hot = F.one_hot(token_ids, num_classes=self.num_embeddings).to(self.embedding.dtype) # batch_size, sequence_length, num_embeddings
        return one_hot @ self.embedding # batch_size, sequence_length, embedding_dim
    

class MyRMSNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.gain = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, device="cpu" if device is None else device, dtype=torch.float32 if dtype is None else dtype)
            )
        )
        self.d_model = d_model
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_sum = torch.sum(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(
            x_sum / self.d_model + self.eps
        )

        result = self.gain * x / rms
        return result.to(in_dtype)


class MySiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * F.sigmoid(x)
    

class MyGLU(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model

        self.W1 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(8 * d_model // 3, d_model, device="cpu" if device is None else device, dtype=torch.float32 if dtype is None else dtype)
            )
        )

        self.W2 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(8 * d_model // 3, d_model, device="cpu" if device is None else device, dtype=torch.float32 if dtype is None else dtype)
            )
        )
    
    def forward(self, x):
        return F.sigmoid(x @ self.W1.T) * (x @ self.W2.T)
    

class MySwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.W1 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_ff, d_model, device="cpu" if device is None else device, dtype=torch.float32 if dtype is None else dtype)
            )
        )

        self.W2 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_ff, device="cpu" if device is None else device, dtype=torch.float32 if dtype is None else dtype)
            )
        )

        self.W3 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_ff, d_model, device="cpu" if device is None else device, dtype=torch.float32 if dtype is None else dtype)
            )
        )


    def forward(self, x):
        w1_out = x @ self.W1.T
        silu_out = w1_out * F.sigmoid(w1_out)
        w3_out = x @ self.W3.T

        return (silu_out * w3_out) @ self.W2.T
    

class MyRoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        row = torch.arange(max_seq_len)
        k = torch.arange(1, d_k//2 + 1)
        col = theta ** ((2-2*k)/d_k)
        
        mat = torch.outer(row, col)

        sin = torch.sin(mat)
        cos = torch.cos(mat)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # x: batch_size, seq_len, d_k
        # token_positions: batch, seq_len

        # batch_size, seq_len, d_k//2
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        idx1 = [i for i in range(0, self.d_k, 2)]
        idx2 = [i+1 for i in range(0, self.d_k, 2)]

        first_pair = x[..., idx1]
        second_pair = x[..., idx2]

        rotated1 = cos * first_pair - sin * second_pair
        rotated2 = sin * first_pair + cos * second_pair

        batch_size, seq_len, _ = x.shape
        return torch.stack([rotated1, rotated2], axis=3).reshape(batch_size, seq_len, -1)
        

def softmax(x: torch.Tensor, dim: int):
    dim_max = torch.max(x, dim=dim, keepdim=True).values
    new_x = x - dim_max

    return torch.exp(new_x) / torch.sum(torch.exp(new_x), dim=dim, keepdim=True)


def scaled_dot_product_attn(q, k, v, mask=None):
    qkT = q @ k.transpose(-1, -2)
    d_model = q.shape[-1]
    qkT = qkT / d_model**0.5

    if mask is not None:
        qkT = torch.where((mask==True), qkT, -torch.inf)
    
    score = softmax(qkT, dim=-1)
    return score @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        self.num_heads = num_heads

        self.Wq = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_model)
            )
        )
        self.Wk = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_model)
            )
        )
        self.Wv = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_model)
            )
        )
        self.Wo = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_model)
            )
        )


    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        Q = x @ self.Wq.T
        K = x @ self.Wk.T
        V = x @ self.Wv.T

        Q = Q.reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        mask = torch.ones(seq_len, seq_len)
        mask = torch.where(torch.tril(mask)==0, False, True)
        attn_out = scaled_dot_product_attn(Q, K, V, mask) # batch_size, num_heads, seq_len, hidden

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1) # batch_size, seq_len, (num_heads*hidden)
        return attn_out @ self.Wo.T