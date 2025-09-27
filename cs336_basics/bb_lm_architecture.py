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