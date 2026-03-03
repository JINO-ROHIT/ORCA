import torch
import torch.nn as nn

def compute_rope_params(head_dim: int, theta_base: int = 10_000, context_length: int = 4096, dtype: torch.dtype = torch.float32):
    inv_freq = 1.0 / theta_base ** (torch.arange(0, head_dim, 2, dtype = dtype) / head_dim)
    positions = torch.arange(0, context_length, dtype = dtype)

    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) # [ctx_len, 1] * [1, head_dim / 2]
    angles = torch.cat([angles, angles], dim = 1) # match to head dim

    sin_angle = torch.sin(angles)
    cos_angle = torch.cos(angles)
    return sin_angle, cos_angle


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    bs, n_head, seq_len, head_dim = x.shape
    x1 = x[..., : head_dim // 2] # ... better than use :, :, :, :head_dim //2
    x2 = x[..., head_dim // 2 :]

    cos = cos[offset: offset + seq_len, :].unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, head_dim)
    sin = sin[offset: offset + seq_len, :].unsqueeze(0).unsqueeze(0)    

    rotated = torch.cat((-x2, x1), dim = -1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype = x.dtype) # okay to use lower precision


# type hint this

class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        x = x.to(torch.float32) #upscale

        variance = x.pow(2).mean(dim = -1, keepdim = True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)