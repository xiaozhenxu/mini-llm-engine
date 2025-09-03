import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        # inv_freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype = torch.float32) / dim))
        i = torch.arange(1, self.dim // 2 + 1, 1, dtype=torch.float32) 
        inv_freqs = 1.0 / (self.base ** ((i - 1) * 2 / dim))
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(
        self,
        x: torch.Tensor,    # (b, num_of_heads, seq_len, head_dim)
    ) -> torch.Tensor:
        print(x.shape)
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, dtype = torch.float32)
        freqs = torch.outer(t, self.inv_freqs)
        
        # 将 freqs 映射到复数域 s, d/2
        freqs_cis = torch.polar(torch.ones_like(freqs, dtype = torch.float32), freqs)

        # 将输入 x 映射到复数域        
        x = x.reshape(*x.shape[:-1], -1, 2)

        x = torch.view_as_complex(x)
     
        x_rotated = x * freqs_cis
        # 将旋转结果映射回实数
        out = torch.view_as_real(x_rotated).flatten(-2)

        return out

        

if __name__ == "__main__":
    x = torch.rand((10, 14, 15, 64), dtype=torch.float32)
    rope = RotaryEmbedding(64, 15)
    print(rope(x).shape)