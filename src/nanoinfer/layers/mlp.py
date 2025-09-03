import torch
import torch.nn as nn
from src.nanoinfer.layers.rmsnorm import RMSNorm
from src.nanoinfer.layers.basics import linear

def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.exp(-x))

class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        post_attention_layernorm_weight: torch.Tensor
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm.load_weight(post_attention_layernorm_weight)
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.post_attention_layernorm(x)
        return linear(silu(linear(x, self.gate_proj)) * linear(x, self.up_proj), self.down_proj)

if __name__ == "__main__":
    x = torch.rand((16, 10, 1536))
    gate_proj = torch.rand((8960, 1536))
    up_proj = torch.rand((8960, 1536))
    down_proj = torch.rand((1536, 8960))

    mlp = Qwen2MLP(1536, 8960, gate_proj, up_proj, down_proj)
    print(mlp(x).shape)