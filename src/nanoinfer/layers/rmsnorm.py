import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: int = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.dim))

    def load_weight(self, weight: torch.Tensor):
        assert weight.shape == (self.dim,)
        self.weight.data = weight.to(self.weight.device)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) + self.eps
        return x * rms * self.weight

if __name__ == "__main__":
    weight = torch.rand(1536)
    input = torch.rand((16, 10, 1536))
    layer_norm = RMSNorm(dim=1536)
    layer_norm.load_weight(weight)
    output = layer_norm(input)
    print(output.shape)