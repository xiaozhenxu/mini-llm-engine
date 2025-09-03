import torch
import torch.nn as nn

def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    if bias is not None:
        return torch.matmul(x, weight.T) + bias
    return torch.matmul(x, weight.T)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_exp = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)