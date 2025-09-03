import torch
import torch.nn as nn
import torch.nn.functional as F
from src.nanoinfer.layers.basics import linear, softmax
from src.nanoinfer.layers.rmsnorm import RMSNorm
from src.nanoinfer.layers.rope import RotaryEmbedding

def causal_mask(
    seq_len: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:   # 默认设置query key的seqLen是一致的
    '''
    | 0 -inf -inf -inf |
    | 0 0    -inf -inf |
    | 0 0    0    -inf |
    | 0 0    0    0    |
    '''
    if device is None:
        device = torch.device("cpu")
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    mask = torch.where(mask == 1,
                       torch.tensor(0.0, device=device, dtype=dtype),
                       torch.tensor(float("-inf"), device=device, dtype=dtype))
    return mask

def scaled_dot_product_attention(
    query: torch.Tensor,            # (BatchSize, NumHeads, SeqLength, HeadDim)
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
    mask: torch.Tensor = None       # (SeqLength, SeqLength)
) -> torch.Tensor:
    if scale is None:
        scale = torch.rsqrt(torch.tensor(query.shape[-1]))
    
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if mask is None:
        assert query.shape[-2] == key.shape[-2]
        mask = causal_mask(query.shape[-2])

    if mask.device != scores.device:
        mask = mask.to(scores.device)
    if mask.dtype != scores.dtype:
        mask = mask.to(scores.dtype)

    scores = scores + mask
    attention_weights = softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value)

def scaled_dot_product_attention_grouped(
    query: torch.Tensor,    # (b, num_attention_heads, s, head_dim)
    key: torch.Tensor,      # (b, num_kv_heads, s, head_dim)
    value: torch.Tensor,    # (b, num_kv_heads, s, head_dim)
    scale: float = None,
    mask: torch.Tensor = None
) -> torch.Tensor:
    batch_size, num_attention_heads, query_seq_length, head_dim = query.shape   # (b, 14, s, 64)
    _, num_key_value_heads, kv_seq_length, _ = key.shape                 # (b, 2, s, 64)

    assert query_seq_length == kv_seq_length
    assert num_attention_heads % num_key_value_heads == 0
    repeats = num_attention_heads // num_key_value_heads

    key = key.unsqueeze(2).expand(-1, -1, repeats, -1, -1)
    value = value.unsqueeze(2).expand(-1, -1, repeats, -1, -1)

    key = key.reshape(batch_size, num_attention_heads, kv_seq_length, head_dim)
    value = value.reshape(batch_size, num_attention_heads, kv_seq_length, head_dim)

    return scaled_dot_product_attention(query, key, value, scale, mask)

class MultiHeadAttention(nn.Module):
    def __init__(self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        q_project: torch.Tensor,
        k_project: torch.Tensor,
        v_project: torch.Tensor,
        q_bias: torch.Tensor,
        k_bias: torch.Tensor,
        v_bias: torch.Tensor,
        o_project: torch.Tensor,
        input_layernorm_weight: torch.Tensor
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.q_project = q_project
        self.k_project = k_project
        self.v_project = v_project
        self.o_project = o_project
        self.q_bias = q_bias
        self.k_bias = k_bias
        self.v_bias = v_bias
        self.input_layernorm = RMSNorm(hidden_size)
        self.input_layernorm.load_weight(input_layernorm_weight)
        self.query_rope = RotaryEmbedding(dim=self.head_dim)
        self.key_rope = RotaryEmbedding(dim=self.head_dim)

    def forward(self,
        x: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = self.input_layernorm(x)
        query = linear(x, self.q_project, self.q_bias)
        key = linear(x, self.k_project, self.k_bias)
        value = linear(x, self.v_project, self.v_bias)
        assert query.shape[-1] == self.num_attention_heads * self.head_dim
        assert key.shape[-1] == self.num_kv_heads * self.head_dim
        assert value.shape[-1] == self.num_kv_heads * self.head_dim

        query = query.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # todo: RoPE 通常期望输入为 (batch_size, seq_len, num_heads, head_dim), 但并不强需
        query = query.transpose(1, 2)   # todo: 优化
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = self.query_rope(query)
        key = self.key_rope(key)

        attention_output = scaled_dot_product_attention_grouped(query, key, value)
        # (b, num_attention_heads, s, head_dim) -> (b, s, num_attention_heads, head_dim)
        attention_output = attention_output.transpose(1, 2)
        # 合并得到 (b, s, hidden_size)
        attention_output = attention_output.reshape(batch_size, seq_len, -1)

        output = linear(attention_output, self.o_project)
        return output

if __name__ == "__main__":
    query = torch.rand((10, 14, 16, 64), dtype = torch.float32)
    key = torch.rand((10, 14, 16, 64), dtype = torch.float32)
    value = torch.rand((10, 14, 16, 64), dtype = torch.float32)

    custom_attention_scores = scaled_dot_product_attention_grouped(query, key, value)
    # base_attention_scores = F.scaled_dot_product_attention(query, key, value)
    # assert torch.allclose(custom_attention_scores, base_attention_scores, atol=1e-5, rtol=1e-5)
    print(custom_attention_scores.shape)