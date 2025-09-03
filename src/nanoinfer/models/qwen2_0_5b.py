import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM
from src.nanoinfer.layers.embedding import Embedding
from src.nanoinfer.layers.attention import MultiHeadAttention
from src.nanoinfer.layers.rmsnorm import RMSNorm
from src.nanoinfer.layers.mlp import Qwen2MLP

@dataclass
class Qwen2Config:
    hidden_size: int = 896
    intermediate_size: int = 4864
    vocab_size: int = 151936
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    num_hidden_layers: int = 24
    torch_dtype: str = "bfloat16"

class Qwen2TransformerBlock(nn.Module):
    def __init__(
        self,
        model_config: Qwen2Config,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        q_bias: torch.Tensor,
        k_bias: torch.Tensor,
        v_bias: torch.Tensor,
        o_proj: torch.Tensor,
        input_layernorm_weight: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        post_attention_layernorm_weight: torch.Tensor
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
                        model_config.num_attention_heads,
                        model_config.num_key_value_heads,
                        model_config.hidden_size,
                        model_config.hidden_size // model_config.num_attention_heads,
                        q_proj, 
                        k_proj, 
                        v_proj,
                        q_bias,
                        k_bias,
                        v_bias,
                        o_proj,
                        input_layernorm_weight
                        )
        self.mlp = Qwen2MLP(
            model_config.hidden_size,
            model_config.intermediate_size,
            gate_proj,
            up_proj,
            down_proj,
            post_attention_layernorm_weight
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attenion_mask: torch.Tensor = None
    ) -> torch.Tensor:
        attention_output = self.self_attn(hidden_states)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + attention_output
        return hidden_states

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config, model: Qwen2ForCausalLM):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config.vocab_size, config.hidden_size)
        self.embedding.load_weight(model.model.embed_tokens.weight.data)
        
        self.layers_inner= []
        for i in range(config.num_hidden_layers):
            wq = model.model.layers[i].self_attn.q_proj.weight.data
            wk = model.model.layers[i].self_attn.k_proj.weight.data
            wv = model.model.layers[i].self_attn.v_proj.weight.data
            wo = model.model.layers[i].self_attn.o_proj.weight.data
            bq = model.model.layers[i].self_attn.q_proj.bias.data
            bk = model.model.layers[i].self_attn.k_proj.bias.data
            bv = model.model.layers[i].self_attn.v_proj.bias.data
            layer = Qwen2TransformerBlock(config, wq, wk, wv, bq, bk, bv, wo,
                                        model.model.layers[i].input_layernorm.weight.data,
                                        model.model.layers[i].mlp.gate_proj.weight.data,
                                        model.model.layers[i].mlp.up_proj.weight.data,
                                        model.model.layers[i].mlp.down_proj.weight.data,
                                        model.model.layers[i].post_attention_layernorm.weight.data)
            self.layers_inner.append(layer)

        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = model.lm_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(x)

        for layer_idx in range(1):
            hidden_states = self.layers_inner[layer_idx](hidden_states)
        
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    Qwen2 = Qwen2ForCausalLM(Qwen2Config(), model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text = "你好，世界"
    inputs= tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(Qwen2(input_ids).shape)