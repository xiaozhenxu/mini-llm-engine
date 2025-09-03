import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from src.nanoinfer.layers.basics import linear

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    def load_weight(self, weight: torch.Tensor):
        assert weight.shape == (self.vocab_size, self.hidden_size)
        self.weight.data = weight.to(self.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

    def as_linear(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight)

def main():
    MODEL_NAEM = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAEM)
    print(model)
    # print(model.model.embed_tokens)
    print(model.config)
    # print(model.model.embed_tokens.weight.data.shape)
    embedding = Embedding(model.config.vocab_size, model.config.hidden_size)

    qwen2_embedding = Embedding(model.config.vocab_size, model.config.hidden_size)
    qwen2_embedding.load_weight(model.model.embed_tokens.weight.data)

    input_ids = torch.randint(low = 0, high = model.config.vocab_size, size = (10, 16))
    print("---embedding test---")
    print(qwen2_embedding(input_ids).shape)
    input_x = torch.rand((10, 16, model.config.hidden_size), dtype=torch.float32)
    print("---embedding as_linear test---")
    print(qwen2_embedding.as_linear(input_x).shape)

if __name__ == "__main__":
    main()