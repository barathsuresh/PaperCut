import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 37000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return x
