import torch
import torch.nn as nn


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size=8192,
        d_model=384,
        n_layers=6,
        n_heads=6,
        context_size=512,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_size = context_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(context_size, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.context_size, "Sequence length exceeds context size"

        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # Causal mask (True = masked)
        mask = torch.triu(
            torch.ones(T, T, device=input_ids.device),
            diagonal=1,
        ).bool()

        for layer in self.layers:
            x = layer(x, src_mask=mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits