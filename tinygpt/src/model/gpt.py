import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        # Pre-norm attention
        residual = x
        x = self.ln1(x)
        
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot product attention with causal masking
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        
        # Reshape and project
        attn_output = attn_output.contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        
        x = residual + attn_output
        
        # Pre-norm FFN
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


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
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
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

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits