"""
Decoder-only transformer with µP (Maximal Update Parameterization).

Adapted from part2/model.py (nanoGPT). Key changes from standard parameterization:
  - nn.Linear used for all hidden matrices; set_base_shapes annotates them with infshape
  - MuReadout replaces nn.Linear for the LM head (applies 1/width output multiplier)
  - Weight tying is disabled (MuReadout and nn.Embedding cannot be tied)
  - Attention scale: 1/d_head instead of 1/sqrt(d_head) (µP transformer rule)
  - make_mup_model() registers base shapes so MuAdamW scales LR ∝ 1/width_mult per layer
  - configure_optimizers() returns MuAdamW

Reference: Yang et al. 2022, "Tensor Programs V" (https://arxiv.org/abs/2203.09789)
Package:   https://github.com/microsoft/mup
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from mup import MuReadout, set_base_shapes, MuAdamW


# µP base width — always Tiny (n_embd=128). All models in the wide series share this base.
_BASE_N_EMBD = 128
_BASE_N_HEAD = 4
_BASE_N_FF   = 512


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttentionMuP(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.dropout  = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.flash:
            # µP requires overall attention scale 1/d_head (not 1/sqrt(d_head)).
            # F.scaled_dot_product_attention internally applies 1/sqrt(d_head),
            # so pre-scaling q by 1/sqrt(d_head) yields combined scale 1/d_head.
            q = q * (self.head_dim ** -0.5)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / self.head_dim)  # µP: 1/d_head
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLPMuP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.n_ff, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.n_ff, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class BlockMuP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionMuP(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLPMuP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int   = 1024
    vocab_size: int   = 4096
    n_layer:    int   = 4
    n_head:     int   = 4
    n_embd:     int   = 128
    n_ff:       int   = 512
    dropout:    float = 0.0
    bias:       bool  = False


class GPTMuP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([BlockMuP(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        # MuReadout for output logits — no weight tying (incompatible with MuReadout)
        self.lm_head = MuReadout(config.n_embd, config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, MuReadout)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Scaled init for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def count_params(self, non_embedding=True):
        """Total parameters, optionally excluding positional embeddings."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Seq len {T} > block_size {self.config.block_size}"
        device = idx.device
        pos = torch.arange(T, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay_params   = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # MuAdamW reads p.infshape (set by set_base_shapes) and multiplies each
        # layer's effective LR by base_width / layer_width, enabling zero-shot transfer.
        return MuAdamW(optim_groups, lr=learning_rate, betas=betas)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break
        return idx


def make_mup_model(config: GPTConfig) -> GPTMuP:
    """
    Construct a µP-configured GPTMuP model.

    Creates a fresh Tiny base model (n_embd=128) and calls set_base_shapes so that:
      - p.infshape is annotated on every parameter → used by MuAdamW for LR scaling
      - Parameters are rescaled to µP init: std ∝ 1/sqrt(width_mult) for "infinite" dims,
        keeping pre-activation variance ≈ O(1) as width grows.

    The base model must have the same n_layer as the target so parameter names align.
    In the wide series, all models have n_layer=4 (same as Tiny), so this always holds.
    """
    model = GPTMuP(config)
    base_config = GPTConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,   # must equal target n_layer for name alignment
        n_head=_BASE_N_HEAD,
        n_embd=_BASE_N_EMBD,
        n_ff=_BASE_N_FF,
        dropout=0.0,
        bias=config.bias,
    )
    base_model = GPTMuP(base_config)
    set_base_shapes(model, base_model, rescale_params=True)
    del base_model
    return model
