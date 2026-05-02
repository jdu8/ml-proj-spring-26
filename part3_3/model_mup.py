"""
GPT with lightweight µP.

Changes from part2/model.py:
  1. Attention scale: 1/d_head  (was 1/sqrt(d_head))
  2. Plain nn.Linear for lm_head — no MuReadout, no weight tying
  3. Standard weight init — no mup-aware fan-in scaling
  4. No mup package dependency; LR scaling is handled in the train script
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg['n_head']
        self.n_embd = cfg['n_embd']
        self.c_attn = nn.Linear(cfg['n_embd'], 3 * cfg['n_embd'])
        self.c_proj = nn.Linear(cfg['n_embd'], cfg['n_embd'])
        self.drop   = nn.Dropout(cfg['dropout'])
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(cfg['block_size'], cfg['block_size']))
                  .view(1, 1, cfg['block_size'], cfg['block_size'])
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # µP change: scale by 1/d_head instead of 1/sqrt(d_head)
        att = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = nn.Linear(cfg['n_embd'], 4 * cfg['n_embd'])
        self.c_proj = nn.Linear(4 * cfg['n_embd'], cfg['n_embd'])
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg['n_embd'])
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg['n_embd'])
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MuGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg['vocab_size'], cfg['n_embd']),
            wpe  = nn.Embedding(cfg['block_size'], cfg['n_embd']),
            drop = nn.Dropout(cfg['dropout']),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg['n_layer'])]),
            ln_f = nn.LayerNorm(cfg['n_embd']),
        ))
        # µP change: plain nn.Linear — no MuReadout, no weight tying
        self.lm_head = nn.Linear(cfg['n_embd'], cfg['vocab_size'], bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos    = torch.arange(T, device=idx.device)
        x      = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg['block_size'] else idx[:, -self.cfg['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cumprobs - F.softmax(sorted_logits, dim=-1) > top_p] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break
        return idx
