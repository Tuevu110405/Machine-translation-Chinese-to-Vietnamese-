import torch
import torch.nn as nn
import torch.nn.functional as F

from .RMSNorm import RMSNorm
from .FFN_SwiGLU import FFN_SwiGLU
from .RoPE import RoPE
from .apply_rope import apply_rope
import math
from typing import Optional, Tuple
# Lưu ý: Bạn cần đảm bảo class RopeConfig đã được định nghĩa hoặc import
# from .config import RopeConfig


class GroupedQueryAttentionRoPE(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout, rope_base):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        # Corrected: Define n_groups
        self.n_groups = self.n_heads // self.n_kv_heads
        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        self.rope = RoPE(self.d_k, base=rope_base)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        B, T_q = q.size(0), q.size(1)
        T_k = k.size(1)

        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T_k, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, T_k, self.n_kv_heads, self.d_k).transpose(1, 2)

        cos_q, sin_q = self.rope(Q, T_q)
        cos_k, sin_k = self.rope(K, T_k)

        Q = apply_rope(Q, cos_q, sin_q)
        K = apply_rope(K, cos_k, sin_k)

        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        if attn_mask is not None:
            # Corrected: Remove unsqueeze calls to allow proper broadcasting of (T, T) mask
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.W_o(out)


class EncoderLayer(nn.Module):
    def __init__(self, config: RopeConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.self_attn = GroupedQueryAttentionRoPE(
            config.d_model,
            config.n_heads,
            config.n_kv_heads,
            config.dropout,
            config.rope_base
        )
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = FFN_SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, src_pad_mask=None):
        x1 = self.ln1(x)
        attn = self.self_attn(x1, x1, x1, key_padding_mask=src_pad_mask)
        x = x + self.dropout(attn)
        x2 = self.ln2(x)
        return x + self.ffn(x2)


class DecoderLayer(nn.Module):
    def __init__(self, config: RopeConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.self_attn = GroupedQueryAttentionRoPE(
            config.d_model,
            config.n_heads,
            config.n_kv_heads,
            config.dropout,
            config.rope_base
        )
        self.ln2 = RMSNorm(config.d_model)
        self.cross_attn = GroupedQueryAttentionRoPE(
            config.d_model, config.n_heads, config.n_kv_heads, config.dropout, config.rope_base
        )
        self.ln3 = RMSNorm(config.d_model)
        self.ffn = FFN_SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, y, enc_out, tgt_pad_mask=None, tgt_casual_mask=None, src_pad_mask=None):
        y1 = self.ln1(y)
        # Use self attention with casual mask
        self_attn = self.self_attn(y1, y1, y1,
                                   key_padding_mask=tgt_pad_mask,
                                   attn_mask=tgt_casual_mask)
        y = y + self.dropout(self_attn)
        y2 = self.ln2(y)
        # Cross attention (K, V is the output of the encoder)
        cross_attn = self.cross_attn(y2, enc_out, enc_out,
                                     key_padding_mask=src_pad_mask)
        y = y + self.dropout(cross_attn)
        y3 = self.ln3(y)
        return y + self.ffn(y3)


class TransformerModel(nn.Module):
    def __init__(self, config: RopeConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=0
                                      )
        self.emb_dropout = nn.Dropout(config.dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.encoder_final_ln = RMSNorm(config.d_model)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.decoder_final_ln = RMSNorm(config.d_model)

        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.emb_scale = math.sqrt(config.d_model)
        self._init_weight()

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor):
        src_pad = (src_ids == 0)
        tgt_pad = (tgt_ids == 0)

        tgt_in = tgt_ids[:, : -1]
        tgt_pad_in = tgt_pad[:, : -1]
        T = tgt_in.size(1)
        tgt_causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), diagonal=1
        )

        src_emb = self.emb_dropout(self.embedding(src_ids) * self.emb_scale)
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_pad)
        enc_out = self.encoder_final_ln(enc_out)

        tgt_emb = self.emb_dropout(self.embedding(tgt_in) * self.emb_scale)
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_pad_in, tgt_causal, src_pad)
        dec_out = self.decoder_final_ln(dec_out)

        return F.linear(dec_out, self.embedding.weight, self.output_bias)
