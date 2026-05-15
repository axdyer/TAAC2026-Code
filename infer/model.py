"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    timestamp: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    seq_recency_stats: dict  # {domain: tensor [B, R]}


SUPPORTED_POS_USER_PAIR_FIDS = {62, 63, 64, 65, 66}
SUPPORTED_SIGNED_USER_PAIR_FIDS = {89, 90, 91}
SUPPORTED_USER_PAIR_FIDS = (
    SUPPORTED_POS_USER_PAIR_FIDS | SUPPORTED_SIGNED_USER_PAIR_FIDS
)
USER_DENSE_EMB_FID = 61
USER_DENSE_INTEREST_FID = 87
USER_DENSE_INTEREST_BLOCK_DIM = 32

SAMPLE_TIME_DENSE_DIM = 6
SAMPLE_TIME_RECENCY_BOUNDARIES = (
    600,       # 10 minutes
    1800,      # 30 minutes
    3600,      # 1 hour
    7200,      # 2 hours
    14400,     # 4 hours
    28800,     # 8 hours
    43200,     # 12 hours
    64800,     # 18 hours
    86400,     # 1 day
    129600,    # 1.5 days
    172800,    # 2 days
    259200,    # 3 days
    345600,    # 4 days
    432000,    # 5 days
)


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        time_attn_bias: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            time_attn_bias: Optional additive attention bias with shape
                broadcastable to (B, num_heads, Lq, Lk).
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask / attn_mask / additive bias to SDPA format
        bool_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            bool_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            bool_attn_mask = bool_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if bool_attn_mask is not None:
                bool_attn_mask = bool_attn_mask & bool_attn
            else:
                bool_attn_mask = bool_attn

        if time_attn_bias is not None:
            if time_attn_bias.dim() != 4:
                raise ValueError(
                    f"time_attn_bias must have 4 dims, got shape {tuple(time_attn_bias.shape)}")
            if time_attn_bias.shape[0] != B or time_attn_bias.shape[1] != self.num_heads:
                raise ValueError(
                    f"time_attn_bias shape {tuple(time_attn_bias.shape)} is incompatible "
                    f"with batch={B}, num_heads={self.num_heads}")
            if time_attn_bias.shape[2] not in (1, Lq) or time_attn_bias.shape[3] != Lk:
                raise ValueError(
                    f"time_attn_bias shape {tuple(time_attn_bias.shape)} must be "
                    f"broadcastable to (B,H,Lq,Lk)=({B},{self.num_heads},{Lq},{Lk})")
            sdpa_attn_mask = time_attn_bias.to(dtype=Q.dtype)
            if time_attn_bias.shape[2] == 1 and Lq != 1:
                sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)
            if bool_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask.masked_fill(
                    ~bool_attn_mask,
                    torch.finfo(Q.dtype).min,
                )
        else:
            sdpa_attn_mask = bool_attn_mask

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        time_attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            time_attn_bias=time_attn_bias,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        time_attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            time_attn_bias=time_attn_bias,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        time_attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full'
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
        time_attn_bias_list: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            tb = time_attn_bias_list[i] if time_attn_bias_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs, time_attn_bias=tb,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            tb = time_attn_bias_list[i] if time_attn_bias_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs, time_attn_bias=tb,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        domain_time_buckets: bool = False,
        use_sample_time_token: bool = False,
        sample_time_ref_ts: int = 0,
        sample_time_timezone_offset_hours: int = 8,
        use_domain_recency_fusion: bool = False,
        recency_stats_dim: int = 0,
        use_time_attention_bias: bool = False,
        time_attention_bias_domains: Optional[List[str]] = None,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        # User dense-derived tokens. fid 61 is a single user embedding token;
        # fid 87 is treated as 10x32 multi-interest embeddings and pooled into
        # a configurable number of tokens; 62-66 / 89-91 pair features are
        # modeled as separate positive/signed tokens.
        user_feat_pair: Optional[List[int]] = None,
        user_dense_87_tokens: int = 2,
        user_int_feature_ids: Optional[List[int]] = None,
        user_dense_feature_specs: Optional[List[Tuple[int, int, int]]] = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.domain_time_buckets = domain_time_buckets
        self.use_sample_time_token = use_sample_time_token
        self.use_domain_recency_fusion = bool(use_domain_recency_fusion)
        self.recency_stats_dim = int(recency_stats_dim)
        self.use_time_attention_bias = bool(use_time_attention_bias)
        self.sample_time_ref_ts = int(sample_time_ref_ts)
        self.sample_time_timezone_offset_seconds = (
            int(sample_time_timezone_offset_hours) * 3600
        )
        if self.domain_time_buckets and self.num_time_buckets <= 0:
            raise ValueError("domain_time_buckets=True requires num_time_buckets > 0")
        if self.use_sample_time_token and self.sample_time_ref_ts <= 0:
            raise ValueError(
                "use_sample_time_token=True requires sample_time_ref_ts to be "
                "a positive Unix timestamp")
        if self.use_domain_recency_fusion and self.recency_stats_dim <= 0:
            raise ValueError(
                "use_domain_recency_fusion=True requires recency_stats_dim > 0")
        if self.use_time_attention_bias:
            if self.num_time_buckets <= 0:
                raise ValueError("use_time_attention_bias=True requires num_time_buckets > 0")
            if seq_encoder_type == 'longer':
                raise ValueError(
                    "use_time_attention_bias=True is not implemented for "
                    "seq_encoder_type='longer' because sequence compression "
                    "changes the key length across blocks")
        if time_attention_bias_domains:
            unknown_bias_domains = sorted(
                set(time_attention_bias_domains) - set(self.seq_domains))
            if unknown_bias_domains:
                raise ValueError(
                    f"time_attention_bias_domains contains unknown domains "
                    f"{unknown_bias_domains}; known domains={self.seq_domains}")
            self.time_attention_bias_domains = list(time_attention_bias_domains)
        else:
            self.time_attention_bias_domains = list(self.seq_domains)
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.user_feat_pair = list(user_feat_pair or [])
        self.user_dense_87_tokens = int(user_dense_87_tokens)
        if self.user_dense_87_tokens < 0:
            raise ValueError(f"user_dense_87_tokens must be >= 0, got {user_dense_87_tokens}")
        if len(set(self.user_feat_pair)) != len(self.user_feat_pair):
            raise ValueError(f"user_feat_pair contains duplicate fids: {self.user_feat_pair}")
        self.has_user_feat_pair = len(self.user_feat_pair) > 0
        self.user_pos_feat_pair = [
            fid for fid in self.user_feat_pair
            if fid in SUPPORTED_POS_USER_PAIR_FIDS
        ]
        self.user_signed_feat_pair = [
            fid for fid in self.user_feat_pair
            if fid in SUPPORTED_SIGNED_USER_PAIR_FIDS
        ]
        self.has_user_pos_pair = len(self.user_pos_feat_pair) > 0
        self.has_user_signed_pair = len(self.user_signed_feat_pair) > 0

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense-derived tokens (if available)
        self.has_user_dense = user_dense_dim > 0
        self.num_user_dense_tokens = 0
        self.user_dense_61_spec: Optional[Tuple[int, int]] = None
        self.user_dense_87_spec: Optional[Tuple[int, int]] = None
        self.user_dense_87_blocks = 0
        if self.has_user_dense:
            if user_dense_feature_specs is None:
                raise ValueError("user_dense_feature_specs must be provided when user_dense_dim > 0")
            dense_by_fid = {
                int(fid): (int(offset), int(length))
                for fid, offset, length in user_dense_feature_specs
            }
            if USER_DENSE_EMB_FID not in dense_by_fid:
                raise ValueError(f"user dense fid {USER_DENSE_EMB_FID} not found in user_dense schema")
            if USER_DENSE_INTEREST_FID not in dense_by_fid:
                raise ValueError(f"user dense fid {USER_DENSE_INTEREST_FID} not found in user_dense schema")

            dense61_offset, dense61_len = dense_by_fid[USER_DENSE_EMB_FID]
            if dense61_len != 256:
                raise ValueError(
                    f"user dense fid {USER_DENSE_EMB_FID} expected length 256, got {dense61_len}"
                )
            self.user_dense_61_spec = (dense61_offset, dense61_len)
            self.user_dense_61_proj = nn.Sequential(
                nn.Linear(dense61_len, d_model),
                nn.LayerNorm(d_model),
            )
            self.num_user_dense_tokens += 1

            dense87_offset, dense87_len = dense_by_fid[USER_DENSE_INTEREST_FID]
            if dense87_len % USER_DENSE_INTEREST_BLOCK_DIM != 0:
                raise ValueError(
                    f"user dense fid {USER_DENSE_INTEREST_FID} length {dense87_len} "
                    f"must be divisible by {USER_DENSE_INTEREST_BLOCK_DIM}"
                )
            self.user_dense_87_spec = (dense87_offset, dense87_len)
            self.user_dense_87_blocks = dense87_len // USER_DENSE_INTEREST_BLOCK_DIM
            if self.user_dense_87_tokens > 0:
                self.user_dense_87_block_proj = nn.Sequential(
                    nn.Linear(USER_DENSE_INTEREST_BLOCK_DIM, d_model),
                    nn.LayerNorm(d_model),
                )
                self.user_dense_87_queries = nn.Parameter(
                    torch.empty(self.user_dense_87_tokens, d_model)
                )
                self.user_dense_87_out_norm = nn.LayerNorm(d_model)
                self.num_user_dense_tokens += self.user_dense_87_tokens

        # ================== User Int/Dense Pair Tokens ==================
        self.user_pos_pair_specs: List[Tuple[int, int, int, int, int]] = []
        self.user_signed_pair_specs: List[Tuple[int, int, int, int, int]] = []
        if self.has_user_feat_pair:
            if not self.has_user_dense:
                raise ValueError("user_feat_pair requires user_dense_dim > 0")
            unsupported = [
                fid for fid in self.user_feat_pair
                if fid not in SUPPORTED_USER_PAIR_FIDS
            ]
            if unsupported:
                raise ValueError(
                    "user_feat_pair supports only positive long-tail fids "
                    f"{sorted(SUPPORTED_POS_USER_PAIR_FIDS)} and signed fids "
                    f"{sorted(SUPPORTED_SIGNED_USER_PAIR_FIDS)}; got unsupported "
                    f"fids {unsupported}."
                )
            if user_int_feature_ids is None:
                raise ValueError("user_int_feature_ids must be provided when user_feat_pair is non-empty")
            if len(user_int_feature_ids) != len(user_int_feature_specs):
                raise ValueError(
                    "user_int_feature_ids length must match user_int_feature_specs "
                    f"length, got {len(user_int_feature_ids)} vs {len(user_int_feature_specs)}"
                )

            int_idx_by_fid = {int(fid): i for i, fid in enumerate(user_int_feature_ids)}

            def _build_pair_branch(
                fids: List[int],
            ) -> Tuple[
                List[Tuple[int, int, int, int, int]],
                List[nn.Embedding],
                List[nn.LayerNorm],
            ]:
                branch_specs: List[Tuple[int, int, int, int, int]] = []
                branch_embs: List[nn.Embedding] = []
                branch_norms: List[nn.LayerNorm] = []
                for fid in fids:
                    if fid not in int_idx_by_fid:
                        raise ValueError(f"user_feat_pair fid {fid} not found in user_int schema")
                    if fid not in dense_by_fid:
                        raise ValueError(f"user_feat_pair fid {fid} not found in user_dense schema")
                    vs, int_offset, int_len = user_int_feature_specs[int_idx_by_fid[fid]]
                    dense_offset, dense_len = dense_by_fid[fid]
                    if int_len != dense_len:
                        raise ValueError(
                            f"user_feat_pair fid {fid} int length {int_len} "
                            f"!= dense length {dense_len}"
                        )
                    if int(vs) <= 0:
                        raise ValueError(f"user_feat_pair fid {fid} has invalid vocab_size={vs}")
                    branch_specs.append((
                        int(fid), int(vs), int(int_offset), int(dense_offset), int(int_len)
                    ))
                    # +2 because 0 is reserved for padding and raw id=0 is kept
                    # as a valid category via shifted_ids = raw_id + 1.
                    branch_embs.append(nn.Embedding(int(vs) + 2, emb_dim, padding_idx=0))
                    branch_norms.append(nn.LayerNorm(emb_dim))
                return branch_specs, branch_embs, branch_norms

            pos_pair_embs: List[nn.Embedding] = []
            pos_pair_norms: List[nn.LayerNorm] = []
            signed_pair_embs: List[nn.Embedding] = []
            signed_pair_norms: List[nn.LayerNorm] = []
            if self.has_user_pos_pair:
                self.user_pos_pair_specs, pos_pair_embs, pos_pair_norms = (
                    _build_pair_branch(self.user_pos_feat_pair)
                )
                self.user_pos_pair_embs = nn.ModuleList(pos_pair_embs)
                self.user_pos_pair_norms = nn.ModuleList(pos_pair_norms)
                self.user_pos_pair_proj = nn.Sequential(
                    nn.Linear(len(self.user_pos_pair_specs) * emb_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                self.num_user_dense_tokens += 1

            if self.has_user_signed_pair:
                self.user_signed_pair_specs, signed_pair_embs, signed_pair_norms = (
                    _build_pair_branch(self.user_signed_feat_pair)
                )
                self.user_signed_pair_embs = nn.ModuleList(signed_pair_embs)
                self.user_signed_pair_norms = nn.ModuleList(signed_pair_norms)
                self.user_signed_pair_proj = nn.Sequential(
                    nn.Linear(len(self.user_signed_pair_specs) * emb_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                self.num_user_dense_tokens += 1

            logging.info(
                f"User int/dense pair tokens enabled for fids={self.user_feat_pair}; "
                f"positive_branch={self.user_pos_feat_pair}, "
                f"signed_branch={self.user_signed_feat_pair}"
            )

        if self.has_user_dense:
            logging.info(
                f"User dense split enabled: fid61_tokens=1, "
                f"fid87_tokens={self.user_dense_87_tokens}, "
                f"pos_pair_tokens={1 if self.has_user_pos_pair else 0}, "
                f"signed_pair_tokens={1 if self.has_user_signed_pair else 0}, "
                f"total_user_dense_tokens={self.num_user_dense_tokens}"
            )

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Sample-level timestamp token. This is separate from sequence
        # recency buckets: it models the request/exposure time itself.
        if self.use_sample_time_token:
            self.sample_time_dense_proj = nn.Sequential(
                nn.Linear(SAMPLE_TIME_DENSE_DIM, d_model),
                nn.LayerNorm(d_model),
            )
            self.sample_time_recency_embedding = nn.Embedding(
                len(SAMPLE_TIME_RECENCY_BOUNDARIES) + 1, d_model
            )
            self.sample_time_norm = nn.LayerNorm(d_model)
            self.register_buffer(
                'sample_time_recency_boundaries',
                torch.tensor(SAMPLE_TIME_RECENCY_BOUNDARIES, dtype=torch.long),
            )

        # Total NS token count
        self.num_ns = (num_user_ns + self.num_user_dense_tokens
                       + (1 if self.use_sample_time_token else 0)
                       + num_item_ns + (1 if self.has_item_dense else 0))

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            if self.domain_time_buckets:
                self.time_embeddings = nn.ModuleDict({
                    domain: nn.Embedding(num_time_buckets, d_model, padding_idx=0)
                    for domain in self.seq_domains
                })
            else:
                self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        if self.use_time_attention_bias:
            self.time_attention_bias_embeddings = nn.ModuleDict({
                domain: nn.Embedding(num_time_buckets, num_heads, padding_idx=0)
                for domain in self.time_attention_bias_domains
            })

        # Explicit per-domain recency statistics can modulate that domain's
        # sequence tokens without adding extra global tokens.
        if self.use_domain_recency_fusion:
            self.domain_recency_projs = nn.ModuleDict({
                domain: nn.Linear(self.recency_stats_dim, 2 * d_model)
                for domain in self.seq_domains
            })
            self.domain_recency_norms = nn.ModuleDict({
                domain: nn.LayerNorm(d_model)
                for domain in self.seq_domains
            })
            self.domain_recency_dropout = nn.Dropout(dropout_rate)

        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.has_user_pos_pair:
            for emb in self.user_pos_pair_embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.has_user_signed_pair:
            for emb in self.user_signed_pair_embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.has_user_dense and self.user_dense_87_tokens > 0:
            nn.init.xavier_normal_(self.user_dense_87_queries.data)

        if self.num_time_buckets > 0:
            if self.domain_time_buckets:
                for emb in self.time_embeddings.values():
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
            else:
                nn.init.xavier_normal_(self.time_embedding.weight.data)
                self.time_embedding.weight.data[0, :] = 0

        if self.use_time_attention_bias:
            for emb in self.time_attention_bias_embeddings.values():
                nn.init.zeros_(emb.weight.data)

        if self.use_sample_time_token:
            nn.init.xavier_normal_(self.sample_time_recency_embedding.weight.data)

        if self.use_domain_recency_fusion:
            for proj in self.domain_recency_projs.values():
                nn.init.zeros_(proj.weight.data)
                nn.init.zeros_(proj.bias.data)

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        if self.has_user_pos_pair:
            for emb, (fid, vs, _, _, _) in zip(self.user_pos_pair_embs, self.user_pos_pair_specs):
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        if self.has_user_signed_pair:
            for emb, (fid, vs, _, _, _) in zip(self.user_signed_pair_embs, self.user_signed_pair_specs):
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time embeddings are always preserved
        if self.num_time_buckets > 0:
            skip_count += len(self.seq_domains) if self.domain_time_buckets else 1
        if self.use_sample_time_token:
            skip_count += 1

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _pool_user_pair_branch(
        self,
        user_int_feats: torch.Tensor,
        user_dense_feats: torch.Tensor,
        pair_embs: nn.ModuleList,
        pair_norms: nn.ModuleList,
        pair_specs: List[Tuple[int, int, int, int, int]],
        signed: bool,
    ) -> torch.Tensor:
        """Build a branch token from aligned user int/dense fids."""
        pair_vecs = []
        for emb, norm, (_, _, int_offset, dense_offset, length) in zip(pair_embs, pair_norms, pair_specs):
            ids = user_int_feats[:, int_offset:int_offset + length].long()
            vals = user_dense_feats[:, dense_offset:dense_offset + length].float()
            if signed:
                valid = vals != 0
                weights = vals.clamp(min=-10.0, max=10.0).unsqueeze(-1)
                weights = weights * valid.float().unsqueeze(-1)
                denom = weights.abs().sum(dim=1).clamp_min(1.0)
            else:
                valid = vals > 0
                weights = torch.log1p(vals.clamp_min(0.0)).unsqueeze(-1)
                weights = weights * valid.float().unsqueeze(-1)
                denom = weights.sum(dim=1).clamp_min(1.0)
            shifted_ids = ids + 1
            shifted_ids = torch.where(valid, shifted_ids, torch.zeros_like(shifted_ids))
            emb_all = emb(shifted_ids)  # (B, L, E)
            pair_vec = (emb_all * weights).sum(dim=1) / denom
            pair_vecs.append(norm(pair_vec))
        pair_concat = torch.cat(pair_vecs, dim=-1)
        if signed:
            return self.user_signed_pair_proj(pair_concat)
        return self.user_pos_pair_proj(pair_concat)

    def _embed_user_pos_feat_pairs(
        self,
        user_int_feats: torch.Tensor,
        user_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Build a positive long-tail pair token for fids 62-66."""
        return self._pool_user_pair_branch(
            user_int_feats=user_int_feats,
            user_dense_feats=user_dense_feats,
            pair_embs=self.user_pos_pair_embs,
            pair_norms=self.user_pos_pair_norms,
            pair_specs=self.user_pos_pair_specs,
            signed=False,
        )

    def _embed_user_signed_feat_pairs(
        self,
        user_int_feats: torch.Tensor,
        user_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Build a signed pair token for fids 89-91."""
        return self._pool_user_pair_branch(
            user_int_feats=user_int_feats,
            user_dense_feats=user_dense_feats,
            pair_embs=self.user_signed_pair_embs,
            pair_norms=self.user_signed_pair_norms,
            pair_specs=self.user_signed_pair_specs,
            signed=True,
        )

    def _make_user_dense_61_token(
        self,
        user_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Project fid 61, a single 256D user embedding, into one NS token."""
        if self.user_dense_61_spec is None:
            raise RuntimeError("user_dense_61_spec is not initialized")
        offset, length = self.user_dense_61_spec
        x61 = user_dense_feats[:, offset:offset + length].float()
        return F.silu(self.user_dense_61_proj(x61)).unsqueeze(1)

    def _make_user_dense_87_tokens(
        self,
        user_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Pool fid 87 10x32 multi-interest embeddings into N NS tokens."""
        if self.user_dense_87_spec is None:
            raise RuntimeError("user_dense_87_spec is not initialized")
        offset, length = self.user_dense_87_spec
        B = user_dense_feats.shape[0]
        x87 = user_dense_feats[:, offset:offset + length].float()
        x87 = x87.view(B, self.user_dense_87_blocks, USER_DENSE_INTEREST_BLOCK_DIM)
        mask = x87.norm(dim=-1) > 1e-6  # (B, K), all-zero blocks are padding

        block_tokens = F.silu(self.user_dense_87_block_proj(x87))  # (B, K, D)
        block_tokens = block_tokens * mask.unsqueeze(-1).to(block_tokens.dtype)

        scores = torch.einsum(
            'td,bkd->btk', self.user_dense_87_queries, block_tokens
        ) / math.sqrt(self.d_model)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * mask.unsqueeze(1).to(attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        tokens = torch.matmul(attn, block_tokens)  # (B, N87, D)
        return F.silu(self.user_dense_87_out_norm(tokens))

    def _make_user_dense_tokens(
        self,
        user_int_feats: torch.Tensor,
        user_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Build all dense-derived user NS tokens."""
        dense_parts = [self._make_user_dense_61_token(user_dense_feats)]
        if self.user_dense_87_tokens > 0:
            dense_parts.append(self._make_user_dense_87_tokens(user_dense_feats))
        if self.has_user_pos_pair:
            pos_pair_token = self._embed_user_pos_feat_pairs(
                user_int_feats, user_dense_feats)
            dense_parts.append(F.silu(pos_pair_token).unsqueeze(1))
        if self.has_user_signed_pair:
            signed_pair_token = self._embed_user_signed_feat_pairs(
                user_int_feats, user_dense_feats)
            dense_parts.append(F.silu(signed_pair_token).unsqueeze(1))
        return torch.cat(dense_parts, dim=1)

    def _make_sample_time_token(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Build one NS token from the sample request/exposure timestamp.

        The recency bucket uses ``max(sample_time_ref_ts - timestamp, 0)`` so
        test rows after the training cutoff land in the newest bucket instead
        of an unseen negative-age region.
        """
        ts_long = timestamps.long()
        local_ts = ts_long + self.sample_time_timezone_offset_seconds
        seconds_in_day = torch.remainder(local_ts, 86400).float()
        hour_idx = torch.div(seconds_in_day, 3600, rounding_mode='floor')
        day_idx = torch.div(local_ts, 86400, rounding_mode='floor')
        day_of_week = torch.remainder(day_idx, 7).float()

        minute_phase = seconds_in_day / 86400.0
        hour_phase = hour_idx / 24.0
        week_phase = day_of_week / 7.0
        two_pi = 2.0 * math.pi
        dense_feats = torch.stack([
            torch.sin(two_pi * minute_phase),
            torch.cos(two_pi * minute_phase),
            torch.sin(two_pi * hour_phase),
            torch.cos(two_pi * hour_phase),
            torch.sin(two_pi * week_phase),
            torch.cos(two_pi * week_phase),
        ], dim=-1)

        age_seconds = torch.clamp(self.sample_time_ref_ts - ts_long, min=0)
        recency_bucket = torch.bucketize(
            age_seconds,
            self.sample_time_recency_boundaries.to(age_seconds.device),
        )
        time_dense = self.sample_time_dense_proj(dense_feats)
        time_recency = self.sample_time_recency_embedding(recency_bucket)
        time_token = self.sample_time_norm(time_dense + time_recency)
        return F.silu(time_token).unsqueeze(1)

    def _embed_seq_domain(
        self,
        domain: str,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        recency_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            if self.domain_time_buckets:
                token_emb = token_emb + self.time_embeddings[domain](time_bucket_ids)
            else:
                token_emb = token_emb + self.time_embedding(time_bucket_ids)

        if self.use_domain_recency_fusion:
            if recency_stats is None:
                raise KeyError(
                    f"use_domain_recency_fusion=True requires recency stats for {domain}")
            fused = self.domain_recency_projs[domain](recency_stats.float())
            gate, delta = fused.chunk(2, dim=-1)
            gate = torch.sigmoid(gate).unsqueeze(1)
            delta = F.silu(delta).unsqueeze(1)
            delta = self.domain_recency_dropout(delta)
            token_emb = self.domain_recency_norms[domain](token_emb + gate * delta)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _make_time_attention_bias(
        self,
        domain: str,
        time_bucket_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_time_attention_bias:
            return None
        if domain not in self.time_attention_bias_embeddings:
            return None
        # (B, L, H) -> (B, H, 1, L), a key-side additive bias shared by all queries.
        bias = self.time_attention_bias_embeddings[domain](time_bucket_ids)
        return bias.permute(0, 2, 1).unsqueeze(2)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        time_attn_bias_list: Optional[List[Optional[torch.Tensor]]] = None,
        apply_dropout: bool = True
    ) -> torch.Tensor:
        """Runs the multi-sequence block stack with dropout and output projection."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
                time_attn_bias_list=time_attn_bias_list,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens: grouped projection
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)   # (B, num_user_groups, D)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)   # (B, num_item_groups, D)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tokens = self._make_user_dense_tokens(
                inputs.user_int_feats, inputs.user_dense_feats)  # (B, Ndense, D)
            ns_parts.append(user_dense_tokens)
        if self.use_sample_time_token:
            ns_parts.append(self._make_sample_time_token(inputs.timestamp))
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)  # (B, 1, D)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        time_attn_bias_list = [] if self.use_time_attention_bias else None
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                domain,
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_recency_stats[domain] if self.use_domain_recency_fusion else None)
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)
            if time_attn_bias_list is not None:
                time_attn_bias_list.append(
                    self._make_time_attention_bias(
                        domain,
                        inputs.seq_time_buckets[domain],
                    )
                )

        # 3. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        # 4. Dropout + MultiSeqHyFormerBlock stack + output projection
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            time_attn_bias_list=time_attn_bias_list,
            apply_dropout=self.training
        )

        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tokens = self._make_user_dense_tokens(
                inputs.user_int_feats, inputs.user_dense_feats)
            ns_parts.append(user_dense_tokens)
        if self.use_sample_time_token:
            ns_parts.append(self._make_sample_time_token(inputs.timestamp))
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        time_attn_bias_list = [] if self.use_time_attention_bias else None
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                domain,
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_recency_stats[domain] if self.use_domain_recency_fusion else None)
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)
            if time_attn_bias_list is not None:
                time_attn_bias_list.append(
                    self._make_time_attention_bias(
                        domain,
                        inputs.seq_time_buckets[domain],
                    )
                )

        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            time_attn_bias_list=time_attn_bias_list,
            apply_dropout=False
        )

        logits = self.clsfier(output)
        return logits, output
