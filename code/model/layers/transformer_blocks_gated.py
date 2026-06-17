from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch import Tensor

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class HeadwiseGatedAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gate_bias: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gate_proj = nn.Linear(dim, num_heads)
        self.gate_bias = gate_bias
        self.reset_gate_bias()

    def reset_gate_bias(self) -> None:
        nn.init.constant_(self.gate_proj.bias, self.gate_bias)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def _masked_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_drop: Optional[nn.Dropout] = None,
    ) -> Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min
            )

        attn = torch.softmax(scores, dim=-1)
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :].bool(), 0.0)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        if attn_drop is not None:
            attn = attn_drop(attn)
        return torch.matmul(attn, v)

    def forward(
        self,
        query: Tensor,
        key_value: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if key_value is None:
            key_value = query

        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(key_value))
        v = self._reshape_heads(self.v_proj(key_value))

        context = self._masked_attention(
            q=q,
            k=k,
            v=v,
            key_padding_mask=key_padding_mask,
            attn_drop=self.attn_drop,
        )

        gate = torch.sigmoid(self.gate_proj(query)).permute(0, 2, 1).unsqueeze(-1)
        context = context * gate

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(query.shape[0], query.shape[1], self.dim)
        context = self.proj_drop(self.out_proj(context))

        if query_padding_mask is not None:
            context = context.masked_fill(query_padding_mask.unsqueeze(-1).bool(), 0.0)
        return context


class ActorLaneGatedAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        lane_gate_bias: float = -1.0,
        out_gate_bias: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lane_gate_proj = nn.Linear(dim, num_heads)
        self.out_gate_proj = nn.Linear(dim, num_heads)
        self.lane_gate_bias = lane_gate_bias
        self.out_gate_bias = out_gate_bias
        self.reset_gate_bias()

    def reset_gate_bias(self) -> None:
        nn.init.constant_(self.lane_gate_proj.bias, self.lane_gate_bias)
        nn.init.constant_(self.out_gate_proj.bias, self.out_gate_bias)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _context(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if key_value.shape[1] == 0:
            batch_size, query_len, _ = query.shape
            return query.new_zeros(batch_size, self.num_heads, query_len, self.head_dim)

        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(key_value))
        v = self._reshape_heads(self.v_proj(key_value))
        return HeadwiseGatedAttention._masked_attention(
            q=q,
            k=k,
            v=v,
            key_padding_mask=key_padding_mask,
            attn_drop=self.attn_drop,
        )

    def forward(
        self,
        actor_query: Tensor,
        actor_padding_mask: Optional[Tensor],
        actor_memory: Tensor,
        actor_memory_padding_mask: Optional[Tensor],
        lane_memory: Tensor,
        lane_memory_padding_mask: Optional[Tensor],
    ) -> Tensor:
        actor_context = self._context(
            query=actor_query,
            key_value=actor_memory,
            key_padding_mask=actor_memory_padding_mask,
        )
        lane_context = self._context(
            query=actor_query,
            key_value=lane_memory,
            key_padding_mask=lane_memory_padding_mask,
        )

        lane_gate = torch.sigmoid(self.lane_gate_proj(actor_query))
        lane_gate = lane_gate.permute(0, 2, 1).unsqueeze(-1)
        out_gate = torch.sigmoid(self.out_gate_proj(actor_query))
        out_gate = out_gate.permute(0, 2, 1).unsqueeze(-1)

        context = (actor_context + lane_gate * lane_context) * out_gate
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(actor_query.shape[0], actor_query.shape[1], self.dim)
        context = self.proj_drop(self.out_proj(context))

        if actor_padding_mask is not None:
            context = context.masked_fill(actor_padding_mask.unsqueeze(-1).bool(), 0.0)
        return context


class GatedEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        lane_gate_bias: float = -1.0,
        out_gate_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.actor_attn = ActorLaneGatedAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            lane_gate_bias=lane_gate_bias,
            out_gate_bias=out_gate_bias,
        )
        self.lane_attn = HeadwiseGatedAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            gate_bias=out_gate_bias,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        src: Tensor,
        num_actor_tokens: int,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src2 = self.norm1(src)
        actor_tokens = src2[:, :num_actor_tokens]
        lane_tokens = src2[:, num_actor_tokens:]

        actor_padding_mask = None
        lane_padding_mask = None
        if key_padding_mask is not None:
            actor_padding_mask = key_padding_mask[:, :num_actor_tokens]
            lane_padding_mask = key_padding_mask[:, num_actor_tokens:]

        actor_update = self.actor_attn(
            actor_query=actor_tokens,
            actor_padding_mask=actor_padding_mask,
            actor_memory=actor_tokens,
            actor_memory_padding_mask=actor_padding_mask,
            lane_memory=lane_tokens,
            lane_memory_padding_mask=lane_padding_mask,
        )

        if lane_tokens.shape[1] > 0:
            lane_update = self.lane_attn(
                query=lane_tokens,
                key_value=src2,
                query_padding_mask=lane_padding_mask,
                key_padding_mask=key_padding_mask,
            )
            attn_update = torch.cat([actor_update, lane_update], dim=1)
        else:
            attn_update = actor_update

        src = src + self.drop_path1(attn_update)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src


