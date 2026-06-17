import torch
import torch.nn as nn
from ..tools import build_mlps
from einops import rearrange, repeat

class Prompt(nn.Module):
    def __init__(self, 
                 in_channels_traj,
                 in_channels_lane, 
                 hidden_dim, 
                 num_heads,
                 qkv_bias = False,
                 attn_drop=0.0):
        super().__init__()

        self.traj_mlp = build_mlps(
            c_in=in_channels_traj,
            mlp_channels=[hidden_dim] * 2,
            ret_before_act=True,
            layer_norm=True
        )
        self.lane_mlp = build_mlps(
            c_in = in_channels_lane,
            mlp_channels = [hidden_dim] * 2,
            ret_before_act = True,
            layer_norm=True
        )
        
        self.attn = torch.nn.MultiheadAttention(
            embed_dim = hidden_dim,
            num_heads = num_heads,
            add_bias_kv=qkv_bias,
            dropout = attn_drop,
            batch_first=True,
        )

        self.fusion_mlp = build_mlps(
            c_in=hidden_dim,
            mlp_channels=[hidden_dim] * 2,
            ret_before_act=True,
            without_norm=True,
        )

    def forward(self, query, key_pos, key_attr, key_padding_mask, mask=None):

        query = rearrange(query, 'b a f d -> b a (f d)')
        query = self.traj_mlp(query)     

        key_pos = rearrange(key_pos, 'b a f d -> b a (f d)')
        key = torch.cat([key_pos, key_attr], dim=-1)
        key = self.lane_mlp(key)         

        p_out = self.attn(
            query=query,
            key=key,
            value=key,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]

        p_out = self.fusion_mlp(p_out)     

        return p_out




