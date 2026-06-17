from typing import Optional, Tuple
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..single_embedding.embedding import MultipleInputEmbedding
from ..single_embedding.embedding import SingleInputEmbedding
from ..single_embedding.utils_s import init_weights

class GlobalInteractor(nn.Module):

    def __init__(self, 
                 historical_steps: int,
                 embed_dim: int,
                 edge_dim: int,
                 num_modes: int = 6,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 rotate: bool = True) -> None:
        super().__init__()
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.rotate = rotate

        if rotate:
            self.rel_embed = MultipleInputEmbedding(in_channels=[edge_dim, edge_dim], out_channel=embed_dim)
        else:
            self.rel_embed = SingleInputEmbedding(in_channel=edge_dim, out_channel=embed_dim)
        self.layers = nn.ModuleList([
            GlobalInteractorLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        self.apply(init_weights)

    def forward(self,
                data,
                local_embed: torch.Tensor,):
        batch_size, num_agents, _ = local_embed.shape
        agent_mask = ~data["x_key_padding_mask"].bool()               

        positions = data['x_positions'][:, :, -1]                     
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(2)     

        rotate_angles = data["x_angles"][:, :, self.historical_steps-1]
        if self.rotate and rotate_angles is not None:
            sin_vals = torch.sin(rotate_angles)
            cos_vals = torch.cos(rotate_angles)
            rotate_mat = torch.stack([
                torch.stack([cos_vals, -sin_vals], dim=-1),
                torch.stack([sin_vals, cos_vals], dim=-1),
            ], dim=-2)
            rel_pos = torch.matmul(rel_pos.unsqueeze(-2), rotate_mat.unsqueeze(1)).squeeze(-2)
            rel_theta = rotate_angles.unsqueeze(1) - rotate_angles.unsqueeze(2)
            rel_theta = torch.stack([torch.cos(rel_theta), torch.sin(rel_theta)], dim=-1)    
            rel_embed = self.rel_embed([rel_pos, rel_theta])                             
        else:
            rel_embed = self.rel_embed(rel_pos)

        pair_mask = agent_mask.unsqueeze(1) & agent_mask.unsqueeze(2)
        rel_embed = rel_embed * pair_mask.unsqueeze(-1).float()

        x = local_embed * agent_mask.unsqueeze(-1).float()         
        for layer in self.layers:
            x = layer(x=x, rel_embed=rel_embed, agent_mask=agent_mask)
        x = self.norm(x) * agent_mask.unsqueeze(-1).float()               
        x = self.multihead_proj(x).view(batch_size, num_agents, self.num_modes, self.embed_dim)
        return x.permute(0, 2, 1, 3).contiguous()
    
class GlobalInteractorLayer(nn.Module):

    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                rel_embed: torch.Tensor,
                agent_mask: Optional[torch.Tensor] = None):
        batch_size, num_agents, _ = x.shape
        agent_mask = agent_mask    
        
        x_norm = self.norm1(x)
        query = self.lin_q_node(x_norm).view(batch_size, num_agents, self.num_heads, self.head_dim)
        key_node = self.lin_k_node(x_norm).view(batch_size, num_agents, self.num_heads, self.head_dim)
        value_node = self.lin_v_node(x_norm).view(batch_size, num_agents, self.num_heads, self.head_dim)

        key_edge = self.lin_k_edge(rel_embed).view(batch_size, num_agents, num_agents, self.num_heads, self.head_dim)
        value_edge = self.lin_v_edge(rel_embed).view(batch_size, num_agents, num_agents, self.num_heads, self.head_dim)

        query = query.unsqueeze(2)
        key = key_node.unsqueeze(1) + key_edge               
        value = value_node.unsqueeze(1) + value_edge          

        scale = self.head_dim ** 0.5
        scores = (query * key).sum(dim=-1) / scale

        key_mask = agent_mask.unsqueeze(1).unsqueeze(-1)            
        query_mask = agent_mask.unsqueeze(2).unsqueeze(-1)
        valid_pair_mask = key_mask & query_mask                     
        scores = scores.masked_fill(~valid_pair_mask, float('-inf'))   

        fully_masked = ~valid_pair_mask.any(dim=2, keepdim=True)      
        scores = scores.masked_fill(fully_masked, 0.0)

        attn = torch.softmax(scores, dim=2)
        attn = self.attn_drop(attn)
        attn = attn * query_mask.float()

        agg = (attn.unsqueeze(-1) * value).sum(dim=2).reshape(batch_size, num_agents, self.embed_dim)
        agg = self.out_proj(agg)
        agg = self.proj_drop(agg)        

        gate = torch.sigmoid(self.lin_ih(agg) + self.lin_hh(x_norm))
        update = agg + gate * (self.lin_self(x_norm) - agg)
        update = update * agent_mask.unsqueeze(-1).float()        

        x = x + update
        x = x + self.mlp(self.norm2(x))                
        x = x * agent_mask.unsqueeze(-1).float() 
        return x
