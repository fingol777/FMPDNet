from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..single_embedding.utils_s import init_weights

class MLPDecoder(nn.Module):

    def __init__(self, 
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 min_scale: float = 1e-3):
        super().__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.min_scale = min_scale

        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
        )
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2),
        )
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2),
        )
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1),
        )
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor,
                data):
        batch_size, num_agents, _ = local_embed.shape
        agent_mask = ~data["x_key_padding_mask"].bool()               

        local_modes = local_embed.unsqueeze(1).expand(-1, self.num_modes, -1, -1)    
        fused = torch.cat([global_embed, local_modes], dim=-1)                       

        pi = self.pi(fused).squeeze(-1).permute(0, 2, 1).contiguous()              
        pi = pi.masked_fill(~agent_mask.unsqueeze(-1), float('-inf'))

        hidden = self.aggr_embed(fused)
        loc = self.loc(hidden).view(batch_size, self.num_modes, num_agents, self.future_steps, 2)  
        scale = F.elu_(self.scale(hidden), alpha=1.0).view(batch_size,
                                                           self.num_modes,
                                                           num_agents,
                                                           self.future_steps,
                                                           2) + 1.0 + self.min_scale
        
        loc = loc.permute(0, 2, 1, 3, 4).contiguous()                
        scale = scale.permute(0, 2, 1, 3, 4).contiguous()             
        pi = pi * agent_mask.unsqueeze(-1).float()                   
        loc = loc * agent_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        scale = scale * agent_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        return loc, pi, scale
        