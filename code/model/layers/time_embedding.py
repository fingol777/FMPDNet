import math
import torch
import torch.nn as  nn

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, bias=True, max_period=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def time_embedding(t, dim, max_period=10000):
        
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        _max_period = self.max_period if self.max_period is not None else 10000
        t_freq = self.time_embedding(t, self.frequency_embedding_size, max_period=_max_period)
        t_emb = self.mlp(t_freq)
        return t_emb
