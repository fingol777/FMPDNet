import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False, layer_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k+1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=False))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
            else:
                if layer_norm:
                    layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.LayerNorm(mlp_channels[k]), nn.ReLU()])
                else:
                    layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)
    

def create_pos_embedding(max_k, d_model):
    pe = torch.zeros(max_k, d_model)
    position = torch.arange(0, max_k, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * 
        (-math.log(10000.0) / d_model)
    )
        
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

def restore_history_trajectory(differenced_history_traj, initial_position):

    restored_history_traj = torch.zeros_like(differenced_history_traj)
    restored_history_traj[:, 0, :] = initial_position 
    
    for t in range(1, differenced_history_traj.shape[1]):
        restored_history_traj[:, t, :] = restored_history_traj[:, t - 1, :] + differenced_history_traj[:, t, :]
    
    return restored_history_traj

def restore_future_trajectory(differenced_future_traj, initial_position, num_steps=30):

    restored_future_traj = torch.zeros_like(differenced_future_traj)
    
    for k in range(differenced_future_traj.shape[1]): 
        for t in range(num_steps):
            restored_future_traj[:, k, t, :] = differenced_future_traj[:, k, t, :] + initial_position

    return restored_future_traj


