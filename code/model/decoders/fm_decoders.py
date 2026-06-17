
import torch
import torch.nn as nn

import numpy as np
import copy

from einops import repeat, rearrange

from ..tools import build_mlps, create_pos_embedding
from ..layers.time_embedding import TimestepEmbedding
from ..layers.dmt_decoder_layer import DMTDecoderLayer

class FM_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cfg = config
        self.embed_dim = self.cfg.embed_dim
        self.t_embedder = TimestepEmbedding(self.embed_dim)
        self.num_decoder_layers = self.cfg.num_decoder_layers
        self.num_future_frames = self.cfg.historical_steps

        _feat_tf_encoder = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.cfg.fm_decoder_heads,
            dim_feedforward=self.embed_dim * 4, dropout=self.cfg.fm_decoder_dropout, batch_first=True
        )
        self.feat_tf_encoder = nn.TransformerEncoder(_feat_tf_encoder, num_layers=2)

        self.feat_fusion_mlp = build_mlps(c_in=self.embed_dim*2, mlp_channels=[self.embed_dim] * 2, 
                                          ret_before_act=True, without_norm=False, layer_norm=True)
        
        self.query_ca_pos_mlp = build_mlps(c_in=self.embed_dim, mlp_channels=[self.embed_dim, self.embed_dim], ret_before_act=True)
        
        fusion_decoder_layer = DMTDecoderLayer(d_model=self.embed_dim, nhead=self.cfg.fm_decoder_heads, 
                                            dim_feedforward=self.embed_dim * 4, 
                                            dropout=self.cfg.fm_decoder_dropout, activation="relu", normalize_before=False,
                                            use_concat_pe_ca=True, normalization_type='layer_norm', bias=True, qk_norm=False)
        self.fusion_decoder_layers = nn.ModuleList([copy.deepcopy(fusion_decoder_layer) for _ in range(self.num_decoder_layers)])

        agent_decoder_layer = DMTDecoderLayer(d_model=self.embed_dim, nhead=self.cfg.fm_decoder_heads, 
                                            dim_feedforward=self.embed_dim * 4, 
                                            dropout=self.cfg.fm_decoder_dropout, activation="relu", normalize_before=False,
                                            use_concat_pe_ca=True, normalization_type='layer_norm', bias=True, qk_norm=False)
        self.agent_decoder_layers = nn.ModuleList([copy.deepcopy(agent_decoder_layer) for _ in range(self.num_decoder_layers)])

        map_decoder_layer = DMTDecoderLayer(d_model=self.embed_dim, nhead=self.cfg.fm_decoder_heads, 
                                            dim_feedforward=self.embed_dim * 4, 
                                            dropout=self.cfg.fm_decoder_dropout, activation="relu", normalize_before=False,
                                            use_concat_pe_ca=True, normalization_type='layer_norm', bias=True, qk_norm=False)
        self.map_decoder_layers = nn.ModuleList([copy.deepcopy(map_decoder_layer) for _ in range(self.num_decoder_layers)])

        if self.cfg.fusion_trans_decode:
            temp_layer = build_mlps(c_in=self.embed_dim * 2, mlp_channels=[self.embed_dim, self.embed_dim], ret_before_act=True)
            self.query_feature_fusion_layers = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
        else:
            temp_layer = build_mlps(c_in=self.embed_dim * 3, mlp_channels=[self.embed_dim, self.embed_dim], ret_before_act=True)
            self.query_feature_fusion_layers = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])

        motion_reg_head =  build_mlps(c_in=self.embed_dim, mlp_channels=[self.embed_dim, self.embed_dim, self.num_future_frames * 4], ret_before_act=True)
        motion_cls_head =  build_mlps(c_in=self.embed_dim, mlp_channels=[self.embed_dim, self.embed_dim, 1], ret_before_act=True)
        self.motion_reg_heads = nn.ModuleList([copy.deepcopy(motion_reg_head) for _ in range(self.num_decoder_layers)])
        self.motion_cls_heads = nn.ModuleList([copy.deepcopy(motion_cls_head) for _ in range(self.num_decoder_layers)])

    def forward(self, y_t_in, fut_traj_embed, short_traj_context_embed, lane_embed, context_pos_embed, t, data):

        B, A, D = fut_traj_embed.shape
        M = lane_embed.shape[1]

        t_embed = self.t_embedder(t)                                       
        t_embed = repeat(t_embed, 'b d -> b a d', a=A)                                 

        query_y_emb = fut_traj_embed                                             
        query_y_pos_embed = context_pos_embed[:, :A, :]                       

        fusion_embed = self.feat_fusion_mlp(torch.cat((query_y_emb, t_embed), dim=-1))   
        center_object_features = self.feat_tf_encoder(fusion_embed)                     

        
        context_agent_feat = short_traj_context_embed          
        context_map_feat = lane_embed                          

        query_mask_agent = ~data["x_key_padding_mask"]          
        context_mask_agent = ~data["x_key_padding_mask"]        
        context_mask_map = ~data["lane_key_padding_mask"]       

        context_agent_pos_embed = context_pos_embed[:, :A, :]                   
        context_map_pos_embed = context_pos_embed[:, -M:, :]                     


        pred_list = []
        query_content = center_object_features                                   
        for layer_idx in range(self.num_decoder_layers):
            agent_query_feature = self.agent_decoder_layers[layer_idx](
                query = query_content,
                context = context_agent_feat,
                query_valid_mask = query_mask_agent,
                context_valid_mask = context_mask_agent,
                query_sa_pos_embeddings = query_y_pos_embed,
                query_ca_pos_embeddings = query_y_pos_embed,
                context_ca_pos_embeddings = context_agent_pos_embed,
                is_first = layer_idx==0,
            )       

            map_query_feature = self.map_decoder_layers[layer_idx](
                query = query_content,
                context = context_map_feat,
                query_valid_mask = query_mask_agent,
                context_valid_mask = context_mask_map,
                query_sa_pos_embeddings = query_y_pos_embed,
                query_ca_pos_embeddings = query_y_pos_embed,
                context_ca_pos_embeddings = context_map_pos_embed,
                is_first = layer_idx==0,
            )       

            query_feature = torch.cat([center_object_features, agent_query_feature, map_query_feature], dim=-1)

            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
            ).view(B, A, -1)                                        

            query_content_t = query_content.view(B * A, -1)
            pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(B, A, self.num_future_frames, 4)   
            pred_list.append(pred_trajs)

        model_out = pred_list[-1]         
        model_out = rearrange(model_out, 'b a f d -> b a (f d)')  

        return model_out

