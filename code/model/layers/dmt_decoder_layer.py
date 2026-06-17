from typing import Optional
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class DMTDecoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 use_concat_pe_ca=True,
                 normalization_type="layer_norm", 
                 bias=True, 
                 qk_norm=False):
        super().__init__()

        self.nhead = nhead
        self.dropout_val = dropout
        self.normalize_before = normalize_before
        self.use_concat_pe_ca = use_concat_pe_ca
        self.normalization_type = normalization_type
        self.qk_norm = qk_norm

        if normalization_type == 'layer_norm':
            norm_layer = nn.LayerNorm(d_model, elementwise_affine=bias)
        else:
            norm_layer = nn.BatchNorm1d(d_model)


        self.sa_qcontent_proj = nn.Linear(d_model, d_model, bias=bias)
        self.sa_qpos_proj = nn.Linear(d_model, d_model, bias=bias)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model, bias=bias)
        self.sa_kpos_proj = nn.Linear(d_model, d_model, bias=bias)
        self.sa_v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.sa_o_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_norm:
            self.sa_q_norm = deepcopy(norm_layer)
            self.sa_k_norm = deepcopy(norm_layer)

        self.norm1 = deepcopy(norm_layer)
        self.dropout1 = nn.Dropout(dropout)


        self.ca_qcontent_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_qpos_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_kpos_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_o_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_norm:
            d_model_ca = d_model * 2 if use_concat_pe_ca else d_model
            if normalization_type == 'layer_norm':
                qk_norm_layer = nn.LayerNorm(d_model_ca, bias=bias)
            else:
                qk_norm_layer = nn.BatchNorm1d(d_model_ca)
            self.ca_q_norm = deepcopy(qk_norm_layer)
            self.ca_k_norm = deepcopy(qk_norm_layer)


        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm2 = deepcopy(norm_layer)
        self.norm3 = deepcopy(norm_layer)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self,
                query,
                context,
                query_valid_mask=None,
                context_valid_mask=None,
                query_sa_pos_embeddings=None,
                query_ca_pos_embeddings=None,
                context_ca_pos_embeddings=None,
                is_causal=False,
                is_first=False):
        
        B, A, D = query.shape
        M = context.shape[1]

        sa_q_content = self.sa_qcontent_proj(query)
        sa_q_pos = self.sa_qpos_proj(query_sa_pos_embeddings)
        sa_k_content = self.sa_kcontent_proj(query)
        sa_k_pos = self.sa_kpos_proj(query_sa_pos_embeddings)
        sa_v = self.sa_v_proj(query)

        sa_q = sa_q_content + sa_q_pos
        sa_k = sa_k_content + sa_k_pos

        if self.qk_norm:
            sa_q = self.sa_q_norm(sa_q)
            sa_k = self.sa_k_norm(sa_k)

        sa_q = sa_q.view(B, A, self.nhead, D // self.nhead).transpose(1, 2)
        sa_k = sa_k.view(B, A, self.nhead, D // self.nhead).transpose(1, 2)
        sa_v = sa_v.view(B, A, self.nhead, D // self.nhead).transpose(1, 2)

        output_sa = torch.nn.functional.scaled_dot_product_attention(sa_q, sa_k, sa_v,
                                                                is_causal=is_causal,
                                                                dropout_p=self.dropout_val if self.training else 0.0)
        output_sa = output_sa.transpose(1, 2).contiguous().view(B, A, D)
        output_sa = self.sa_o_proj(output_sa)

        query = query + self.dropout1(output_sa)
        query = self.norm1(query)


        ca_q_content = self.ca_qcontent_proj(query)                       
        ca_q_pos = self.ca_qpos_sine_proj(query_ca_pos_embeddings)       

        if len(context_valid_mask.shape) == 3:
            assert context_valid_mask.shape[1] == A

            context_valid_mask_ = context_valid_mask.any(dim=1)
            context_invalid_mask_ = torch.logical_not(context_valid_mask_)
            context_invalid_mask_ = context_invalid_mask_.unsqueeze(-1)     

            ca_k_content = self.ca_kcontent_proj(context)
            ca_k_content = torch.masked_fill(ca_k_content, context_invalid_mask_, 0.0)    

            ca_v = self.ca_v_proj(context)
            ca_v = torch.masked_fill(ca_v, context_invalid_mask_, 0.0)

            ca_k_pos = self.ca_kpos_proj(context_ca_pos_embeddings)
            ca_k_pos = torch.masked_fill(ca_k_pos, context_invalid_mask_, 0.0)
        elif len(context_valid_mask.shape) == 2:
            valid_context = context[context_valid_mask]  

            ca_k_content_valid = self.ca_kcontent_proj(valid_context)
            ca_k_content = context.new_zeros(B, M, D)
            ca_k_content[context_valid_mask] = ca_k_content_valid          

            ca_v_valid = self.ca_v_proj(valid_context)
            ca_v = context.new_zeros(B, M, D)
            ca_v[context_valid_mask] = ca_v_valid                           

            ca_valid_pos = context_ca_pos_embeddings[context_valid_mask]
            ca_k_pos_valid = self.ca_kpos_proj(ca_valid_pos)
            ca_k_pos = context_ca_pos_embeddings.new_zeros(B, M, D)
            ca_k_pos[context_valid_mask] = ca_k_pos_valid                   
        else:
            raise ValueError("Invalid context_valid_mask shape.")
        
        if is_first:
            ca_q_pos_from_sa = self.ca_qpos_proj(query_sa_pos_embeddings)
            ca_q = ca_q_content + ca_q_pos_from_sa                         
            ca_k = ca_k_content + ca_k_pos                                
        else:
            ca_q = ca_q_content      
            ca_k = ca_k_content      

        if self.use_concat_pe_ca:
            # concatenating positional embeddings
            ca_q = torch.cat([ca_q, ca_q_pos], dim=-1)  
            ca_k = torch.cat([ca_k, ca_k_pos], dim=-1)  
            D_cat = D * 2
        else:
            # adding positional embeddings
            ca_q = ca_q + ca_q_pos  
            ca_k = ca_k + ca_k_pos  
            D_cat = D

        # QK norm if needed
        if self.qk_norm:
            ca_q = self.ca_q_norm(ca_q)
            ca_k = self.ca_k_norm(ca_k)

        ca_q = ca_q.view(B, A, self.nhead, D_cat//self.nhead).transpose(1, 2)   
        ca_k = ca_k.view(B, M, self.nhead, D_cat//self.nhead).transpose(1, 2)   
        ca_v = ca_v.view(B, M, self.nhead, D//self.nhead).transpose(1, 2)       
        if len(context_valid_mask.shape) == 3:
            attn_mask_bool = context_valid_mask[:, None, :, :].expand(B, self.nhead, A, M)
        elif len(context_valid_mask.shape) == 2:
            attn_mask_bool = context_valid_mask[:, None, None, :].expand(B, self.nhead, A, M)
        else:
            raise ValueError("Invalid context_valid_mask shape.")
        
        ca_output = torch.nn.functional.scaled_dot_product_attention(ca_q, ca_k, ca_v, 
                                                                attn_mask=attn_mask_bool,
                                                                dropout_p=self.dropout_val if self.training else 0.0)  
        ca_output = ca_output.transpose(1, 2).contiguous().view(B, A, D)
        ca_output = self.ca_o_proj(ca_output)

        query = query + self.dropout2(ca_output)
        query = self.norm2(query)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(tgt2)
        query = self.norm3(query)     

        return query