import torch
import torch.nn as nn

from ..tools import build_mlps

class Short_traj_encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in = in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = build_mlps(
            c_in = hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )

        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels],
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None

    def forward(self, polylines, polylines_mask):
       
        B, A, T, D = polylines.shape

        polylines_feat_valid = self.pre_mlps(polylines[polylines_mask])
        polylines_feat = polylines.new_zeros(B, A, T, polylines_feat_valid.shape[-1])
        polylines_feat[polylines_mask] = polylines_feat_valid

        pooled_feature = polylines_feat.max(dim=2)[0]
        polylines_feat = torch.cat((polylines_feat, pooled_feature[:, :, None, :].repeat(1, 1, T, 1)), dim=-1)

        polylines_feat_valid = self.mlps(polylines_feat[polylines_mask])
        feature_buffers = polylines_feat.new_zeros(B, A, T, polylines_feat_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feat_valid

        feature_buffers = feature_buffers.max(dim=2)[0]    # (B, A, D_E)

        if self.out_mlps is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])
            feature_buffers = feature_buffers.new_zeros(B, A, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers