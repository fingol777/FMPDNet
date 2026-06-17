from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .decoders.fm_decoders import FM_Decoder
from .decoders.fut_decoder import MultimodalDecoder
from .decoders.mlp_decoder import MLPDecoder
from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.global_interactor import GlobalInteractor
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.prompt_layer import Prompt
from .layers.short_traj_embedding import Short_traj_encoder
from .layers.transformer_blocks_gated import (
    ActorLaneGatedAttention,
    GatedEncoderBlock,
    HeadwiseGatedAttention,
)


def pad_t_like_x(t, x):
    
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class FlowMatcher(nn.Module):

    def __init__(
        self,
        config,
        embed_dim: int = 128,
        encoder_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path: float = 0.2,
        future_steps: int = 30,
    ) -> None:
        super().__init__()
        self.cfg = config

        self._build_encoder(
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
        )
        self._build_history_modules(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self._build_prediction_modules(
            embed_dim=embed_dim,
            num_heads=num_heads,
            future_steps=future_steps,
        )

        self.cross_entropy = nn.CrossEntropyLoss()
        self.initialize_weights()

    def _build_encoder(
        self,
        embed_dim: int,
        encoder_depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_path: float,
    ) -> None:
        self.hist_embed = AgentEmbeddingLayer(4, embed_dim // 4, drop_path_rate=drop_path)
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList(
            GatedEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=drop_path_rates[i],
                lane_gate_bias=-1.0,
                out_gate_bias=1.0,
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(3, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

    def _build_history_modules(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool,
    ) -> None:
        self.short_traj_encoder = Short_traj_encoder(
            in_channels=4,
            hidden_dim=embed_dim,
            num_layers=5,
            num_pre_layers=3,
        )
        self.prompt_creater = Prompt(
            in_channels_traj=8,
            in_channels_lane=43,
            hidden_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.s_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )
        self.t_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )
        self.w_p = nn.Parameter(torch.tensor(1.0))
        self.w_f = nn.Parameter(torch.tensor(1.0))

    def _build_prediction_modules(
        self,
        embed_dim: int,
        num_heads: int,
        future_steps: int,
    ) -> None:
        self.global_interactor = GlobalInteractor(
            historical_steps=self.cfg.historical_steps,
            embed_dim=embed_dim,
            edge_dim=self.cfg.edge_dim,
            num_modes=self.cfg.num_modes,
            num_heads=num_heads,
            num_layers=self.cfg.num_global_layers,
            dropout=0.1,
            rotate=self.cfg.rotate,
        )
        self.aggr_embed = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.fm_decoder = FM_Decoder(self.cfg)
        self.fut_decoder = MultimodalDecoder(
            embed_dim=embed_dim,
            future_steps=future_steps,
        )
        self.fut_dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, future_steps * 2),
        )
        self.global_decoder = MLPDecoder(
            local_channels=embed_dim,
            global_channels=embed_dim,
            future_steps=future_steps,
            num_modes=self.cfg.num_modes,
        )

    def initialize_weights(self) -> None:
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)
        self.apply(self._init_weights)

        for module in self.modules():
            if isinstance(module, (ActorLaneGatedAttention, HeadwiseGatedAttention)):
                module.reset_gate_bias()

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.cfg.device)
        ckpt = checkpoint["model"]
        if len(ckpt) > 0 and list(ckpt.keys())[0].startswith("model."):
            state_dict = {
                k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
            }
        else:
            state_dict = ckpt
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def pre_encoder(self, hist_feat, data):
        
        batch_size, num_agents, hist_len, feat_dim = hist_feat.shape

        actor_feat = self._encode_agents(
            hist_feat=hist_feat,
            hist_key_padding_mask=data["x_key_padding_mask"],
            batch_size=batch_size,
            num_agents=num_agents,
            hist_len=hist_len,
            feat_dim=feat_dim,
        )
        lane_feat = self._encode_lanes(data)
        pos_embed = self._build_context_pos_embed(data)

        actor_feat = actor_feat + self.actor_type_embed[data["x_attr"][..., 0].long()]
        num_lanes = lane_feat.shape[1]
        lane_feat = lane_feat + self.lane_type_embed.repeat(batch_size, num_lanes, 1)

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1) + pos_embed
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]],
            dim=1,
        )
        for block in self.blocks:
            x_encoder = block(
                x_encoder,
                num_actor_tokens=num_agents,
                key_padding_mask=key_padding_mask,
            )
        return self.norm(x_encoder)

    def _encode_agents(
        self,
        hist_feat,
        hist_key_padding_mask,
        batch_size: int,
        num_agents: int,
        hist_len: int,
        feat_dim: int,
    ):
        hist_feat = hist_feat.contiguous().view(batch_size * num_agents, hist_len, feat_dim)
        flat_padding_mask = hist_key_padding_mask.view(batch_size * num_agents)

        actor_feat = self.hist_embed(
            hist_feat[~flat_padding_mask].permute(0, 2, 1).contiguous()
        )
        padded_actor_feat = torch.zeros(
            batch_size * num_agents,
            actor_feat.shape[-1],
            device=actor_feat.device,
        )
        padded_actor_feat[~flat_padding_mask] = actor_feat
        return padded_actor_feat.view(batch_size, num_agents, actor_feat.shape[-1])

    def _encode_lanes(self, data):
        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, ~lane_padding_mask[..., None]],
            dim=-1,
        )
        batch_size, num_lanes, lane_len, lane_dim = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, lane_len, lane_dim).contiguous())
        return lane_feat.view(batch_size, num_lanes, -1)

    def _build_context_pos_embed(self, data):
        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, 19], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        return self.pos_embed(torch.cat([x_centers, x_angles], dim=-1))

    def _build_history_features(self, data):
        hist_padding_mask = data["x_padding_mask"][:, :, : self.cfg.historical_steps]
        return torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )

    def _build_short_history_context(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        coor_x_last_two = data["x"][:, :, -2:, :]
        vel_x_last_two = data["x_velocity_diff"][:, :, -2:]
        mask_last_two = data["x_padding_mask"][:, :, 18:20]
        short_hist_feat = torch.cat(
            [
                coor_x_last_two,
                vel_x_last_two[..., None],
                ~mask_last_two[..., None],
            ],
            dim=-1,
        )
        short_context = self.short_traj_encoder(short_hist_feat, mask_last_two)
        return short_hist_feat, short_context

    def _create_prompt(self, short_hist_feat, data):
        return self.prompt_creater(
            query=short_hist_feat,
            key_pos=data["lane_positions"],
            key_attr=data["lane_attr"],
            key_padding_mask=data["lane_key_padding_mask"],
        )

    def _fuse_prompt_and_history(self, prompt, hist_embed):
        weights = F.softmax(torch.stack([self.w_p, self.w_f]), dim=0)
        fused_embed = torch.cat([weights[0] * prompt, weights[1] * hist_embed], dim=-1)
        return self.s_mlp(fused_embed)

    def _predict_future(self, local_embed, data):
        global_embed = self.global_interactor(data, local_embed)
        fut_y, fut_pi, _ = self.global_decoder(
            local_embed=local_embed,
            global_embed=global_embed,
            data=data,
        )
        return {
            "y_agent_hat": fut_y[:, 0, :, :, :],
            "pi_agent": fut_pi[:, 0, :],
            "y_others_hat": fut_y[:, 1:, 0, :, :],
        }

    def get_precond_coef(self, t):
        coef_1 = t.pow(2) * self.cfg.sigma_data**2 + (1 - t).pow(2)
        alpha_t = t * self.cfg.sigma_data**2 / coef_1
        beta_t = (1 - t) * self.cfg.sigma_data / coef_1.sqrt()
        return alpha_t, beta_t

    def fm_wrapper_func(self, x_t, t, model_out):
        wrapper = self.cfg.fm_wrapper
        if wrapper == "direct":
            return model_out
        if wrapper == "velocity":
            t = pad_t_like_x(t, x_t)
            return x_t + (1 - t) * model_out
        if wrapper in ("precond", "percond"):
            t = pad_t_like_x(t, x_t)
            alpha_t, beta_t = self.get_precond_coef(t)
            return alpha_t * x_t + beta_t * model_out
        raise ValueError(f"unknown fm_wrapper {wrapper}")

    def get_loss_input(self, y_start_k):
        batch_size = y_start_k.shape[0]
        if self.cfg.t_schedule == "uniform":
            t = torch.rand((batch_size,), device=self.cfg.device)
        elif self.cfg.t_schedule == "logit_normal":
            t_normal = (
                torch.randn((batch_size,), device=self.cfg.device) * self.cfg.logit_norm_std
                + self.cfg.logit_norm_mean
            )
            t = torch.sigmoid(t_normal)
        else:
            raise ValueError(f"unknown t_schedule {self.cfg.t_schedule}")

        assert t.min() >= 0 and t.max() <= 1
        noise = torch.randn_like(y_start_k)
        x_t, u_t = self.fwd_sample_t(x0=noise, x1=y_start_k, t=t)

        if self.cfg.objective == "pred_data":
            target = y_start_k
        elif self.cfg.objective == "pred_vel":
            target = u_t
        else:
            raise ValueError(f"unknown objective {self.cfg.objective}")

        return t, x_t, u_t, target, self.get_reweighting(t)

    def fwd_sample_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0, x1 - x0

    def get_reweighting(self, t, wrapper=None):
        wrapper = self.cfg.fm_wrapper if wrapper is None else wrapper
        if wrapper == "direct":
            loss_weight = torch.ones_like(t)
        elif wrapper == "velocity":
            loss_weight = 1.0 / (1 - t) ** 2
        elif wrapper in ("precond", "percond"):
            _, beta_t = self.get_precond_coef(t)
            loss_weight = 1.0 / beta_t**2
        else:
            raise ValueError(f"unknown fm_wrapper {wrapper}")

        if self.cfg.fm_rew_sqrt:
            loss_weight = loss_weight.sqrt()
        return loss_weight.clamp(min=1e-4, max=1e4)

    def forward(self, data):
        hist_feat = self._build_history_features(data)
        batch_size, num_agents, hist_len, feat_dim = hist_feat.shape

        hist_traj_normalized = rearrange(hist_feat, "b a f d -> b a (f d)")
        t, y_t, _, _, loss_weight = self.get_loss_input(y_start_k=hist_traj_normalized)

        y_t_in = self._maybe_drop_noisy_history(y_t, t)
        model_out = self._predict_denoised_history(y_t_in, t, data)
        denoised_y = self.fm_wrapper_func(y_t, t, model_out)
        denoised_y = rearrange(
            denoised_y,
            "b a (f d) -> b a f d",
            f=self.cfg.historical_steps,
        )

        loss_reg = self._compute_history_loss(
            denoised_y=denoised_y,
            hist_traj_normalized=hist_traj_normalized,
            hist_len=hist_len,
            feat_dim=feat_dim,
            loss_weight=loss_weight,
            data=data,
        )
        hist_pred_x_embed = self._encode_reconstructed_history(denoised_y, data, num_agents)
        hist_true_x_embed = self._encode_true_history(hist_feat, data, num_agents)

        short_hist_feat, _ = self._build_short_history_context(data)
        prompt = self._create_prompt(short_hist_feat, data)
        hist_pred_x_embed = self._fuse_prompt_and_history(prompt, hist_pred_x_embed)

        loss_align = self._compute_alignment_loss(
            student_embed=hist_pred_x_embed,
            teacher_embed=hist_true_x_embed,
            batch_size=batch_size,
            num_agents=num_agents,
        )

        future_out = self._predict_future(hist_pred_x_embed, data)
        future_losses = self._compute_future_losses(future_out, data)

        loss = (
            self.cfg.w_reg * loss_reg
            + self.cfg.fut_weight_reg * future_losses["agent_reg_loss"]
            + self.cfg.fut_weight_cls * future_losses["agent_cls_loss"]
            + self.cfg.fut_weight_others * future_losses["others_reg_loss"]
            + self.cfg.w_align * loss_align
        )

        return {
            "loss": loss,
            "loss_reg": loss_reg.item(),
            "agent_reg_loss": future_losses["agent_reg_loss"].item(),
            "agent_cls_loss": future_losses["agent_cls_loss"].item(),
            "others_reg_loss": future_losses["others_reg_loss"].item(),
            "loss_align": loss_align.item(),
        }

    def _maybe_drop_noisy_history(self, y_t, t):
        if not (self.cfg.train and self.cfg.use_drop_mask):
            return y_t

        midpoint, slope = self.cfg.drop_logi_m, self.cfg.drop_logi_k
        drop_prob = 1 / (1 + torch.exp(-slope * (t - midpoint)))
        drop_prob = drop_prob[:, None, None]
        return y_t.masked_fill(torch.rand_like(drop_prob) < drop_prob, 0.0)

    def _predict_denoised_history(self, y_t_in, t, data):
        batch_size, num_agents = y_t_in.shape[:2]
        hist_noise_feat = y_t_in.view(batch_size, num_agents, self.cfg.historical_steps, -1)
        x_encoder = self.pre_encoder(hist_noise_feat, data)

        _, short_context = self._build_short_history_context(data)
        return self.fm_decoder(
            y_t_in=y_t_in,
            fut_traj_embed=x_encoder[:, :num_agents, :],
            short_traj_context_embed=short_context,
            lane_embed=x_encoder[:, num_agents:, :],
            context_pos_embed=self._build_context_pos_embed(data),
            t=t,
            data=data,
        )

    def _compute_history_loss(
        self,
        denoised_y,
        hist_traj_normalized,
        hist_len: int,
        feat_dim: int,
        loss_weight,
        data,
    ):
        hist_traj_metric = rearrange(
            hist_traj_normalized,
            "b a (f d) -> b a f d",
            f=hist_len,
            d=feat_dim,
        )
        error_per_agent = (denoised_y - hist_traj_metric).norm(dim=-1).mean(dim=-1)
        error_per_scene = error_per_agent.mean(dim=-1)

        if self.cfg.best_mode == "scene":
            loss_reg_b = error_per_scene
        elif self.cfg.best_mode == "agent":
            loss_reg_b = error_per_agent.mean(dim=-1)
        else:
            raise ValueError(f"unknown best_mode {self.cfg.best_mode}")

        return (loss_reg_b * loss_weight).mean()

    def _encode_reconstructed_history(self, denoised_y, data, num_agents: int):
        x_encoder = self.pre_encoder(denoised_y, data)
        return x_encoder[:, :num_agents, :]

    def _encode_true_history(self, hist_feat, data, num_agents: int):
        x_encoder = self.pre_encoder(hist_feat, data)
        return x_encoder[:, :num_agents, :]

    def _compute_alignment_loss(
        self,
        student_embed,
        teacher_embed,
        batch_size: int,
        num_agents: int,
    ):
        student_embed = student_embed.view(batch_size * num_agents, -1)
        teacher_embed = self.t_mlp(teacher_embed).view(batch_size * num_agents, -1)
        return self.cross_entropy(
            F.softmax(student_embed, dim=-1),
            F.softmax(teacher_embed, dim=-1),
        )

    def _compute_future_losses(self, future_out, data):
        y_agent = data["y"][:, 0]
        y_others = data["y"][:, 1:]

        y_agent_hat = future_out["y_agent_hat"]
        pi_agent = future_out["pi_agent"]
        y_others_hat = future_out["y_others_hat"]

        l2_norm = torch.norm(y_agent_hat[..., :2] - y_agent.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_agent_hat[torch.arange(y_agent_hat.shape[0]), best_mode]

        others_reg_mask = ~data["x_padding_mask"][:, 1:, 20:]
        return {
            "agent_reg_loss": F.smooth_l1_loss(y_hat_best[..., :2], y_agent),
            "agent_cls_loss": F.cross_entropy(input=pi_agent, target=best_mode.detach()),
            "others_reg_loss": F.smooth_l1_loss(
                y_others_hat[others_reg_mask],
                y_others[others_reg_mask],
            ),
        }

    @torch.no_grad()
    def future_traj_predictor(self, data):
        batch_size, num_agents, hist_steps = data["x"].shape[:3]
        assert hist_steps == self.cfg.historical_steps

        hist_pred, _, _, _ = self.sample(
            data,
            self.cfg.denoising_head_preds,
            return_all_states=False,
        )
        hist_pred_feature = hist_pred.view(batch_size, num_agents, hist_steps, -1)
        x_encoder_past = self.pre_encoder(hist_pred_feature, data)
        hist_pred_x_embed = x_encoder_past[:, :num_agents, :]

        short_hist_feat, _ = self._build_short_history_context(data)
        prompt = self._create_prompt(short_hist_feat, data)
        hist_pred_x_embed = self._fuse_prompt_and_history(prompt, hist_pred_x_embed)

        return self._predict_future(hist_pred_x_embed, data)

    @torch.no_grad()
    def sample(self, data, num_trajs, return_all_states=False):
        del num_trajs
        batch_size, num_agents, hist_steps = data["x"].shape[:3]
        feat_dim = 4
        y_t = torch.randn(
            (batch_size, num_agents, hist_steps * feat_dim),
            device=self.cfg.device,
        )

        y_data_t_list = []
        y_t_list = []
        t_list, dt_list = self._build_sampling_schedule()

        for cur_t, cur_dt in zip(t_list, dt_list):
            y_t, y_data = self.bwd_sample_t(
                y_t=y_t,
                t=cur_t,
                dt=cur_dt,
                data=data,
                flag_print=False,
            )
            y_data_t_list.append(y_data)
            if return_all_states:
                y_t_list.append(y_t)

        y_data_t_list = torch.stack(y_data_t_list, dim=1)
        t_tensor = torch.tensor(t_list, device=self.cfg.device)
        if return_all_states:
            y_t_list = torch.stack(y_t_list, dim=1)

        return y_t, y_data_t_list, t_tensor, y_t_list

    def _build_sampling_schedule(self):
        if self.cfg.solver == "euler":
            dt = 1.0 / self.cfg.sampling_steps
            t_list = dt * np.arange(self.cfg.sampling_steps)
            dt_list = dt * np.ones(self.cfg.sampling_steps)
            return t_list, dt_list

        if self.cfg.solver == "lin_poly":
            n_steps_lin = self.cfg.sampling_steps // 2
            n_steps_poly = self.cfg.sampling_steps - n_steps_lin
            dt_lin = 1.0 / self.cfg.lin_poly_long_step
            t_lin = dt_lin * np.arange(n_steps_lin)

            t_poly_start = t_lin[-1] + dt_lin
            t_poly_end = 1.0
            t_poly = self._polynomially_spaced_points(
                t_poly_start,
                t_poly_end,
                n_steps_poly + 1,
                p=self.cfg.lin_poly_p,
            )
            dt_poly = np.diff(t_poly)

            dt_list = np.concatenate([dt_lin * np.ones(n_steps_lin), dt_poly]).tolist()
            t_list = np.concatenate([t_lin, t_poly[:-1]]).tolist()
            return t_list, dt_list

        raise ValueError(f"unknown solver {self.cfg.solver}")

    @staticmethod
    def _polynomially_spaced_points(a, b, num_points, p=2):
        return [
            a + (b - a) * ((idx - 1) ** p) / ((num_points - 1) ** p)
            for idx in range(1, num_points + 1)
        ]

    @torch.inference_mode()
    def bwd_sample_t(self, y_t, t, dt, data, flag_print):
        batch_size = y_t.shape[0]
        batched_t = torch.full(
            (batch_size,),
            t,
            device=self.cfg.device,
            dtype=torch.float,
        )
        pred_vel, pred_data = self.model_predictions(y_t, data, batched_t, flag_print)
        return y_t + pred_vel * dt, pred_data

    def model_predictions(self, y_t, data, t, flag_print):
        del flag_print
        model_out = self._predict_denoised_history(y_t, t, data)
        y_data_t = self.fm_wrapper_func(y_t, t, model_out)
        pred_vel = self.predict_vel_from_data(y_data_t, y_t, t)
        return pred_vel, y_data_t

    def predict_vel_from_data(self, x1, xt, t):
        t = pad_t_like_x(t, x1)
        return (x1 - xt) / (1 - t)
