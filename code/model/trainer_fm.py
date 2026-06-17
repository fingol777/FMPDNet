import argparse
import os
import logging
import torch
import torch.nn as nn
import pdb
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torchmetrics import MetricCollection
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from .model_fm import FlowMatcher
from datamodule.av1_dataset import Av1Dataset, collate_fn
from utils.optim import WarmupCosLR
from metrics.min_ADE import minADE
from metrics.min_FDE import minFDE
from metrics.mr import MR

class Trainer_fm:
    def __init__(self, config):
        self.config = config
        self.embed_dim = config.embed_dim
        self.historical_steps = config.historical_steps
        self.future_steps = config.future_steps
        self.encoder_depth = config.encoder_depth
        self.decoder_depth = config.decoder_depth
        self.num_heads = config.num_heads
        self. mlp_ratio = config.mlp_ratio
        self.qkv_bias = config.qkv_bias
        self.drop_path = config.drop_path
        self.actor_mask_ratio = config.actor_mask_ratio
        self.lane_mask_ratio = config.lane_mask_ratio
        self.epochs = config.epochs
        self.warmup_epochs = config.warmup_epochs
        self.lr = config.lr
        self.loss_weight = config.loss_weight
        self.weight_decay = float(config.weight_decay)
        self.training = config.train
        self.device = self.config.device

        self.pretrained_weights = self.config.pretrained_weights
        self.freeze_backbone = self.config.freeze_backbone

        self.scanario_steps = 0
        self.steps = 0

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1).to(self.device),
                "minFDE1": minFDE(k=1).to(self.device),
                "MR1": MR(k=1).to(self.device),
                "minADE6": minADE(k=6).to(self.device),
                "minFDE6": minFDE(k=6).to(self.device),
                "MR6": MR(k=6).to(self.device),
                "minADE3": minADE(k=3).to(self.device),
                "minFDE3": minFDE(k=3).to(self.device),
                "MR3": MR(k=3).to(self.device),
            }
        )
        self.test_metrics = metrics.clone(prefix="test_")
        self._build()

    def _build(self):
        self._build_log()
        self._build_model()
        self._build_train_loader()
        self._build_val_loader()
        self._build_test_loader()
        self._build_optimizer()
    
    def train(self):
        self.model.train()
        start_epoch = 1

        for epoch in range(start_epoch, self.epochs+1):
            self.model.train()
            pbar = tqdm(self.train_data_loader)
            total_loss = 0.0
            batch_num = 0.0
            running_loss = 0.0
            for i, data in enumerate(pbar):
                data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}
                self.optimizer.zero_grad()
                out = self.model(data)
                train_loss = out["loss"]
                hist_reg_loss = out["loss_reg"]
                agent_reg_loss = out["agent_reg_loss"]
                agent_cls_loss = out["agent_cls_loss"]
                others_reg_loss = out["others_reg_loss"]
                align_loss = out["loss_align"]
                pbar.set_description(
                    f"train_loss:{train_loss.item()}, "
                    f"hist_loss:{hist_reg_loss},"
                    f"agent_reg:{agent_reg_loss}, "
                    f"agent_cls:{agent_cls_loss}, "
                    f"ohters_reg:{others_reg_loss},"
                    f"align_loss:{align_loss},"
                )
                train_loss.backward()
                self.optimizer.step()

                total_loss += float(train_loss.item())  
                running_loss += train_loss.item()
                batch_num += 1

            epoch_loss = total_loss / max(batch_num, 1)
            self.logger.info(f"Epoch:{epoch}, average_loss:{epoch_loss}")

            if epoch % self.config.save_steps == 0:
               self.model.eval()
               checkpoint = {
                    'model': self.model.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'optimizer': self.optimizer.state_dict(),
                    'loss': epoch_loss,
               }
               torch.save(checkpoint, os.path.join(self.config.save_path, f"FM_{self.config.dataset}_best.pt"))
               self.logger.info(f'saved the model for epoch {epoch} with loss {total_loss / batch_num}')
               self.model.train()

            self.scheduler.step()
        

    def test(self):
        self.model.eval()
        self.logger.info("Start testing...")
        batch_count = 0

        with torch.no_grad():
            pbar = tqdm(self.test_data_loader)
            for batch_idx, data in enumerate(pbar):
                data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}
                out = self.model.future_traj_predictor(data)                
                metrics = self.test_metrics.update(out, data["y"][:, 0])
                batch_count += 1

                y_agent_hat = out["y_agent_hat"]                         
                pi_agent = out["pi_agent"]                              
                scenario_ids = data["scenario_id"]                       
                track_ids = data["track_id"]
                batch = len(track_ids)

                origin = data["origin"].view(batch, 1, 1, 2).double()     
                theta = data["theta"].double()                             
                rotate_mat = torch.stack(
                    [
                        torch.cos(theta),
                        torch.sin(theta),
                        -torch.sin(theta),
                        torch.cos(theta),
                    ],
                    dim=1,
                ).reshape(batch, 2, 2)                                  

                with torch.no_grad():
                    global_y_agent_hat = (
                        torch.matmul(y_agent_hat[..., :2].double(), rotate_mat.unsqueeze(1))
                        + origin
                    )
                
                global_y_agent_hat = global_y_agent_hat.detach().cpu().numpy()       
                pi_agent = pi_agent.detach().cpu().numpy()                          
                
            final_metrics = self.test_metrics.compute()
            avg_ade_1 = final_metrics["test_minADE1"].item()
            avg_ade_6 = final_metrics["test_minADE6"].item()
            avg_fde_1 = final_metrics["test_minFDE1"].item()
            avg_fde_6 = final_metrics["test_minFDE6"].item()
            avg_mr_1 = final_metrics["test_MR1"].item()
            avg_mr_6 = final_metrics["test_MR6"].item()
            avg_ade_3 = final_metrics["test_minADE3"].item()
            avg_fde_3 = final_metrics["test_minFDE3"].item()
            avg_mr_3= final_metrics["test_MR3"].item()
            
            self.logger.info(f"avegera_ADE_1: {avg_ade_1:.4f},"
                             f"average_FDE_1: {avg_fde_1:.4f},"
                             f"avg_mr_1: {avg_mr_1:.4f},")
            self.logger.info(f"average_ADE_6: {avg_ade_6:.4f},"
                             f"average_FDE_6: {avg_fde_6:.4f},"
                             f"avg_mr_6: {avg_mr_6:.4f},")
            self.logger.info(f"avegera_ADE_3: {avg_ade_3:.4f},"
                             f"average_FDE_3: {avg_fde_3:.4f},"
                             f"avg_mr_3: {avg_mr_3:.4f},")
            
    def _build_model(self):
        config = self.config
        model = FlowMatcher(
                    config = self.config,
                    embed_dim = self.embed_dim,
                    encoder_depth = self.encoder_depth,
                    num_heads = self.num_heads,
                    mlp_ratio = self.mlp_ratio,
                    qkv_bias = self.qkv_bias,
                    drop_path = self.drop_path,
                    future_steps = self.future_steps,
                )
        self.model = model.to(config.device)

        if self.pretrained_weights is not None:
            self.model.load_from_checkpoint(self.pretrained_weights)
            print('Sucessed to load pretrained weights')

        if self.freeze_backbone:
            self.freeze_pretrained_layers()
            print('have freeze the pretrained model')
            
        if config.test:
            self.checkpoint = torch.load(os.path.join(config.save_path, f"FM_{self.config.dataset}_best_{self.config.test_epoch}.pt"))
            self.model.load_state_dict(self.checkpoint['model'])
            self.logger.info(f'load test model for {self.config.test_epoch}')

        print("> Model Built")

    def freeze_pretrained_layers(self):
        frozen_count = 0

        layers_freeze = [
            'hist_embed',
            'lane_embed',
            'blocks',
            'norm',
            'pos_embed',
            'actor_type_embed',
            'lane_type_embed',
        ]

        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layers_freeze):
                param.requires_grad = False
                frozen_count += 1

        print(f'freeze {frozen_count} layers')
    
    def _build_train_loader(self):
        self.train_batch_size = self.config.train_batch_size
        self.num_workers = self.config.num_workers
        self.pin_memory = self.config.pin_memory
        self.train_dataset = Av1Dataset(
            data_root=self.config.data_root,
            cached_split="train"
        )

        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
        print(f">train data loader! {len(self.train_data_loader)}")

    def _build_val_loader(self):
        self.val_batch_size = self.config.val_batch_size
        self.num_workers = self.config.num_workers
        self.pin_memory = self.config.pin_memory
        self.val_dataset = Av1Dataset(
            data_root=self.config.data_root,
            cached_split="val"
        )

        self.val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
        print(f">val data loader! {len(self.val_data_loader)}")

    def _build_test_loader(self):
        self.test_batch_size = self.config.test_batch_size
        self.num_workers = self.config.num_workers
        self.pin_memory = self.config.pin_memory
        self.test_dataset = Av1Dataset(
            data_root=self.config.data_root, 
            cached_split="test"
        )
        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
        print(f">test loader build! {len(self.test_data_loader)}")

    def _build_optimizer(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.model.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = WarmupCosLR(
            optimizer=self.optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        print("> Optimizer built!")

    def _build_log(self):
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d")

        if self.config.train:
            log_file = os.path.join(self.config.log_dir, f"train_FM_{self.config.dataset}_{timestamp}.log")
        elif self.config.val:
            log_file = os.path.join(self.config.log_dir, f"val_FM_{self.config.dataset}_{timestamp}.log")
        elif self.config.test:
            log_file = os.path.join(self.config.log_dir, f"test_FM_{self.config.dataset}_{timestamp}.log")

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_hanlder = logging.FileHandler(log_file, encoding='utf-8')
            file_hanlder.setLevel(logging.INFO)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_hanlder.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_hanlder)
            self.logger.addHandler(console_handler)
        print("> log ready!")
