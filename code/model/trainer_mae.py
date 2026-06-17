import argparse
import os
import logging
import torch
import torch.nn as nn
import pdb
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from .model_mae import ModelMAE
from datamodule.av1_dataset import Av1Dataset, collate_fn
from utils.optim import WarmupCosLR
from metrics.min_ADE import ADE
from metrics.min_FDE import FDE

class Trainer_mae:
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

        self.min_ADE_x = ADE()
        self.min_FDE_x = FDE()
        self.min_ADE_y = ADE()
        self.min_FDE_y = FDE()
        self.min_ADE_l = ADE()
        self.min_FDE_l = FDE()
        self._build()

    def train(self):
        start_epoch = 1
        for epoch in range(start_epoch, self.epochs+1):
            self.model.train()
            pbar = tqdm(self.train_data_loader)
            total_loss = 0.0
            batch_num = 0.0
            for i, data in enumerate(pbar):
                data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}
                self.optimizer.zero_grad()
                out = self.model(data)
                train_loss = out["loss"]
                hist_loss = out["hist_loss"]
                future_loss = out["future_loss"]
                lane_pred_loss = out["lane_pred_loss"]

                pbar.set_description(f"Epoch {epoch}, toatl: {train_loss.item():.2f}")
                train_loss.backward()
                self.optimizer.step()

                total_loss += float(train_loss.item())  
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
                    'loss': epoch_loss
               }
               torch.save(checkpoint, os.path.join(self.config.save_path, f"MAE_{self.config.dataset}_best.pt"))
               self.logger.info(f'saved the model for epoch {epoch} with loss {total_loss / batch_num}')
               self.model.train()
            self.scheduler.step()

    def test(self):
        self.model.eval()
        self.logger.info("Start testing...")

        self.min_ADE_x.reset()
        self.min_FDE_x.reset()
        self.min_ADE_y.reset()
        self.min_FDE_y.reset()
        self.min_ADE_l.reset()
        self.min_FDE_l.reset()

        with torch.no_grad():
            pbar = tqdm(self.test_data_loader)
            for batch_idx, data in enumerate(pbar):
                data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}
                out = self.model(data)
                x_hat = out["x_hat"].view(-1, 20, 2)
                y_hat = out["y_hat"].view(-1, 30, 2)
                lane_pred = out["lane_hat"].view(-1, 20, 2)

                hist_pred_mask = out["hist_pred_mask"]
                future_pred_mask = out["future_pred_mask"]
                lane_pred_mask = out["lane_pred_mask"]

                x = (data["x_positions"] - data["x_centers"].unsqueeze(-2)).view(-1, 20, 2)
                x_reg_mask = ~data["x_padding_mask"][:, :, :20]
                x_reg_mask[~hist_pred_mask] = False
                x_reg_mask = x_reg_mask.view(-1, 20)
                self.min_ADE_x.update(x_hat[x_reg_mask], x[x_reg_mask])
                self.min_FDE_x.update(x_hat[x_reg_mask], x[x_reg_mask])

                y = data["y"].view(-1, 30, 2)
                reg_mask = ~data["x_padding_mask"][:, :, 20:]
                reg_mask[~future_pred_mask] = False
                reg_mask = reg_mask.view(-1, 30)
                self.min_ADE_y.update(y_hat[reg_mask], y[reg_mask])
                self.min_FDE_y.update(y_hat[reg_mask], y[reg_mask])

                lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
                lane_normalized = lane_normalized.view(-1, 20, 2)
                lane_reg_mask = ~data["lane_padding_mask"]
                lane_reg_mask[~lane_pred_mask] = False    
                lane_reg_mask = lane_reg_mask.view(-1, 20)

                self.min_ADE_l.update(lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask])
                self.min_FDE_l.update(lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask])

            min_ade_x = self.min_ADE_x.compute()
            min_fde_x = self.min_FDE_x.compute()
            min_ade_y = self.min_ADE_y.compute()
            min_fde_y = self.min_FDE_y.compute()
            min_ade_l = self.min_ADE_l.compute()
            min_fde_l = self.min_FDE_l.compute()
            self.logger.info(
                f"min_ade_x:{min_ade_x}, min_fde_x:{min_fde_x}"
                f"min_ade_y:{min_ade_y}, min_fde_y:{min_fde_y}"
                f"min_ade_l:{min_ade_l}, min_fde_l:{min_fde_l}"
            )
            
    def _build(self):
        self._build_log()
        self._build_model()
        self._build_train_loader()
        self._build_test_loader()
        self._build_optimizer()

    def _build_model(self):
        config = self.config
        model = ModelMAE(
            embed_dim=self.embed_dim,
            encoder_depth=self.encoder_depth,
            decoder_depth=self.decoder_depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_path=self.drop_path,
            actor_mask_ratio=self.actor_mask_ratio,
            lane_mask_ratio=self.lane_mask_ratio,
            history_steps=self.historical_steps,
            future_step=self.future_steps,
            loss_weight=self.loss_weight,
            training = self.training
        )
        self.model = model.cuda()

        if config.test:
            self.checkpoint = torch.load(os.path.join(config.save_path, f"MAE_{self.config.dataset}_best.pt"))
            self.model.load_state_dict(self.checkpoint['model'])
            self.logger.info('load test model')

        print("> Model Built")


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
            log_file = os.path.join(self.config.log_dir, f"train_MAE_{self.config.dataset}_{timestamp}.log")
        elif self.config.val:
            log_file = os.path.join(self.config.log_dir, f"val_MAE_{self.config.dataset}_{timestamp}.log")
        elif self.config.test:
            log_file = os.path.join(self.config.log_dir, f"test_MAE_{self.config.dataset}_{timestamp}.log")

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