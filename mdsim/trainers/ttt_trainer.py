import datetime
import errno
import logging
import os
import random
import subprocess
from abc import ABC
from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import mdsim
from mdsim.common import distutils
from mdsim.common.data_parallel import (
    BalancedBatchSampler,
    OCPDataParallel,
    ParallelCollater,
)
from mdsim.common.registry import registry
from mdsim.common.utils import save_checkpoint

from mdsim.modules.evaluator import Evaluator
from mdsim.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from mdsim.modules.loss import DDPLoss, L2MAELoss
from mdsim.modules.normalizer import Normalizer
from mdsim.modules.scheduler import LRScheduler

from mdsim.trainers.trainer import Trainer
from mdsim.common.ttt_utils import gemnet_freeze_amount_to_fn
    

@registry.register_trainer("ttt_trainer")
class TTTTrainer(Trainer):
    """
    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """
    def __init__(
        self,
        *args,
        ttt_params,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.ttt_params = ttt_params

        self.ttt_steps = self.ttt_params.get("ttt_steps", 1)
        self.ttt_lr = self.ttt_params.get("ttt_lr", 1e-4)
        self.ttt_momentum = self.ttt_params.get("ttt_momentum", 0.9)
        self.ttt_weight_decay = self.ttt_params.get("ttt_weight_decay", 0.0)
        self.ttt_opt = self.ttt_params.get("ttt_opt", "sgd")
        self.freeze_head_checkpoint = self.ttt_params.get("freeze_head_checkpoint", None)
        self.skip_ttt = self.ttt_params.get("skip_ttt", False)
        self.ttt_freeze_amount = self.ttt_params.get("ttt_freeze_amount", "min")
        self.ttt_save_checkpoint_name = self.ttt_params.get("ttt_save_checkpoint_name", None)
        self.print_every_ttt = self.ttt_params.get("print_every_ttt", 10)
        self.save_ttt_results = self.ttt_params.get("save_ttt_results", False)
        self.save_ttt_results_name = self.ttt_params.get("save_ttt_results_name", '')
        self.freeze_prior_head = self.ttt_params.get("freeze_prior_head", True)
        self.use_lr_scheduler = self.ttt_params.get("use_lr_scheduler", False)
        self.ttt_lr_scheduler = None
        self.load_normalizers = self.ttt_params.get("load_normalizers", False)

        self.train_dataset.return_classical = True
        self.train_dataset.noise_classical_scale = self.ttt_params.get("noise_classical_scale", None)
        
    def load_checkpoint(self, checkpoint_path):

        self.checkpoint_path = checkpoint_path

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        if not self.finetuning:
            self.epoch = checkpoint.get("epoch", 0)
            self.step = checkpoint.get("step", 0)
            self.elapsed = checkpoint.get("elapsed", 0)
            if self.elapsed > 0:
                logging.info(f"trained time: {self.elapsed}")
        # Load model, optimizer, normalizer state dict.
        # if trained with ddp and want to load in non-ddp, modify keys from
        # module.module.. -> module..
        first_key = next(iter(checkpoint["state_dict"]))
        if (
            not distutils.initialized() or self.config["noddp"]
        ) and first_key.split(".")[1] == "module":
            # No need for OrderedDict since dictionaries are technically ordered
            # since Python 3.6 and officially ordered since Python 3.7
            new_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
            self.model.load_state_dict(new_dict)
        elif distutils.initialized() and first_key.split(".")[1] != "module":
            new_dict = {
                f"module.{k}": v for k, v in checkpoint["state_dict"].items()
            }
            self.model.load_state_dict(new_dict)
        else:
            self.model.load_state_dict(checkpoint["state_dict"])
            
        if self.freeze_rep:
            if self.config['model'] == 'gemnet_t':
                for name, param in self.model.named_parameters():
                    if gemnet_freeze_amount_to_fn[self.ttt_freeze_amount](name) == True:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                raise NotImplementedError('freeze_rep not implemented for ttt for this model')
                
        if self.reset_head:
            if self.config['model'] == 'gemnet_t':
                raise NotImplementedError('reset_head not implemented for gemnet_t')
            else:
                print("RESETTING HEAD!")
                self.model.module.interactions[-1].reset_parameters()
                torch.nn.init.xavier_uniform_(self.model.module.lin1.weight)
                self.model.module.lin1.bias.data.fill_(0)
                torch.nn.init.xavier_uniform_(self.model.module.lin2.weight)
                self.model.module.lin2.bias.data.fill_(0)
    
    
    def load_freeze_head(self):
        """
        Load the head of the model that was used on the prior.
        """
        assert self.freeze_head_checkpoint is not None

        checkpoint = torch.load(self.freeze_head_checkpoint)
        if self.load_normalizers:
            for key in checkpoint["normalizers"]:
                if key in self.normalizers:
                    self.normalizers[key].load_state_dict(
                        checkpoint["normalizers"][key]
                    )

        freeze_head_state_dict = checkpoint['state_dict']

        for name, param in self.model.named_parameters():
            if gemnet_freeze_amount_to_fn[self.ttt_freeze_amount](name) == False:
                param.data = freeze_head_state_dict[name].data.clone()
                if self.freeze_prior_head:
                    param.requires_grad = False
        
    def load_qm_head(self):
        """
        Load the head used on the qm data
        """
        checkpoint = torch.load(self.checkpoint_path)

        if self.load_normalizers:
            for key in checkpoint["normalizers"]:
                if key in self.normalizers:
                    self.normalizers[key].load_state_dict(
                        checkpoint["normalizers"][key]
                    )

        ft_head_state_dict = checkpoint['state_dict']

        for name, param in self.model.named_parameters():
            if gemnet_freeze_amount_to_fn[self.ttt_freeze_amount](name) == False:
                param.data = ft_head_state_dict[name].data.clone()
        
        
    @torch.no_grad()
    def validate(self, split="val", disable_tqdm=False, max_points=None, skip_ttt_for_logging=False):
        if distutils.is_master():
            logging.info(f"Evaluating on {split}.")
        if self.is_hpo:
            disable_tqdm = True

        if not self.skip_ttt and not skip_ttt_for_logging:
            self.load_freeze_head()
            self.train()
            self.load_qm_head()
        
        self.model.eval()
        
        max_points = 1000
        if self.ema is not None:
            logging.info("Forcing EMA to None!")
            self.ema = None

        evaluator, metrics = Evaluator(task=self.name), {}
        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader
        batch_size = self.config["optim"].get(
                        "eval_batch_size", self.config["optim"]["batch_size"])
        if max_points is None:
            max_points = len(loader) * batch_size
        for i, batch in tqdm(
            enumerate(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
            total=np.ceil(max_points // batch_size)
        ):
            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)
            
            if max_points and (i+1) * batch_size >= max_points:
                break
            
        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()

        
        if self.ttt_save_checkpoint_name is not None:
            self.save(
                metrics=metrics,
                checkpoint_file=self.ttt_save_checkpoint_name,
                training_state=False,
                force_save=True,
                save_path=os.path.dirname(self.checkpoint_path),
            )
        
        return metrics

    def _compute_loss(self, out, batch_list):
        loss = []

        # Energy loss.
        if not self.no_energy:
            energy_target = torch.cat(
                [batch.y.to(self.device) for batch in batch_list], dim=0
            )
            
            if self.normalizer.get("normalize_labels", False):
                energy_target = self.normalizers["target"].norm(energy_target, mean_energy_per_system=batch_list[0].energy_mean if 'energy_mean' in batch_list[0] else None, std_energy_per_system=batch_list[0].energy_std if 'energy_std' in batch_list[0] else None)
            energy_mult = self.config["optim"].get("energy_coefficient", 1)
            loss.append(
                energy_mult * self.loss_fn["energy"](out["energy"], energy_target)
            )

        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            if self.normalizer.get("normalize_labels", False):
                force_target = self.normalizers["grad_target"].norm(
                    force_target
                )

            tag_specific_weights = self.config["task"].get(
                "tag_specific_weights", []
            )
            if tag_specific_weights != []:
                # handle tag specific weights as introduced in forcenet
                assert len(tag_specific_weights) == 3

                batch_tags = torch.cat(
                    [
                        batch.tags.float().to(self.device)
                        for batch in batch_list
                    ],
                    dim=0,
                )
                weight = torch.zeros_like(batch_tags)
                weight[batch_tags == 0] = tag_specific_weights[0]
                weight[batch_tags == 1] = tag_specific_weights[1]
                weight[batch_tags == 2] = tag_specific_weights[2]

                loss_force_list = torch.abs(out["forces"] - force_target)
                train_loss_force_unnormalized = torch.sum(
                    loss_force_list * weight.view(-1, 1)
                )
                train_loss_force_normalizer = 3.0 * weight.sum()

                # add up normalizer to obtain global normalizer
                distutils.all_reduce(train_loss_force_normalizer)

                # perform loss normalization before backprop
                train_loss_force_normalized = train_loss_force_unnormalized * (
                    distutils.get_world_size() / train_loss_force_normalizer
                )
                loss.append(train_loss_force_normalized)

            else:
                # Force coefficient = 30 has been working well for us.
                force_mult = self.config["optim"].get("force_coefficient", 30)
                if self.config["task"].get("train_on_free_atoms", False):
                    fixed = torch.cat(
                        [batch.fixed.to(self.device) for batch in batch_list]
                    )
                    mask = fixed == 0
                    loss.append(
                        force_mult
                        * self.loss_fn["force"](
                            out["forces"][mask], force_target[mask]
                        )
                    )
                else:
                    loss.append(
                        force_mult
                        * self.loss_fn["force"](out["forces"], force_target)
                    )
        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")           
                
        loss = sum(loss)
        
        return loss

    
    def get_ttt_opt(self):
        if self.ttt_opt == 'adam':
            ttt_optimizer = optim.Adam(self.model.parameters(), lr=self.ttt_lr, weight_decay=self.ttt_weight_decay)
        elif self.ttt_opt == 'sgd':
            ttt_optimizer = optim.SGD(self.model.parameters(), lr=self.ttt_lr, momentum=self.ttt_momentum, weight_decay=self.ttt_weight_decay)
        else:
            raise NotImplementedError(f'Optimizer {self.ttt_opt} not implemented')

        if self.use_lr_scheduler:
            self.ttt_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ttt_optimizer, mode='min', factor=0.5, patience=100, threshold=0.01, threshold_mode='rel', cooldown=100, min_lr=1e-6, eps=1e-08, verbose=True)
        
        return ttt_optimizer


    def train(self, disable_eval_tqdm=False):
        
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        self.optimizer = self.get_ttt_opt()

        logging.info(f"Starting {self.ttt_steps} steps of test time training!")

        start_epoch = self.step // len(self.train_loader)

        ttt_losses = []
        val_metrics_list = []

        for epoch_int in range(
            1 # 1 epoch for ttt
        ):
            epoch_start_time = time.time()
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in tqdm(range(self.ttt_steps)):

                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(self.train_loader)
                    batch = next(train_loader_iter)
                    logging.info("Restarting the training loader!")
                
                # Forward, loss, backward.
                with torch.enable_grad():
                    with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                        out = self._forward(batch)
                        loss = self._compute_loss(out, batch)
                
                loss = self.scaler.scale(loss) if self.scaler else loss
                
                self._backward(loss)
                if self.ttt_lr_scheduler: 
                    self.ttt_lr_scheduler.step(loss)

                if i % self.print_every_ttt == 0:
                    logging.info(f"Step {i} loss: {loss.item()}")
                    ttt_losses.append(loss.item())

                    if self.save_ttt_results:
                        self.load_qm_head()
                        val_metrics = self.validate(split="val", disable_tqdm=disable_eval_tqdm, skip_ttt_for_logging=True)
                        self.load_freeze_head()
                        val_metrics_list.append(val_metrics)

                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

        if self.save_ttt_results:
            import pickle
            with open(f'ttt_results_{self.save_ttt_results_name}.pkl', 'wb') as f:
                pickle.dump({'ttt_losses': ttt_losses, 'val_metrics_list': val_metrics_list}, f)

            
    
    def check_diff(self):
        c = torch.load(self.freeze_head_checkpoint)['state_dict']
        m = torch.load(self.checkpoint_path)['state_dict']

        for n, p in self.model.named_parameters():
            print(n, torch.allclose(p, c[n]), torch.allclose(p, m[n]))