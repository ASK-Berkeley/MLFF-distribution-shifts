import bisect
import logging
import pickle
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from tqdm import tqdm
import os
from mdsim.common.registry import registry
from mdsim.datasets.lmdb_dataset import LmdbDataset


@registry.register_dataset("multi")
class MultiMoleculeDataset(Dataset):
    r"""Dataset class to load from multiple molecules

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(
        self,
        config,
        normalize_force=False,
        normalize_energy=True,
        transform=None,
        percentages=None,
        return_classical=False,
        val=False,
    ):
        super(MultiMoleculeDataset, self).__init__()

        self.config = config
        self.noise_classical_scale = self.config.get("noise_classical_scale", None)
        self.noise_scale_f_std = self.config.get("noise_scale_f_std", 1.1)
        if "return_classical" in self.config:
            # Overriding the default return_classical from config!
            self.return_classical = self.config["return_classical"]
        else:
            self.return_classical = return_classical

        self.pop_classical = self.config.get("pop_classical", False)

        if self.pop_classical and self.return_classical:
            raise ValueError(
                "pop_classical and return_classical cannot be used together!"
            )

        configs = [{"src": src} for src in self.config["src"]]

        self.lmbd_datasets = [LmdbDataset(c) for c in configs]

        logging.info(f"Calculating mean and std for datasets...")
        for dataset in self.lmbd_datasets:
            dataset.calculate_mean_std_energy()

        if percentages is not None:
            self.lmbd_datasets = [
                Subset(dataset, self.get_indices(len(dataset), percentage))
                for dataset, percentage in zip(self.lmbd_datasets, percentages)
            ]

        flip = -1 if config.get("force_norm_flip", False) else 1
        if config.get("force_norm_cutoff", None) is not None:
            assert len(self.lmbd_datasets) == 1
            base_path = configs[0]['src']
            fn_path = f'{base_path}/force_norm_idx_cutoff={config["force_norm_cutoff"]}_flip={flip}.npy'
            
            if os.path.isfile(fn_path):
                idices_per_dataset = np.load(fn_path)
            else:
                cutoff = config["force_norm_cutoff"]
                idices_per_dataset = []

                for d in self.lmbd_datasets:
                    idx = []
                    for i, x in tqdm(enumerate(d)):
                        if x.force.norm(dim=-1).mean() * flip < cutoff * flip:
                            idx.append(i)
                    idices_per_dataset.append(idx)

                np.save(fn_path, idices_per_dataset)
            self.lmbd_datasets = [Subset(self.lmbd_datasets[i], idices_per_dataset[i]) for i in range(len(self.lmbd_datasets))]

        self.dataset = torch.utils.data.ConcatDataset(self.lmbd_datasets)
       

    def get_indices(self, length, percentage):
        idx = np.random.permutation(length)[: int(length * percentage)]
        return idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        obj = self.dataset[idx]

        if self.return_classical:
            # Used to simulate a worse prior
            if self.noise_classical_scale is not None:
                obj.energy_classical = (
                    obj.energy_classical
                    + torch.randn_like(obj.energy_classical)
                    * self.noise_classical_scale
                    * obj.energy_classical_std
                )
                obj.forces_classical = (
                    obj.forces_classical
                    + torch.randn_like(obj.forces_classical)
                    * self.noise_classical_scale
                    * self.noise_scale_f_std
                )

            if (
                "forces_classical" in obj
            ):  # joint training dataset so set target to classical
                obj.force = torch.tensor(obj.forces_classical)
                obj.y = torch.tensor(obj.energy_classical)
                obj.energy_mean = obj.energy_classical_mean
                obj.energy_std = obj.energy_classical_std

        if self.pop_classical:
            if "forces_classical" in obj:
                obj.pop("forces_classical")
                obj.pop("energy_classical")
                if "energy_classical_mean" in obj:
                    obj.pop("energy_classical_mean")
                    obj.pop("energy_classical_std")
                
        obj.y = torch.tensor(obj.y)
        obj.cell = (torch.eye(3) * 1000.0).unsqueeze(dim=0)
        obj.energy_mean = torch.tensor(obj.energy_mean)
        obj.energy_std = torch.tensor(obj.energy_std)
        obj.fixed = torch.zeros(obj.natoms, dtype=torch.bool)

        return obj

    def close_db(self):
        for dataset in self.lmbd_datasets:
            if type(dataset) == LmdbDataset:
                dataset.close_db()
            else:
                dataset.dataset.close_db()
