import torch


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None, element_wise=False):
        """tensor is taken as a sample to calculate the mean and std"""
        
        self.element_wise = element_wise
        if element_wise:
            self.mean = {}
            self.std = {}
            for k in mean:
                torch.tensor(mean[k]).to(device)
                torch.tensor(std[k]).to(device)
            
            return

        if tensor is None and mean is None:
            return

        if device is None:
            device = "cpu"

        if tensor is not None:
            self.mean = torch.mean(tensor, dim=0).to(device)
            self.std = torch.std(tensor, dim=0).to(device)
            return

        if mean is not None and std is not None:
            self.mean = torch.tensor(mean).to(device)
            self.std = torch.tensor(std).to(device)

    def to(self, device):

        if self.element_wise:
            for k in self.mean:
                self.mean[k] = self.mean[k].to(device)
                self.std[k] = self.std[k].to(device)
            return

        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def norm(self, tensor, element=None, mean_energy_per_system=None, std_energy_per_system=None):
        
        if mean_energy_per_system is not None and std_energy_per_system is not None:
            return (tensor - mean_energy_per_system) / std_energy_per_system
        
        if self.element_wise:
            return (tensor - self.mean[element]) / self.std[element]
        
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor, element=None, mean_energy_per_system=None, std_energy_per_system=None):
        
        if mean_energy_per_system is not None and std_energy_per_system is not None:
            return normed_tensor * std_energy_per_system + mean_energy_per_system
        
        if self.element_wise:
            return normed_tensor * self.std[element] + self.mean[element]

        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):

        if self.element_wise:
            self.mean = {}
            self.std = {}
            for k in state_dict["mean"]:
                self.mean[k] = state_dict["mean"][k].to(self.mean.device)
                self.std[k] = state_dict["std"][k].to(self.mean.device)
            return
        
        self.mean = state_dict["mean"].to(self.mean.device)
        self.std = state_dict["std"].to(self.mean.device)
