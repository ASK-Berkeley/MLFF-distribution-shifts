from mace.calculators import mace_off
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric
from torch_geometric.nn import radius_graph

import numpy as np
from tqdm import tqdm

from mdsim.datasets.lmdb_dataset import LmdbDataset

from matplotlib import pyplot as plt
from ase import Atoms
import ase.io

import pickle 

from mdsim.md.ase_utils import data_to_atoms
import torch

import argparse


def get_eigvals(edge_index):
    
    edge_index = torch_geometric.utils.add_remaining_self_loops(edge_index)[0]
    laplacian = torch_geometric.utils.get_laplacian(edge_index, normalization='sym')
    L = to_scipy_sparse_matrix(laplacian[0], laplacian[1])
    eigvals = np.linalg.eigvals(L.toarray())
    return eigvals

def get_eigval_distribution(data, radius=5.0):
    edge_index = radius_graph(
                data.pos,
                r=radius,
                max_num_neighbors=50,
            )
    
    eigvals = get_eigvals(edge_index)

    return eigvals.astype(np.float32)

# Distance heuristic between eigenvalue distributions
def get_percent_within_pt1(data, r=5.0, center=1.0, width=0.1):
    ev = get_eigval_distribution(data, radius=r)
    return np.sum(np.abs(ev - center) < width)/len(ev)



def get_rad_err(calc, radii, data, opt_eig_percent=0.55, eig_center=1.0, eig_width=0.1):

        # Find the best radius for the given system    
        # (i.e. the radius that yields a connectivity that best matches the training data)
        connectivities = []
        for rad in radii:    
            rad = float(rad)
            connectivities.append(get_percent_within_pt1(data, r=rad, center=eig_center, width=eig_width))

        best_r = radii[np.argmin(np.abs(connectivities - opt_eig_percent))]

        # Perform inference with the optimal radius cutoff
        calc.r_max = best_r 
        data.cell = (torch.eye(3) * 1000).unsqueeze(0)
        atoms = data_to_atoms(data)
        atoms.set_calculator(calc)

        f_pred = atoms.get_forces()
        f_true = data.force.numpy()
        return np.abs(f_pred - f_true).mean()

def main(args):

    # Load calculator
    calc = mace_off(model=args.model, device=args.device, default_dtype='float32')

    # Load dataset
    cfg = {'src' : args.data_path}
    dataset = LmdbDataset(cfg)

    radii = np.linspace(args.r_min, args.r_max, args.n_r)

    errors = []
    for i in tqdm(range(args.n_test)):
        data = dataset[i]
        errors.append(get_rad_err(calc, radii, data, opt_eig_percent=args.opt_eig_percent, eig_center=args.eig_center, eig_width=args.eig_width))
    
    print(f'Mean error: {np.mean(errors)}')
    print(f'Std error: {np.std(errors)}')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--r_min', default=3, type=int)
    parser.add_argument('--r_max', default=9, type=int)
    parser.add_argument('--n_r', default=10, type=int)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model', type=str, default='medium')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_test', type=int, default=1000)
    # Heuristic used for radius selection to measure distance between eigenvalue distributions
    # (needs to be refined for different datasets)
    parser.add_argument('--opt_eig_percent', type=float, default=0.55) 
    parser.add_argument('--eig_center', type=float, default=1.0)
    parser.add_argument('--eig_width', type=float, default=0.1)

    args = parser.parse_args()
    main(args)