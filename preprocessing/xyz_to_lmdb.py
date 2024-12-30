import os
import argparse
from pathlib import Path
import pdb
import pickle
from sqlite3 import SQLITE_IOERR_TRUNCATE

import lmdb
import numpy as np
from tqdm import tqdm
from urllib import request as request
from sklearn.model_selection import train_test_split

from arrays_to_graphs import AtomsToGraphs

import torch
import ase.io

import pickle

from sgdml.intf.ase_calc import SGDMLCalculator




def write_to_lmdb(xyz_path, db_path, split_name, ref_energy=False, calc_classical=False, sgdml_path=False):

    # For the wB97M-D3(BJ)/def2-TZVPPD level of theory with PSI4
    an_2_atomic_reference_energy = {35: -70045.28385080204, 6: -1030.5671648271828, 17: -12522.649269035726, 9: -2715.318528602957, 1: -13.571964772646918, 53: -8102.524593409054, 7: -1486.3750255780376, 8: -2043.933693071156, 15: -9287.407133426237, 16: -10834.4844708122}


    a2g = AtomsToGraphs(
        max_neigh=1000,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_distances=False,
        r_edges=False,
        device='cpu'
    )


    print(f'Loading xyz file {xyz_path}....')
    atoms = ase.io.read(xyz_path, ':')
    
    R = [a.positions for a in atoms]
    forces = [a.get_forces() for a in atoms]
    energies = [a.get_potential_energy() for a in atoms]
    
    z = [a.get_atomic_numbers() for a in atoms]
    

    if args.ref_energy:
        atomic_reference_energies = [sum([an_2_atomic_reference_energy[an] for an in an_list]) for an_list in z]
        atomic_reference_energies = np.array(atomic_reference_energies)
        energies = np.array(energies) - atomic_reference_energies
    
    energies_classical = None
    forces_classical = None
    if calc_classical:
        # You might have to double check the units for your use case if using sGDML
        # (see http://docs.sgdml.org/applications.html)
        # Can also replace this with a different prior
        sgdml_calc = SGDMLCalculator(sgdml_path)
        energies_classical = [sgdml_calc.get_potential_energy(a) for a in atoms]
        forces_classical = [sgdml_calc.get_forces(a) for a in atoms]


    
    atomic_numbers = z 
    positions = R
    
    force = forces
    energy = np.array(energies)
    energy = energy.reshape(-1, 1)  # Reshape energy into 2D array
    
    lengths = np.ones(3)[None, :] * 30.
    angles = np.ones(3)[None, :] * 90.
    

    norm_stats = {
        'e_mean': energy.mean(),
        'e_std': energy.std(),
        'f_mean': np.concatenate([f.flatten() for f in force]).mean(),
        'f_std': np.concatenate([f.flatten() for f in force]).std(),
    }

    if calc_classical:
        norm_stats['e_classical_mean'] = np.array(energies_classical).mean()
        norm_stats['e_classical_std'] = np.array(energies_classical).std()
        norm_stats['f_classical_mean'] = np.concatenate([f.flatten() for f in forces_classical]).mean()
        norm_stats['f_classical_std'] = np.concatenate([f.flatten() for f in forces_classical]).std()

    save_path = Path(db_path)
    save_path.mkdir(parents=True, exist_ok=True)
    np.save(save_path / 'metadata', norm_stats)
        
    print(f'processing split {split_name}.')
    save_path = Path(db_path) / split_name
    save_path.mkdir(parents=True, exist_ok=True)
    db = lmdb.open(
        str(save_path / 'data.lmdb'),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    for idx in tqdm(range(len(positions))):
        #natoms = np.array([positions.shape[1]] * 1, dtype=np.int64)
        natoms = np.array([len(positions[idx])] * 1, dtype=np.int64)
        data = a2g.convert(natoms, positions[idx], atomic_numbers[idx], 
                        lengths, angles, energy[idx], force[idx])
        data.sid = 0
        data.fid = idx

        if calc_classical:
            data.energy_classical = torch.tensor(energies_classical[idx]).float()
            data.forces_classical = torch.tensor(forces_classical[idx]).float()

        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default='/data/shared/spice/maceoff_split')
    parser.add_argument('--xyz_path', type=str, default='/data/shared/spice/test_large_neut_all.xyz')
    parser.add_argument('--split_name', type=str, default='train')
    
    # Whether or not to calculate classical targets with sGDML
    # (calc can be modified to use a different prior)
    # To see how to train an sgdml model, see http://docs.sgdml.org/index.html
    parser.add_argument('--calc-classical', type=bool, default=False, action='store_true')
    parser.add_argument('--sgdml-path', type=str, default=None)
    
    # For spice, calculate the reference energy
    parser.add_argument('--ref-energy', type=bool, default=False, action='store_true') 
    
    args = parser.parse_args()
    
    write_to_lmdb(args.xyz_path, args.db_path, args.split_name, ref_energy=args.ref_energy, calc_classical=args.calc_classical, sgdml_path=args.sgdml_path)