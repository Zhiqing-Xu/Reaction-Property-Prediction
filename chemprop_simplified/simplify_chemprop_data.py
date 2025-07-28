import threading
import logging
from collections import OrderedDict,defaultdict
from random import Random
from typing import Dict, Iterator, List, Optional, Union, Tuple
import os
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem
from tap import Tap 
from tqdm import tqdm
from simplify_constants import StandardScaler, AtomBondScaler
from simplify_chemprop_features import make_mol,get_features_generator,BatchMolGraph, MolGraph,is_explicit_h, is_reaction, is_adding_hs, is_mol, is_keeping_atom_map,load_features, load_valid_atom_or_bond_features, is_mol
import csv
import ctypes
from logging import Logger
import pickle
from tqdm import tqdm
from typing_extensions import Literal
import torch
Metric = Literal['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein', 'f1', 'mcc', 'bounded_rmse', 'bounded_mae', 'bounded_mse']

# from scaffold
from typing import Dict, List, Set, Tuple, Union
import warnings
from rdkit.Chem.Scaffolds import MurckoScaffold

from simplify_chemprop_features import load_features, load_valid_atom_or_bond_features, is_mol
# Cache of graph featurizations
CACHE_GRAPH = True
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}

# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]] = {}
def cache_graph() -> bool:
    r"""Returns whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    return CACHE_GRAPH

def set_cache_graph(cache_graph: bool) -> None:
    r"""Sets whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph

def empty_cache():
    r"""Empties the cache of :class:`~chemprop.features.MolGraph` and RDKit molecules."""
    SMILES_TO_GRAPH.clear()
    SMILES_TO_MOL.clear()

def cache_mol() -> bool:
    r"""Returns whether RDKit molecules will be cached."""
    return CACHE_MOL

def set_cache_mol(cache_mol: bool) -> None:
    r"""Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol

class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    def __init__(self,
                 smiles: List[str],
                 targets: List[Optional[float]] = None,
                 atom_targets: List[Optional[float]] = None,
                 bond_targets: List[Optional[float]] = None,
                 row: OrderedDict = None,
                 data_weight: float = None,
                 gt_targets: List[bool] = None,
                 lt_targets: List[bool] = None,
                 features: np.ndarray = None,
                 features_generator: List[str] = None,
                 phase_features: List[float] = None,
                 atom_features: np.ndarray = None,
                 atom_descriptors: np.ndarray = None,
                 bond_features: np.ndarray = None,
                 bond_descriptors: np.ndarray = None,
                 raw_constraints: np.ndarray = None,
                 constraints: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False):
        """
        :param smiles: A list of the SMILES strings for the molecules.
        :param targets: A list of targets for the molecule (contains None for unknown target values).
        :param atom_targets: A list of targets for the atomic properties.
        :param bond_targets: A list of targets for the bond properties.
        :param row: The raw CSV row containing the information for this molecule.
        :param data_weight: Weighting of the datapoint for the loss function.
        :param gt_targets: Indicates whether the targets are an inequality regression target of the form ">x".
        :param lt_targets: Indicates whether the targets are an inequality regression target of the form "<x".
        :param features: A numpy array containing additional features (e.g., Morgan fingerprint).
        :param features_generator: A list of features generators to use.
        :param phase_features: A one-hot vector indicating the phase of the data, as used in spectra data.
        :param atom_descriptors: A numpy array containing additional atom descriptors to featurize the molecule.
        :param bond_descriptors: A numpy array containing additional bond descriptors to featurize the molecule.
        :param raw_constraints: A numpy array containing all user-provided atom/bond-level constraints in input data.
        :param constraints: A numpy array containing atom/bond-level constraints that are used in training. Param constraints is a subset of param raw_constraints.
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features.
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features.

        """
        self.smiles = smiles
        self.targets = targets
        self.atom_targets = atom_targets
        self.bond_targets = bond_targets
        self.row = row
        self.features = features
        self.features_generator = features_generator
        self.phase_features = phase_features
        self.atom_descriptors = atom_descriptors
        self.bond_descriptors = bond_descriptors
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.constraints = constraints
        self.raw_constraints = raw_constraints
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features
        self.is_mol_list = [is_mol(s) for s in smiles]
        self.is_reaction_list = [is_reaction(x) for x in self.is_mol_list]
        self.is_explicit_h_list = [is_explicit_h(x) for x in self.is_mol_list]
        self.is_adding_hs_list = [is_adding_hs(x) for x in self.is_mol_list]
        self.is_keeping_atom_map_list = [is_keeping_atom_map(x) for x in self.is_mol_list]

        if data_weight is not None:
            self.data_weight = data_weight
        if gt_targets is not None:
            self.gt_targets = gt_targets
        if lt_targets is not None:
            self.lt_targets = lt_targets

        # Generate additional features if given a generator
        if self.features_generator is not None:
            if self.features is None:
                self.features = []
            else:
                self.features = list(self.features)

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                for m, reaction in zip(self.mol, self.is_reaction_list):
                    if not reaction:
                        if m is not None and m.GetNumHeavyAtoms() > 0:
                            self.features.extend(features_generator(m))
                        # for H2
                        elif m is not None and m.GetNumHeavyAtoms() == 0:
                            # not all features are equally long, so use methane as dummy molecule to determine length
                            self.features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))                           
                    else:
                        if m[0] is not None and m[1] is not None and m[0].GetNumHeavyAtoms() > 0:
                            self.features.extend(features_generator(m[0]))
                        elif m[0] is not None and m[1] is not None and m[0].GetNumHeavyAtoms() == 0:
                            self.features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))   
                    

            self.features = np.array(self.features)

        # Fix nans in features
        replace_token = 0
        if self.features is not None:
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Fix nans in atom_descriptors
        if self.atom_descriptors is not None:
            self.atom_descriptors = np.where(np.isnan(self.atom_descriptors), replace_token, self.atom_descriptors)

        # Fix nans in atom_features
        if self.atom_features is not None:
            self.atom_features = np.where(np.isnan(self.atom_features), replace_token, self.atom_features)

        # Fix nans in bond_descriptors
        if self.bond_descriptors is not None:
            self.bond_descriptors = np.where(np.isnan(self.bond_descriptors), replace_token, self.bond_descriptors)

        # Fix nans in bond_features
        if self.bond_features is not None:
            self.bond_features = np.where(np.isnan(self.bond_features), replace_token, self.bond_features)

        # Save a copy of the raw features and targets to enable different scaling later on
        self.raw_features, self.raw_targets, self.raw_atom_targets, self.raw_bond_targets = \
            self.features, self.targets, self.atom_targets, self.bond_targets
        self.raw_atom_descriptors, self.raw_atom_features, self.raw_bond_descriptors, self.raw_bond_features = \
            self.atom_descriptors, self.atom_features, self.bond_descriptors, self.bond_features

    @property
    def mol(self) -> List[Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]]:
        """Gets the corresponding list of RDKit molecules for the corresponding SMILES list."""
        mol = make_mols(self.smiles, self.is_reaction_list, self.is_explicit_h_list, self.is_adding_hs_list, self.is_keeping_atom_map_list)
        if cache_mol():
            for s, m in zip(self.smiles, mol):
                SMILES_TO_MOL[s] = m

        return mol

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in the :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return len(self.smiles)

    @property
    def number_of_atoms(self) -> int:
        """
        Gets the number of atoms in the :class:`MoleculeDatapoint`.

        :return: A list of number of atoms for each molecule.
        """
        return [len(self.mol[i].GetAtoms()) for i in range(self.number_of_molecules)]

    @property
    def number_of_bonds(self) -> List[int]:
        """
        Gets the number of bonds in the :class:`MoleculeDatapoint`.

        :return: A list of number of bonds for each molecule.
        """
        return [len(self.mol[i].GetBonds()) for i in range(self.number_of_molecules)]

    @property
    def bond_types(self) -> List[List[float]]:
        """
        Gets the bond types in the :class:`MoleculeDatapoint`.

        :return: A list of bond types for each molecule.
        """
        return [[b.GetBondTypeAsDouble() for b in self.mol[i].GetBonds()] for i in range(self.number_of_molecules)]

    def set_features(self, features: np.ndarray) -> None:
        """
        Sets the features of the molecule.

        :param features: A 1D numpy array of features for the molecule.
        """
        self.features = features

    def set_atom_descriptors(self, atom_descriptors: np.ndarray) -> None:
        """
        Sets the atom descriptors of the molecule.

        :param atom_descriptors: A 1D numpy array of atom descriptors for the molecule.
        """
        self.atom_descriptors = atom_descriptors

    def set_atom_features(self, atom_features: np.ndarray) -> None:
        """
        Sets the atom features of the molecule.

        :param atom_features: A 1D numpy array of atom features for the molecule.
        """
        self.atom_features = atom_features

    def set_bond_descriptors(self, bond_descriptors: np.ndarray) -> None:
        """
        Sets the atom descriptors of the molecule.

        :param bond_descriptors: A 1D numpy array of bond descriptors for the molecule.
        """
        self.bond_descriptors = bond_descriptors

    def set_bond_features(self, bond_features: np.ndarray) -> None:
        """
        Sets the bond features of the molecule.

        :param bond_features: A 1D numpy array of bond features for the molecule.
        """
        self.bond_features = bond_features

    def extend_features(self, features: np.ndarray) -> None:
        """
        Extends the features of the molecule.

        :param features: A 1D numpy array of extra features for the molecule.
        """
        self.features = np.append(self.features, features) if self.features is not None else features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        self.features, self.targets, self.atom_targets, self.bond_targets = \
            self.raw_features, self.raw_targets, self.raw_atom_targets, self.raw_bond_targets
        self.atom_descriptors, self.atom_features, self.bond_descriptors, self.bond_features = \
            self.raw_atom_descriptors, self.raw_atom_features, self.raw_bond_descriptors, self.raw_bond_features

class MoleculeDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to their attributes."""

    def __init__(self, data: List[MoleculeDatapoint]):
        r"""
        :param data: A list of :class:`MoleculeDatapoint`\ s.
        """
        self._data = data
        self._batch_graph = None
        self._random = Random()

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Returns a list containing the SMILES list associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
        """
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]

        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]], List[Tuple[Chem.Mol, Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]]]:
        """
        Returns a list of the RDKit molecules associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned RDKit molecules to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of RDKit molecules, depending on :code:`flatten`.
        """
        if flatten:
            return [mol for d in self._data for mol in d.mol]

        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in each :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    @property
    def number_of_atoms(self) -> List[List[int]]:
        """
        Gets the number of atoms in each :class:`MoleculeDatapoint`.

        :return: A list of number of atoms for each molecule.
        """
        return [d.number_of_atoms for d in self._data]

    @property
    def number_of_bonds(self) -> List[List[int]]:
        """
        Gets the number of bonds in each :class:`MoleculeDatapoint`.

        :return: A list of number of bonds for each molecule.
        """
        return [d.number_of_bonds for d in self._data]

    @property
    def bond_types(self) -> List[List[float]]:
        """
        Gets the bond types in each :class:`MoleculeDatapoint`.

        :return: A list of bond types for each molecule.
        """
        return [d.bond_types for d in self._data]

    @property
    def is_atom_bond_targets(self) -> bool:
        """
        Gets the Boolean whether this is atomic/bond properties prediction.

        :return: A Boolean value.
        """
        if self._data[0].atom_targets is None and self._data[0].bond_targets is None:
            return False
        else:
            return True




    def batch_graph(self) -> List[BatchMolGraph]:
        r"""
        Constructs a :class:`~chemprop.features.BatchMolGraph` with the graph featurization of all the molecules.

        .. note::
           The :class:`~chemprop.features.BatchMolGraph` is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to :meth:`batch_graph`. This means that if the underlying
           set of :class:`MoleculeDatapoint`\ s changes, then the returned :class:`~chemprop.features.BatchMolGraph`
           will be incorrect for the underlying data.

        :return: A list of :class:`~chemprop.features.BatchMolGraph` containing the graph featurization of all the
                 molecules in each :class:`MoleculeDatapoint`.
        """
        if self._batch_graph is None:
            self._batch_graph = []

            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for s, m in zip(d.smiles, d.mol):
                    if s in SMILES_TO_GRAPH:
                        mol_graph = SMILES_TO_GRAPH[s]
                    else:
                        if len(d.smiles) > 1 and (d.atom_features is not None or d.bond_features is not None):
                            raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                                      'per input (i.e., number_of_molecules = 1).')

                        mol_graph = MolGraph(m, d.atom_features, d.bond_features,
                                             overwrite_default_atom_features=d.overwrite_default_atom_features,
                                             overwrite_default_bond_features=d.overwrite_default_bond_features)
                        if cache_graph():
                            SMILES_TO_GRAPH[s] = mol_graph
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)

            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]

        return self._batch_graph




    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def phase_features(self) -> List[np.ndarray]:
        """
        Returns the phase features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the phase features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].phase_features is None:
            return None

        return [d.phase_features for d in self._data]

    def atom_features(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_features is None:
            return None

        return [d.atom_features for d in self._data]

    def atom_descriptors(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_descriptors is None:
            return None

        return [d.atom_descriptors for d in self._data]

    def bond_features(self) -> List[np.ndarray]:
        """
        Returns the bond features associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the bond features
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].bond_features is None:
            return None

        return [d.bond_features for d in self._data]

    def bond_descriptors(self) -> List[np.ndarray]:
        """
        Returns the bond descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the bond descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].bond_descriptors is None:
            return None

        return [d.bond_descriptors for d in self._data]

    def constraints(self) -> List[np.ndarray]:
        """
        Return the constraints applied in atomic/bond properties prediction.
        """
        constraints = []
        for d in self._data:
            if d.constraints is None :
                natom_targets = len(d.atom_targets) if d.atom_targets is not None else 0
                nbond_targets = len(d.bond_targets) if d.bond_targets is not None else 0
                ntargets = natom_targets + nbond_targets
                constraints.append([None] * ntargets)
            else:
                constraints.append(d.constraints)
        return constraints

    def data_weights(self) -> List[float]:
        """
        Returns the loss weighting associated with each datapoint.
        """
        if not hasattr(self._data[0], 'data_weight'):
            return [1. for d in self._data]

        return [d.data_weight for d in self._data]

    def atom_bond_data_weights(self) -> List[List[float]]:
        """
        Returns the loss weighting associated with each datapoint for atomic/bond properties prediction.
        """
        targets = self.targets()
        data_weights = self.data_weights()
        atom_bond_data_weights = [[] for _ in targets[0]]
        for i, tb in enumerate(targets):
            weight = data_weights[i]
            for j, x in enumerate(tb): 
                atom_bond_data_weights[j] += [1. * weight] * len(x)

        return atom_bond_data_weights

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.targets for d in self._data]
    
    def mask(self) -> List[List[bool]]:
        """
        Returns whether the targets associated with each molecule and task are present.

        :return: A list of list of booleans associated with targets.
        """
        targets = self.targets()
        if self.is_atom_bond_targets:
            mask = []
            for dt in zip(*targets):
                dt = np.concatenate(dt)
                mask.append([x is not None for x in dt])
        else:
            mask = [[t is not None for t in dt] for dt in targets]
            mask = list(zip(*mask))
        return mask

    def gt_targets(self) -> List[np.ndarray]:
        """
        Returns indications of whether the targets associated with each molecule are greater-than inequalities.
        
        :return: A list of lists of booleans indicating whether the targets in those positions are greater-than inequality targets.
        """
        if not hasattr(self._data[0], 'gt_targets'):
            return None

        return [d.gt_targets for d in self._data]

    def lt_targets(self) -> List[np.ndarray]:
        """
        Returns indications of whether the targets associated with each molecule are less-than inequalities.
        
        :return: A list of lists of booleans indicating whether the targets in those positions are less-than inequality targets.
        """
        if not hasattr(self._data[0], 'lt_targets'):
            return None

        return [d.lt_targets for d in self._data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the additional features vector associated with the molecules.

        :return: The size of the additional features vector.
        """
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def atom_descriptors_size(self) -> int:
        """
        Returns the size of custom additional atom descriptors vector associated with the molecules.

        :return: The size of the additional atom descriptor vector.
        """
        return len(self._data[0].atom_descriptors[0]) \
            if len(self._data) > 0 and self._data[0].atom_descriptors is not None else None

    def atom_features_size(self) -> int:
        """
        Returns the size of custom additional atom features vector associated with the molecules.

        :return: The size of the additional atom feature vector.
        """
        return len(self._data[0].atom_features[0]) \
            if len(self._data) > 0 and self._data[0].atom_features is not None else None

    def bond_descriptors_size(self) -> int:
        """
        Returns the size of custom additional bond descriptors vector associated with the molecules.

        :return: The size of the additional bond descriptor vector.
        """
        return len(self._data[0].bond_descriptors[0]) \
            if len(self._data) > 0 and self._data[0].bond_descriptors is not None else None

    def bond_features_size(self) -> int:
        """
        Returns the size of custom additional bond features vector associated with the molecules.

        :return: The size of the additional bond feature vector.
        """
        return len(self._data[0].bond_features[0]) \
            if len(self._data) > 0 and self._data[0].bond_features is not None else None

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0,
                           scale_atom_descriptors: bool = False, scale_bond_descriptors: bool = False) -> StandardScaler:
        """
        Normalizes the features of the dataset using a :class:`~chemprop.data.StandardScaler`.

        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each feature independently.

        If a :class:`~chemprop.data.StandardScaler` is provided, it is used to perform the normalization.
        Otherwise, a :class:`~chemprop.data.StandardScaler` is first fit to the features in this dataset
        and is then used to perform the normalization.

        :param scaler: A fitted :class:`~chemprop.data.StandardScaler`. If it is provided it is used,
                       otherwise a new :class:`~chemprop.data.StandardScaler` is first fitted to this
                       data and is then used.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        :param scale_atom_descriptors: If the features that need to be scaled are atom features rather than molecule.
        :param scale_bond_descriptors: If the features that need to be scaled are bond features rather than molecule.
        :return: A fitted :class:`~chemprop.data.StandardScaler`. If a :class:`~chemprop.data.StandardScaler`
                 is provided as a parameter, this is the same :class:`~chemprop.data.StandardScaler`. Otherwise,
                 this is a new :class:`~chemprop.data.StandardScaler` that has been fit on this dataset.
        """
        if len(self._data) == 0 or \
                (self._data[0].features is None and not scale_bond_descriptors and not scale_atom_descriptors):
            return None

        if scaler is None:
            if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
                features = np.vstack([d.raw_atom_descriptors for d in self._data])
            elif scale_atom_descriptors and not self._data[0].atom_features is None:
                features = np.vstack([d.raw_atom_features for d in self._data])
            elif scale_bond_descriptors and not self._data[0].bond_descriptors is None:
                features = np.vstack([d.raw_bond_descriptors for d in self._data])
            elif scale_bond_descriptors and not self._data[0].bond_features is None:
                features = np.vstack([d.raw_bond_features for d in self._data])
            else:
                features = np.vstack([d.raw_features for d in self._data])
            scaler = StandardScaler(replace_nan_token=replace_nan_token)
            scaler.fit(features)

        if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
            for d in self._data:
                d.set_atom_descriptors(scaler.transform(d.raw_atom_descriptors))
        elif scale_atom_descriptors and not self._data[0].atom_features is None:
            for d in self._data:
                d.set_atom_features(scaler.transform(d.raw_atom_features))
        elif scale_bond_descriptors and not self._data[0].bond_descriptors is None:
            for d in self._data:
                d.set_bond_descriptors(scaler.transform(d.raw_bond_descriptors))
        elif scale_bond_descriptors and not self._data[0].bond_features is None:
            for d in self._data:
                d.set_bond_features(scaler.transform(d.raw_bond_features))
        else:
            for d in self._data:
                d.set_features(scaler.transform(d.raw_features.reshape(1, -1))[0])

        return scaler

    def normalize_targets(self) -> StandardScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.
        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.
        This should only be used for regression datasets.
        :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
        """
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

        return scaler

    def normalize_atom_bond_targets(self) -> AtomBondScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.AtomBondScaler`.

        The :class:`~chemprop.data.AtomBondScaler` subtracts the mean and divides by the standard deviation
        for each task independently.

        This should only be used for regression datasets.

        :return: A :class:`~chemprop.data.AtomBondScaler` fitted to the targets.
        """
        atom_targets = self._data[0].atom_targets
        bond_targets = self._data[0].bond_targets
        n_atom_targets = len(atom_targets) if atom_targets is not None else 0
        n_bond_targets = len(bond_targets) if bond_targets is not None else 0
        n_atoms, n_bonds = self.number_of_atoms, self.number_of_bonds

        targets = [d.raw_targets for d in self._data]
        targets = [np.concatenate(x).reshape([-1, 1]) for x in zip(*targets)]
        scaler = AtomBondScaler(
            n_atom_targets=n_atom_targets,
            n_bond_targets=n_bond_targets,
        ).fit(targets)
        scaled_targets = scaler.transform(targets)
        for i in range(n_atom_targets):
            scaled_targets[i] = np.split(np.array(scaled_targets[i]).flatten(), np.cumsum(np.array(n_atoms)))[:-1]
        for i in range(n_bond_targets):
            scaled_targets[i+n_atom_targets] = np.split(np.array(scaled_targets[i+n_atom_targets]).flatten(), np.cumsum(np.array(n_bonds)))[:-1]
        scaled_targets = np.array(scaled_targets, dtype=object).T
        self.set_targets(scaled_targets)

        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        if not len(self._data) == len(targets):
            raise ValueError(
                "number of molecules and targets must be of same length! "
                f"num molecules: {len(self._data)}, num targets: {len(targets)}"
            )
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]

class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([any(target == 1 for target in datapoint.targets) for datapoint in dataset])

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length

def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    r"""
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.

    Additionally, precomputes the :class:`~chemprop.features.BatchMolGraph` for the constructed
    :class:`MoleculeDataset`.

    :param data: A list of :class:`MoleculeDatapoint`\ s.
    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\ s.
    """
    data = MoleculeDataset(data)
    data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

    return data

class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def gt_targets(self) -> List[List[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')
        
        if not hasattr(self._dataset[0],'gt_targets'):
            return None

        return [self._dataset[index].gt_targets for index in self._sampler]

    @property
    def lt_targets(self) -> List[List[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        if not hasattr(self._dataset[0],'lt_targets'):
            return None

        return [self._dataset[index].lt_targets for index in self._sampler]


    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()

def make_mols(smiles: List[str], reaction_list: List[bool], keep_h_list: List[bool], add_h_list: List[bool], keep_atom_map_list: List[bool]):
    """
    Builds a list of RDKit molecules (or a list of tuples of molecules if reaction is True) for a list of smiles.

    :param smiles: List of SMILES strings.
    :param reaction_list: List of booleans whether the SMILES strings are to be treated as a reaction.
    :param keep_h_list: List of booleans whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h_list: List of booleasn whether to add hydrogens to the input smiles.
    :param keep_atom_map_list: List of booleasn whether to keep the original atom mapping.
    :return: List of RDKit molecules or list of tuple of molecules.
    """
    mol = []
    for s, reaction, keep_h, add_h, keep_atom_map in zip(smiles, reaction_list, keep_h_list, add_h_list, keep_atom_map_list):
        if reaction:
            mol.append(SMILES_TO_MOL[s] if s in SMILES_TO_MOL else (make_mol(s.split(">")[0], keep_h, add_h, keep_atom_map), make_mol(s.split(">")[-1], keep_h, add_h, keep_atom_map)))
        else:
            mol.append(SMILES_TO_MOL[s] if s in SMILES_TO_MOL else make_mol(s, keep_h, add_h, keep_atom_map))
    return mol


class CommonArgs(Tap):
    """:class:`CommonArgs` contains arguments that are used in both :class:`TrainArgs` and :class:`PredictArgs`."""

    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    number_of_molecules: int = 1
    """Number of molecules in each input to the model.
    This must equal the length of :code:`smiles_columns` (if not :code:`None`)."""
    checkpoint_dir: str = None
    """Directory from which to load model checkpoints (walks directory and ensembles all models that are found)."""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pt` files)."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    features_generator: List[str] = None
    """Method(s) of generating additional features."""
    features_path: List[str] = None
    """Path(s) to features to use in FNN (instead of features_generator)."""
    phase_features_path: str = None
    """Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype."""
    no_features_scaling: bool = False
    """Turn off scaling of features."""
    max_data_size: int = None
    """Maximum number of data points to load."""
    num_workers: int = 8
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 50
    """Batch size."""
    atom_descriptors: Literal['feature', 'descriptor'] = None
    """
    Custom extra atom descriptors.
    :code:`feature`: used as atom features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned atomic representation.
    """
    atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    bond_descriptors: Literal['feature', 'descriptor'] = None
    """
    Custom extra bond descriptors.
    :code:`feature`: used as bond features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned bond representation.
    """
    bond_descriptors_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""
    no_cache_mol: bool = False
    """
    Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).
    """
    empty_cache: bool = False
    """
    Whether to empty all caches before training or predicting. This is necessary if multiple jobs are run within a single script and the atom or bond features change.
    """
    constraints_path: str = None
    """
    Path to constraints applied to atomic/bond properties prediction.
    """

    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)
        self._atom_features_size = 0
        self._bond_features_size = 0
        self._atom_descriptors_size = 0
        self._bond_descriptors_size = 0
        self._atom_constraints = []
        self._bond_constraints = []

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    @property
    def features_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional molecule-level features.
        """
        return not self.no_features_scaling

    @features_scaling.setter
    def features_scaling(self, features_scaling: bool) -> None:
        self.no_features_scaling = not features_scaling

    @property
    def atom_features_size(self) -> int:
        """The size of the atom features."""
        return self._atom_features_size

    @atom_features_size.setter
    def atom_features_size(self, atom_features_size: int) -> None:
        self._atom_features_size = atom_features_size

    @property
    def atom_descriptors_size(self) -> int:
        """The size of the atom descriptors."""
        return self._atom_descriptors_size

    @atom_descriptors_size.setter
    def atom_descriptors_size(self, atom_descriptors_size: int) -> None:
        self._atom_descriptors_size = atom_descriptors_size

    @property
    def bond_features_size(self) -> int:
        """The size of the atom features."""
        return self._bond_features_size

    @bond_features_size.setter
    def bond_features_size(self, bond_features_size: int) -> None:
        self._bond_features_size = bond_features_size

    @property
    def bond_descriptors_size(self) -> int:
        """The size of the bond descriptors."""
        return self._bond_descriptors_size

    @bond_descriptors_size.setter
    def bond_descriptors_size(self, bond_descriptors_size: int) -> None:
        self._bond_descriptors_size = bond_descriptors_size

    def configure(self) -> None:
        self.add_argument('--gpu', choices=list(range(torch.cuda.device_count())))
        self.add_argument('--features_generator', choices=get_available_features_generators())

    def process_args(self) -> None:
        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Validate features
        if self.features_generator is not None and 'rdkit_2d_normalized' in self.features_generator and self.features_scaling:
            raise ValueError('When using rdkit_2d_normalized features, --no_features_scaling must be specified.')

        # Validate atom descriptors
        if (self.atom_descriptors is None) != (self.atom_descriptors_path is None):
            raise ValueError('If atom_descriptors is specified, then an atom_descriptors_path must be provided '
                             'and vice versa.')

        if self.atom_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                      'per input (i.e., number_of_molecules = 1).')

        # Validate bond descriptors
        if (self.bond_descriptors is None) != (self.bond_descriptors_path is None):
            raise ValueError('If bond_descriptors is specified, then an bond_descriptors_path must be provided '
                             'and vice versa.')

        if self.bond_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError('Bond descriptors are currently only supported with one molecule '
                                      'per input (i.e., number_of_molecules = 1).')

        set_cache_mol(not self.no_cache_mol)

        if self.empty_cache:
            empty_cache()

class TrainArgs(CommonArgs):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""

    # General arguments
    data_path: str
    """Path to data CSV file."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    By default, uses all columns except the SMILES column and the :code:`ignore_columns`.
    """
    ignore_columns: List[str] = None
    """Name of the columns to ignore when :code:`target_columns` is not provided."""
    dataset_type: Literal['regression', 'classification', 'multiclass', 'spectra']
    """Type of dataset. This determines the default loss function used during training."""
    loss_function: Literal['mse', 'bounded_mse', 'binary_cross_entropy', 'cross_entropy', 'mcc', 'sid', 'wasserstein', 'mve', 'evidential', 'dirichlet'] = None
    """Choice of loss function. Loss functions are limited to compatible dataset types."""
    multiclass_num_classes: int = 3
    """Number of classes when running multiclass classification."""
    separate_val_path: str = None
    """Path to separate val set, optional."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    spectra_phase_mask_path: str = None
    """Path to a file containing a phase mask array, used for excluding particular regions in spectra predictions."""
    data_weights_path: str = None
    """Path to weights for each molecule in the training data, affecting the relative weight of molecules in the loss function"""
    target_weights: List[float] = None
    """Weights associated with each target, affecting the relative weight of targets in the loss function. Must match the number of target columns."""
    split_type: Literal['random', 'scaffold_balanced', 'predetermined', 'crossval', 'cv', 'cv-no-test', 'index_predetermined', 'random_with_repeated_smiles'] = 'random'
    """Method of splitting the data into train/val/test."""
    split_sizes: List[float] = None
    """Split proportions for train/validation/test sets."""
    split_key_molecule: int = 0
    """The index of the key molecule used for splitting when multiple molecules are present and constrained split_type is used, like scaffold_balanced or random_with_repeated_smiles.
       Note that this index begins with zero for the first molecule."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    folds_file: str = None
    """Optional file of fold labels."""
    val_fold_index: int = None
    """Which fold to use as val for leave-one-out cross val."""
    test_fold_index: int = None
    """Which fold to use as test for leave-one-out cross val."""
    crossval_index_dir: str = None
    """Directory in which to find cross validation index files."""
    crossval_index_file: str = None
    """Indices of files to use as train/val/test. Overrides :code:`--num_folds` and :code:`--seed`."""
    seed: int = 0
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    pytorch_seed: int = 0
    """Seed for PyTorch randomness (e.g., random initial weights)."""
    metric: Metric = None
    """
    Metric to use during evaluation. It is also used with the validation set for early stopping.
    Defaults to "auc" for classification, "rmse" for regression, and "sid" for spectra.
    """
    extra_metrics: List[Metric] = []
    """Additional metrics to use to evaluate the model. Not used for early stopping."""
    save_dir: str = None
    """Directory where model checkpoints will be saved."""
    checkpoint_frzn: str = None
    """Path to model checkpoint file to be loaded for overwriting and freezing weights."""
    save_smiles_splits: bool = False
    """Save smiles for each train/val/test splits for prediction convenience later."""
    test: bool = False
    """Whether to skip training and only test the model."""
    quiet: bool = False
    """Skip non-essential print statements."""
    log_frequency: int = 10
    """The number of batches between each logging of the training loss."""
    show_individual_scores: bool = False
    """Show all scores for individual targets, not just average, at the end."""
    cache_cutoff: float = 10000
    """
    Maximum number of molecules in dataset to allow caching.
    Below this number, caching is used and data loading is sequential.
    Above this number, caching is not used and data loading is parallel.
    Use "inf" to always cache.
    """
    save_preds: bool = False
    """Whether to save test split predictions during training."""
    resume_experiment: bool = False
    """
    Whether to resume the experiment.
    Loads test results from any folds that have already been completed and skips training those folds.
    """

    # Model arguments
    bias: bool = False
    """Whether to add bias to linear layers."""
    hidden_size: int = 300
    """Dimensionality of hidden layers in MPN."""
    depth: int = 3
    """Number of message passing steps."""
    bias_solvent: bool = False
    """Whether to add bias to linear layers for solvent MPN if :code:`reaction_solvent` is True."""
    hidden_size_solvent: int = 300
    """Dimensionality of hidden layers in solvent MPN if :code:`reaction_solvent` is True."""
    depth_solvent: int = 3
    """Number of message passing steps for solvent if :code:`reaction_solvent` is True."""
    mpn_shared: bool = False
    """Whether to use the same message passing neural network for all input molecules
    Only relevant if :code:`number_of_molecules > 1`"""
    dropout: float = 0.0
    """Dropout probability."""
    activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'
    """Activation function."""
    atom_messages: bool = False
    """Centers messages on atoms instead of on bonds."""
    undirected: bool = False
    """Undirected edges (always sum the two relevant bond vectors)."""
    ffn_hidden_size: int = None
    """Hidden dim for higher-capacity FFN (defaults to hidden_size)."""
    ffn_num_layers: int = 2
    """Number of layers in FFN after MPN encoding."""
    features_only: bool = False
    """Use only the additional features in an FFN, no graph network."""
    separate_val_features_path: List[str] = None
    """Path to file with features for separate val set."""
    separate_test_features_path: List[str] = None
    """Path to file with features for separate test set."""
    separate_val_phase_features_path: str = None
    """Path to file with phase features for separate val set."""
    separate_test_phase_features_path: str = None
    """Path to file with phase features for separate test set."""
    separate_val_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    separate_val_bond_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_bond_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    separate_val_constraints_path: str = None
    """Path to file with constraints for separate val set."""
    separate_test_constraints_path: str = None
    """Path to file with constraints for separate test set."""
    config_path: str = None
    """
    Path to a :code:`.json` file containing arguments. Any arguments present in the config file
    will override arguments specified via the command line or by the defaults.
    """
    ensemble_size: int = 1
    """Number of models in ensemble."""
    aggregation: Literal['mean', 'sum', 'norm'] = 'mean'
    """Aggregation scheme for atomic vectors into molecular vectors"""
    aggregation_norm: int = 100
    """For norm aggregation, number by which to divide summed up atomic features"""
    reaction: bool = False
    """
    Whether to adjust MPNN layer to take reactions as input instead of molecules.
    """
    reaction_mode: Literal['reac_prod', 'reac_diff', 'prod_diff', 'reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance'] = 'reac_diff'
    """
    Choices for construction of atom and bond features for reactions
    :code:`reac_prod`: concatenates the reactants feature with the products feature.
    :code:`reac_diff`: concatenates the reactants feature with the difference in features between reactants and products.
    :code:`prod_diff`: concatenates the products feature with the difference in features between reactants and products.
    :code:`reac_prod_balance`: concatenates the reactants feature with the products feature, balances imbalanced reactions.
    :code:`reac_diff_balance`: concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions.
    :code:`prod_diff_balance`: concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions.
    """
    reaction_solvent: bool = False
    """
    Whether to adjust the MPNN layer to take as input a reaction and a molecule, and to encode them with separate MPNNs.
    """
    explicit_h: bool = False
    """
    Whether H are explicitly specified in input (and should be kept this way). This option is intended to be used
    with the :code:`reaction` or :code:`reaction_solvent` options, and applies only to the reaction part.
    """
    adding_h: bool = False
    """
    Whether RDKit molecules will be constructed with adding the Hs to them. This option is intended to be used
    with Chemprop's default molecule or multi-molecule encoders, or in :code:`reaction_solvent` mode where it applies to the solvent only.
    """
    is_atom_bond_targets: bool = False
    """
    whether this is atomic/bond properties prediction.
    """
    keeping_atom_map: bool = False
    """
    Whether RDKit molecules keep the original atom mapping. This option is intended to be used when providing atom-mapped SMILES with
    the :code:`is_atom_bond_targets`.
    """
    no_shared_atom_bond_ffn: bool = False
    """
    Whether the FFN weights for atom and bond targets should be independent between tasks.
    """
    weights_ffn_num_layers: int = 2
    """
    Number of layers in FFN for determining weights used in constrained targets.
    """
    no_adding_bond_types: bool = False
    """
    Whether the bond types determined by RDKit molecules added to the output of bond targets. This option is intended to be used
    with the :code:`is_atom_bond_targets`.
    """

    # Training arguments
    epochs: int = 30
    """Number of epochs to run."""
    warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 1e-4
    """Initial learning rate."""
    max_lr: float = 1e-3
    """Maximum learning rate."""
    final_lr: float = 1e-4
    """Final learning rate."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    class_balance: bool = False
    """Trains with an equal number of positives and negatives in each batch."""
    spectra_activation: Literal['exp', 'softplus'] = 'exp'
    """Indicates which function to use in dataset_type spectra training to constrain outputs to be positive."""
    spectra_target_floor: float = 1e-8
    """Values in targets for dataset type spectra are replaced with this value, intended to be a small positive number used to enforce positive values."""
    evidential_regularization: float = 0
    """Value used in regularization for evidential loss function. The default value recommended by Soleimany et al.(2021) is 0.2. 
    Optimal value is dataset-dependent; it is recommended that users test different values to find the best value for their model."""
    overwrite_default_atom_features: bool = False
    """
    Overwrites the default atom descriptors with the new ones instead of concatenating them.
    Can only be used if atom_descriptors are used as a feature.
    """
    no_atom_descriptor_scaling: bool = False
    """Turn off atom feature scaling."""
    overwrite_default_bond_features: bool = False
    """
    Overwrites the default bond descriptors with the new ones instead of concatenating them.
    Can only be used if bond_descriptors are used as a feature.
    """
    no_bond_descriptor_scaling: bool = False
    """Turn off atom feature scaling."""
    frzn_ffn_layers: int = 0
    """
    Overwrites weights for the first n layers of the ffn from checkpoint model (specified checkpoint_frzn),
    where n is specified in the input.
    Automatically also freezes mpnn weights.
    """
    freeze_first_only: bool = False
    """
    Determines whether or not to use checkpoint_frzn for just the first encoder.
    Default (False) is to use the checkpoint to freeze all encoders.
    (only relevant for number_of_molecules > 1, where checkpoint model has number_of_molecules = 1)
    """

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None

    @property
    def metrics(self) -> List[str]:
        """The list of metrics used for evaluation. Only the first is used for early stopping."""
        return [self.metric] + self.extra_metrics

    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {'rmse', 'mae', 'mse', 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein', 'bounded_mse', 'bounded_mae', 'bounded_rmse'}

    @property
    def use_input_features(self) -> bool:
        """Whether the model is using additional molecule-level features."""
        return self.features_generator is not None or self.features_path is not None or self.phase_features_path is not None

    @property
    def num_lrs(self) -> int:
        """The number of learning rates to use (currently hard-coded to 1)."""
        return 1

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        """Index sets used for splitting data into train/validation/test during cross-validation"""
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        """A list of names of the tasks being trained on."""
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0

    @property
    def features_size(self) -> int:
        """The dimensionality of the additional molecule-level features."""
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        """The size of the training data set."""
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size

    @property
    def atom_descriptor_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional atom features."
        """
        return not self.no_atom_descriptor_scaling

    @property
    def bond_descriptor_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional bond features."
        """
        return not self.no_bond_descriptor_scaling
    
    @property
    def shared_atom_bond_ffn(self) -> bool:
        """
        Whether the FFN weights for atom and bond targets should be shared between tasks.
        """
        return not self.no_shared_atom_bond_ffn

    @property
    def adding_bond_types(self) -> bool:
        """
        Whether the bond types determined by RDKit molecules should be added to the output of bond targets.
        """
        return not self.no_adding_bond_types

    @property
    def atom_constraints(self) -> List[bool]:
        """
        A list of booleans indicating whether constraints applied to output of atomic properties.
        """
        if self.is_atom_bond_targets and self.constraints_path:
            if not self._atom_constraints:
                header = chemprop.data.utils.get_header(self.constraints_path)
                self._atom_constraints = [target in header for target in self.atom_targets]
        else:
            self._atom_constraints = [False] * len(self.atom_targets)
        return self._atom_constraints

    @property
    def bond_constraints(self) -> List[bool]:
        """
        A list of booleans indicating whether constraints applied to output of bond properties.
        """
        if self.is_atom_bond_targets and self.constraints_path:
            if not self._bond_constraints:
                header = chemprop.data.utils.get_header(self.constraints_path)
                self._bond_constraints = [target in header for target in self.bond_targets]
        else:
            self._bond_constraints = [False] * len(self.bond_targets)
        return self._bond_constraints

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()

        global temp_save_dir  # Prevents the temporary directory from being deleted upon function return

        # Adapt the number of molecules for reaction_solvent mode
        if self.reaction_solvent is True and self.number_of_molecules != 2:
            raise ValueError('In reaction_solvent mode, --number_of_molecules 2 must be specified.')

        # Process SMILES columns
        self.smiles_columns = chemprop.data.utils.preprocess_smiles_columns(
            path=self.data_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        # Load config file
        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)

        # Determine the target_columns when training atomic and bond targets
        if self.is_atom_bond_targets:
            self.atom_targets, self.bond_targets, self.molecule_targets = chemprop.data.utils.get_mixed_task_names(
                path=self.data_path,
                smiles_columns=self.smiles_columns,
                target_columns=self.target_columns,
                ignore_columns=self.ignore_columns,
                keep_h=self.explicit_h,
                add_h=self.adding_h,
                keep_atom_map=self.keeping_atom_map,
            )
            self.target_columns = self.atom_targets + self.bond_targets
            # self.target_columns = self.atom_targets + self.bond_targets + self.molecule_targets  # TODO: Support mixed targets
        else:
            self.atom_targets, self.bond_targets = [], []

        # Check whether atomic/bond constraints have been applied on the correct dataset_type
        if self.constraints_path:
            if not self.is_atom_bond_targets:
                raise ValueError('Constraints on atomic/bond targets can only be used in atomic/bond properties prediction.')
            if self.dataset_type != 'regression':
                raise ValueError(f'In atomic/bond properties prediction, atomic/bond constraints are not supported for {self.dataset_type}.')

        # Check whether the number of input columns is one for the atomic/bond mode
        if self.is_atom_bond_targets:
            if self.number_of_molecules != 1:
                raise ValueError('In atomic/bond properties prediction, exactly one smiles column must be provided.')

        # Check whether the number of input columns is two for the reaction_solvent mode
        if self.reaction_solvent is True and len(self.smiles_columns) != 2:
            raise ValueError('In reaction_solvent mode, exactly two smiles column must be provided (one for reactions, and one for molecules)')

        # Validate reaction/reaction_solvent mode
        if self.reaction is True and self.reaction_solvent is True:
            raise ValueError('Only reaction or reaction_solvent mode can be used, not both.')

        # Create temporary directory as save directory if not provided
        if self.save_dir is None:
            temp_save_dir = TemporaryDirectory()
            self.save_dir = temp_save_dir.name

        # Fix ensemble size if loading checkpoints
        if self.checkpoint_paths is not None and len(self.checkpoint_paths) > 0:
            self.ensemble_size = len(self.checkpoint_paths)

        # Process and validate metric and loss function
        if self.metric is None:
            if self.dataset_type == 'classification':
                self.metric = 'auc'
            elif self.dataset_type == 'multiclass':
                self.metric = 'cross_entropy'
            elif self.dataset_type == 'spectra':
                self.metric = 'sid'
            elif self.dataset_type == 'regression' and self.loss_function == 'bounded_mse':
                self.metric = 'bounded_mse'
            elif self.dataset_type == 'regression':
                self.metric = 'rmse'
            else:
                raise ValueError(f'Dataset type {self.dataset_type} is not supported.')

        if self.metric in self.extra_metrics:
            raise ValueError(f'Metric {self.metric} is both the metric and is in extra_metrics. '
                             f'Please only include it once.')

        for metric in self.metrics:
            if not any([(self.dataset_type == 'classification' and metric in ['auc', 'prc-auc', 'accuracy', 'binary_cross_entropy', 'f1', 'mcc']),
                        (self.dataset_type == 'regression' and metric in ['rmse', 'mae', 'mse', 'r2', 'bounded_rmse', 'bounded_mae', 'bounded_mse']),
                        (self.dataset_type == 'multiclass' and metric in ['cross_entropy', 'accuracy', 'f1', 'mcc']),
                        (self.dataset_type == 'spectra' and metric in ['sid', 'wasserstein'])]):
                raise ValueError(f'Metric "{metric}" invalid for dataset type "{self.dataset_type}".')

        if self.loss_function is None:
            if self.dataset_type == 'classification':
                self.loss_function = 'binary_cross_entropy'
            elif self.dataset_type == 'multiclass':
                self.loss_function = 'cross_entropy'
            elif self.dataset_type == 'spectra':
                self.loss_function = 'sid'
            elif self.dataset_type == 'regression':
                self.loss_function = 'mse'
            else:
                raise ValueError(f'Default loss function not configured for dataset type {self.dataset_type}.')

        if self.loss_function != 'bounded_mse' and any(metric in ['bounded_mse', 'bounded_rmse', 'bounded_mae'] for metric in self.metrics):
            raise ValueError('Bounded metrics can only be used in conjunction with the regression loss function bounded_mse.')

        # Validate class balance
        if self.class_balance and self.dataset_type != 'classification':
            raise ValueError('Class balance can only be applied if the dataset type is classification.')

        # Validate features
        if self.features_only and not (self.features_generator or self.features_path):
            raise ValueError('When using features_only, a features_generator or features_path must be provided.')

        # Handle FFN hidden size
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size

        # Handle MPN variants
        if self.atom_messages and self.undirected:
            raise ValueError('Undirected is unnecessary when using atom_messages '
                             'since atom_messages are by their nature undirected.')

        # Validate split type settings
        if not (self.split_type == 'predetermined') == (self.folds_file is not None) == (self.test_fold_index is not None):
            raise ValueError('When using predetermined split type, must provide folds_file and test_fold_index.')

        if not (self.split_type == 'crossval') == (self.crossval_index_dir is not None):
            raise ValueError('When using crossval split type, must provide crossval_index_dir.')

        if not (self.split_type in ['crossval', 'index_predetermined']) == (self.crossval_index_file is not None):
            raise ValueError('When using crossval or index_predetermined split type, must provide crossval_index_file.')

        if self.split_type in ['crossval', 'index_predetermined']:
            with open(self.crossval_index_file, 'rb') as rf:
                self._crossval_index_sets = pickle.load(rf)
            self.num_folds = len(self.crossval_index_sets)
            self.seed = 0

        # Validate split size entry and set default values
        if self.split_sizes is None:
            if self.separate_val_path is None and self.separate_test_path is None: # separate data paths are not provided
                self.split_sizes = [0.8, 0.1, 0.1]
            elif self.separate_val_path is not None and self.separate_test_path is None: # separate val path only
                self.split_sizes = [0.8, 0., 0.2]
            elif self.separate_val_path is None and self.separate_test_path is not None: # separate test path only
                self.split_sizes = [0.8, 0.2, 0.]
            else: # both separate data paths are provided
                self.split_sizes = [1., 0., 0.]

        else:
            if not np.isclose(sum(self.split_sizes), 1):
                raise ValueError(f'Provided split sizes of {self.split_sizes} do not sum to 1.')
            if any([size < 0 for size in self.split_sizes]):
                raise ValueError(f'Split sizes must be non-negative. Received split sizes: {self.split_sizes}')


            if len(self.split_sizes) not in [2, 3]:
                raise ValueError(f'Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s).')

            if self.separate_val_path is None and self.separate_test_path is None:  # separate data paths are not provided
                if len(self.split_sizes) != 3:
                    raise ValueError(f'Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s).')
                if self.split_sizes[0] == 0.:
                    raise ValueError(f'Provided split size for train split must be nonzero. Received split size {self.split_sizes[0]}')
                if self.split_sizes[1] == 0.:
                    raise ValueError(f'Provided split size for validation split must be nonzero. Received split size {self.split_sizes[1]}')

            elif self.separate_val_path is not None and self.separate_test_path is None: # separate val path only
                if len(self.split_sizes) == 2: # allow input of just 2 values
                    self.split_sizes = [self.split_sizes[0], 0., self.split_sizes[1]]
                if self.split_sizes[0] == 0.:
                    raise ValueError('Provided split size for train split must be nonzero.')
                if self.split_sizes[1] != 0.:
                    raise ValueError(f'Provided split size for validation split must be 0 because validation set is provided separately. Received split size {self.split_sizes[1]}')

            elif self.separate_val_path is None and self.separate_test_path is not None: # separate test path only
                if len(self.split_sizes) == 2: # allow input of just 2 values
                    self.split_sizes = [self.split_sizes[0], self.split_sizes[1], 0.]
                if self.split_sizes[0] == 0.:
                    raise ValueError('Provided split size for train split must be nonzero.')
                if self.split_sizes[1] == 0.:
                    raise ValueError('Provided split size for validation split must be nonzero.')
                if self.split_sizes[2] != 0.:
                    raise ValueError(f'Provided split size for test split must be 0 because test set is provided separately. Received split size {self.split_sizes[2]}')


            else: # both separate data paths are provided
                if self.split_sizes != [1., 0., 0.]:
                    raise ValueError(f'Separate data paths were provided for val and test splits. Split sizes should not also be provided. Received split sizes: {self.split_sizes}')

        # Test settings
        if self.test:
            self.epochs = 0

        # Validate features are provided for separate validation or test set for each of the kinds of additional features
        for (features_argument, base_features_path, val_features_path, test_features_path) in [
            ('`--features_path`', self.features_path, self.separate_val_features_path, self.separate_test_features_path),
            ('`--phase_features_path`', self.phase_features_path, self.separate_val_phase_features_path, self.separate_test_phase_features_path),
            ('`--atom_descriptors_path`', self.atom_descriptors_path, self.separate_val_atom_descriptors_path, self.separate_test_atom_descriptors_path),
            ('`--bond_descriptors_path`', self.bond_descriptors_path, self.separate_val_bond_descriptors_path, self.separate_test_bond_descriptors_path),
            ('`--constraints_path`', self.constraints_path, self.separate_val_constraints_path, self.separate_test_constraints_path)
        ]:
            if base_features_path is not None:
                if self.separate_val_path is not None and val_features_path is None:
                    raise ValueError(f'Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the separate validation set.')
                if self.separate_test_path is not None and test_features_path is None:
                    raise ValueError(f'Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the separate test set.')

        # validate extra atom descriptor options
        if self.overwrite_default_atom_features and self.atom_descriptors != 'feature':
            raise NotImplementedError('Overwriting of the default atom descriptors can only be used if the'
                                      'provided atom descriptors are features.')

        if not self.atom_descriptor_scaling and self.atom_descriptors is None:
            raise ValueError('Atom descriptor scaling is only possible if additional atom features are provided.')

        # validate extra bond descriptor options
        if self.overwrite_default_bond_features and self.bond_descriptors != 'feature':
            raise NotImplementedError('Overwriting of the default bond descriptors can only be used if the'
                                      'provided bond descriptors are features.')

        if not self.bond_descriptor_scaling and self.bond_descriptors is None:
            raise ValueError('Bond descriptor scaling is only possible if additional bond features are provided.')

        if self.bond_descriptors == 'descriptor' and not self.is_atom_bond_targets:
            raise NotImplementedError('Bond descriptors as descriptor can only be used with `--is_atom_bond_targets`.')

        # normalize target weights
        if self.target_weights is not None:
            avg_weight = sum(self.target_weights)/len(self.target_weights)
            self.target_weights = [w/avg_weight for w in self.target_weights]
            if min(self.target_weights) < 0:
                raise ValueError('Provided target weights must be non-negative.')

        # check if key molecule index is outside of the number of molecules
        if self.split_key_molecule >= self.number_of_molecules:
            raise ValueError('The index provided with the argument `--split_key_molecule` must be less than the number of molecules. Note that this index begins with 0 for the first molecule. ')

class PredictArgs(CommonArgs):
    """:class:`PredictArgs` includes :class:`CommonArgs` along with additional arguments used for predicting with a Chemprop model."""

    test_path: str
    """Path to CSV file containing testing data for which predictions will be made."""
    preds_path: str
    """Path to CSV or PICKLE file where predictions will be saved."""
    drop_extra_columns: bool = False
    """Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns."""
    ensemble_variance: bool = False
    """Deprecated. Whether to calculate the variance of ensembles as a measure of epistemic uncertainty. If True, the variance is saved as an additional column for each target in the preds_path."""
    individual_ensemble_predictions: bool = False
    """Whether to return the predictions made by each of the individual models rather than the average of the ensemble"""
    # Uncertainty arguments
    uncertainty_method: Literal[
        'mve',
        'ensemble',
        'evidential_epistemic',
        'evidential_aleatoric',
        'evidential_total',
        'classification',
        'dropout',
        'spectra_roundrobin',
    ] = None
    """The method of calculating uncertainty."""
    calibration_method: Literal['zscaling', 'tscaling', 'zelikman_interval', 'mve_weighting', 'platt', 'isotonic'] = None
    """Methods used for calibrating the uncertainty calculated with uncertainty method."""
    evaluation_methods: List[str] = None
    """The methods used for evaluating the uncertainty performance if the test data provided includes targets.
    Available methods are [nll, miscalibration_area, ence, spearman] or any available classification or multiclass metric."""
    evaluation_scores_path: str = None
    """Location to save the results of uncertainty evaluations."""
    uncertainty_dropout_p: float = 0.1
    """The probability to use for Monte Carlo dropout uncertainty estimation."""
    dropout_sampling_size: int = 10
    """The number of samples to use for Monte Carlo dropout uncertainty estimation. Distinct from the dropout used during training."""
    calibration_interval_percentile: float = 95
    """Sets the percentile used in the calibration methods. Must be in the range (1,100)."""
    regression_calibrator_metric: Literal['stdev', 'interval'] = None
    """Regression calibrators can output either a stdev or an inverval. """
    calibration_path: str = None
    """Path to data file to be used for uncertainty calibration."""
    calibration_features_path: List[str] = None
    """Path to features data to be used with the uncertainty calibration dataset."""
    calibration_phase_features_path: str = None
    """ """
    calibration_atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    calibration_bond_descriptors_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""

    @property
    def ensemble_size(self) -> int:
        """The number of models in the ensemble."""
        return len(self.checkpoint_paths)

    def process_args(self) -> None:
        super(PredictArgs, self).process_args()

        if self.regression_calibrator_metric is None:
            if self.calibration_method == 'zelikman_interval':
                self.regression_calibrator_metric = 'interval'
            else:
                self.regression_calibrator_metric = 'stdev'

        if self.uncertainty_method == 'dropout' and version.parse(torch.__version__) < version.parse('1.9.0'):
            raise ValueError('Dropout uncertainty is only supported for pytorch versions >= 1.9.0')

        self.smiles_columns = chemprop.data.utils.preprocess_smiles_columns(
            path=self.test_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')

        if self.ensemble_variance:
            if self.uncertainty_method in ['ensemble', None]:
                warn(
                    'The `--ensemble_variance` argument is deprecated and should \
                        be replaced with `--uncertainty_method ensemble`.',
                    DeprecationWarning,
                )
                self.uncertainty_method = 'ensemble'
            else:
                raise ValueError(
                    f'Only one uncertainty method can be used at a time. \
                        The arguement `--ensemble_variance` was provided along \
                        with the uncertainty method {self.uncertainty_method}. The `--ensemble_variance` \
                        argument is deprecated and should be replaced with `--uncertainty_method ensemble`.'
                )

        if self.calibration_interval_percentile <= 1 or self.calibration_interval_percentile >= 100:
            raise ValueError('The calibration interval must be a percentile value in the range (1,100).')

        if self.uncertainty_dropout_p < 0 or self.uncertainty_dropout_p > 1:
            raise ValueError('The dropout probability must be in the range (0,1).')

        if self.dropout_sampling_size <= 1:
            raise ValueError('The argument `--dropout_sampling_size` must be an integer greater than 1.')

        # Validate that features provided for the prediction test set are also provided for the calibration set
        for (features_argument, base_features_path, cal_features_path) in [
            ('`--features_path`', self.features_path, self.calibration_features_path),
            ('`--phase_features_path`', self.phase_features_path, self.calibration_phase_features_path),
            ('`--atom_descriptors_path`', self.atom_descriptors_path, self.calibration_atom_descriptors_path),
            ('`--bond_descriptors_path`', self.bond_descriptors_path, self.calibration_bond_descriptors_path)
        ]:
            if base_features_path is not None and self.calibration_path is not None and cal_features_path is None:
                raise ValueError(f'Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the calibration dataset.')


## from data.utils
# Increase maximum size of field in the csv processing for the current architecture
csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.
    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def preprocess_smiles_columns(path: str,
                              smiles_columns: Union[str, List[str]] = None,
                              number_of_molecules: int = 1) -> List[str]:
    """
    Preprocesses the :code:`smiles_columns` variable to ensure that it is a list of column
    headings corresponding to the columns in the data file holding SMILES. Assumes file has a header.
    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules with associated SMILES for each
                           data point.
    :return: The preprocessed version of :code:`smiles_columns` which is guaranteed to be a list.
    """

    if smiles_columns is None:
        if os.path.isfile(path):
            columns = get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None]*number_of_molecules
    else:
        if isinstance(smiles_columns, str):
            smiles_columns = [smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError('Length of smiles_columns must match number_of_molecules.')
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError('Provided smiles_columns do not match the header of data file.')

    return smiles_columns


def get_task_names(path: str,
                   smiles_columns: Union[str, List[str]] = None,
                   target_columns: List[str] = None,
                   ignore_columns: List[str] = None) -> List[str]:
    """
    Gets the task names from a data CSV file.
    If :code:`target_columns` is provided, returns `target_columns`.
    Otherwise, returns all columns except the :code:`smiles_columns`
    (or the first column, if the :code:`smiles_columns` is None) and
    the :code:`ignore_columns`.
    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_columns` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :return: A list of task names.
    """
    if target_columns is not None:
        return target_columns

    columns = get_header(path)

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    ignore_columns = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))

    target_names = [column for column in columns if column not in ignore_columns]

    return target_names


def get_mixed_task_names(path: str,
                         smiles_columns: Union[str, List[str]] = None,
                         target_columns: List[str] = None,
                         ignore_columns: List[str] = None,
                         keep_h: bool = None,
                         add_h: bool = None,
                         keep_atom_map: bool = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Gets the task names for atomic, bond, and molecule targets separately from a data CSV file.

    If :code:`target_columns` is provided, returned lists based off `target_columns`.
    Otherwise, returned lists based off all columns except the :code:`smiles_columns`
    (or the first column, if the :code:`smiles_columns` is None) and
    the :code:`ignore_columns`.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_columns` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h: Boolean whether to add hydrogens to the input smiles.
    :param keep_atom_map: Boolean whether to keep the original atom mapping.
    :return: A tuple containing the task names of atomic, bond, and molecule properties separately.
    """
    columns = get_header(path)

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    ignore_columns = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))

    if target_columns is not None:
        target_names =  target_columns
    else:
        target_names = [column for column in columns if column not in ignore_columns]

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            atom_target_names, bond_target_names, molecule_target_names = [], [], []
            smiles = [row[c] for c in smiles_columns]
            mol = make_mol(smiles[0], keep_h, add_h, keep_atom_map)
            for column in target_names:
                value = row[column]
                value = value.replace('None', 'null')
                target = np.array(json.loads(value))

                is_atom_target, is_bond_target, is_molecule_target = False, False, False
                if len(target.shape) == 0:
                    is_molecule_target = True
                elif len(target.shape) == 1:
                    if len(mol.GetAtoms()) == len(mol.GetBonds()):
                        break
                    elif len(target) == len(mol.GetAtoms()):  # Atom targets saved as 1D list
                        is_atom_target = True
                    elif len(target) == len(mol.GetBonds()):  # Bond targets saved as 1D list
                        is_bond_target = True
                elif len(target.shape) == 2:  # Bond targets saved as 2D list
                    is_bond_target = True
                else:
                    raise ValueError('Unrecognized targets of column {column} in {path}.')
                
                if is_atom_target:
                    atom_target_names.append(column)
                elif is_bond_target:
                    bond_target_names.append(column)
                elif is_molecule_target:
                    molecule_target_names.append(column)
            if len(atom_target_names) + len(bond_target_names) + len(molecule_target_names) == len(target_names):
                break

    return atom_target_names, bond_target_names, molecule_target_names


def get_data_weights(path: str) -> List[float]:
    """
    Returns the list of data weights for the loss function as stored in a CSV file.

    :param path: Path to a CSV file.
    :return: A list of floats containing the data weights.
    """
    weights = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for line in reader:
            weights.append(float(line[0]))
    # normalize the data weights
    avg_weight = sum(weights) / len(weights)
    weights = [w / avg_weight for w in weights]
    if min(weights) < 0:
        raise ValueError('Data weights must be non-negative for each datapoint.')
    return weights


def get_constraints(path: str,
                    target_columns: List[str],
                    save_raw_data: bool = False) -> Tuple[List[float], List[float]]:
    """
    Returns lists of data constraints for the atomic/bond targets as stored in a CSV file.

    :param path: Path to a CSV file.
    :param target_columns: Name of the columns containing target values.
    :param save_raw_data: Whether to save all user-provided atom/bond-level constraints in input data,
                          which will be used to construct constraints files for each train/val/test split
                          for prediction convenience later.
    :return: Lists of floats containing the data constraints.
    """
    constraints_data = []
    reader = pd.read_csv(path)
    reader_columns = reader.columns.tolist()
    if len(reader_columns) != len(set(reader_columns)):
        raise ValueError(f'There are duplicates in {path}.')
    for target in target_columns:
        if target in reader_columns:
            constraints_data.append(reader[target].values)
        else:
            constraints_data.append([None] * len(reader))
    constraints_data = np.transpose(constraints_data)  # each is num_data x num_targets

    if save_raw_data:
        raw_constraints_data = []
        for target in reader_columns:
            raw_constraints_data.append(reader[target].values)
        raw_constraints_data = np.transpose(raw_constraints_data)  # each is num_data x num_columns
    else:
        raw_constraints_data = None
    
    return constraints_data, raw_constraints_data


def get_smiles(path: str,
               smiles_columns: Union[str, List[str]] = None,
               number_of_molecules: int = 1,
               header: bool = True,
               flatten: bool = False
               ) -> Union[List[str], List[List[str]]]:
    """
    Returns the SMILES from a data CSV file.

    :param path: Path to a CSV file.
    :param smiles_columns: A list of the names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules for each data point. Not necessary if
                                the names of smiles columns are previously processed.
    :param header: Whether the CSV file contains a header.
    :param flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
    :return: A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
    """
    if smiles_columns is not None and not header:
        raise ValueError('If smiles_column is provided, the CSV file must have a header.')

    if (isinstance(smiles_columns, str) or smiles_columns is None) and header:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns, number_of_molecules=number_of_molecules)

    with open(path) as f:
        if header:
            reader = csv.DictReader(f)
        else:
            reader = csv.reader(f)
            smiles_columns = list(range(number_of_molecules))

        smiles = [[row[c] for c in smiles_columns] for row in reader]

    if flatten:
        smiles = [smile for smiles_list in smiles for smile in smiles_list]

    return smiles


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :return: A :class:`~chemprop.data.MoleculeDataset` with only the valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in tqdm(data)
                            if all(s != '' for s in datapoint.smiles) and all(m is not None for m in datapoint.mol)
                            and all(m.GetNumHeavyAtoms() > 0 for m in datapoint.mol if not isinstance(m, tuple))
                            and all(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() > 0 for m in datapoint.mol if isinstance(m, tuple))])


def get_invalid_smiles_from_file(path: str = None,
                                 smiles_columns: Union[str, List[str]] = None,
                                 header: bool = True,
                                 reaction: bool = False,
                                 ) -> Union[List[str], List[List[str]]]:
    """
    Returns the invalid SMILES from a data CSV file.

    :param path: Path to a CSV file.
    :param smiles_columns: A list of the names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param header: Whether the CSV file contains a header.
    :param reaction: Boolean whether the SMILES strings are to be treated as a reaction.
    :return: A list of lists of SMILES, for the invalid SMILES in the file.
    """
    smiles = get_smiles(path=path, smiles_columns=smiles_columns, header=header)

    invalid_smiles = get_invalid_smiles_from_list(smiles=smiles, reaction=reaction)

    return invalid_smiles


def get_invalid_smiles_from_list(smiles: List[List[str]], reaction: bool = False) -> List[List[str]]:
    """
    Returns the invalid SMILES from a list of lists of SMILES strings.

    :param smiles: A list of list of SMILES.
    :param reaction: Boolean whether the SMILES strings are to be treated as a reaction.
    :return: A list of lists of SMILES, for the invalid SMILES among the lists provided.
    """
    invalid_smiles = []

    # If the first SMILES in the column is a molecule, the remaining SMILES in the same column should all be a molecule.
    # Similarly, if the first SMILES in the column is a reaction, the remaining SMILES in the same column should all
    # correspond to reaction. Therefore, get `is_mol_list` only using the first element in smiles.
    is_mol_list = [is_mol(s) for s in smiles[0]]
    is_reaction_list = [True if not x and reaction else False for x in is_mol_list]
    is_explicit_h_list = [False for x in is_mol_list]  # set this to False as it is not needed for invalid SMILES check
    is_adding_hs_list = [False for x in is_mol_list]  # set this to False as it is not needed for invalid SMILES check
    keep_atom_map_list = [False for x in is_mol_list]  # set this to False as it is not needed for invalid SMILES check

    for mol_smiles in smiles:
        mols = make_mols(smiles=mol_smiles, reaction_list=is_reaction_list, keep_h_list=is_explicit_h_list,
                         add_h_list=is_adding_hs_list, keep_atom_map_list=keep_atom_map_list)
        if any(s == '' for s in mol_smiles) or \
           any(m is None for m in mols) or \
           any(m.GetNumHeavyAtoms() == 0 for m in mols if not isinstance(m, tuple)) or \
           any(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() == 0 for m in mols if isinstance(m, tuple)):

            invalid_smiles.append(mol_smiles)

    return invalid_smiles


def get_data(path: str,
             smiles_columns: Union[str, List[str]] = None,
             target_columns: List[str] = None,
             ignore_columns: List[str] = None,
             skip_invalid_smiles: bool = True,
             args: Union[TrainArgs, PredictArgs] = None,
             data_weights_path: str = None,
             features_path: List[str] = None,
             features_generator: List[str] = None,
             phase_features_path: str = None,
             atom_descriptors_path: str = None,
             bond_descriptors_path: str = None,
             constraints_path: str = None,
             max_data_size: int = None,
             store_row: bool = False,
             logger: Logger = None,
             loss_function: str = None,
             skip_none_targets: bool = False) -> MoleculeDataset:
    """
    Gets SMILES and target values from a CSV file.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_column` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles using :func:`filter_invalid_smiles`.
    :param args: Arguments, either :class:`~chemprop.args.TrainArgs` or :class:`~chemprop.args.PredictArgs`.
    :param data_weights_path: A path to a file containing weights for each molecule in the loss function.
    :param features_path: A list of paths to files containing features. If provided, it is used
                          in place of :code:`args.features_path`.
    :param features_generator: A list of features generators to use. If provided, it is used
                               in place of :code:`args.features_generator`.
    :param phase_features_path: A path to a file containing phase features as applicable to spectra.
    :param atom_descriptors_path: The path to the file containing the custom atom descriptors.
    :param bond_descriptors_path: The path to the file containing the custom bond descriptors.
    :param constraints_path: The path to the file containing constraints applied to different atomic/bond properties.
    :param max_data_size: The maximum number of data points to load.
    :param logger: A logger for recording output.
    :param store_row: Whether to store the raw CSV row in each :class:`~chemprop.data.data.MoleculeDatapoint`.
    :param skip_none_targets: Whether to skip targets that are all 'None'. This is mostly relevant when --target_columns
                              are passed in, so only a subset of tasks are examined.
    :param loss_function: The loss function to be used in training.
    :return: A :class:`~chemprop.data.MoleculeDataset` containing SMILES and target values along
             with other info such as additional features when desired.
    """
    debug = logger.debug if logger is not None else print

    if args is not None:
        # Prefer explicit function arguments but default to args if not provided
        smiles_columns = smiles_columns if smiles_columns is not None else args.smiles_columns
        target_columns = target_columns if target_columns is not None else args.target_columns
        ignore_columns = ignore_columns if ignore_columns is not None else args.ignore_columns
        features_path = features_path if features_path is not None else args.features_path
        features_generator = features_generator if features_generator is not None else args.features_generator
        phase_features_path = phase_features_path if phase_features_path is not None else args.phase_features_path
        atom_descriptors_path = atom_descriptors_path if atom_descriptors_path is not None \
            else args.atom_descriptors_path
        bond_descriptors_path = bond_descriptors_path if bond_descriptors_path is not None \
            else args.bond_descriptors_path
        constraints_path = constraints_path if constraints_path is not None else args.constraints_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        loss_function = loss_function if loss_function is not None else args.loss_function

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    max_data_size = max_data_size or float('inf')

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    if phase_features_path is not None:
        phase_features = load_features(phase_features_path)
        for d_phase in phase_features:
            if not (d_phase.sum() == 1 and np.count_nonzero(d_phase) == 1):
                raise ValueError('Phase features must be one-hot encoded.')
        if features_data is not None:
            features_data = np.concatenate((features_data, phase_features), axis=1)
        else:  # if there are no other molecular features, phase features become the only molecular features
            features_data = np.array(phase_features)
    else:
        phase_features = None

    # Load constraints
    if constraints_path is not None:
        constraints_data, raw_constraints_data = get_constraints(
            path=constraints_path,
            target_columns=args.target_columns,
            save_raw_data=args.save_smiles_splits
        )
    else:
        constraints_data = None
        raw_constraints_data = None

    # Load data weights
    if data_weights_path is not None:
        data_weights = get_data_weights(data_weights_path)
    else:
        data_weights = None

    # By default, the targets columns are all the columns except the SMILES column
    if target_columns is None:
        target_columns = get_task_names(
            path=path,
            smiles_columns=smiles_columns,
            target_columns=target_columns,
            ignore_columns=ignore_columns,
        )

    # Find targets provided as inequalities
    if loss_function == 'bounded_mse':
        gt_targets, lt_targets = get_inequality_targets(path=path, target_columns=target_columns)
    else:
        gt_targets, lt_targets = None, None

    # Load data
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if any([c not in fieldnames for c in smiles_columns]):
            raise ValueError(f'Data file did not contain all provided smiles columns: {smiles_columns}. Data file field names are: {fieldnames}')
        if any([c not in fieldnames for c in target_columns]):
            raise ValueError(f'Data file did not contain all provided target columns: {target_columns}. Data file field names are: {fieldnames}')

        all_smiles, all_targets, all_atom_targets, all_bond_targets, all_rows, all_features, all_phase_features, all_constraints_data, all_raw_constraints_data, all_weights, all_gt, all_lt = [], [], [], [], [], [], [], [], [], [], [], []
        for i, row in enumerate(tqdm(reader)):
            smiles = [row[c] for c in smiles_columns]

            targets, atom_targets, bond_targets = [], [], []
            for column in target_columns:
                value = row[column]
                if value in ['', 'nan']:
                    targets.append(None)
                elif '>' in value or '<' in value:
                    if loss_function == 'bounded_mse':
                        targets.append(float(value.strip('<>')))
                    else:
                        raise ValueError('Inequality found in target data. To use inequality targets (> or <), the regression loss function bounded_mse must be used.')
                elif '[' in value or ']' in value:
                    value = value.replace('None', 'null')
                    target = np.array(json.loads(value))
                    if len(target.shape) == 1 and column in args.atom_targets:  # Atom targets saved as 1D list
                        atom_targets.append(target)
                        targets.append(target)
                    elif len(target.shape) == 1 and column in args.bond_targets:  # Bond targets saved as 1D list
                        bond_targets.append(target)
                        targets.append(target)
                    elif len(target.shape) == 2:  # Bond targets saved as 2D list
                        bond_target_arranged = []
                        mol = make_mol(smiles[0], args.explicit_h, args.adding_h, args.keeping_atom_map)
                        for bond in mol.GetBonds():
                            bond_target_arranged.append(target[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                        bond_targets.append(np.array(bond_target_arranged))
                        targets.append(np.array(bond_target_arranged))
                    else:
                        raise ValueError(f'Unrecognized targets of column {column} in {path}.')
                else:
                    targets.append(float(value))

            # Check whether all targets are None and skip if so
            if skip_none_targets and all(x is None for x in targets):
                continue

            all_smiles.append(smiles)
            all_targets.append(targets)
            all_atom_targets.append(atom_targets)
            all_bond_targets.append(bond_targets)

            if features_data is not None:
                all_features.append(features_data[i])

            if phase_features is not None:
                all_phase_features.append(phase_features[i])

            if constraints_data is not None:
                all_constraints_data.append(constraints_data[i])

            if raw_constraints_data is not None:
                all_raw_constraints_data.append(raw_constraints_data[i])

            if data_weights is not None:
                all_weights.append(data_weights[i])

            if gt_targets is not None:
                all_gt.append(gt_targets[i])

            if lt_targets is not None:
                all_lt.append(lt_targets[i])

            if store_row:
                all_rows.append(row)

            if len(all_smiles) >= max_data_size:
                break

        atom_features = None
        atom_descriptors = None
        if args is not None and args.atom_descriptors is not None:
            try:
                descriptors = load_valid_atom_or_bond_features(atom_descriptors_path, [x[0] for x in all_smiles])
            except Exception as e:
                raise ValueError(f'Failed to load or validate custom atomic descriptors or features: {e}')

            if args.atom_descriptors == 'feature':
                atom_features = descriptors
            elif args.atom_descriptors == 'descriptor':
                atom_descriptors = descriptors

        bond_features = None
        bond_descriptors = None
        if args is not None and args.bond_descriptors is not None:
            try:
                descriptors = load_valid_atom_or_bond_features(bond_descriptors_path, [x[0] for x in all_smiles])
            except Exception as e:
                raise ValueError(f'Failed to load or validate custom bond descriptors or features: {e}')

            if args.bond_descriptors == 'feature':
                bond_features = descriptors
            elif args.bond_descriptors == 'descriptor':
                bond_descriptors = descriptors

        data = MoleculeDataset([
            MoleculeDatapoint(
                smiles=smiles,
                targets=targets,
                atom_targets=all_atom_targets[i] if atom_targets else None,
                bond_targets=all_bond_targets[i] if bond_targets else None,
                row=all_rows[i] if store_row else None,
                data_weight=all_weights[i] if data_weights is not None else None,
                gt_targets=all_gt[i] if gt_targets is not None else None,
                lt_targets=all_lt[i] if lt_targets is not None else None,
                features_generator=features_generator,
                features=all_features[i] if features_data is not None else None,
                phase_features=all_phase_features[i] if phase_features is not None else None,
                atom_features=atom_features[i] if atom_features is not None else None,
                atom_descriptors=atom_descriptors[i] if atom_descriptors is not None else None,
                bond_features=bond_features[i] if bond_features is not None else None,
                bond_descriptors=bond_descriptors[i] if bond_descriptors is not None else None,
                constraints=all_constraints_data[i] if constraints_data is not None else None,
                raw_constraints=all_raw_constraints_data[i] if raw_constraints_data is not None else None,
                overwrite_default_atom_features=args.overwrite_default_atom_features if args is not None else False,
                overwrite_default_bond_features=args.overwrite_default_bond_features if args is not None else False
            ) for i, (smiles, targets) in tqdm(enumerate(zip(all_smiles, all_targets)),
                                            total=len(all_smiles))
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def get_data_from_smiles(smiles: List[List[str]],
                         skip_invalid_smiles: bool = True,
                         logger: Logger = None,
                         features_generator: List[str] = None) -> MoleculeDataset:
    """
    Converts a list of SMILES to a :class:`~chemprop.data.MoleculeDataset`.

    :param smiles: A list of lists of SMILES with length depending on the number of molecules.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles using :func:`filter_invalid_smiles`
    :param logger: A logger for recording output.
    :param features_generator: List of features generators.
    :return: A :class:`~chemprop.data.MoleculeDataset` with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=smile,
            row=OrderedDict({'smiles': smile}),
            features_generator=features_generator
        ) for smile in smiles
    ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               key_molecule_index: int = 0,
               seed: int = 0,
               num_folds: int = 1,
               args: TrainArgs = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    r"""
    Splits data into training, validation, and test splits.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param key_molecule_index: For data with multiple molecules, this sets which molecule will be considered during splitting.
    :param seed: The random seed to use before shuffling data.
    :param num_folds: Number of folds to create (only needed for "cv" split type).
    :param args: A :class:`~chemprop.args.TrainArgs` object.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Split sizes do not sum to 1. Received train/val/test splits: {sizes}")
    if any([size < 0 for size in sizes]):
        raise ValueError(f"Split sizes must be non-negative. Received train/val/test splits: {sizes}")

    random = Random(seed)

    if args is not None:
        folds_file, val_fold_index, test_fold_index = \
            args.folds_file, args.val_fold_index, args.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None

    if split_type == 'crossval':
        index_set = args.crossval_index_sets[args.seed]
        data_split = []
        for split in range(3):
            split_indices = []
            for index in index_set[split]:
                with open(os.path.join(args.crossval_index_dir, f'{index}.pkl'), 'rb') as rf:
                    split_indices.extend(pickle.load(rf))
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type in {'cv', 'cv-no-test'}:
        if num_folds <= 1 or num_folds > len(data):
            raise ValueError(f'Number of folds for cross-validation must be between 2 and the number of valid datapoints ({len(data)}), inclusive.')

        random = Random(0)

        indices = np.tile(np.arange(num_folds), 1 + len(data) // num_folds)[:len(data)]
        random.shuffle(indices)
        test_index = seed % num_folds
        val_index = (seed + 1) % num_folds

        train, val, test = [], [], []
        for d, index in zip(data, indices):
            if index == test_index and split_type != 'cv-no-test':
                test.append(d)
            elif index == val_index:
                val.append(d)
            else:
                train.append(d)

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'index_predetermined':
        split_indices = args.crossval_index_sets[args.seed]

        if len(split_indices) != 3:
            raise ValueError('Split indices must have three splits: train, validation, and test')

        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'predetermined':
        if not val_fold_index and sizes[2] != 0:
            raise ValueError('Test size must be zero since test set is created separately '
                             'and we want to put all other data in train and validation')

        if folds_file is None:
            raise ValueError('arg "folds_file" can not be None!')
        if test_fold_index is None:
            raise ValueError('arg "test_fold_index" can not be None!')

        try:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f, encoding='latin1')  # in case we're loading indices from python2

        log_scaffold_stats(data, all_fold_indices, logger=logger)

        folds = [[data[i] for i in fold_indices] for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != test_fold_index and (val_fold_index is None or i != val_fold_index):
                train_val.extend(folds[i])

        if val_fold_index is not None:
            train = train_val
        else:
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, key_molecule_index=key_molecule_index, seed=seed, logger=logger)

    elif split_type == 'random_with_repeated_smiles':  # Use to constrain data with the same smiles go in the same split.
        smiles_dict = defaultdict(set)
        for i, smiles in enumerate(data.smiles()):
            smiles_dict[smiles[key_molecule_index]].add(i)
        index_sets = list(smiles_dict.values())
        random.seed(seed)
        random.shuffle(index_sets)
        train, val, test = [], [], []
        train_size = int(sizes[0] * len(data))
        val_size = int(sizes[1] * len(data))
        for index_set in index_sets:
            if len(train)+len(index_set) <= train_size:
                train += index_set
            elif len(val) + len(index_set) <= val_size:
                val += index_set
            else:
                test += index_set
        train = [data[i] for i in train]
        val = [data[i] for i in val]
        test = [data[i] for i in test]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'random':
        indices = list(range(len(data)))
        random.shuffle(indices)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = [data[i] for i in indices[:train_size]]
        val = [data[i] for i in indices[train_size:train_val_size]]
        test = [data[i] for i in indices[train_val_size:]]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')


def get_class_sizes(data: MoleculeDataset, proportion: bool = True) -> List[List[float]]:
    """
    Determines the proportions of the different classes in a classification dataset.

    :param data: A classification :class:`~chemprop.data.MoleculeDataset`.
    :param proportion: Choice of whether to return proportions for class size or counts.
    :return: A list of lists of class proportions. Each inner list contains the class proportions for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if data.is_atom_bond_targets:
                for target in targets[i][task_num]:
                    if targets[i][task_num] is not None:
                        valid_targets[task_num].append(target)
            else:
                if targets[i][task_num] is not None:
                    valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        if set(np.unique(task_targets)) > {0, 1}:
            raise ValueError('Classification dataset must only contains 0s and 1s.')
        if proportion:
            try:
                ones = np.count_nonzero(task_targets) / len(task_targets)
            except ZeroDivisionError:
                ones = float('nan')
                print('Warning: class has no targets')
            class_sizes.append([1 - ones, ones])
        else:  # counts
            ones = np.count_nonzero(task_targets)
            class_sizes.append([len(task_targets) - ones, ones])

    return class_sizes



