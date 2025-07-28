#!/usr/bin/env python
# coding: utf-8
#====================================================================================================#
# The following code ensures the code work properly in 
# MS VS, MS VS CODE and jupyter notebook on both Linux and Windows.
import os 
import sys
from os import path
from sys import platform
from pathlib import Path

if __name__ == "__main__":
    print("\n\n")
    print("="*80)
    if os.name == 'nt' or platform == 'win32':
        print("Running on Windows")
        if 'ptvsd' in sys.modules:
            print("Running in Visual Studio")

    if os.name != 'nt' and platform != 'win32':
        print("Not Running on Windows")

    if "__file__" in globals().keys():
        print('CurrentDir: ', os.getcwd())
        try:
            os.chdir(os.path.dirname(__file__))
        except:
            print("Problems with navigating to the file dir.")
        print('CurrentDir: ', os.getcwd())
    else:
        print("Running in python jupyter notebook.")
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                print('workbookDir: ' + workbookDir)
                os.chdir(workbookDir)
        except:
            print("Problems with navigating to the workbook dir.")
#====================================================================================================#
# Imports

import scipy
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
#--------------------------------------------------#
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.Fingerprints import FingerprintMols

#--------------------------------------------------#
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.weight_norm import weight_norm

#--------------------------------------------------#
from itertools import zip_longest

#--------------------------------------------------#
from typing import Union, List, Tuple, Dict, Any, Optional

#--------------------------------------------------#
# from rxnmapper import RXNMapper
# rxn_mapper = RXNMapper()

#--------------------------------------------------#
from AP_convert import Get_Unique_SMILES, MolFromSmiles_ZX

from ZX02_nn_utils import StandardScaler
from ZX02_nn_utils import build_optimizer, build_lr_scheduler
from ZX03_rxn_mpnn_args import TrainArgs
from ZX04_funcs import onek_encoding_unk
from ZX05_loss_functions import get_loss_func

#--------------------------------------------------#
from rxn05A_Batch_Mol_Graph import Feat_params, MolGraph, BatchMolGraph
from rxn05A_Batch_Mol_Graph import set_extra_atom_fdim, set_extra_bond_fdim
from rxn05A_Batch_Mol_Graph import is_mol, is_reaction, is_explicit_h, is_adding_hs, is_keeping_atom_map, make_mols



###################################################################################################################
###################################################################################################################
# Molecule Settings

# Unique SMILES
GetUnqSmi  = Get_Unique_SMILES(isomericSmiles = True , kekuleSmiles = False, canonical = False, SMARTS_bool = False, removeAtomMapNumber = True)










#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   `7MMM.     ,MMF'       `7MM    `7MM"""Yb.            mm                                db             mm                                           #
#     MMMb    dPMM           MM      MM    `Yb.          MM                                               MM                                           #
#     M YM   ,M MM  ,pW"Wq.  MM      MM     `Mb  ,6"Yb.mmMMmm  ,6"Yb. `7MMpdMAo.  ,pW"Wq.`7MM `7MMpMMMb.mmMMmm                                         #
#     M  Mb  M' MM 6W'   `Wb MM      MM      MM 8)   MM  MM   8)   MM   MM   `Wb 6W'   `Wb MM   MM    MM  MM                                           #
#     M  YM.P'  MM 8M     M8 MM      MM     ,MP  ,pm9MM  MM    ,pm9MM   MM    M8 8M     M8 MM   MM    MM  MM                                           #
#     M  `YM'   MM YA.   ,A9 MM      MM    ,dP' 8M   MM  MM   8M   MM   MM   ,AP YA.   ,A9 MM   MM    MM  MM                                           #
#   .JML. `'  .JMML.`Ybmd9'.JMML.  .JMMmmmdP'   `Moo9^Yo.`Mbmo`Moo9^Yo. MMbmmd'   `Ybmd9'.JMML.JMML  JMML.`Mbmo                                        #
#                                                                       MM                                                                             #
#                                                                     .JMML.                                                                           #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#





###################################################################################################################
#  __              `7MMM.     ,MMF'          db                   .g8"""bgd `7MM                                  #
#  MM                MMMb    dPMM                               .dP'     `M   MM                                  #
#  MM   `MM.         M YM   ,M MM   ,6"Yb. `7MM `7MMpMMMb.      dM'       `   MM   ,6"Yb.  ,pP"Ybd ,pP"Ybd        #
#  MM     `Mb.       M  Mb  M' MM  8)   MM   MM   MM    MM      MM            MM  8)   MM  8I   `" 8I   `"        #
#  MMMMMMMMMMMMD     M  YM.P'  MM   ,pm9MM   MM   MM    MM      MM.           MM   ,pm9MM  `YMMMa. `YMMMa.        #
#          ,M'       M  `YM'   MM  8M   MM   MM   MM    MM      `Mb.     ,'   MM  8M   MM  L.   I8 L.   I8        #
#        .M'       .JML. `'  .JMML.`Moo9^Yo.JMML.JMML  JMML.      `"bmmmd'  .JMML.`Moo9^Yo.M9mmmP' M9mmmP'        #
###################################################################################################################

class MoleculeDatapoint:

    #====================================================================================================#
    # MoleculeDatapoint contains a single molecule/reaction and its associated features and targets.
    def __init__(self,
                 smiles                          : List[str]                   , 
                 features                        : np.ndarray  = None          , 
                 phase_features                  : List[float] = None          , 
                 atom_features                   : np.ndarray  = None          , 
                 atom_descriptors                : np.ndarray  = None          , 
                 bond_features                   : np.ndarray  = None          , 
                 bond_descriptors                : np.ndarray  = None          , 
                 overwrite_default_atom_features : bool        = False         , 
                 overwrite_default_bond_features : bool        = False         , 
                 PARAMS                          : Feat_params = Feat_params() , ) :

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # smiles           : A list of the SMILES or GEN_SMILES string(s) representing molecule or reaction.
        #                       - Actually, it is a list containing only one string.
        #                       - The string represents A SINGLE molecule or a reaction.
        # features         : A numpy array containing additional features (e.g., Morgan fingerprint).
        # phase_features   : A one-hot vector indicating the phase of the data, as used in spectra data.
        # atom_features    : A numpy array containing additional atom features to featurize the molecule
        # atom_descriptors : A numpy array containing additional atom descriptors to featurize the molecule.
        # bond_features    : A numpy array containing additional bond features to featurize the molecule.
        # bond_descriptors : A numpy array containing additional bond descriptors to featurize the molecule.

        self.smiles = smiles # self.smiles is a list!

        self.features                         =  features
        self.phase_features                   =  phase_features
        self.atom_descriptors                 =  atom_descriptors
        self.bond_descriptors                 =  bond_descriptors
        self.atom_features                    =  atom_features
        self.bond_features                    =  bond_features
        self.overwrite_default_atom_features  =  overwrite_default_atom_features
        self.overwrite_default_bond_features  =  overwrite_default_bond_features
        self.PARAMS                           =  PARAMS 

        self.is_mol_list              = [is_mol(s)                        for s in smiles           ]
        self.is_reaction_list         = [is_reaction        (x, PARAMS)   for x in self.is_mol_list ]
        self.is_explicit_h_list       = [is_explicit_h      (x, PARAMS)   for x in self.is_mol_list ]
        self.is_adding_hs_list        = [is_adding_hs       (x, PARAMS)   for x in self.is_mol_list ]
        self.is_keeping_atom_map_list = [is_keeping_atom_map(x, PARAMS)   for x in self.is_mol_list ]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
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


        # Save a copy of the raw features and targets to enable different scaling later on. TODO: Is this necessary?
        self.raw_features         = self.features
        self.raw_atom_descriptors = self.atom_descriptors
        self.raw_bond_descriptors = self.bond_descriptors
        self.raw_atom_features    = self.atom_features
        self.raw_bond_features    = self.bond_features


    #====================================================================================================#
    # Get RDKIT mol object from SMILES input string.
    @property
    def mol(self) -> List[Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]]:
        # Gets the corresponding list of RDKit molecules for the corresponding SMILES list.
        mol = make_mols( input_str          = self.smiles                   ,
                         reaction_list      = self.is_reaction_list         ,
                         keep_h_list        = self.is_explicit_h_list       ,
                         add_h_list         = self.is_adding_hs_list        ,
                         keep_atom_map_list = self.is_keeping_atom_map_list , ) 
        return mol

    @property
    def number_of_molecules(self) -> int:
        # Gets the number of molecules in the class MoleculeDatapoint.
        return len(self.smiles)

    #====================================================================================================#
    # Other Functions.
    def set_features(self, features: np.ndarray) -> None:
        # Sets the features of the molecule.
        # features: A 1D numpy array of features for the molecule.
        self.features = features


    def set_atom_descriptors(self, atom_descriptors: np.ndarray) -> None:
        # Sets the atom descriptors of the molecule.
        # atom_descriptors: A 1D numpy array of features for the molecule.
        self.atom_descriptors = atom_descriptors


    def set_atom_features(self, atom_features: np.ndarray) -> None:
        # Sets the atom features of the molecule.
        # atom_features: A 1D numpy array of features for the molecule.
        self.atom_features = atom_features


    def set_bond_features(self, bond_features: np.ndarray) -> None:
        # Sets the bond features of the molecule.
        # bond_features: A 1D numpy array of features for the molecule.
        self.bond_features = bond_features


    def extend_features(self, features: np.ndarray) -> None:
        # Extends the features of the molecule.
        # features: A 1D numpy array of extra features for the molecule.
        self.features = np.append(self.features, features) if self.features is not None else features


    def num_tasks(self) -> int:
        # For Future Use.
        # Returns the number of prediction tasks.
        # return: The number of tasks.
        return len(self.targets)


    def set_targets(self, targets: List[Optional[float]]):
        # For Future Use.
        # Sets the targets of a molecule.
        # targets: A list of floats containing the targets.
        self.targets = targets

























#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   `7MMM.     ,MMF'       `7MM    `7MM"""Yb.            mm                              mm                                                            #
#     MMMb    dPMM           MM      MM    `Yb.          MM                              MM                                                            #
#     M YM   ,M MM  ,pW"Wq.  MM      MM     `Mb  ,6"Yb.mmMMmm  ,6"Yb.  ,pP"Ybd  .gP"Ya mmMMmm                                                          #
#     M  Mb  M' MM 6W'   `Wb MM      MM      MM 8)   MM  MM   8)   MM  8I   `" ,M'   Yb  MM                                                            #
#     M  YM.P'  MM 8M     M8 MM      MM     ,MP  ,pm9MM  MM    ,pm9MM  `YMMMa. 8M""""""  MM                                                            #
#     M  `YM'   MM YA.   ,A9 MM      MM    ,dP' 8M   MM  MM   8M   MM  L.   I8 YM.    ,  MM                                                            #
#   .JML. `'  .JMML.`Ybmd9'.JMML.  .JMMmmmdP'   `Moo9^Yo.`Mbmo`Moo9^Yo.M9mmmP'  `Mbmmd'  `Mbmo                                                         #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

# ``MoleculeDataset`` contains a list of classes MoleculeDatapoint's with access to their attributes.

class MoleculeDataset(Dataset):

    #====================================================================================================#
    # Constructor.
    def __init__(self                                                          , 
                 data                       : List[MoleculeDatapoint]          , 
                 X_TrainArgs                : TrainArgs                = None  , 
                 X_extra_cmpd_features      : Dict                     = None  , 
                 X_extra_atom_bond_features : Dict                     = None  , ):
        
        # data: A list of classes MoleculeDatapoint's.
        self._data                       = data
        self._batch_graph                = None
        self._random                     = random.Random()
        self.X_TrainArgs                 = X_TrainArgs
        self.X_extra_cmpd_features       = X_extra_cmpd_features 
        self.X_extra_atom_bond_features  = X_extra_atom_bond_features  

    def __len__(self) -> int:
        # Returns the length of the dataset (i.e., the number of molecules).
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        # Gets one or more class MoleculeDatapoint's via an index or slice.
        # param 'item': An index (int) or a slice object.
        # return: A class MoleculeDatapoint if an int is provided or a list of class MoleculeDatapoint if a slice is provided.
        return self._data[item]
    
    #====================================================================================================#
    # Key Functions.
    def smiles(self, flatten_bool: bool = False) -> Union[List[str], List[List[str]]]:
        # Returns a list containing the SMILES list associated with each class MoleculeDatapoint.
        # param 'flatten': Whether to flatten the returned SMILES to a list instead of a list of lists.
        # return: A list of SMILES or a list of lists of SMILES, depending on vars 'flatten'.
        if flatten_bool:
            return [smiles for d in self._data for smiles in d.smiles]
        return [d.smiles for d in self._data]
    

    def mols(self, flatten_bool: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]], List[Tuple[Chem.Mol, Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]]]:
        # Returns a list of the RDKit molecules associated with each  CLASS 'MoleculeDatapoint'.
        # param 'flatten': Whether to flatten the returned RDKit molecules to a list instead of a list of lists.
        # return: A list of SMILES or a list of lists of RDKit molecules, depending on code 'flatten'.
        if flatten_bool:
            return [mol for d in self._data for mol in d.mol]
        return [d.mol for d in self._data]
    

    @property
    def number_of_molecules(self) -> int:
        # Gets the number of molecules in each  CLASS 'MoleculeDatapoint'.
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    #====================================================================================================#
    # Trivial Functions. 
    def features(self) -> List[np.ndarray]:
        # Returns the features associated with each molecule (if they exist).
        # return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        if len(self._data) == 0 or self._data[0].features is None:
            return None
        return [d.features for d in self._data]


    def phase_features(self) -> List[np.ndarray]:
        # Returns the phase features associated with each molecule (if they exist).
        # return: A list of 1D numpy arrays containing the phase features for each molecule or None if there are no features.
        if len(self._data) == 0 or self._data[0].phase_features is None:
            return None
        return [d.phase_features for d in self._data]

    def atom_features(self) -> List[np.ndarray]:
        # Returns the atom descriptors associated with each molecule (if they exit).
        # return: A list of 2D numpy arrays containing the atom descriptors
        #         for each molecule or None if there are no features.
        if len(self._data) == 0 or self._data[0].atom_features is None:
            return None
        return [d.atom_features for d in self._data]

    def atom_descriptors(self) -> List[np.ndarray]:
        # Returns the atom descriptors associated with each molecule (if they exit).
        # return: A list of 2D numpy arrays containing the atom descriptors
        #         for each molecule or None if there are no features.
        if len(self._data) == 0 or self._data[0].atom_descriptors is None:
            return None
        return [d.atom_descriptors for d in self._data]

    def bond_features(self) -> List[np.ndarray]:
        # Returns the bond features associated with each molecule (if they exit).
        # return: A list of 2D numpy arrays containing the bond features
        #          for each molecule or None if there are no features.
        if len(self._data) == 0 or self._data[0].bond_features is None:
            return None
        return [d.bond_features for d in self._data]
    
    def bond_descriptors(self) -> List[np.ndarray]:
        # Returns the bond descriptors associated with each molecule (if they exit).
        # return: A list of 2D numpy arrays containing the bond descriptors
        #         for each molecule or None if there are no features.
        if len(self._data) == 0 or self._data[0].bond_descriptors is None:
            return None
        return [d.bond_descriptors for d in self._data]
    
    def features_size(self) -> int:
        # Returns the size of the additional features vector associated with the molecules.
        # return: The size of the additional features vector.
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def atom_descriptors_size(self) -> int:
        # Returns the size of custom additional atom descriptors vector associated with the molecules.
        # > return: The size of the additional atom descriptor vector.
        return len(self._data[0].atom_descriptors[0]) \
            if len(self._data) > 0 and self._data[0].atom_descriptors is not None else None

    def atom_features_size(self) -> int:
        # Returns the size of custom additional atom features vector associated with the molecules.
        # > return: The size of the additional atom feature vector.
        return len(self._data[0].atom_features[0]) \
            if len(self._data) > 0 and self._data[0].atom_features is not None else None

    def bond_features_size(self) -> int:
        # Returns the size of custom additional bond features vector associated with the molecules.
        # > return: The size of the additional bond feature vector.
        return len(self._data[0].bond_features[0]) \
            if len(self._data) > 0 and self._data[0].bond_features is not None else None

    def bond_descriptors_size(self) -> int:
        # Returns the size of custom additional bond descriptors vector associated with the molecules.
        # > return: The size of the additional bond descriptor vector.
        return len(self._data[0].bond_descriptors[0]) \
            if len(self._data) > 0 and self._data[0].bond_descriptors is not None else None

    #====================================================================================================#
    def normalize_features(self, 
                           scaler                 : StandardScaler = None, 
                           replace_nan_token      : int            = 0,
                           scale_atom_descriptors : bool           = False, 
                           scale_bond_features    : bool           = False, ) -> StandardScaler:
        
        # Normalizes the features of the dataset using a class 'nn_utils.StandardScalar'.

        # The class chemprop.data.StandardScaler subtracts the mean and divides by the standard deviation
        # for each feature independently.

        # If a class 'nn_utils.StandardScalar' is provided, it is used to perform the normalization.
        # Otherwise, a class 'nn_utils.StandardScalar' is first fit to the features in this dataset
        # and is then used to perform the normalization.

        # param scaler: A fitted class 'nn_utils.StandardScalar'. If it is provided it is used,
        #               otherwise a new class 'nn_utils.StandardScalar' is first fitted to this
        #               data and is then used.
        # param replace_nan_token      : A token to use to replace NaN entries in the features.
        # param scale_atom_descriptors : If the features that need to be scaled are atom features rather than molecule.
        # param scale_bond_features    : If the features that need to be scaled are bond descriptors rather than molecule.

        # return: A fitted class 'nn_utils.StandardScalar'. If a class 'nn_utils.StandardScalar'
        #         is provided as a parameter, this is the same class 'nn_utils.StandardScalar'. Otherwise,
        #         this is a new class 'nn_utils.StandardScalar' that has been fit on this dataset.

        if len(self._data) == 0 or (self._data[0].features is None and not scale_bond_features and not scale_atom_descriptors):
            return None

        if scaler is None:
            if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
                features = np.vstack([d.raw_atom_descriptors for d in self._data])
            elif scale_atom_descriptors and not self._data[0].atom_features is None:
                features = np.vstack([d.raw_atom_features for d in self._data])
            elif scale_bond_features:
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
        elif scale_bond_features:
            for d in self._data:
                d.set_bond_features(scaler.transform(d.raw_bond_features))
        else:
            for d in self._data:
                d.set_features(scaler.transform(d.raw_features.reshape(1, -1))[0])
        return scaler


    ###################################################################################################################
    # __                `7MM"""Yp,          mm         `7MM         .g8"""bgd                          `7MM           #
    # MM                  MM    Yb          MM           MM       .dP'     `M                            MM           #
    # MM   `MM.           MM    dP  ,6"Yb.mmMMmm ,p6"bo  MMpMMMb. dM'       ` `7Mb,od8 ,6"Yb. `7MMpdMAo. MMpMMMb.     #
    # MM     `Mb.         MM"""bg. 8)   MM  MM  6M'  OO  MM    MM MM            MM' "'8)   MM   MM   `Wb MM    MM     #
    # MMMMMMMMMMMMD       MM    `Y  ,pm9MM  MM  8M       MM    MM MM.    `7MMF' MM     ,pm9MM   MM    M8 MM    MM     #
    #         ,M'         MM    ,9 8M   MM  MM  YM.    , MM    MM `Mb.     MM   MM    8M   MM   MM   ,AP MM    MM     #
    #       .M'         .JMMmmmd9  `Moo9^Yo.`MbmoYMbmd'.JMML  JMML. `"bmmmdPY .JMML.  `Moo9^Yo. MMbmmd'.JMML  JMML.   #
    #                                                                                           MM                    #
    #                                                                                         .JMML.                  #
    ###################################################################################################################

    def batch_graph(self) -> List[BatchMolGraph]:

        print("start batch_graph!")

        # Constructs a  CLASS '~chemprop.features.BatchMolGraph' with the graph featurization of all the molecules.

        # - note:
        #   The  CLASS '~chemprop.features.BatchMolGraph' is cached in after the first time it is computed
        #   and is simply accessed upon subsequent calls to method ``batch_graph``. This means that if the underlying
        #   set of  CLASS ``MoleculeDatapoint`` \'s changes, then the returned  CLASS '~chemprop.features.BatchMolGraph'
        #   will be incorrect for the underlying data.

        # > return: A list of  CLASS ``~chemprop.features.BatchMolGraph`` containing the graph featurization of all the
        #           molecules in each  CLASS ``MoleculeDatapoint`.

        if self._batch_graph is None:

            self._batch_graph = []
            mol_graphs        = [] # shape: (num_cmpd, 1, 1)

            for one_rxn_cmpd_data in self._data: # one_rxn_cmpd_data contains a MoleculeDatapoint object (empty features of one compound).
                mol_graphs_list = []

                for one_rxn_cmpd_smiles, one_rxn_cmpd_mol in zip(one_rxn_cmpd_data.smiles, one_rxn_cmpd_data.mol): 
                    print("one_rxn_cmpd_smiles: ", one_rxn_cmpd_smiles)
                    print("one_rxn_cmpd_mol: ", one_rxn_cmpd_mol)


                    #--------------------------------------------------#
                    if len(one_rxn_cmpd_data.smiles) > 1 and (one_rxn_cmpd_data.atom_features is not None or one_rxn_cmpd_data.bond_features is not None):
                        raise NotImplementedError('Atom descriptors are currently only supported with one '
                                                  'molecule per input (i.e., number_of_molecules = 1).')
                    #--------------------------------------------------#
                    mol_graph = MolGraph(one_mol                         =  one_rxn_cmpd_mol                                  , 
                                         one_GEN_SMILES                  =  one_rxn_cmpd_smiles                               ,
                                         atom_features_extra             =  one_rxn_cmpd_data.atom_features                   , 
                                         bond_features_extra             =  one_rxn_cmpd_data.bond_features                   , 
                                         overwrite_default_atom_features =  one_rxn_cmpd_data.overwrite_default_atom_features , 
                                         overwrite_default_bond_features =  one_rxn_cmpd_data.overwrite_default_bond_features , 
                                         X_extra_cmpd_features           =  self.X_extra_cmpd_features                        , 
                                         X_extra_atom_bond_features      =  self.X_extra_atom_bond_features                   , 
                                         X_TrainArgs                     =  self.X_TrainArgs                                  , 
                                         PARAMS                          =  one_rxn_cmpd_data.PARAMS                          , )

                    mol_graphs_list.append(mol_graph)

                mol_graphs.append(mol_graphs_list)

            # For future use, here use a more complicated expression.
            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]
            #print("len(self._batch_graph): ", len(self._batch_graph))  # len = 1

        return self._batch_graph























#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#       .g8""8q.     mm  `7MM                             `7MM"""YMM                               mm     db                                           #
#     .dP'    `YM.   MM    MM                               MM    `7                               MM                                                  #
#     dM'      `MM mmMMmm  MMpMMMb.  .gP"Ya `7Mb,od8        MM   d `7MM  `7MM `7MMpMMMb.  ,p6"bo mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.                        #
#     MM        MM   MM    MM    MM ,M'   Yb  MM' "'        MM""MM   MM    MM   MM    MM 6M'  OO   MM     MM 6W'   `Wb MM    MM                        #
#     MM.      ,MP   MM    MM    MM 8M""""""  MM            MM   Y   MM    MM   MM    MM 8M        MM     MM 8M     M8 MM    MM                        #
#     `Mb.    ,dP'   MM    MM    MM YM.    ,  MM            MM       MM    MM   MM    MM YM.    ,  MM     MM YA.   ,A9 MM    MM                        #
#       `"bmmd"'     `Mbm.JMML  JMML.`Mbmmd'.JMML.        .JMML.     `Mbod"YML.JMML  JMML.YMbmd'   `Mbmo.JMML.`Ybmd9'.JMML  JMML.                      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

def filter_invalid_smiles(X_molecule_dataset: MoleculeDataset, X_TrainArgs: TrainArgs) -> MoleculeDataset:
    # Filters out invalid SMILES.
    # - X_molecule_dataset   : A  CLASS `~chemprop.data.MoleculeDataset`.
    # > return               : A  CLASS `~chemprop.data.MoleculeDataset` with only the valid molecules.

    '''
    return MoleculeDataset(
                            [datapoint for datapoint in data if 
                                    all(s != '' for s in datapoint.smiles)                                           
                                and all(m is not None for m in datapoint.mol)                                        
                                and all(m.GetNumHeavyAtoms() > 0 for m in datapoint.mol if not isinstance(m, tuple)) 
                                and all(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() > 0 for m in datapoint.mol if isinstance(m, tuple))
                            ]            , 
                            X_TrainArgs  ,
                            )
                            '''

    valid_data = []

    for datapoint in X_molecule_dataset:
        smiles = datapoint.smiles
        mol    = datapoint.mol

        if all(s != '' for s in smiles)                                                               and \
           all(m is not None for m in mol)                                                            and \
           all(m.GetNumHeavyAtoms() > 0 for m in mol if not isinstance(m, tuple))                     and \
           all(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() > 0 for m in mol if isinstance(m, tuple)) :
           
                valid_data.append(datapoint)


    if X_TrainArgs is not None:
        return MoleculeDataset(valid_data                                     , 
                               X_TrainArgs                                    , 
                               X_molecule_dataset.X_extra_cmpd_features       , 
                               X_molecule_dataset.X_extra_atom_bond_features  , )
    else:
        return MoleculeDataset(valid_data                                     , 
                               X_molecule_dataset.X_TrainArgs                 , 
                               X_molecule_dataset.X_extra_cmpd_features       , 
                               X_molecule_dataset.X_extra_atom_bond_features  , )



















#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#     .g8"""bgd `7MMM.     ,MMF`7MM"""Mq`7MM"""Yb.       `7MM"""Mq. `YMM'   `MP'`7MN.   `7MF'    `7MM"""Yb.      db  MMP""MM""YMM  db                  #
#   .dP'     `M   MMMb    dPMM   MM   `MM.MM    `Yb.       MM   `MM.  VMb.  ,P    MMN.    M        MM    `Yb.   ;MM: P'   MM   `7 ;MM:                 #
#   dM'       `   M YM   ,M MM   MM   ,M9 MM     `Mb       MM   ,M9    `MM.M'     M YMb   M        MM     `Mb  ,V^MM.     MM     ,V^MM.                #
#   MM            M  Mb  M' MM   MMmmdM9  MM      MM       MMmmdM9       MMb      M  `MN. M        MM      MM ,M  `MM     MM    ,M  `MM                #
#   MM.           M  YM.P'  MM   MM       MM     ,MP       MM  YM.     ,M'`Mb.    M   `MM.M        MM     ,MP AbmmmqMA    MM    AbmmmqMA               #
#   `Mb.     ,'   M  `YM'   MM   MM       MM    ,dP'       MM   `Mb.  ,P   `MM.   M     YMM        MM    ,dP'A'     VML   MM   A'     VML              #
#     `"bmmmd'  .JML. `'  .JMML.JMML.   .JMMmmmdP'       .JMML. .JMM.MM:.  .:MMa.JML.    YM      .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.            #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

# Format smiles set and compound features into class MoleculeDataset.
def RXN_CMPD_Dataset_Main(X_tr_GEN_SMILES         : List        , # Generalized SMILES (compound smiles or reaction smiles string)
                          X_ts_GEN_SMILES         : List        , # Generalized SMILES (compound smiles or reaction smiles string)
                          X_va_GEN_SMILES         : List        , # Generalized SMILES (compound smiles or reaction smiles string)
                          X_extra_cmpd_features   : Dict        , 
                          X_TrainArgs             : TrainArgs   , 
                          data_folder             : Path = None , 
                          extra_cmpd_feature_file : str  = None , 
                          ):

    #====================================================================================================#
    # Initialize the ``MoleculeDataset`` settings.
    PARAMS = Feat_params()

    # Change the default parameters in the ``MoleculeDataset`` settings if necessary.
    if X_TrainArgs.reaction:
        PARAMS.REACTION = X_TrainArgs.reaction
        if X_TrainArgs.reaction:
            PARAMS.EXTRA_ATOM_FDIM = PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1
            PARAMS.EXTRA_BOND_FDIM = PARAMS.BOND_FDIM
            PARAMS.REACTION_MODE   = X_TrainArgs.reaction_mode

    elif X_TrainArgs.reaction_solvent:
        PARAMS.REACTION = True
        PARAMS.EXTRA_ATOM_FDIM = PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1
        PARAMS.EXTRA_BOND_FDIM = PARAMS.BOND_FDIM
        PARAMS.REACTION_MODE   = X_TrainArgs.reaction_mode


    #====================================================================================================#
    # Use customized ATOM/BOND-level features.
    if extra_cmpd_feature_file != None :
        # Get Compound Encodings from X02 pickles.
        with open( data_folder / extra_cmpd_feature_file, 'rb') as extra_cmpd_features:
            X_extra_atom_bond_features = pickle.load(extra_cmpd_features)
    else:
        X_extra_atom_bond_features = None


    #====================================================================================================#
    # Parse X_extra_atom_bond_features.
    '''
    extra_feature_smiles_list                = X_extra_atom_bond_features["smiles_list"               ]
    extra_feature_Atom_Attributes_list       = X_extra_atom_bond_features["Atom_Attributes_list"      ] # shape: (num_mol, ~num_atom, extra_atom_feature_dim)
    extra_feature_Atom_RXN_Portion_list      = X_extra_atom_bond_features["Atom_RXN_Portion_list"     ] # Used in HGNN only
    extra_feature_Bond_Adjacency_Matrix_list = X_extra_atom_bond_features["Bond_Adjacency_Matrix_list"] # Used in HGNN only
    extra_feature_Bond_Attributes_list       = X_extra_atom_bond_features["Bond_Attributes_list"      ] # Used in HGNN only
    extra_feature_Bond_general_info_list     = X_extra_atom_bond_features["Bond_general_info_list"    ] # shape: (num_mol, ~num_bond, extra_bond_feature_dim)

    X_extra_atom_bond_features = dict([])
    for i, (smls, a_ft, a_rp, b_aj, b_ft, b_if) in enumerate(zip(extra_feature_smiles_list                , \
                                                                 extra_feature_Atom_Attributes_list       , \
                                                                 extra_feature_Atom_RXN_Portion_list      , \
                                                                 extra_feature_Bond_Adjacency_Matrix_list , \
                                                                 extra_feature_Bond_Attributes_list       , \
                                                                 extra_feature_Bond_general_info_list     ,  )):
        
        X_extra_atom_bond_features[smls] = [a_ft, b_if] # TODO : Add more features if needed.
    
    X_TrainArgs.scale_bond_features    = False 
    X_TrainArgs.scale_atom_descriptors = False 

    X_TrainArgs.Extra_Atom_Feature_Dim = len(extra_feature_Atom_Attributes_list[0][0])
    #X_TrainArgs.Extra_Bond_Feature_Dim = len(extra_feature_Bond_general_info_list[0][0])

    # Are those extra bond features really used here?
    print("X_TrainArgs.Extra_Atom_Feature_Dim  : ", X_TrainArgs.Extra_Atom_Feature_Dim)
    print("X_TrainArgs.Extra_Bond_Feature_Dim  : ", X_TrainArgs.Extra_Bond_Feature_Dim)
    '''

    #====================================================================================================#
    # Do NOT use customized atom-level features (for compound input only).
    '''
    X_tr_atom_features    = X_ts_atom_features    = X_va_atom_features    = None
    X_tr_atom_descriptors = X_ts_atom_descriptors = X_va_atom_descriptors = None
    X_tr_bond_features    = X_ts_bond_features    = X_va_bond_features    = None
    '''

    #====================================================================================================#
    # 
    empty_atom_bond_features = {smls : [None, None] for smls in X_extra_cmpd_features }

    # Initialize EMPTY Compound/Reaction - Level Features. 
    #   - TODO Compound Level Features are to be replaced by ``ECFP6 Count Encodings``. 
    #   - TODO Reaction Level Features are to be replaced by some encodings (some ``ECFP6-based encodings`` to be programed).
    #   - TODO The above two can be performed before MoleculeDatapoint() is called or inside the MoleculeDatapoint() class.
    #   - TODO d-MPNN does these two steps before the MoleculeDatapoint() class is called, this can be modified here. 
    features_tr = [None] * len(X_tr_GEN_SMILES) # **EMPTY** extra Compound/Reaction - Level Features.
    features_ts = [None] * len(X_ts_GEN_SMILES) # **EMPTY** extra Compound/Reaction - Level Features.
    features_va = [None] * len(X_va_GEN_SMILES) # **EMPTY** extra Compound/Reaction - Level Features.

    #====================================================================================================#
    # 
    #X_tr_GEN_SMILES = [  rxn_mapper.get_attention_guided_atom_maps([one_GEN_SMILES, ])[0]["mapped_rxn"]  for one_GEN_SMILES in X_tr_GEN_SMILES]
    #X_ts_GEN_SMILES = [  rxn_mapper.get_attention_guided_atom_maps([one_GEN_SMILES, ])[0]["mapped_rxn"]  for one_GEN_SMILES in X_ts_GEN_SMILES]
    #X_va_GEN_SMILES = [  rxn_mapper.get_attention_guided_atom_maps([one_GEN_SMILES, ])[0]["mapped_rxn"]  for one_GEN_SMILES in X_va_GEN_SMILES]

    #====================================================================================================#
    # 
    X_tr_smiles_dataset = \
        MoleculeDataset([ MoleculeDatapoint(smiles                          = [smiles,]                                    , 
                                            features                        = features_tr[i]                               , # Extra Compound/Reaction - Level Features;
                                            phase_features                  = None                                         , # Unused in most cases;
                                            atom_features                   = None                                         , # Extra Atom - Level Features;
                                            atom_descriptors                = None                                         , 
                                            bond_features                   = None                                         , # Extra Bond - Level Features;
                                            bond_descriptors                = None                                         , 
                                            overwrite_default_atom_features = X_TrainArgs.overwrite_default_atom_features  , # Whether to overwrite default atom features;
                                            overwrite_default_bond_features = X_TrainArgs.overwrite_default_bond_features  , # Whether to overwrite default bond features;
                                            PARAMS                          = PARAMS                                       , 
                                            )
                          for i, smiles in enumerate(X_tr_GEN_SMILES) ], 
                        X_TrainArgs                                    , 
                        X_extra_cmpd_features       = X_extra_cmpd_features        , # Extra Compound/Reaction Level Features Dict;
                        X_extra_atom_bond_features  = X_extra_atom_bond_features   , # Extra Atom/Bond Level Features Dict;
                       )



    X_ts_smiles_dataset = \
        MoleculeDataset([ MoleculeDatapoint(smiles                          = [smiles,]                                    , 
                                            features                        = features_ts[i]                               , 
                                            phase_features                  = None                                         , 
                                            atom_features                   = None                                         , 
                                            atom_descriptors                = None                                         , 
                                            bond_features                   = None                                         , 
                                            bond_descriptors                = None                                         , 
                                            overwrite_default_atom_features = X_TrainArgs.overwrite_default_atom_features  , 
                                            overwrite_default_bond_features = X_TrainArgs.overwrite_default_bond_features  , 
                                            PARAMS                          = PARAMS                                       , 
                                            )
                          for i, smiles in enumerate(X_ts_GEN_SMILES) ], 
                        X_TrainArgs                                    , 
                        X_extra_cmpd_features       = X_extra_cmpd_features        , 
                        X_extra_atom_bond_features  = X_extra_atom_bond_features   , 
                       )


    X_va_smiles_dataset = \
        MoleculeDataset([ MoleculeDatapoint(smiles                          = [smiles,]                                    , 
                                            features                        = features_va[i]                               , 
                                            phase_features                  = None                                         , 
                                            atom_features                   = None                                         , 
                                            atom_descriptors                = None                                         , 
                                            bond_features                   = None                                         , 
                                            bond_descriptors                = None                                         , 
                                            overwrite_default_atom_features = X_TrainArgs.overwrite_default_atom_features  , 
                                            overwrite_default_bond_features = X_TrainArgs.overwrite_default_bond_features  , 
                                            PARAMS                          = PARAMS                                       , 
                                            )
                          for i, smiles in enumerate(X_va_GEN_SMILES) ], 
                        X_TrainArgs                                    , 
                        X_extra_cmpd_features       = X_extra_cmpd_features        , 
                        X_extra_atom_bond_features  = X_extra_atom_bond_features   , 
                       )


    #====================================================================================================#
    # Validate the datasets.
    original_data_len   = len(X_tr_smiles_dataset)
    original_data_len   = len(X_ts_smiles_dataset)
    original_data_len   = len(X_va_smiles_dataset)

    X_tr_smiles_dataset = filter_invalid_smiles(X_tr_smiles_dataset, X_TrainArgs = X_TrainArgs)
    X_ts_smiles_dataset = filter_invalid_smiles(X_ts_smiles_dataset, X_TrainArgs = X_TrainArgs)
    X_va_smiles_dataset = filter_invalid_smiles(X_va_smiles_dataset, X_TrainArgs = X_TrainArgs)

    if len(X_tr_smiles_dataset) < original_data_len:
        print(f'Warning: {original_data_len - len(X_tr_smiles_dataset)} SMILES are invalid.')
    if len(X_ts_smiles_dataset) < original_data_len:
        print(f'Warning: {original_data_len - len(X_ts_smiles_dataset)} SMILES are invalid.')
    if len(X_va_smiles_dataset) < original_data_len:
        print(f'Warning: {original_data_len - len(X_va_smiles_dataset)} SMILES are invalid.')

    #====================================================================================================#
    # Update X_TrainArgs. 
    #   - The dimensions updated below are not currently in use.

    X_TrainArgs.features_size = X_tr_smiles_dataset.features_size()

    if X_TrainArgs.atom_descriptors == 'descriptor':
        X_TrainArgs.atom_descriptors_size = X_tr_smiles_dataset.atom_descriptors_size()
        X_TrainArgs.ffn_hidden_size += X_TrainArgs.atom_descriptors_size 

    elif X_TrainArgs.atom_descriptors == 'feature':
        X_TrainArgs.atom_features_size = X_tr_smiles_dataset.atom_features_size()
        set_extra_atom_fdim(X_TrainArgs.atom_features_size, PARAMS)
        
    if X_TrainArgs.bond_descriptors == 'descriptor':
        X_TrainArgs.bond_descriptors_size = X_tr_smiles_dataset.bond_descriptors_size()
        
    elif X_TrainArgs.bond_descriptors == 'feature':
        X_TrainArgs.bond_features_size = X_tr_smiles_dataset.bond_features_size()
        set_extra_bond_fdim(X_TrainArgs.bond_features_size)



    # if X_TrainArgs.bond_descriptors_path is not None:
    #     X_TrainArgs.bond_features_size = X_tr_smiles_dataset.bond_features_size()
    #     set_extra_bond_fdim(X_TrainArgs.bond_features_size, PARAMS)


    if X_TrainArgs.features_scaling:
        features_scaler = X_tr_smiles_dataset.normalize_features(replace_nan_token=0)
        X_va_smiles_dataset.normalize_features(features_scaler)
        X_ts_smiles_dataset.normalize_features(features_scaler)
    else:
        features_scaler = None


    if X_TrainArgs.atom_descriptor_scaling and X_TrainArgs.atom_descriptors is not None:
        atom_descriptor_scaler = X_tr_smiles_dataset.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
        X_va_smiles_dataset.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        X_ts_smiles_dataset.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None


    if X_TrainArgs.bond_descriptor_scaling and X_TrainArgs.bond_descriptors is not None:
        bond_descriptor_scaler = X_tr_smiles_dataset.normalize_features(replace_nan_token=0, scale_bond_descriptors=True)
        X_va_smiles_dataset.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
        X_ts_smiles_dataset.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
    else:
        bond_descriptor_scaler = None

    X_TrainArgs.train_data_size = len(X_tr_smiles_dataset)

    #====================================================================================================#
    # Validate the size of arguments.


    print("X_TrainArgs.features_scaling        : " , X_TrainArgs.features_scaling       )
    print("X_TrainArgs.atom_descriptor_scaling : " , X_TrainArgs.atom_descriptor_scaling)
    print("X_TrainArgs.bond_features_size      : " , X_TrainArgs.bond_features_size     )
    print("X_TrainArgs.bond_feature_scaling    : " , X_TrainArgs.bond_descriptor_scaling   )
    print()


    """    
    mole_features = np.vstack([d.raw_features         for d in X_tr_smiles_dataset._data])
    atom_features = np.vstack([d.raw_atom_features    for d in X_tr_smiles_dataset._data])
    bond_features = np.vstack([d.raw_bond_features    for d in X_tr_smiles_dataset._data])
    atom_descript = np.vstack([d.raw_atom_descriptors for d in X_tr_smiles_dataset._data])

    print("mole_features: ", mole_features.shape, mole_features)
    print("atom_features: ", atom_features.shape, atom_features)
    print("bond_features: ", bond_features.shape, bond_features)
    print("atom_descript: ", atom_descript.shape, atom_descript)
    print()
    """

    
    # mole_features = [d.raw_features         for d in X_tr_smiles_dataset._data]
    # atom_features = [d.raw_atom_features    for d in X_tr_smiles_dataset._data]
    # bond_features = [d.raw_bond_features    for d in X_tr_smiles_dataset._data]
    # atom_descript = [d.raw_atom_descriptors for d in X_tr_smiles_dataset._data]

    # print("mole_features: ", len(mole_features), mole_features)
    # print("atom_features: ", len(atom_features), atom_features)
    # print("bond_features: ", len(bond_features), bond_features)
    # print("atom_descript: ", len(atom_descript), atom_descript)
    # print()
    
    # for one_mol_atom_features in atom_features:
    #     print(one_mol_atom_features, "\n", np.linalg.norm(one_mol_atom_features, np.inf), "\n", np.linalg.norm(one_mol_atom_features, 2) )


    # print( "len(X_tr_smiles_dataset.batch_graph())               : " , len(X_tr_smiles_dataset.batch_graph())              )
    # print( "X_tr_smiles_dataset.batch_graph()[0].f_atoms         : " , X_tr_smiles_dataset.batch_graph()[0].f_atoms        )
    # print( "X_tr_smiles_dataset.batch_graph()[0].f_atoms.size()  : " , X_tr_smiles_dataset.batch_graph()[0].f_atoms.size() )
    print( "X_tr_smiles_dataset.features_size()                    : " , X_tr_smiles_dataset.features_size()                 ) 
    print( "X_tr_smiles_dataset.atom_features_size()               : " , X_tr_smiles_dataset.atom_features_size()            ) # None
    print( "X_tr_smiles_dataset.bond_features_size()               : " , X_tr_smiles_dataset.bond_features_size()            ) # None
    print( "X_TrainArgs.train_data_size                            : " , X_TrainArgs.train_data_size                         ) 
    print()

    return X_tr_smiles_dataset, X_ts_smiles_dataset, X_va_smiles_dataset, PARAMS


































#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#      .g8"""bgd                           `7MM      `7MMF'                           `7MM           OO                                                #
#    .dP'     `M                             MM        MM                               MM           88                                                #
#    dM'       `   ,pW"Wq.   ,pW"Wq.    ,M""bMM        MM        `7MM  `7MM   ,p6"bo    MM  ,MP'     ||                                                #
#    MM           6W'   `Wb 6W'   `Wb ,AP    MM        MM          MM    MM  6M'  OO    MM ;Y        ||                                                #
#    MM.    `7MMF'8M     M8 8M     M8 8MI    MM        MM      ,   MM    MM  8M         MM;Mm        ''                                                #
#    `Mb.     MM  YA.   ,A9 YA.   ,A9 `Mb    MM        MM     ,M   MM    MM  YM.    ,   MM `Mb.      __                                                #
#      `"bmmmdPY   `Ybmd9'   `Ybmd9'   `Wbmd"MML.    .JMMmmmmMMM   `Mbod"YML. YMbmd'  .JMML. YA.     MM                                                #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

if __name__ == "__main__":
    print()











#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
###################################################################################################################
###################################################################################################################
#====================================================================================================#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#--------------------------------------------------#
#------------------------------

#                  __                  A             M       
#      `MM.        MM                 MMM            M       
#        `Mb.      MM   `MM.         MMMMM           M       
# MMMMMMMMMMMMD    MM     `Mb.     ,MA:M:AM.     `7M'M`MF'   
#         ,M'      MMMMMMMMMMMMD       M           VAMAV     
#       .M'                ,M'         M            VVV      
#                        .M'           M             V       






