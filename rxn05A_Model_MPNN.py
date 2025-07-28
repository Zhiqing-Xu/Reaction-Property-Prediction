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
import numpy as np
import pandas as pd
import seaborn as sns
#--------------------------------------------------#
from tokenize import Double
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
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.weight_norm import weight_norm
#--------------------------------------------------#
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
#--------------------------------------------------#
import pickle
import random
import threading
#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection, Iterator
#--------------------------------------------------#
from functools import reduce

#--------------------------------------------------#
from ZX02_nn_utils import StandardScaler
from ZX02_nn_utils import initialize_weights
from ZX02_nn_utils import get_activation_function
from ZX02_nn_utils import build_optimizer, build_lr_scheduler
from ZX02_nn_utils import *

from ZX03_rxn_mpnn_args import TrainArgs
from ZX04_funcs import onek_encoding_unk
from ZX05_loss_functions import get_loss_func

#--------------------------------------------------#
from rxn05A_Batch_Mol_Graph import *
from rxn05A_RXN_CMPD_Dataset import * 

#--------------------------------------------------#



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MM"""Yb.      db  MMP""MM""YMM  db     `7MMF'        .g8""8q.      db     `7MM"""Yb. `7MM"""YMM  `7MM"""Mq.                                M      #
#    MM    `Yb.   ;MM: P'   MM   `7 ;MM:      MM        .dP'    `YM.   ;MM:      MM    `Yb. MM    `7    MM   `MM.                               M      #
#    MM     `Mb  ,V^MM.     MM     ,V^MM.     MM        dM'      `MM  ,V^MM.     MM     `Mb MM   d      MM   ,M9                                M      #
#    MM      MM ,M  `MM     MM    ,M  `MM     MM        MM        MM ,M  `MM     MM      MM MMmmMM      MMmmdM9                             `7M'M`MF'  #
#    MM     ,MP AbmmmqMA    MM    AbmmmqMA    MM      , MM.      ,MP AbmmmqMA    MM     ,MP MM   Y  ,   MM  YM.                               VAM,V    #
#    MM    ,dP'A'     VML   MM   A'     VML   MM     ,M `Mb.    ,dP'A'     VML   MM    ,dP' MM     ,M   MM   `Mb.                              VVV     #
#  .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.JMMmmmmMMM   `"bmmd"'.AMA.   .AMMA.JMMmmmdP' .JMMmmmmMMM .JMML. .JMM.                              V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

# 
def construct_molecule_batch(X_molecule_dataset: MoleculeDataset, X_TrainArgs: TrainArgs) -> MoleculeDataset: 

    # Constructs a `MoleculeDataset` class from a list of `MoleculeDatapoint` classes.
    # Additionally, precomputes the `chemprop.features.BatchMolGraph` class for the constructed.

    # MoleculeDataset class. 
    # param data: A list of `MoleculeDatapoint` classes.
    # return: A `MoleculeDataset` class containing all the `MoleculeDatapoint` classes.

    if X_TrainArgs is not None:
        X_molecule_dataset = MoleculeDataset(X_molecule_dataset                              , 
                                             X_TrainArgs                                     , 
                                             X_molecule_dataset.X_extra_cmpd_features        , 
                                             X_molecule_dataset.X_extra_atom_bond_features   , )
    else:
        X_molecule_dataset = MoleculeDataset(X_molecule_dataset                              , 
                                             X_molecule_dataset.X_TrainArgs                  , 
                                             X_molecule_dataset.X_extra_cmpd_features        , 
                                             X_molecule_dataset.X_extra_atom_bond_features   , )

    X_molecule_dataset.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

    return X_molecule_dataset



# Machine Learning Ready Dataset.
class MPNN_Dataset(data.Dataset):
    # MPNN_Dataset is for processing and organizing data that being sent to the model. 
    # In order to process molecules, use Molecule Dataloader to load molecule data (in Z05G_Cpd_Data).
    def __init__(self, cmpd_dataset    : MoleculeDataset , 
                       target          : List            , 
                       X_TrainArgs     : TrainArgs       , ): 

        super().__init__()
        self.cmpd_dataset  = cmpd_dataset
        self.target        = target
        self.X_TrainArgs   = X_TrainArgs

    def __len__(self):
        return len(self.cmpd_dataset)

    def __getitem__(self, idx):
        return self.cmpd_dataset[idx],  \
               self.target      [idx]                 

    def collate_fn(self, batch : List[Tuple[Any, ...]]) -> Dict[str, Any]:
        cmpd_dataset, target = zip(*batch)
        cmpd_dataset = MoleculeDataset(cmpd_dataset                                    , 
                                       self.X_TrainArgs                                , 
                                       self.cmpd_dataset.X_extra_cmpd_features         , 
                                       self.cmpd_dataset.X_extra_atom_bond_features    , )

        cmpd_dataset = construct_molecule_batch(cmpd_dataset, self.X_TrainArgs) 
        return {'cmpd_dataset'    : cmpd_dataset                                      , # MoleculeDataset  ;
                #'cmpd_feature'   : torch.tensor(np.array(cmpd_dataset.features()))   , # MoleculeDataset  ;
                'y_property'      : torch.tensor(np.array(list(target)))              , # torch.LongTensor ;
                }



# DataLoader.
def get_MPNN_DataLoader(X_tr_GEN_SMILES_dataset, y_tr,
                        X_va_GEN_SMILES_dataset, y_va,
                        X_ts_GEN_SMILES_dataset, y_ts,
                        batch_size, X_TrainArgs):

    X_y_tr = MPNN_Dataset(X_tr_GEN_SMILES_dataset, y_tr, X_TrainArgs)
    X_y_va = MPNN_Dataset(X_va_GEN_SMILES_dataset, y_va, X_TrainArgs)
    X_y_ts = MPNN_Dataset(X_ts_GEN_SMILES_dataset, y_ts, X_TrainArgs)

    train_loader = data.DataLoader(X_y_tr, batch_size, True,  collate_fn = X_y_tr.collate_fn)
    valid_loader = data.DataLoader(X_y_va, batch_size, False, collate_fn = X_y_va.collate_fn)
    test_loader  = data.DataLoader(X_y_ts, batch_size, False, collate_fn = X_y_ts.collate_fn)
    
    return train_loader, valid_loader, test_loader











#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MM"""Mq. `YMM'   `MP'`7MN.   `7MF'      .g8"""bgd `7MMM.     ,MMF'`7MM"""Mq.`7MM"""Yb.      `7MM"""YMM  `7MN.   `7MF' .g8"""bgd            M      #
#    MM   `MM.  VMb.  ,P    MMN.    M      .dP'     `M   MMMb    dPMM    MM   `MM. MM    `Yb.      MM    `7    MMN.    M .dP'     `M            M      #
#    MM   ,M9    `MM.M'     M YMb   M      dM'       `   M YM   ,M MM    MM   ,M9  MM     `Mb      MM   d      M YMb   M dM'       `            M      #
#    MMmmdM9       MMb      M  `MN. M      MM            M  Mb  M' MM    MMmmdM9   MM      MM      MMmmMM      M  `MN. M MM                 `7M'M`MF'  #
#    MM  YM.     ,M'`Mb.    M   `MM.M      MM.           M  YM.P'  MM    MM        MM     ,MP      MM   Y  ,   M   `MM.M MM.                  VAM,V    #
#    MM   `Mb.  ,P   `MM.   M     YMM      `Mb.     ,'   M  `YM'   MM    MM        MM    ,dP'      MM     ,M   M     YMM `Mb.     ,'           VVV     #
#  .JMML. .JMM.MM:.  .:MMa.JML.    YM        `"bmmmd'  .JML. `'  .JMML..JMML.    .JMMmmmdP'      .JMMmmmmMMM .JML.    YM   `"bmmmd'             V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Model: (ZXCaution: Need to specify mpn_shared in the arguments for compound protein interaction training.)

class RXN_CMPD_Encoder(nn.Module): # Modified from CLASS `MPNEncoder` from `chemprop` package.

    # Class `RXN_CMPD_Encoder` is a directed message passing neural network with the following design of the MPNN:
    #   - Use "ECFP6 Count Encodings of molecular structures" as compound features.
    #   - Use "Morgan Count Encoding of molecular substructures" as atom features. 
    #   - Use "Atom rdkit profiles" as atom features.
    #   - Use "Bond rdkit profiles" as bond features.
    #   - Use "Compound profiles" as EXTRA compound features?

    def __init__(self, 
                 X_TrainArgs : TrainArgs          , 
                 atom_fdim   : int                , 
                 bond_fdim   : int                , 
                 hidden_size : int        = None  ,
                 bias        : bool       = None  , 
                 depth       : int        = None  , ):
        
        # - X_TrainArgs  : A `X_TrainArgs` object containing model arguments.
        # - atom_fdim    : Atom feature vector dimension.
        # - bond_fdim    : Bond feature vector dimension.
        # - hidden_size  : Hidden layers dimension.
        # - bias         : Whether to add bias to linear layers.
        # - depth        : Number of message passing steps.


        #====================================================================================================#
        # Model Settings.
        super(RXN_CMPD_Encoder, self).__init__()

        # Get model hyperparams from TrainArgs:
        self.hidden_size      = hidden_size          or X_TrainArgs.hidden_size
        self.bias             = bias                 or X_TrainArgs.bias
        self.depth            = depth                or X_TrainArgs.depth
        self.device           = X_TrainArgs.device
        self.dropout          = X_TrainArgs.dropout
        self.undirected       = X_TrainArgs.undirected
        self.atom_messages    = X_TrainArgs.atom_messages
        self.aggregation      = X_TrainArgs.aggregation
        self.aggregation_norm = X_TrainArgs.aggregation_norm

        self.layers_per_message = 1

        # For Constructing the model
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim


        # Dropout
        self.dropout_layer = nn.Dropout(p = self.dropout)

        # Activation
        self.act_func = get_activation_function(X_TrainArgs.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad = False)

        #====================================================================================================#
        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias = self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        #====================================================================================================#
        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias = self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        #====================================================================================================#
        # layer after concatenating the descriptors if X_TrainArgs.atom_descriptors == descriptors
        if X_TrainArgs.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = X_TrainArgs.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)

        if X_TrainArgs.bond_descriptors == 'descriptor':
            self.bond_descriptors_size = X_TrainArgs.bond_descriptors_size
            self.bond_descriptors_layer = nn.Linear(self.hidden_size + self.bond_descriptors_size,
                                                    self.hidden_size + self.bond_descriptors_size,)


    ###################################################################################################################
    #   __               .dMMM                                                        `7MM                            #
    #   MM               dM`                                                            MM                            #
    #   MM   `MM.       mMMmm  ,pW"Wq. `7Mb,od8 `7M'    ,A    `MF',6"Yb. `7Mb,od8  ,M""bMM                            #
    #   MM     `Mb.      MM   6W'   `Wb  MM' "'   VA   ,VAA   ,V 8)   MM   MM' "',AP    MM                            #
    #   MMMMMMMMMMMMD    MM   8M     M8  MM        VA ,V  VA ,V   ,pm9MM   MM    8MI    MM                            #
    #           ,M'      MM   YA.   ,A9  MM         VVV    VVV   8M   MM   MM    `Mb    MM                            #
    #         .M'      .JMML.  `Ybmd9' .JMML.        W      W    `Moo9^Yo.JMML.   `Wbmd"MML.                          #
    ###################################################################################################################
    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None,
                bond_descriptors_batch: List[np.ndarray] = None) -> torch.Tensor:

        # Encodes a batch of molecular graphs.
        # - mol_graph              : A  CLASS `BatchMolGraph` representing a batch of molecular graphs.
        # - atom_descriptors_batch : A list of numpy arrays containing additional atomic descriptors.
        # - bond_descriptors_batch : A list of numpy arrays containing additional bond descriptors
        # > return                 : A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.

        #====================================================================================================#
        # 
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch   # padding the first with 0 to match the atom_hiddens
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis = 0)).double().to(self.device)


        f_atoms    , \
        f_bonds    , \
        a2b        , \
        b2a        , \
        b2revb     , \
        a_scope    , \
        b_scope    , \
            = mol_graph.get_components(atom_messages = self.atom_messages)


        f_atoms  =  f_atoms.double().to(self.device)
        f_bonds  =  f_bonds.double().to(self.device)
        a2b      =  a2b.to(self.device)
        b2a      =  b2a.to(self.device)
        b2revb   =  b2revb.to(self.device)


        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)


        #====================================================================================================#
        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)                                                   # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)                                                   # num_bonds x hidden_size

        message = self.act_func(input)                                                  # num_bonds x hidden_size

        #====================================================================================================#
        # Message passing
        for depth in range(self.depth - 1):

            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)                           # num_atoms x max_num_bonds x hidden
                nei_f_bonds   = index_select_ND(f_bonds, a2b)                           # num_atoms x max_num_bonds x bond_fdim
                nei_message   = torch.cat((nei_a_message, nei_f_bonds), dim = 2)        # num_atoms x max_num_bonds x hidden + bond_fdim
                message       = nei_message.sum(dim = 1)                                # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)                           # num_atoms x max_num_bonds x hidden
                a_message     = nei_a_message.sum(dim=1)                                # num_atoms x hidden
                rev_message   = message[b2revb]                                         # num_bonds x hidden
                message       = a_message[b2a] - rev_message                            # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)                                    # num_bonds x hidden_size
            message = self.dropout_layer(message)                                       # num_bonds x hidden


        #====================================================================================================#
        # atom hidden
        a2x           = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)                                   # num_atoms x max_num_bonds x hidden
        a_message     = nei_a_message.sum(dim=1)                                        # num_atoms x hidden
        a_input       = torch.cat([f_atoms, a_message], dim=1)                          # num_atoms x (atom_fdim + hidden)
        atom_hiddens  = self.act_func(self.W_o(a_input))                                # num_atoms x hidden
        atom_hiddens  = self.dropout_layer(atom_hiddens)                                # num_atoms x hidden


        #====================================================================================================#
        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(f'The number of atoms is different from the length of the extra atom features')

            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)     # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)                    # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.dropout_layer(atom_hiddens)                             # num_atoms x (hidden + descriptor size)


        #====================================================================================================#
        # Readout
        mol_vecs = []
        
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs                                                                 # num_molecules x hidden













#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                                                             
# `7MM"""Mq. `YMM'   `MP'`7MN.   `7MF'      .g8"""bgd `7MMM.     ,MMF'`7MM"""Mq.`7MM"""Yb.       `7MMM.     ,MMF'              `7MM         `7MM       #
#   MM   `MM.  VMb.  ,P    MMN.    M      .dP'     `M   MMMb    dPMM    MM   `MM. MM    `Yb.       MMMb    dPMM                  MM           MM       #
#   MM   ,M9    `MM.M'     M YMb   M      dM'       `   M YM   ,M MM    MM   ,M9  MM     `Mb       M YM   ,M MM  ,pW"Wq.    ,M""bMM  .gP"Ya   MM       #
#   MMmmdM9       MMb      M  `MN. M      MM            M  Mb  M' MM    MMmmdM9   MM      MM       M  Mb  M' MM 6W'   `Wb ,AP    MM ,M'   Yb  MM       #
#   MM  YM.     ,M'`Mb.    M   `MM.M      MM.           M  YM.P'  MM    MM        MM     ,MP       M  YM.P'  MM 8M     M8 8MI    MM 8M""""""  MM       #
#   MM   `Mb.  ,P   `MM.   M     YMM      `Mb.     ,'   M  `YM'   MM    MM        MM    ,dP'       M  `YM'   MM YA.   ,A9 `Mb    MM YM.    ,  MM       #
# .JMML. .JMM.MM:.  .:MMa.JML.    YM        `"bmmmd'  .JML. `'  .JMML..JMML.    .JMMmmmdP'       .JML. `'  .JMML.`Ybmd9'   `Wbmd"MML.`Mbmmd'.JMML.     #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# 

class RXN_CMPD_MPNN(nn.Module): # Modified from CLASS `MPN` from `chemprop` package.

    # RXN_CMPD_MPNN calls RXN_CMPD_Encoder to learn molecule representations and send to the main model.
    def __init__(self                                , 
                 X_TrainArgs : TrainArgs             , 
                 atom_fdim   : int         = None    , 
                 bond_fdim   : int         = None    , 
                 PARAMS      : Feat_params = None    , ) :
        
        # - X_TrainArgs : A  CLASS `TrainArgs` object containing model arguments.
        # - atom_fdim   : Atom feature vector dimension.
        # - bond_fdim   : Bond feature vector dimension.

        super(RXN_CMPD_MPNN, self).__init__()

        #====================================================================================================#
        # 
        self.PARAMS           = PARAMS
        self.X_TrainArgs      = X_TrainArgs
        self.reaction         = X_TrainArgs.reaction
        self.reaction_solvent = X_TrainArgs.reaction_solvent

        self.atom_fdim = atom_fdim or get_atom_fdim(overwrite_default_atom = X_TrainArgs.overwrite_default_atom_features , 
                                                    is_reaction            = (self.reaction or self.reaction_solvent)    , 
                                                    PARAMS                 = self.PARAMS                                 , 
                                                    X_TrainArgs            = self.X_TrainArgs                            , 
                                                    )


        self.bond_fdim = bond_fdim or get_bond_fdim(atom_messages           = self.X_TrainArgs.atom_messages              , 
                                                    overwrite_default_bond  = X_TrainArgs.overwrite_default_bond_features , 
                                                    overwrite_default_atom  = X_TrainArgs.overwrite_default_atom_features , 
                                                    is_reaction             = (self.reaction or self.reaction_solvent)    , 
                                                    PARAMS                  = self.PARAMS                                 , 
                                                    X_TrainArgs             = self.X_TrainArgs                            , 
                                                    )

        print("self.atom_fdim: ", self.atom_fdim)
        print("self.bond_fdim: ", self.bond_fdim)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # 
        self.device                          = X_TrainArgs.device

        self.features_only                   = X_TrainArgs.features_only
        self.use_input_features              = X_TrainArgs.use_input_features

        self.atom_descriptors                = X_TrainArgs.atom_descriptors
        self.bond_descriptors                = X_TrainArgs.bond_descriptors
        self.overwrite_default_atom_features = X_TrainArgs.overwrite_default_atom_features
        self.overwrite_default_bond_features = X_TrainArgs.overwrite_default_bond_features


        #====================================================================================================#
        #                                     mm                          .dMMM                mm            #
        #      `MM.                           MM                          dM`                  MM            #
        #        `Mb.     .gP"Ya `7M'   `MF'mmMMmm `7Mb,od8 ,6"Yb.       mMMmm.gP"Ya   ,6"Yb.mmMMmm          #
        # MMMMMMMMMMMMD  ,M'   Yb  `VA ,V'    MM     MM' "'8)   MM        MM ,M'   Yb 8)   MM  MM            #
        #         ,M'    8M""""""    XMX      MM     MM     ,pm9MM        MM 8M""""""  ,pm9MM  MM            #
        #       .M'      YM.    ,  ,V' VA.    MM     MM    8M   MM        MM YM.    , 8M   MM  MM            #
        #                 `Mbmmd'.AM.   .MA.  `Mbmo.JMML.  `Moo9^Yo.    .JMML.`Mbmmd' `Moo9^Yo.`Mbmo         #
        #====================================================================================================#
        # Use additional (extra) molecule-level features only.
        if self.features_only: # Use only the additional features in an FFN, no graph network.
            return


        #====================================================================================================#
        #                                                      ,M'                          `7MM             #
        #      `MM.                                            MV                             MM             #
        #        `Mb.    `7Mb,od8 `7M'   `MF'`7MMpMMMb.       AW   `7MMpMMMb.pMMMb.  ,pW"Wq.  MM  .gP"Ya     #
        # MMMMMMMMMMMMD    MM' "'   `VA ,V'    MM    MM      ,M'     MM    MM    MM 6W'   `Wb MM ,M'   Yb    #
        #         ,M'      MM         XMX      MM    MM      MV      MM    MM    MM 8M     M8 MM 8M""""""    #
        #       .M'        MM       ,V' VA.    MM    MM     AW       MM    MM    MM YA.   ,A9 MM YM.    ,    #
        #                .JMML.   .AM.   .MA..JMML  JMML.  ,M'     .JMML  JMML  JMML.`Ybmd9'.JMML.`Mbmmd'    #
        #====================================================================================================#
        # Reaction OR Compound Inputs
        if not self.reaction_solvent:
            if X_TrainArgs.mpn_shared:
                self.RXN_CMPD_Model = nn.ModuleList( [ RXN_CMPD_Encoder(X_TrainArgs    , 
                                                                 self.atom_fdim , 
                                                                 self.bond_fdim , ) , ] * X_TrainArgs.number_of_molecules )
            else:
                self.RXN_CMPD_Model = nn.ModuleList( [ RXN_CMPD_Encoder(X_TrainArgs    , 
                                                                 self.atom_fdim , 
                                                                 self.bond_fdim , ) for _ in range(X_TrainArgs.number_of_molecules) ] )
            
            return

        #====================================================================================================#
        #                                     ,M'                 `7MM                               mm      #
        #       `MM.                          MV                    MM                               MM      #
        #         `Mb.   `7M'    ,A    `MF'  AW   ,pP"Ybd  ,pW"Wq.  MM `7M'   `MF'.gP"Ya `7MMpMMMb.mmMMmm    #
        #  MMMMMMMMMMMMD   VA   ,VAA   ,V   ,M'   8I   `" 6W'   `Wb MM   VA   ,V ,M'   Yb  MM    MM  MM      #
        #          ,M'      VA ,V  VA ,V    MV    `YMMMa. 8M     M8 MM    VA ,V  8M""""""  MM    MM  MM      #
        #        .M'         VVV    VVV    AW     L.   I8 YA.   ,A9 MM     VVV   YM.    ,  MM    MM  MM      #
        #                     W      W    ,M'     M9mmmP'  `Ybmd9'.JMML.    W     `Mbmmd'.JMML  JMML.`Mbmo   #
        #====================================================================================================#
        # Reaction AND Solvent Inputs
        if self.reaction_solvent:

            self.RXN_CMPD_Model = RXN_CMPD_Encoder(X_TrainArgs, self.atom_fdim, self.bond_fdim)


            # Set separate atom_fdim and bond_fdim for solvent molecules
            self.atom_fdim_solvent = get_atom_fdim(overwrite_default_atom = X_TrainArgs.overwrite_default_atom_features ,
                                                   is_reaction            = False                                       , 
                                                   PARAMS                 = PARAMS                                      ,
                                                   X_TrainArgs            = X_TrainArgs                                 , )
            

            self.bond_fdim_solvent = get_bond_fdim(overwrite_default_atom  = X_TrainArgs.overwrite_default_atom_features ,
                                                   overwrite_default_bond  = X_TrainArgs.overwrite_default_bond_features ,
                                                   atom_messages           = X_TrainArgs.atom_messages                   ,
                                                   is_reaction             = False                                       , 
                                                   PARAMS                  = PARAMS                                      ,
                                                   X_TrainArgs             = X_TrainArgs                                 , )
            
            self.RXN_CMPD_Model_solvent = RXN_CMPD_Encoder(X_TrainArgs                     , 
                                                           self.atom_fdim_solvent          , 
                                                           self.bond_fdim_solvent          ,
                                                           X_TrainArgs.hidden_size_solvent , 
                                                           X_TrainArgs.bias_solvent        , 
                                                           X_TrainArgs.depth_solvent       , )
            
            return

    ###################################################################################################################
    #   __               .dMMM                                                        `7MM                            #
    #   MM               dM`                                                            MM                            #
    #   MM   `MM.       mMMmm  ,pW"Wq. `7Mb,od8 `7M'    ,A    `MF',6"Yb. `7Mb,od8  ,M""bMM                            #
    #   MM     `Mb.      MM   6W'   `Wb  MM' "'   VA   ,VAA   ,V 8)   MM   MM' "',AP    MM                            #
    #   MMMMMMMMMMMMD    MM   8M     M8  MM        VA ,V  VA ,V   ,pm9MM   MM    8MI    MM                            #
    #           ,M'      MM   YA.   ,A9  MM         VVV    VVV   8M   MM   MM    `Mb    MM                            #
    #         .M'      .JMML.  `Ybmd9' .JMML.        W      W    `Moo9^Yo.JMML.   `Wbmd"MML.                          #
    ###################################################################################################################
    def forward(self,
                BatchMolGraph_list        :  List[BatchMolGraph]           ,
                additional_features_list  :  List[np.ndarray]     = None   ,
                atom_descriptors_batch    :  List[np.ndarray]     = None   ,
                atom_features_batch       :  List[np.ndarray]     = None   ,
                bond_descriptors_batch    :  List[np.ndarray]     = None   ,
                bond_features_batch       :  List[np.ndarray]     = None   , ) -> torch.FloatTensor:

        # Encodes a batch of molecules.

        # BatchMolGraph_list: A list of list of SMILES, a list of list of RDKit molecules, or a list of  CLASS `BatchMolGraph`.
                                  # The outer list or BatchMolGraph is of length `num_molecules` (number of datapoints in batch).
                                  # the inner list is of length `number_of_molecules` (number of molecules per datapoint).

        # - additional_features_list : A list of numpy arrays containing additional features (molecule-level).
        # - atom_descriptors_batch   : A list of numpy arrays containing additional atom descriptors.
        # - atom_features_batch      : A list of numpy arrays containing additional atom features.
        # - bond_features_batch      : A list of numpy arrays containing additional bond features.
        # > return                   : A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.


        #====================================================================================================#
        #                                     mm                          .dMMM                mm            #
        #      `MM.                           MM                          dM`                  MM            #
        #        `Mb.     .gP"Ya `7M'   `MF'mmMMmm `7Mb,od8 ,6"Yb.       mMMmm.gP"Ya   ,6"Yb.mmMMmm          #
        # MMMMMMMMMMMMD  ,M'   Yb  `VA ,V'    MM     MM' "'8)   MM        MM ,M'   Yb 8)   MM  MM            #
        #         ,M'    8M""""""    XMX      MM     MM     ,pm9MM        MM 8M""""""  ,pm9MM  MM            #
        #       .M'      YM.    ,  ,V' VA.    MM     MM    8M   MM        MM YM.    , 8M   MM  MM            #
        #                 `Mbmmd'.AM.   .MA.  `Mbmo.JMML.  `Moo9^Yo.    .JMML.`Mbmmd' `Moo9^Yo.`Mbmo         #
        #====================================================================================================#
        # Use additional (extra) molecule-level features only.
        if self.use_input_features:
            additional_features_list = torch.from_numpy(np.stack(additional_features_list)).double().to(self.device)
            if self.features_only:
                return additional_features_list



        #====================================================================================================#
        #                                                      ,M'                          `7MM             #
        #      `MM.                                            MV                             MM             #
        #        `Mb.    `7Mb,od8 `7M'   `MF'`7MMpMMMb.       AW   `7MMpMMMb.pMMMb.  ,pW"Wq.  MM  .gP"Ya     #
        # MMMMMMMMMMMMD    MM' "'   `VA ,V'    MM    MM      ,M'     MM    MM    MM 6W'   `Wb MM ,M'   Yb    #
        #         ,M'      MM         XMX      MM    MM      MV      MM    MM    MM 8M     M8 MM 8M""""""    #
        #       .M'        MM       ,V' VA.    MM    MM     AW       MM    MM    MM YA.   ,A9 MM YM.    ,    #
        #                .JMML.   .AM.   .MA..JMML  JMML.  ,M'     .JMML  JMML  JMML.`Ybmd9'.JMML.`Mbmmd'    #
        #====================================================================================================#
        # Reaction OR Compound Inputs
        # 
        if self.atom_descriptors == 'descriptor' or self.bond_descriptors == 'descriptor':

            if len(BatchMolGraph_list) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with '
                                          'one molecule per input (i.e., number_of_molecules = 1).')
            
            if len(BatchMolGraph_list) == 1: # This can be seen as a check of not using reaction_solvent as inputs.
                rxn_cmpd_encodings = [RXN_CMPD_MODEL_x(BatchMolGraph_x, atom_descriptors_batch, bond_descriptors_batch) 
                                        for RXN_CMPD_MODEL_x, BatchMolGraph_x in zip(self.RXN_CMPD_Model, BatchMolGraph_list)]
                
        # 
        if self.atom_descriptors != 'descriptor' and self.bond_descriptors != 'descriptor':

            if not self.reaction_solvent:
                rxn_cmpd_encodings = [RXN_CMPD_MODEL_x(BatchMolGraph_x) 
                                      for RXN_CMPD_MODEL_x, BatchMolGraph_x in zip(self.RXN_CMPD_Model, BatchMolGraph_list)]
                

        #====================================================================================================#
        #                                     ,M'                 `7MM                               mm      #
        #       `MM.                          MV                    MM                               MM      #
        #         `Mb.   `7M'    ,A    `MF'  AW   ,pP"Ybd  ,pW"Wq.  MM `7M'   `MF'.gP"Ya `7MMpMMMb.mmMMmm    #
        #  MMMMMMMMMMMMD   VA   ,VAA   ,V   ,M'   8I   `" 6W'   `Wb MM   VA   ,V ,M'   Yb  MM    MM  MM      #
        #          ,M'      VA ,V  VA ,V    MV    `YMMMa. 8M     M8 MM    VA ,V  8M""""""  MM    MM  MM      #
        #        .M'         VVV    VVV    AW     L.   I8 YA.   ,A9 MM     VVV   YM.    ,  MM    MM  MM      #
        #                     W      W    ,M'     M9mmmP'  `Ybmd9'.JMML.    W     `Mbmmd'.JMML  JMML.`Mbmo   #
        #====================================================================================================#
        # Reaction AND Solvent Inputs
        if self.reaction_solvent and self.atom_descriptors != 'descriptor' and self.bond_descriptors != 'descriptor':
                rxn_cmpd_encodings = []
                for BatchMolGraph_x in BatchMolGraph_list:
                    if BatchMolGraph_x.is_reaction:
                        rxn_cmpd_encodings.append(self.RXN_CMPD_Model(BatchMolGraph_x))
                    else:
                        rxn_cmpd_encodings.append(self.RXN_CMPD_Model_solvent(BatchMolGraph_x))



        #====================================================================================================#
        # Prepare the output representation of the molecule(s)/reaction(s)/solvent(s).
        output = rxn_cmpd_encodings[0] if len(rxn_cmpd_encodings) == 1 else torch.cat(rxn_cmpd_encodings, dim=1)


        #====================================================================================================#
        # Deal with additional features (concatenating it to the processed rxn/cmpd representations).
        if self.use_input_features:
            if len(additional_features_list.shape) == 1:
                additional_features_list = additional_features_list.view(1, -1)

            output = torch.cat([output, additional_features_list], dim = 1)

        #====================================================================================================#
        return output
















#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#     MMP""MM""YMM   .g8""8q.   `7MM"""Mq.     `7MMM.     ,MMF'  .g8""8q.   `7MM"""Yb.   `7MM"""YMM  `7MMF'                                     M      #
#     P'   MM   `7 .dP'    `YM.   MM   `MM.      MMMb    dPMM  .dP'    `YM.   MM    `Yb.   MM    `7    MM                                       M      #
#          MM      dM'      `MM   MM   ,M9       M YM   ,M MM  dM'      `MM   MM     `Mb   MM   d      MM                                       M      #
#          MM      MM        MM   MMmmdM9        M  Mb  M' MM  MM        MM   MM      MM   MMmmMM      MM                                   `7M'M`MF'  #
#          MM      MM.      ,MP   MM             M  YM.P'  MM  MM.      ,MP   MM     ,MP   MM   Y  ,   MM      ,                              VAM,V    #
#          MM      `Mb.    ,dP'   MM             M  `YM'   MM  `Mb.    ,dP'   MM    ,dP'   MM     ,M   MM     ,M                               VVV     #
#        .JMML.      `"bmmd"'   .JMML.         .JML. `'  .JMML.  `"bmmd"'   .JMMmmmdP'   .JMMmmmmMMM .JMMmmmmMMM                                V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

class RXN_CP_MPNN_Model(nn.Module): # Modified from CLASS `MoleculeModel` from `chemprop` package.

    # A class RXN_CP_MPNN_Model is a model which contains: 
    #  -  a Message Passing Network for parsing molecule structures, 
    #  -  a feed-forward network for predicting interactions.
    
    def __init__(self, 
                 X_TrainArgs : TrainArgs             , 
                 PARAMS      : Feat_params  = None   ,
                 cmpd_dim    : int          = None   , 
                 last_hid    : int          = 1024   , 
                 dropout     : Double       = 0.1    , ):
        

        super(RXN_CP_MPNN_Model, self).__init__()
        self.PARAMS             = PARAMS
        self.X_TrainArgs        = X_TrainArgs

        #--------------------------------------------------#
        # Get training essentials.
        self.device             = X_TrainArgs.device
        self.loss_function      = X_TrainArgs.loss_function

        #--------------------------------------------------#
        # Classification parameters for training.
        self.multiclass         = X_TrainArgs.dataset_type == 'multiclass'
        self.classification     = X_TrainArgs.dataset_type == 'classification'
        
        # Classification parameters for neural network.
        self.output_size        = X_TrainArgs.num_tasks
        self.output_size       *= X_TrainArgs.multiclass_num_classes if self.multiclass else 1
        self.sigmoid            = nn.Sigmoid() if self.classification else None
        self.multiclass_softmax = nn.Softmax(dim=2) if self.multiclass else None

        #--------------------------------------------------#
        # Other parameters for neural networks.
        self.output_size        = X_TrainArgs.num_tasks

        #--------------------------------------------------#
        # Select cmpd enc to use and adjust cmpd_dim accordingly.
        self.get_rxn_cmpd_encodings(X_TrainArgs, PARAMS)


        # cmpd_encodings_dim:
        if X_TrainArgs.features_only:
            #cmpd_dim = X_TrainArgs.features_size #?
            cmpd_dim = cmpd_dim

        if not X_TrainArgs.features_only:
            mpn_enc_dim = X_TrainArgs.hidden_size * X_TrainArgs.number_of_molecules
            cmpd_dim = mpn_enc_dim + cmpd_dim if X_TrainArgs.use_input_features else mpn_enc_dim

        print("cmpd_dim: ", cmpd_dim)

        #====================================================================================================#
        # Top Model Layers

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # FFN Layers
        self.fc_1 = nn.Linear(int(cmpd_dim), last_hid)
        self.fc_2 = nn.Linear(last_hid, last_hid)
        self.fc_3 = nn.Linear(last_hid, 1)
        self.cls = nn.Sigmoid()
        #====================================================================================================#





    def get_rxn_cmpd_encodings(self, 
                               X_TrainArgs : TrainArgs   = None ,
                               PARAMS      : Feat_params = None , ) -> None:
        #--------------------------------------------------#
        # Creates the A Encoder for getting molecule encodings.
        self.RXN_CMPD_MPNN_Encodings = RXN_CMPD_MPNN(X_TrainArgs = self.X_TrainArgs  ,
                                                     atom_fdim   = None              ,
                                                     bond_fdim   = None              ,
                                                     PARAMS      = self.PARAMS       ,)

        #--------------------------------------------------#
        # Load pre-trained parameters to the model.
        if X_TrainArgs.checkpoint_frzn is not None:
            #------------------------------
            if X_TrainArgs.freeze_first_only:
            # Freeze only the first encoder.
                for param in list(self.RXN_CMPD_MPNN_Encodings.RXN_CMPD_Model.children())[0].parameters():
                    param.requires_grad = False
            #------------------------------
            else: 
            # Freeze all encoders.
                for param in self.RXN_CMPD_MPNN_Encodings.parameters():
                    param.requires_grad = False





    def forward(self, 
                cmpd_dataset    : MoleculeDataset ,
                ) -> torch.FloatTensor:

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # features_batch         : List[np.ndarray] = None, A list of numpy arrays containing additional features.
        # atom_descriptors_batch : List[np.ndarray] = None, A list of numpy arrays containing additional atom descriptors.
        # atom_features_batch    : List[np.ndarray] = None, A list of numpy arrays containing additional atom features.
        # bond_features_batch    : List[np.ndarray] = None, A list of numpy arrays containing additional bond features.
        # return                 : The output contains a list of property predictions
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Data Final Preparations:
        #------------------------------
        # seqs_embeddings shall be sent to cuda device now.
        # ...
        #------------------------------
        # cmpd_encodings shall be converted to MolGraph List.
        BatchMolGraph_list            = cmpd_dataset.batch_graph()        #-> List[BatchMolGraph]
        additional_features_list      = cmpd_dataset.features()           #-> List[np.ndarray]
        cmpd_atom_descriptors_list    = cmpd_dataset.atom_descriptors()   #-> List[np.ndarray]
        cmpd_bond_descriptors_list    = cmpd_dataset.bond_descriptors()   #-> List[np.ndarray]
        cmpd_atom_features_list       = cmpd_dataset.atom_features()      #-> List[np.ndarray]
        cmpd_bond_features_list       = cmpd_dataset.bond_features()      #-> List[np.ndarray]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # RXN/CMPD Encodings.

        # BatchMolGraph_list        
        # additional_features_list  
        # atom_descriptors_batch    
        # atom_features_batch       
        # bond_descriptors_batch    
        # bond_features_batch       


        cmpd_encodings = self.RXN_CMPD_MPNN_Encodings(BatchMolGraph_list         , 
                                                      additional_features_list   , 
                                                      cmpd_atom_descriptors_list ,
                                                      cmpd_atom_features_list    , 
                                                      cmpd_bond_descriptors_list ,
                                                      cmpd_bond_features_list    , )

        output = cmpd_encodings
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # FFN Layers
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        output = nn.functional.relu(output)
        output = self.fc_3(output)

        return output, cmpd_encodings




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

if __name__ == "__main__":

    Step_code = "rxn05A_"
    print("*" * 50)
    print(Step_code + " Done!") 






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








