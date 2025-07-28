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
import matplotlib.pyplot as plt

#--------------------------------------------------#
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

#--------------------------------------------------#
from itertools import zip_longest
from collections import OrderedDict

#--------------------------------------------------#
import threading

#--------------------------------------------------#
from typing import Any, Dict, List, Tuple, Union, Sequence, Optional, Collection, Iterator

#--------------------------------------------------#
from AP_convert import MolFromSmiles_ZX
from AP_convert import Get_Unique_SMILES

from ZX02_nn_utils       import StandardScaler
from ZX02_nn_utils       import build_optimizer, build_lr_scheduler
from ZX03_rxn_mpnn_args        import TrainArgs
from ZX04_funcs          import onek_encoding_unk
from ZX05_loss_functions import get_loss_func

#--------------------------------------------------#






###################################################################################################################
###################################################################################################################
# Molecule Settings

# Unique SMILES
GetUnqSmi  = Get_Unique_SMILES(isomericSmiles = True , kekuleSmiles = False, canonical = False, SMARTS_bool = False, removeAtomMapNumber = True)







#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   `7MMM.     ,MMF'       `7MM      `7MM"""YMM                 mm                         db                   mm     db                              #
#     MMMb    dPMM           MM        MM    `7                 MM                                              MM                                     #
#     M YM   ,M MM  ,pW"Wq.  MM        MM   d  .gP"Ya   ,6"Yb.mmMMmm `7MM  `7MM `7Mb,od8 `7MM  M"""MMV  ,6"Yb.mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.           #
#     M  Mb  M' MM 6W'   `Wb MM        MM""MM ,M'   Yb 8)   MM  MM     MM    MM   MM' "'   MM  '  AMV  8)   MM  MM     MM 6W'   `Wb MM    MM           #
#     M  YM.P'  MM 8M     M8 MM        MM   Y 8M""""""  ,pm9MM  MM     MM    MM   MM       MM    AMV    ,pm9MM  MM     MM 8M     M8 MM    MM           #
#     M  `YM'   MM YA.   ,A9 MM        MM     YM.    , 8M   MM  MM     MM    MM   MM       MM   AMV  , 8M   MM  MM     MM YA.   ,A9 MM    MM           #
#   .JML. `'  .JMML.`Ybmd9'.JMML.    .JMML.    `Mbmmd' `Moo9^Yo.`Mbmo  `Mbod"YML.JMML.   .JMML.AMMmmmM `Moo9^Yo.`Mbmo.JMML.`Ybmd9'.JMML  JMML.         #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Checked !
class Feat_params: 
    # A class holding molecule featurization parameters as attributes.

    def __init__(self) -> None:

        # Atom feature sizes
        self.MAX_ATOMIC_NUM = 100 # This shall NOT limit the molecule size. 
        self.ATOM_FEATURES = {  'atomic_num'    : list(range(self.MAX_ATOMIC_NUM)),
                                'degree'        : [0, 1, 2, 3, 4, 5 ],
                                'formal_charge' : [-1, -2, 1, 2, 0  ],
                                'chiral_tag'    : [0, 1, 2, 3       ],
                                'num_Hs'        : [0, 1, 2, 3, 4    ],
                                'hybridization' : [Chem.rdchem.HybridizationType.SP    ,
                                                   Chem.rdchem.HybridizationType.SP2   ,
                                                   Chem.rdchem.HybridizationType.SP3   ,
                                                   Chem.rdchem.HybridizationType.SP3D  ,
                                                   Chem.rdchem.HybridizationType.SP3D2 , ], }

        # Distance feature sizes
        self.PATH_DISTANCE_BINS    = list(range(10))
        self.THREE_D_DISTANCE_MAX  = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM       = sum(len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM       = 14
        self.EXTRA_BOND_FDIM = 0
        self.REACTION_MODE   = None
        self.EXPLICIT_H      = False
        self.REACTION        = False
        self.ADDING_H        = False
        self.KEEP_ATOM_MAP   = True

        #print("Feat_params -> self.ATOM_FEATURES : ", self.ATOM_FDIM)










#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# `7MM"""YMM                 mm                         db                   mm     db                       `7MM"""YMM                                #
#   MM    `7                 MM                                              MM                                MM    `7                                #
#   MM   d  .gP"Ya   ,6"Yb.mmMMmm `7MM  `7MM `7Mb,od8 `7MM  M"""MMV  ,6"Yb.mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.      MM   d `7MM  `7MM `7MMpMMMb.  ,p6"bo    #
#   MM""MM ,M'   Yb 8)   MM  MM     MM    MM   MM' "'   MM  '  AMV  8)   MM  MM     MM 6W'   `Wb MM    MM      MM""MM   MM    MM   MM    MM 6M'  OO    #
#   MM   Y 8M""""""  ,pm9MM  MM     MM    MM   MM       MM    AMV    ,pm9MM  MM     MM 8M     M8 MM    MM      MM   Y   MM    MM   MM    MM 8M         #
#   MM     YM.    , 8M   MM  MM     MM    MM   MM       MM   AMV  , 8M   MM  MM     MM YA.   ,A9 MM    MM      MM       MM    MM   MM    MM YM.    ,   #
# .JMML.    `Mbmmd' `Moo9^Yo.`Mbmo  `Mbod"YML.JMML.   .JMML.AMMmmmM `Moo9^Yo.`Mbmo.JMML.`Ybmd9'.JMML  JMML.  .JMML.     `Mbod"YML.JMML  JMML.YMbmd'    #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Checked !
def atom_features(atom: Chem.rdchem.Atom, 
                  functional_groups: List[int] = None, 
                  PARAMS: Feat_params = Feat_params()) -> List[Union[bool, int, float]]:

    # Builds a feature vector for an atom.
    # - atom              : An RDKit atom.
    # - functional_groups : A k-hot vector indicating the functional groups the atom belongs to.
    # > return            : A list containing the atom features.


    # Parameter object for reference throughout this module
    #PARAMS = Feat_params()

    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features =  onek_encoding_unk(atom.GetAtomicNum() - 1       ,     PARAMS.ATOM_FEATURES['atomic_num'])        + \
                    onek_encoding_unk(atom.GetTotalDegree()         ,     PARAMS.ATOM_FEATURES['degree'])            + \
                    onek_encoding_unk(atom.GetFormalCharge()        ,     PARAMS.ATOM_FEATURES['formal_charge'])     + \
                    onek_encoding_unk(int(atom.GetChiralTag())      ,     PARAMS.ATOM_FEATURES['chiral_tag'])        + \
                    onek_encoding_unk(int(atom.GetTotalNumHs())     ,     PARAMS.ATOM_FEATURES['num_Hs'])            + \
                    onek_encoding_unk(int(atom.GetHybridization())  ,     PARAMS.ATOM_FEATURES['hybridization'])     + \
                    [1 if atom.GetIsAromatic() else 0]                                                               + \
                    [atom.GetMass() * 0.01]  # scaled to about the same range as other features

        if functional_groups is not None: # Default to None.
            features += functional_groups

    return features

#====================================================================================================#
def bond_features(bond: Chem.rdchem.Bond,
                  PARAMS: Feat_params = Feat_params()) -> List[Union[bool, int, float]]:
    
    # Builds a feature vector for a bond.
    # - bond   : An RDKit bond.
    # > return : A list containing the bond features.


    # Parameter object for reference throughout this module
    #PARAMS = Feat_params()

    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1) 
    else:
        bt = bond.GetBondType()
        fbond = [0,  # bond is not None
                 bt == Chem.rdchem.BondType.SINGLE,
                 bt == Chem.rdchem.BondType.DOUBLE,
                 bt == Chem.rdchem.BondType.TRIPLE,
                 bt == Chem.rdchem.BondType.AROMATIC,
                (bond.GetIsConjugated() if bt is not None else 0),
                (bond.IsInRing() if bt is not None else 0)]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    return fbond

#====================================================================================================#
# get atom fearture dimension.
def get_atom_fdim(overwrite_default_atom : bool                     = False                      , 
                  is_reaction            : bool                     = False                      , 
                  PARAMS                 : Feat_params              = Feat_params()              ,
                  X_TrainArgs            : TrainArgs                = None                       ,
                  ) -> int:

    # Gets the dimensionality of the atom feature vector.
    # - overwrite_default_atom  : Whether to overwrite the default atom descriptors
    # - is_reaction             : Whether to add :code:'EXTRA_ATOM_FDIM' for reaction input when code 'REACTION_MODE' is not None
    # > return                  : The dimensionality of the atom feature vector.

    #PARAMS = Feat_params()

    if PARAMS.REACTION_MODE:
        return (not overwrite_default_atom) * PARAMS.ATOM_FDIM         + \
               (is_reaction               ) * PARAMS.EXTRA_ATOM_FDIM
    
    else:
        return (not overwrite_default_atom) * PARAMS.ATOM_FDIM         + \
               (1)                          * PARAMS.EXTRA_ATOM_FDIM

#====================================================================================================#
# get bond fearture dimension.
def get_bond_fdim(atom_messages          : bool                     = False                      ,
                  overwrite_default_bond : bool                     = False                      ,
                  overwrite_default_atom : bool                     = False                      ,
                  is_reaction            : bool                     = False                      , 
                  PARAMS                 : Feat_params              = Feat_params()              ,
                  X_TrainArgs            : TrainArgs                = None                       ,
                  ) -> int:
    
    # Gets the dimensionality of the bond feature vector.
    # - atom_messages          : Whether atom messages are being used. 
    #                               If atom messages are used, 
    #                               then the bond feature vector only contains bond features.
    #                               Otherwise it contains both atom and bond features.
    # - overwrite_default_bond : Whether to overwrite the default bond descriptors
    # - overwrite_default_atom : Whether to overwrite the default atom descriptors
    # - is_reaction            : Whether to add :code:'EXTRA_BOND_FDIM' for reaction input when :code:'REACTION_MODE:' is not None
    # return: The dimensionality of the bond feature vector.


    #PARAMS = Feat_params()


    '''
    return  X_TrainArgs.Extra_Bond_Feature_Dim               + \
            (not overwrite_default_bond) * PARAMS.BOND_FDIM  + \
            (not atom_messages) * get_atom_fdim(overwrite_default_atom = overwrite_default_atom     , 
                                                is_reaction            = is_reaction                ,
                                                PARAMS                 = PARAMS                     ,
                                                X_TrainArgs            = X_TrainArgs                , )
                                                '''
    

    print("PARAMS.REACTION_MODE               : " , PARAMS.REACTION_MODE                                                )
    print("overwrite_default_bond             : " , overwrite_default_bond                                              )
    print("PARAMS.BOND_FDIM                   : " , PARAMS.BOND_FDIM                                                    )
    print("is_reaction                        : " , is_reaction                                                         )
    print("X_TrainArgs.Extra_Bond_Feature_Dim : " , X_TrainArgs.Extra_Bond_Feature_Dim                                  )
    print("not atom_messages                  : " , not atom_messages                                                   )
    print("get_atom_fdim                      : " , get_atom_fdim(overwrite_default_atom = overwrite_default_atom , 
                                                                  is_reaction            = is_reaction            , 
                                                                  PARAMS                 = PARAMS                 , 
                                                                  X_TrainArgs            = X_TrainArgs            , )   )


    if PARAMS.REACTION_MODE:
        return (not overwrite_default_bond) * PARAMS.BOND_FDIM                                                  + \
               (is_reaction               ) * PARAMS.EXTRA_BOND_FDIM                                            + \
               (not atom_messages         ) * get_atom_fdim(overwrite_default_atom = overwrite_default_atom , 
                                                            is_reaction            = is_reaction            , 
                                                            PARAMS                 = PARAMS                 , 
                                                            X_TrainArgs            = X_TrainArgs            , )
    
    else:
        return (not overwrite_default_bond) * PARAMS.BOND_FDIM                                                  + \
               (1                         ) * PARAMS.EXTRA_BOND_FDIM                                            + \
               (not atom_messages         ) * get_atom_fdim(overwrite_default_atom = overwrite_default_atom , 
                                                            is_reaction            = is_reaction            , 
                                                            PARAMS                 = PARAMS                 , 
                                                            X_TrainArgs            = X_TrainArgs            , )




#====================================================================================================#
def set_extra_atom_fdim(extra, PARAMS: Feat_params = Feat_params()):
    # Change the dimensionality of the atom feature vector.
    PARAMS.EXTRA_ATOM_FDIM = extra
    return


#====================================================================================================#
def set_extra_bond_fdim(extra, PARAMS: Feat_params = Feat_params()):
    # Change the dimensionality of the bond feature vector.
    PARAMS.EXTRA_BOND_FDIM = extra
    return



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   `7MMM.     ,MMF'        `7MM                  `7MMM.     ,MMF'       `7MM                                                                          #
#     MMMb    dPMM            MM                    MMMb    dPMM           MM                                                                          #
#     M YM   ,M MM   ,6"Yb.   MM  ,MP'.gP"Ya        M YM   ,M MM  ,pW"Wq.  MM  ,pP"Ybd                                                                 #
#     M  Mb  M' MM  8)   MM   MM ;Y  ,M'   Yb       M  Mb  M' MM 6W'   `Wb MM  8I   `"                                                                 #
#     M  YM.P'  MM   ,pm9MM   MM;Mm  8M""""""       M  YM.P'  MM 8M     M8 MM  `YMMMa.                                                                 #
#     M  `YM'   MM  8M   MM   MM `Mb.YM.    ,       M  `YM'   MM YA.   ,A9 MM  L.   I8                                                                 #
#   .JML. `'  .JMML.`Moo9^Yo.JMML. YA.`Mbmmd'     .JML. `'  .JMML.`Ybmd9'.JMML.M9mmmP'                                                                 #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Functions for converting RXN/SMILES strings to Chem.Mol objects.
def make_mol(smiles_x       : str         , 
             keep_h         : bool        , 
             add_h          : bool        , 
             keep_atom_map  : bool        , 
             use_ZX_setting : bool = True , 
             isomericSmiles : bool = True , ) :

    # Builds an RDKit molecule from a SMILES string.
    
    # - smiles_x      : SMILES string.
    # - keep_h        : Boolean whether to keep hydrogens in the input smiles.
    #                      - This does not add hydrogens.
    #                      - It only keeps them if they are specified.
    # - add_h         : Boolean whether to add hydrogens to the input smiles.
    # - keep_atom_map : Boolean whether to keep the original atom mapping.
    # > return        : RDKit molecule.

    if not use_ZX_setting:
        params          = Chem.SmilesParserParams()
        params.removeHs = not keep_h if not keep_atom_map else False
        mol_x           = Chem.MolFromSmiles(smiles_x, params)

        if add_h:
            mol_x = Chem.AddHs(mol_x)

        if keep_atom_map and mol_x is not None:
            atom_map_numbers = tuple(atom.GetAtomMapNum() for atom in mol_x.GetAtoms())
            for idx, map_num in enumerate(atom_map_numbers):
                if idx + 1 != map_num:
                    new_order = np.argsort(atom_map_numbers).tolist()
                    return Chem.rdmolops.RenumberAtoms(mol_x, new_order)
        elif not keep_atom_map and mol_x is not None:
            for atom in mol_x.GetAtoms():
                atom.SetAtomMapNum(0)

    else:
        mol_x = Chem.MolFromSmiles(smiles_x)
        try: 
            Chem.MolToSmiles(mol_x, isomericSmiles = isomericSmiles)
        except:
            print ("!!!!! Problematic SMILES (MolToSmiles): ", smiles_x)

    return mol_x

#====================================================================================================#
def make_mols(input_str          : List[str ] , 
              reaction_list      : List[bool] , 
              keep_h_list        : List[bool] , 
              add_h_list         : List[bool] , 
              keep_atom_map_list : List[bool] , ):

    # Builds a list of RDKit molecules (or a list of tuples of molecules if reaction is True) for a list of smiles.

    # - input_str          : List of SMILES strings.
    # - reaction_list      : List of booleans whether the SMILES strings are to be treated as a reaction.
    # - keep_h_list        : List of booleans whether to keep hydrogens in the input smiles. 
    #                           - This does not add hydrogens.
    #                           - It only keeps them if they are specified.
    # - add_h_list         : List of booleasn whether to add hydrogens to the input smiles.
    # - keep_atom_map_list : List of booleasn whether to keep the original atom mapping.
    # > return             : List of RDKit molecules or list of tuple of molecules.

    print(input_str)

    mol = []

    for s, is_reaction, keep_h, add_h, keep_atom_map in zip(input_str, reaction_list, keep_h_list, add_h_list, keep_atom_map_list):

        if is_reaction:
            mol.append( 
                      ( 
                        make_mol(smiles_x       = s.split(">")[ 0] , 
                                 keep_h         = keep_h           , 
                                 add_h          = add_h            , 
                                 keep_atom_map  = keep_atom_map    , 
                                 use_ZX_setting = True             , 
                                 isomericSmiles = True             , ) ,

                        make_mol(smiles_x       = s.split(">")[-1] , 
                                 keep_h         = keep_h           , 
                                 add_h          = add_h            , 
                                 keep_atom_map  = keep_atom_map    , 
                                 use_ZX_setting = True             , 
                                 isomericSmiles = True             , ) ,
                      )
                      )
            
        else:
            mol.append(
                       make_mol(smiles_x       = s                , 
                                keep_h         = keep_h           , 
                                add_h          = add_h            , 
                                keep_atom_map  = keep_atom_map    , 
                                use_ZX_setting = True             , 
                                isomericSmiles = True             , ) ,
                      )
            
    return mol


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  MMP""MM""YMM       db              db          `7MM       `7MM"""YMM                               mm     db                                        #
#  P'   MM   `7                                     MM         MM    `7                               MM                                               #
#       MM `7Mb,od8 `7MM `7M'   `MF'`7MM   ,6"Yb.   MM         MM   d `7MM  `7MM `7MMpMMMb.  ,p6"bo mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.  ,pP"Ybd            #
#       MM   MM' "'   MM   VA   ,V    MM  8)   MM   MM         MM""MM   MM    MM   MM    MM 6M'  OO   MM     MM 6W'   `Wb MM    MM  8I   `"            #
#       MM   MM       MM    VA ,V     MM   ,pm9MM   MM         MM   Y   MM    MM   MM    MM 8M        MM     MM 8M     M8 MM    MM  `YMMMa.            #
#       MM   MM       MM     VVV      MM  8M   MM   MM         MM       MM    MM   MM    MM YM.    ,  MM     MM YA.   ,A9 MM    MM  L.   I8            #
#     .JMML.JMML.   .JMML.    W     .JMML.`Moo9^Yo.JMML.     .JMML.     `Mbod"YML.JMML  JMML.YMbmd'   `Mbmo.JMML.`Ybmd9'.JMML  JMML.M9mmmP'            #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

# Trivial Functions.
def is_mol(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]) -> bool:
    # Checks whether an input is a molecule (Chem.Mol / smiles).
    if isinstance(mol, str) and ">" not in mol:    # input is a smiles string
        return True
    elif isinstance(mol, Chem.Mol):                # input is a Chem.Mol
        return True
    else:                                          # input is not a Chem.Mol or a smiles string
        return False


def is_reaction(is_mol: bool = True, PARAMS: Feat_params = Feat_params()) -> bool:
    # Returns whether to use reactions as input.
    if is_mol:
        return False
    if PARAMS.REACTION:
        return True
    return False

def is_explicit_h(is_mol: bool = True, PARAMS: Feat_params = Feat_params()) -> bool:
    # Returns whether to retain explicit Hs (for reactions only).
    if not is_mol:
        return PARAMS.EXPLICIT_H
    return False

def is_adding_hs(is_mol: bool = True, PARAMS: Feat_params = Feat_params()) -> bool:
    # Returns whether to add explicit Hs to the mol (not for reactions).
    if is_mol:
        return PARAMS.ADDING_H
    return False

def is_keeping_atom_map(is_mol: bool = True, PARAMS: Feat_params = Feat_params()) -> bool:
    # Returns whether to keep the original atom mapping (not for reactions).
    if is_mol:
        return PARAMS.KEEP_ATOM_MAP
    return False










#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   `7MMM.     ,MMF'       `7MM         .g8"""bgd                          `7MM             `7MM"""YMM                                                 #
#     MMMb    dPMM           MM       .dP'     `M                            MM               MM    `7                                                 #
#     M YM   ,M MM  ,pW"Wq.  MM       dM'       ` `7Mb,od8 ,6"Yb. `7MMpdMAo. MMpMMMb.         MM   d `7MM  `7MM `7MMpMMMb.  ,p6"bo                     #
#     M  Mb  M' MM 6W'   `Wb MM       MM            MM' "'8)   MM   MM   `Wb MM    MM         MM""MM   MM    MM   MM    MM 6M'  OO                     #
#     M  YM.P'  MM 8M     M8 MM       MM.    `7MMF' MM     ,pm9MM   MM    M8 MM    MM         MM   Y   MM    MM   MM    MM 8M                          #
#     M  `YM'   MM YA.   ,A9 MM       `Mb.     MM   MM    8M   MM   MM   ,AP MM    MM         MM       MM    MM   MM    MM YM.    ,                    #
#   .JML. `'  .JMML.`Ybmd9'.JMML.       `"bmmmdPY .JMML.  `Moo9^Yo. MMbmmd'.JMML  JMML.     .JMML.     `Mbod"YML.JMML  JMML.YMbmd'                     #
#                                                                   MM                                                                                 #
#                                                                 .JMML.                                                                               #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#



def map_reac_to_prod(mol_reac: Chem.Mol, mol_prod: Chem.Mol):
    """
    Build a dictionary of mapping atom indices in the reactants to the products.

    :param mol_reac: An RDKit molecule of the reactants.
    :param mol_prod: An RDKit molecule of the products.
    :return: A dictionary of corresponding reactant and product atom indices.
    """
    only_prod_ids = []
    prod_map_to_id = {}
    mapnos_reac = set([atom.GetAtomMapNum() for atom in mol_reac.GetAtoms()]) 
    for atom in mol_prod.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            prod_map_to_id[mapno] = atom.GetIdx()
            if mapno not in mapnos_reac:
                only_prod_ids.append(atom.GetIdx()) 
        else:
            only_prod_ids.append(atom.GetIdx())
    only_reac_ids = []
    reac_id_to_prod_id = {}
    for atom in mol_reac.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            try:
                reac_id_to_prod_id[atom.GetIdx()] = prod_map_to_id[mapno]
            except KeyError:
                only_reac_ids.append(atom.GetIdx())
        else:
            only_reac_ids.append(atom.GetIdx())
    return reac_id_to_prod_id, only_prod_ids, only_reac_ids




def atom_features_zeros(atom: Chem.rdchem.Atom, PARAMS: Feat_params, ) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom containing only the atom number information.

    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            [0] * (PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1) #set other features to zero
    return features






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    `7MMM.     ,MMF'       `7MM        .g8"""bgd                          `7MM                                                                        #
#      MMMb    dPMM           MM      .dP'     `M                            MM                                                                        #
#      M YM   ,M MM  ,pW"Wq.  MM      dM'       ` `7Mb,od8 ,6"Yb. `7MMpdMAo. MMpMMMb.                                                                  #
#      M  Mb  M' MM 6W'   `Wb MM      MM            MM' "'8)   MM   MM   `Wb MM    MM                                                                  #
#      M  YM.P'  MM 8M     M8 MM      MM.    `7MMF' MM     ,pm9MM   MM    M8 MM    MM                                                                  #
#      M  `YM'   MM YA.   ,A9 MM      `Mb.     MM   MM    8M   MM   MM   ,AP MM    MM                                                                  #
#    .JML. `'  .JMML.`Ybmd9'.JMML.      `"bmmmdPY .JMML.  `Moo9^Yo. MMbmmd'.JMML  JMML.                                                                #
#                                                                   MM                                                                                 #
#                                                                 .JMML.                                                                               #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class MolGraph:

    # Class 'MolGraph' represents the graph structure and featurization of a single molecule.
    # A MolGraph computes the following attributes:

    # vars 'n_atoms'       : The number of atoms in the molecule.
    # vars 'n_bonds'       : The number of bonds in the molecule.
    # vars 'f_atoms'       : A mapping from an atom index to a list of atom features.
    # vars 'f_bonds'       : A mapping from a bond index to a list of bond features.
    # vars 'a2b'           : A mapping from an atom index to a list of incoming bond indices.
    # vars 'b2a'           : A mapping from a bond index to the index of the atom the bond originates from.
    # vars 'b2revb'        : A mapping from a bond index to the index of the reverse bond.
    # vars 'is_mol'        : A boolean whether the input is a molecule.
    # vars 'is_reaction'   : A boolean whether the molecule is a reaction.
    # vars 'is_explicit_h' : A boolean whether to retain explicit Hs (for reaction mode)
    # vars 'is_adding_hs'  : A boolean whether to add explicit Hs (not for reaction mode)
    # vars 'overwrite_default_atom_features': A boolean to overwrite default atom descriptors.
    # vars 'overwrite_default_bond_features': A boolean to overwrite default bond descriptors.


    def __init__(self, 
                 one_mol                         : Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]       ,
                 one_GEN_SMILES                  : str             = None                                ,
                 atom_features_extra             : np.ndarray      = None                                ,
                 bond_features_extra             : np.ndarray      = None                                ,
                 overwrite_default_atom_features : bool            = False                               ,
                 overwrite_default_bond_features : bool            = False                               ,
                 X_extra_cmpd_features           : Dict            = None                                ,
                 X_extra_atom_bond_features      : Dict            = None                                ,
                 X_TrainArgs                     : TrainArgs       = None                                , 
                 PARAMS                          : Feat_params     = Feat_params()                       ,
                 ):

        # mol                             :  A SMILES or an RDKit molecule.
        # atom_features_extra             :  A list of 2D numpy array containing additional atom features to featurize the molecule
        # bond_features_extra             :  A list of 2D numpy array containing additional bond features to featurize the molecule
        # overwrite_default_atom_features :  Boolean to overwrite default atom features by atom_features instead of concatenating
        # overwrite_default_bond_features :  Boolean to overwrite default bond features by bond_features instead of concatenating

        self.is_mol        =  is_mol(one_mol)

        self.is_reaction   =  is_reaction   (self.is_mol, PARAMS)
        self.is_explicit_h =  is_explicit_h (self.is_mol, PARAMS)
        self.is_adding_hs  =  is_adding_hs  (self.is_mol, PARAMS)

        self.reaction_mode =  X_TrainArgs.reaction_mode
        print("X_TrainArgs.reaction_mode: ", X_TrainArgs.reaction_mode)


        #====================================================================================================#
        # Convert SMILES to RDKit molecule if necessary, this is always TURE here to ensure SMILES are sent to here. 

        mol_x = one_mol


        self.n_atoms = 0   # number of atoms
        self.n_bonds = 0   # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b     = []  # mapping from atom index to incoming bond indices                           , old name: " a 2 b "
        self.b2a     = []  # mapping from bond index to the index of the atom the bond is coming from   , old name: " b 2 a "
        self.b2revb  = []  # mapping from bond index to the index of the reverse bond                   , old name: " b 2 revb "
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features
        self.X_TrainArgs                     = X_TrainArgs
        self.PARAMS                          = PARAMS

        #====================================================================================================#
        # Input is a reaction.
        if self.is_reaction:
            if atom_features_extra is not None:
                raise NotImplementedError('Extra atom features are currently not supported for reactions')
            if bond_features_extra is not None:
                raise NotImplementedError('Extra bond features are currently not supported for reactions')

            mol_reac = mol_x[0]
            mol_prod = mol_x[1]
            ri2pi, pio, rio = map_reac_to_prod(mol_reac, mol_prod)
           
            # Get atom features
            if self.reaction_mode in ['reac_diff','prod_diff', 'reac_prod']:
                #Reactant: regular atom features for each atom in the reactants, as well as zero features for atoms that are only in the products (indices in pio)
                f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [atom_features_zeros(mol_prod.GetAtomWithIdx(index), PARAMS) for index in pio]
                
                #Product: regular atom features for each atom that is in both reactants and products (not in rio), other atom features zero,
                #regular features for atoms that are only in the products (indices in pio)
                f_atoms_prod = [atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])) if atom.GetIdx() not in rio else
                                atom_features_zeros(atom, PARAMS) for atom in mol_reac.GetAtoms()] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
            else: #balance
                #Reactant: regular atom features for each atom in the reactants, copy features from product side for atoms that are only in the products (indices in pio)
                f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
                
                #Product: regular atom features for each atom that is in both reactants and products (not in rio), copy features from reactant side for
                #other atoms, regular features for atoms that are only in the products (indices in pio)
                f_atoms_prod = [atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])) if atom.GetIdx() not in rio else
                                atom_features(atom) for atom in mol_reac.GetAtoms()] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]

            if self.reaction_mode in ['reac_diff', 'prod_diff', 'reac_diff_balance', 'prod_diff_balance']:
                f_atoms_diff = [list(map(lambda x, y: x - y, ii, jj)) for ii, jj in zip(f_atoms_prod, f_atoms_reac)]
            if self.reaction_mode in ['reac_prod', 'reac_prod_balance']:
                self.f_atoms = [x+y[PARAMS.MAX_ATOMIC_NUM+1:] for x,y in zip(f_atoms_reac, f_atoms_prod)]
            elif self.reaction_mode in ['reac_diff', 'reac_diff_balance']:
                self.f_atoms = [x+y[PARAMS.MAX_ATOMIC_NUM+1:] for x,y in zip(f_atoms_reac, f_atoms_diff)]
            elif self.reaction_mode in ['prod_diff', 'prod_diff_balance']:
                self.f_atoms = [x+y[PARAMS.MAX_ATOMIC_NUM+1:] for x,y in zip(f_atoms_prod, f_atoms_diff)]
            self.n_atoms = len(self.f_atoms)
            n_atoms_reac = mol_reac.GetNumAtoms()

            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    if a1 >= n_atoms_reac and a2 >= n_atoms_reac: # Both atoms only in product
                        bond_prod = mol_prod.GetBondBetweenAtoms(pio[a1 - n_atoms_reac], pio[a2 - n_atoms_reac])
                        if self.reaction_mode in ['reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance']:
                            bond_reac = bond_prod
                        else:
                            bond_reac = None
                    elif a1 < n_atoms_reac and a2 >= n_atoms_reac: # One atom only in product
                        bond_reac = None
                        if a1 in ri2pi.keys():
                            bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[a1], pio[a2 - n_atoms_reac])
                        else:
                            bond_prod = None # Atom atom only in reactant, the other only in product
                    else:
                        bond_reac = mol_reac.GetBondBetweenAtoms(a1, a2)
                        if a1 in ri2pi.keys() and a2 in ri2pi.keys():
                            bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[a1], ri2pi[a2]) #Both atoms in both reactant and product
                        else:
                            if self.reaction_mode in ['reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance']:
                                if a1 in ri2pi.keys() or a2 in ri2pi.keys():
                                    bond_prod = None # One atom only in reactant
                                else:
                                    bond_prod = bond_reac # Both atoms only in reactant
                            else:    
                                bond_prod = None # One or both atoms only in reactant

                    if bond_reac is None and bond_prod is None:
                        continue

                    f_bond_reac = bond_features(bond_reac)
                    f_bond_prod = bond_features(bond_prod)
                    if self.reaction_mode in ['reac_diff', 'prod_diff', 'reac_diff_balance', 'prod_diff_balance']:
                        f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
                    if self.reaction_mode in ['reac_prod', 'reac_prod_balance']:
                        f_bond = f_bond_reac + f_bond_prod
                    elif self.reaction_mode in ['reac_diff', 'reac_diff_balance']:
                        f_bond = f_bond_reac + f_bond_diff
                    elif self.reaction_mode in ['prod_diff', 'prod_diff_balance']:
                        f_bond = f_bond_prod + f_bond_diff
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2  




        #====================================================================================================#
        # Input is a molecule.
        '''
        if not self.is_reaction:
            # Get atom features
            self.f_atoms = [atom_features(atom) for atom in mol_x.GetAtoms()]
            if atom_features_extra is not None:
                if overwrite_default_atom_features:
                    self.f_atoms = [descs.tolist() for descs in atom_features_extra]
                else:
                    self.f_atoms = [f_atoms + descs.tolist() for f_atoms, descs in zip(self.f_atoms, atom_features_extra)]
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # Get number of atoms
            self.n_atoms = len(self.f_atoms)
            if atom_features_extra is not None and len(atom_features_extra) != self.n_atoms:
                raise ValueError(f'The number of atoms in {Chem.MolToSmiles(mol_x)} is different from the length of the extra atom features.')
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol_x.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue

                    f_bond = bond_features(bond)
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr

                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2

            if bond_features_extra is not None and len(bond_features_extra) != self.n_bonds / 2:
                raise ValueError(f'The number of bonds in {Chem.MolToSmiles(mol_x)} is different from the length of the extra bond features.')
                '''
        #====================================================================================================#













#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#     `7MM"""Yp,          mm         `7MM            `7MMM.     ,MMF'       `7MM        .g8"""bgd                          `7MM                        #
#       MM    Yb          MM           MM              MMMb    dPMM           MM      .dP'     `M                            MM                        #
#       MM    dP  ,6"Yb.mmMMmm ,p6"bo  MMpMMMb.        M YM   ,M MM  ,pW"Wq.  MM      dM'       ` `7Mb,od8 ,6"Yb. `7MMpdMAo. MMpMMMb.                  #
#       MM"""bg. 8)   MM  MM  6M'  OO  MM    MM        M  Mb  M' MM 6W'   `Wb MM      MM            MM' "'8)   MM   MM   `Wb MM    MM                  #
#       MM    `Y  ,pm9MM  MM  8M       MM    MM        M  YM.P'  MM 8M     M8 MM      MM.    `7MMF' MM     ,pm9MM   MM    M8 MM    MM                  #
#       MM    ,9 8M   MM  MM  YM.    , MM    MM        M  `YM'   MM YA.   ,A9 MM      `Mb.     MM   MM    8M   MM   MM   ,AP MM    MM                  #
#     .JMMmmmd9  `Moo9^Yo.`MbmoYMbmd'.JMML  JMML.    .JML. `'  .JMML.`Ybmd9'.JMML.      `"bmmmdPY .JMML.  `Moo9^Yo. MMbmmd'.JMML  JMML.                #
#                                                                                                                   MM                                 #
#                                                                                                                 .JMML.                               #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class BatchMolGraph:

    # A class 'BatchMolGraph' represents the graph structure and featurization of a batch of molecules.
    # A BatchMolGraph contains the attributes of a  CLASS 'MolGraph' plus:

    # vars 'atom_fdim': The dimensionality of the atom feature vector.
    # vars 'bond_fdim': The dimensionality of the bond feature vector (technically the combined atom/bond features).
    # vars 'a_scope': A list of tuples indicating the start and end atom indices for each molecule.
    # vars 'b_scope': A list of tuples indicating the start and end bond indices for each molecule.
    # vars 'max_num_bonds': The maximum number of bonds neighboring an atom in this batch.
    # vars 'b2b': (Optional) A mapping from a bond index to incoming bond indices.
    # vars 'a2a': (Optional): A mapping from an atom index to neighboring atom indices.

    def __init__(self, mol_graphs: List[MolGraph]):
        # param mol_graphs: A list of  CLASS 'MolGraph'\ s from which to construct the CLASS 'BatchMolGraph'.

        self.overwrite_default_atom_features = mol_graphs[0].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs[0].overwrite_default_bond_features


        self.is_reaction = mol_graphs[0].is_reaction
        self.X_TrainArgs = mol_graphs[0].X_TrainArgs
        self.PARAMS      = mol_graphs[0].PARAMS


        self.atom_fdim = get_atom_fdim(overwrite_default_atom = self.overwrite_default_atom_features , 
                                       is_reaction            = self.is_reaction                     , 
                                       PARAMS                 = self.PARAMS                          , 
                                       X_TrainArgs            = self.X_TrainArgs                     , 
                                       )

        print("self.atom_fdim: ", self.atom_fdim)




        self.bond_fdim = get_bond_fdim(atom_messages           = self.X_TrainArgs.atom_messages       , 
                                       overwrite_default_bond  = self.overwrite_default_bond_features , 
                                       overwrite_default_atom  = self.overwrite_default_atom_features , 
                                       is_reaction             = self.is_reaction                     , 
                                       PARAMS                  = self.PARAMS                          , 
                                       X_TrainArgs             = self.X_TrainArgs                     , 
                                       )
        
        print("self.bond_fdim: ", self.bond_fdim)



        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        print(len(f_atoms[0]))
        print(len(f_atoms[1]))

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:

        # Returns the components of the  CLASS 'BatchMolGraph'.
        # The returned components are, in order:

        # vars 'f_atoms'
        # vars 'f_bonds'
        # vars 'a2b'
        # vars 'b2a'
        # vars 'b2revb'
        # vars 'a_scope'
        # vars 'b_scope'

        # param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                             # vector to contain only bond features rather than both atom and bond features.
        # return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                # and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).

        if atom_messages:
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages          = atom_messages                        ,
                                                     overwrite_default_atom = self.overwrite_default_atom_features ,
                                                     overwrite_default_bond = self.overwrite_default_bond_features ,
                                                     X_TrainArgs            = self.X_TrainArgs                     , ):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        # Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.
        # return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        # Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.
        # return: A PyTorch tensor containing the mapping from each atom index to all the neighboring atom indices.
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a










###################################################################################################################
#     `7MMM.     ,MMF'       `7MM        mm                                                  `7MM                 #
#       MMMb    dPMM           MM        MM                                                    MM                 #
#       M YM   ,M MM  ,pW"Wq.  MM      mmMMmm ,pW"Wq.      .P"Ybmmm `7Mb,od8 ,6"Yb. `7MMpdMAo. MMpMMMb.           #
#       M  Mb  M' MM 6W'   `Wb MM        MM  6W'   `Wb    :MI  I8     MM' "'8)   MM   MM   `Wb MM    MM           #
#       M  YM.P'  MM 8M     M8 MM        MM  8M     M8     WmmmP"     MM     ,pm9MM   MM    M8 MM    MM           #
#       M  `YM'   MM YA.   ,A9 MM        MM  YA.   ,A9    8M          MM    8M   MM   MM   ,AP MM    MM           #
#     .JML. `'  .JMML.`Ybmd9'.JMML.      `Mbmo`Ybmd9'      YMMMMMb  .JMML.  `Moo9^Yo. MMbmmd'.JMML  JMML.         #
#                                                         6'     dP                   MM                          #
#                                                         Ybmmmd'                   .JMML.                        #
###################################################################################################################
# 
def mol2graph(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
              atom_features_batch: List[np.array] = (None,),
              bond_features_batch: List[np.array] = (None,),
              overwrite_default_atom_features: bool = False,
              overwrite_default_bond_features: bool = False,
              X_TrainArgs: TrainArgs = None, 
              ) -> BatchMolGraph:

    # Converts a list of SMILES or RDKit molecules to a  CLASS `BatchMolGraph` containing the batch of molecular graphs.
    # param mols                             :  A list of SMILES or a list of RDKit molecules.
    # param atom_features_batch              :  A list of 2D numpy array containing additional atom features to featurize the molecule
    # param bond_features_batch              :  A list of 2D numpy array containing additional bond features to featurize the molecule
    # param overwrite_default_atom_features  :  Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating
    # param overwrite_default_bond_features  :  Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating

    # return: A BatchMolGraph CLASS containing the combined molecular graph for the molecules.

    return BatchMolGraph([MolGraph(mol, af, bf,
                                   overwrite_default_atom_features = overwrite_default_atom_features,
                                   overwrite_default_bond_features = overwrite_default_bond_features,
                                   X_TrainArgs                     = X_TrainArgs                    , )
                          for mol, af, bf in zip_longest(mols, atom_features_batch, bond_features_batch)])












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

