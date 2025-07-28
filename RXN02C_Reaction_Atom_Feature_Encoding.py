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
#--------------------------------------------------#
from rdkit import Chem

#--------------------------------------------------#
import sys
import time
import numpy
import pickle
import typing
import itertools
from tqdm import tqdm
from copy import deepcopy
from pprint import pprint
from typing import Optional, Union, Tuple, Type, Set
#--------------------------------------------------#
import numpy as np
import pandas as pd
#--------------------------------------------------#
from PIL import Image
#from cairosvg import svg2png

#--------------------------------------------------#
from HG_figure import *
from HG_rdkit import *
#--------------------------------------------------#
from AP_convert import Get_Unique_SMILES
#GetUnqSmi = Get_Unique_SMILES(isomericSmiles = True , canonical = True, SMARTS_bool = False)
GetUnqSmi = Get_Unique_SMILES(isomericSmiles = False, canonical = True, SMARTS_bool = False)



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def get_all_smiles_from_file(smiles_file):
    smiles_list = []
    with open(smiles_file) as f:
        lines = f.readlines()
        for one_line in lines:
            smiles_list.append(one_line.replace("\n", ""))
    return smiles_list

#============================================================================================================================#
def get_rxn_portion_from_smiles_list(smiles_list        : list  , 
                                     rxn_portion_SMARTS : tuple , ) \
                                     -> Tuple[dict, list]: # { smiles_str : mapping_list}, [mapping_list_i]
    #--------------------------------------------------#
    # Pattern Matching
    smiles_unannotated = []
    substrate_mapping_dict = dict([])
    for one_smiles in smiles_list:
        substrate_mapping_list = patterns_list_retrieving_AP(one_smiles, substructure_smarts_list = rxn_portion_SMARTS)
        for i in range(len(substrate_mapping_list)):
            if len(substrate_mapping_list[i]) != 0:
                substrate_mapping_dict[one_smiles] = substrate_mapping_list[i]  
        if one_smiles not in substrate_mapping_dict:
            smiles_unannotated.append(one_smiles)
            substrate_mapping_dict[one_smiles] = []
            #print("unannotated smiles found!")
            
    #--------------------------------------------------#
    # Hard Coding for identifying substructure.
        '''
        if len(substrate_mapping_list[0])!=0:
            substrate_mapping_dict[one_smiles] = substrate_mapping_list[0]
        elif len(substrate_mapping_list[1])!=0:
            substrate_mapping_dict[one_smiles] = substrate_mapping_list[1]
        elif len(substrate_mapping_list[2])!=0:
            substrate_mapping_dict[one_smiles] = substrate_mapping_list[2]
        else:
            smiles_unannotated.append(one_smiles)
            print("unannotated smiles found!")
            '''


    #[print(i, substrate_mapping_dict[i]) for i in substrate_mapping_dict ]
    #--------------------------------------------------#
    # plot unannotated smiles
    print("number of smiles failed to identify reacting portion: ", len(smiles_unannotated))
    '''
    plot_smiles_list(   smiles_list = smiles_unannotated,
                        fig_folder = output_folder, 
                        img_size = (500, 500), 
                        molsPerRow = 5, 
                        separate_img = False, 
                        fig_name = dataset_nme)
                        '''
    #--------------------------------------------------#
    # Get rxn_portion
    substrate_rxn_portion_dict = dict([])
    substrate_rxn_portion_list = []
    for one_smiles in substrate_mapping_dict:

        mapping_set = set()
        for mapping in (substrate_mapping_dict[one_smiles]):
            mapping_set = mapping_set.union(set(mapping))
        substrate_rxn_portion_dict[one_smiles] = list(mapping_set)
        substrate_rxn_portion_list.append(list(mapping_set))
    return substrate_rxn_portion_dict, substrate_rxn_portion_list




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def RXN02C_HG01(dataset_nme, rxn_portion_SMARTS_dict, input_folder, output_folder, output_nme, output_temp_folder, smiles_file):
    #============================================================================================================================#
    # Get all SMILES.
    smiles_list = get_all_smiles_from_file(input_folder / smiles_file)
    '''
    plot_smiles_list(   smiles_list = smiles_list,
                        fig_folder = output_folder, 
                        img_size = (500, 500), 
                        molsPerRow = 5, 
                        separate_img = True, 
                        fig_name = dataset_nme)
                        '''
    #--------------------------------------------------#
    # Preprocess SMILES list 
    # (FOR TEMPORARY USE ONLY, SHOULD HAVE PERFORM THIS AT THE DATA CLEANING STEP !!)
    smiles_list_proc = []
    print("\n\nProcessing SMILES list...")
    for i, smiles_x in enumerate(smiles_list):
        print(i, "out of", len(smiles_list))
        if smiles_x not in ["[HH]", ]:
            mol_x      = Chem.MolFromSmiles(smiles_x)
            # for atom in mol_x.GetAtoms():
            #     atom.SetFormalCharge(0)
            mol_x      = Chem.RemoveAllHs(mol_x)
            smiles_x   = Chem.MolToSmiles(mol_x)
            smiles_list_proc.append(smiles_x)
        else:
            smiles_list_proc.append(smiles_x)
    #--------------------------------------------------#
    # Get substrate_rxn_portion_dict
    rxn_portion_SMARTS = rxn_portion_SMARTS_dict[dataset_nme] if dataset_nme in rxn_portion_SMARTS_dict else ()
    substrate_rxn_portion_dict, _ = get_rxn_portion_from_smiles_list(smiles_list_proc, rxn_portion_SMARTS)
    #[print(i, substrate_rxn_portion_dict[i]) for i in substrate_rxn_portion_dict ]

    #============================================================================================================================#
    # Obtain graph components for each smiles.
    # components list:
    # 1. Atom_Attributes             size: (n_node, n_attr)               type: numpy.array
    # 2. Atom_RXN_Portion            size: (n_rxn_portion)                type: List
    # 3. Bond_Adjacency_Matrix       size: (n_node, n_node)               type: numpy.array
    # 4. Bond_Attributes             size: (n_node, n_node, dim_attr)     type: numpy.array
    # 5. Node_info
    Atom_Attributes_list       = []
    Atom_RXN_Portion_list      = []
    Bond_Adjacency_Matrix_list = []
    Bond_Attributes_list       = []
    Bond_general_info_list     = []

    #print("locals(): ", locals())

    all_Morgan_list = smiles_list_to_all_Morgan_list(smiles_list_proc, radius = 2)

    print("\n\nProcessing Atom Level Features...")
    for i, one_smiles in enumerate(smiles_list_proc):
        #--------------------------------------------------#
        print(i, "out of", len(smiles_list_proc))
        #--------------------------------------------------#
        # 1. Get Atom_Attributes
        begin = time.time()
        
        Atom_Attributes_list.append(smiles_to_nodes_encodings(one_smiles, smiles_list_proc, radius = 2, all_Morgan_list = all_Morgan_list))
        Atom_RXN_Portion_list.append(substrate_rxn_portion_dict[one_smiles])

        Bond_Adjacency_Matrix, Bond_Attributes, Bond_general_info = smiles_to_bond_matrices(one_smiles)
        Bond_Adjacency_Matrix_list.append(Bond_Adjacency_Matrix)
        Bond_Attributes_list.append(Bond_Attributes)
        Bond_general_info_list.append(Bond_general_info)

        print(time.time() - begin)


    Graph_Attributes = ["Atom_Attributes_list", 
                        "Atom_RXN_Portion_list", 
                        "Bond_Adjacency_Matrix_list", 
                        "Bond_Attributes_list", ]


    Cmpd_Graph_Attributes = { "smiles_list"                : smiles_list                ,
                              "Atom_Attributes_list"       : Atom_Attributes_list       ,
                              "Atom_RXN_Portion_list"      : Atom_RXN_Portion_list      ,
                              "Bond_Adjacency_Matrix_list" : Bond_Adjacency_Matrix_list ,
                              "Bond_Attributes_list"       : Bond_Attributes_list       ,
                              "Bond_general_info_list"     : Bond_general_info_list     ,
                              }

    pickle.dump(Cmpd_Graph_Attributes, open(output_folder / output_nme, "wb"))

    return
###################################################################################################################
###################################################################################################################








#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
if __name__ == '__main__':
    
    # Args
    #--------------------------------------------------#
    # Inputs
    Step_code = "RXN02C_"
    dataset_nme_list = ["sample_reaction_dataset" ,        # 0
                        "Reaction_Energy"         ,        # 1
                        "Log_RateConstant"        ,        # 2
                        ""                        ,        # 99

                        ]

    dataset_nme     = dataset_nme_list[2]
    #--------------------------------------------------#
    # Directory and Files
    data_folder        =  Path("RXN_DataProcessing/")
    input_folder       =  Path("RXN_DataProcessing/")
    output_folder      =  Path("RXN_DataProcessing/")
    output_temp_folder =  Path("RXN_DataProcessing/RXN02_temp/")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_temp_folder):
        os.makedirs(output_temp_folder)

    output_nme = Step_code + dataset_nme + "_Cmpd_Graph_Attributes.p"

    #--------------------------------------------------#
    # Get all SMILES
    smiles_file     = "RXN00_" + dataset_nme + "_compounds_smiles.smiles"
    smiles_list = get_all_smiles_from_file(input_folder / smiles_file)
    # Print SMILES with length <= 4. 
    for i in smiles_list:
        if len(i) <= 4:
            print(i)
    
    #--------------------------------------------------#
    # Interested Substructures
    rxn_portion_SMARTS_dict = { "phosphatase": ("[P](=[O])([OH])([OH])", "[O]([P](=[O])([OH]))([P](=[O])([OH]))", "[P](=[O])([OH])"),
                                "kinase"     : "[#15]",
                                "halogenase" : "[#15]",
                                "esterase"   : "[#15]",
                                                            } 


    #====================================================================================================#
    # Main
    #--------------------------------------------------#
    RXN02C_HG01(dataset_nme, rxn_portion_SMARTS_dict, input_folder, output_folder, output_nme, output_temp_folder, smiles_file)
    #--------------------------------------------------#
    # Test 1
    '''
    one_smiles = "*N[C@@H](CSSC[C@H]([NH2+]*)C(*)=O)C(*)=O"
    Bond_Adjacency_Matrix, Bond_Attributes, Bond_general_info = smiles_to_bond_matrices(one_smiles)
    print(Bond_Adjacency_Matrix)
    print(Bond_Attributes)
    print(len(Bond_general_info))
    '''

    # Test 2
    '''
    Cmpd_Graph_Attributes = pickle.load(open(output_folder / output_nme, 'rb'))
    for i in list(Cmpd_Graph_Attributes.keys()):
        print(i, len(Cmpd_Graph_Attributes[i]))
        '''
    #--------------------------------------------------#
    print("*" * 50)
    print(Step_code + " Done!") 
 