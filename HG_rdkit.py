#!/usr/bin/env python
# coding: utf-8
###################################################################################################################
###################################################################################################################
# The following code ensures the code work properly in 
# MS VS, MS VS CODE and jupyter notebook on both Linux and Windows.
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
if __name__ == "__main__":
    print("="*80)
    if os.name == 'nt' or platform == 'win32':
        print("Running on Windows")
        if 'ptvsd' in sys.modules:
            print("Running in Visual Studio")
#--------------------------------------------------#
    if os.name != 'nt' and platform != 'win32':
        print("Not Running on Windows")
#--------------------------------------------------#
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
# from alfabet import model
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
from typing import Optional, Union, Tuple, Type, Set, List, Dict
#--------------------------------------------------#
import numpy as np
import pandas as pd
#--------------------------------------------------#
from PIL import Image
# from cairosvg import svg2png

#--------------------------------------------------#
from HG_figure import *
from AP_convert import MolFromSmiles_ZX
from AP_convert import Get_Unique_SMILES

#GetUnqSmi = Get_Unique_SMILES(isomericSmiles = False, SMARTS_bool = False)
#GetUnqSmi = Get_Unique_SMILES(isomericSmiles = True, canonical = True, SMARTS_bool = False)
GetUnqSmi = Get_Unique_SMILES(isomericSmiles = False, canonical = True, SMARTS_bool = False)

#--------------------------------------------------#
# Directory
input_folder = Path("HG_data/")
output_folder = Path("HG_results/")
output_temp_folder = Path("HG_results/HG_temp/")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(output_temp_folder):
    os.makedirs(output_temp_folder)

#--------------------------------------------------#
# bond_type_SMARTS_dict
bond_type_SMARTS_dict = {1.0 : "-", 2.0 : "=", 3.0 : "#", 1.5 : ":"}



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MMF'   `7MF' mm     db `7MM             `7MM"""YMM                               mm     db                              
#    MM       M   MM          MM               MM    `7                               MM                                     
#    MM       M mmMMmm `7MM   MM  ,pP"Ybd      MM   d `7MM  `7MM `7MMpMMMb.  ,p6"bo mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.  ,pP"Ybd  
#    MM       M   MM     MM   MM  8I   `"      MM""MM   MM    MM   MM    MM 6M'  OO   MM     MM 6W'   `Wb MM    MM  8I   `"  
#    MM       M   MM     MM   MM  `YMMMa.      MM   Y   MM    MM   MM    MM 8M        MM     MM 8M     M8 MM    MM  `YMMMa.  
#    YM.     ,M   MM     MM   MM  L.   I8      MM       MM    MM   MM    MM YM.    ,  MM     MM YA.   ,A9 MM    MM  L.   I8  
#     `bmmmmd"'   `Mbmo.JMML.JMML.M9mmmP'    .JMML.     `Mbod"YML.JMML  JMML.YMbmd'   `Mbmo.JMML.`Ybmd9'.JMML  JMML.M9mmmP'  
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Read SMILES files.
def get_all_smiles_from_file(smiles_file):
    smiles_list = []
    with open(smiles_file) as f:
        lines = f.readlines()
        for one_line in lines:
            smiles_list.append(one_line.replace("\n", ""))
    return smiles_list

#============================================================================================================================#
# Print the DataFrame obtained.
def beautiful_print(df, row = 30, col = 8, width = 120):
    # Print the dataset in a well-organized format.
    with pd.option_context('display.max_rows', row, 
                           'display.min_rows', row, 
                           'display.max_columns', col, 
                           #"display.max_colwidth", None,
                           "display.width", width,
                           "expand_frame_repr", True,
                           "max_seq_items", None,):  # more options can be specified
        # Once the display.max_rows is exceeded, 
        # the display.min_rows options determines 
        # how many rows are shown in the truncated repr.
        print(df)
    return 

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#         db     `7MM"""Mq.     `7MM"""YMM                               mm     db                             
#        ;MM:      MM   `MM.      MM    `7                               MM                                    
#       ,V^MM.     MM   ,M9       MM   d `7MM  `7MM `7MMpMMMb.  ,p6"bo mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.  ,pP"Ybd 
#      ,M  `MM     MMmmdM9        MM""MM   MM    MM   MM    MM 6M'  OO   MM     MM 6W'   `Wb MM    MM  8I   `" 
#      AbmmmqMA    MM             MM   Y   MM    MM   MM    MM 8M        MM     MM 8M     M8 MM    MM  `YMMMa. 
#     A'     VML   MM             MM       MM    MM   MM    MM YM.    ,  MM     MM YA.   ,A9 MM    MM  L.   I8 
#   .AMA.   .AMMA.JMML.         .JMML.     `Mbod"YML.JMML  JMML.YMbmd'   `Mbmo.JMML.`Ybmd9'.JMML  JMML.M9mmmP' 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Old Functions from Anneal Path.
#============================================================================================================================#
def unique_canonical_smiles_AP(smiles_x, isomericSmiles = False): # unis()
    # SMILES -> R-mol -> SMARTS -> R-mol -> SMILES
    mol_x=Chem.MolFromSmiles(smiles_x)
    try:
        mol_x_unique=Chem.MolFromSmarts(Chem.MolToSmarts(mol_x))
        unique_smiles=Chem.MolToSmiles(mol_x_unique, isomericSmiles = isomericSmiles)
    except Exception:
        print ("!!!!! Problematic input SMILES string !!!!!")
        unique_smiles=smiles_x
    return unique_smiles

#============================================================================================================================#
def unique_canonical_smiles_list_AP(list_x, isomericSmiles = False): # unis_l()
    # SMILES -> R-mol -> SMARTS -> R-mol -> SMILES
    new_list=[]
    for one_smiles in list_x:
        new_list.append(unique_canonical_smiles_AP(one_smiles, isomericSmiles))
    return new_list

#============================================================================================================================#
def canonical_smiles_AP(smiles_x, isomericSmiles = False): # cans()
    # SMILES -> R-mol -> SMILES
    mol_x=Chem.MolFromSmiles(smiles_x)
    try:
        unique_smiles=Chem.MolToSmiles(mol_x, isomericSmiles = isomericSmiles)
    except Exception:
        #print ("problematic")
        unique_smiles=smiles_x
    return unique_smiles

#============================================================================================================================#
def canonical_smiles_list_AP(list_x, isomericSmiles = False): # cans_l()
    # SMILES -> R-mol -> SMILES
    new_list=[]
    for one_smiles in list_x:
        new_list.append(canonical_smiles_AP(one_smiles, isomericSmiles))
    return new_list

#============================================================================================================================#
def MolFromSmiles_AP(smiles_x, bad_ss_dict, isomericSmiles = False):
    # SMILES -> R-mol
    mol_x=Chem.MolFromSmiles(smiles_x)
    try: 
        Chem.MolToSmiles(mol_x, isomericSmiles = isomericSmiles)
    except:
        mol_x=Chem.MolFromSmarts(bad_ss_dict[smiles_x])
        #mol_x=Chem.MolFromSmarts("O")
    return mol_x

#============================================================================================================================#
def MolToSmiles_AP(mol_x, bad_ss_dict, isomericSmiles = False):
    # R-mol -> SMARTS -> R-mol -> SMILES
    # !!!!!: Converting to smarts and back to rdkit-format to ensure uniqueness before converting to smiles
    mol_x_unique=Chem.MolFromSmarts(Chem.MolToSmarts(mol_x))
    smiles_x=Chem.MolToSmiles(mol_x_unique, isomericSmiles = isomericSmiles)
    try:
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles_x), isomericSmiles = isomericSmiles)
    except Exception:
        bad_ss_dict[smiles_x]=Chem.MolToSmarts(mol_x_unique)
    return smiles_x

#============================================================================================================================#
def pattern_matching_AP(cmpd_smiles, substructure_smarts, bad_ss_dict={}):
    #####----------return boolean variable (if the compound match the subgraph)
    mol_x = MolFromSmiles_AP(cmpd_smiles, bad_ss_dict)
    try:
        pattern_matching = mol_x.HasSubstructMatch(Chem.MolFromSmarts(substructure_smarts))
    except Exception:
        pattern_matching = False
    return pattern_matching


#============================================================================================================================#
def pattern_retrieving_AP(cmpd_smiles, substructure_smarts, bad_ss_dict={}):
    #####----------return mappings of substructures
    mol_x = MolFromSmiles_AP(cmpd_smiles, bad_ss_dict)
    try:
        pattern_matching = pattern_matching_AP(cmpd_smiles, substructure_smarts, bad_ss_dict)
        if pattern_matching:
            substructure_mapping = mol_x.GetSubstructMatches(Chem.MolFromSmarts(substructure_smarts))
        else:
            substructure_mapping = tuple([])
    except Exception:
        substructure_mapping = tuple([])
    return substructure_mapping

#============================================================================================================================#
def patterns_list_retrieving_AP(cmpd_smiles, substructure_smarts_list, bad_ss_dict={}):
    #####----------return mappings of substructures
    mol_x = MolFromSmiles_AP(cmpd_smiles, bad_ss_dict)
    try:
        substructure_mapping_list = []
        for substructure_smarts in substructure_smarts_list:
            pattern_matching = pattern_matching_AP(cmpd_smiles, substructure_smarts, bad_ss_dict)
            if pattern_matching:
                substructure_mapping = mol_x.GetSubstructMatches(Chem.MolFromSmarts(substructure_smarts))
                substructure_mapping_list.append(substructure_mapping)
            else:
                substructure_mapping_list.append( tuple([]) )
    except Exception:
        substructure_mapping_list = [ tuple([]) for i in range(len(substructure_mapping_list)) ]
        print("patterns_list_retrieving_AP ERROR, ", cmpd_smiles, "\nsubstructure_mapping_list: ", substructure_mapping_list)
    return substructure_mapping_list








#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# `7MM"""Mq.`7MMF'        .g8""8q.  MMP""MM""YMM   `7MM"""YMM `7MMF'   `7MF'`7MN.   `7MF' .g8"""bgd MMP""MM""YMM `7MMF' .g8""8q. `7MN.   `7MF'.M"""bgd #
#   MM   `MM. MM        .dP'    `YM.P'   MM   `7     MM    `7   MM       M    MMN.    M .dP'     `M P'   MM   `7   MM .dP'    `YM. MMN.    M ,MI    "Y #
#   MM   ,M9  MM        dM'      `MM     MM          MM   d     MM       M    M YMb   M dM'       `      MM        MM dM'      `MM M YMb   M `MMb.     #
#   MMmmdM9   MM        MM        MM     MM          MM""MM     MM       M    M  `MN. M MM               MM        MM MM        MM M  `MN. M   `YMMNq. #
#   MM        MM      , MM.      ,MP     MM          MM   Y     MM       M    M   `MM.M MM.              MM        MM MM.      ,MP M   `MM.M .     `MM #
#   MM        MM     ,M `Mb.    ,dP'     MM          MM         YM.     ,M    M     YMM `Mb.     ,'      MM        MM `Mb.    ,dP' M     YMM Mb     dM #
# .JMML.    .JMMmmmmMMM   `"bmmd"'     .JMML.      .JMML.        `bmmmmd"'  .JML.    YM   `"bmmmd'     .JMML.    .JMML. `"bmmd"' .JML.    YM P"Ybmmd"  #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#######################################################################################################################################
#######################################################################################################################################
# 
def plot_smiles_list(smiles_list, fig_folder = Path("./"), img_size = (520, 520), molsPerRow = 5, separate_img = False, fig_name = ""):
    #r_mol_list = [Chem.MolFromSmiles(one_smiles) for one_smiles in smiles_list]
    r_mol_list = [MolFromSmiles_ZX(one_smiles, isomericSmiles = False) for one_smiles in smiles_list]
    #print(r_mol_list)
    image_list = []
    if separate_img == True :
        for one_smiles in smiles_list:
            mol_x = Chem.MolFromSmiles(one_smiles)
            img = Draw.MolsToGridImage([mol_x, ], molsPerRow = 1, subImgSize = img_size, returnPNG=False)
            output_png_path = fig_folder / "./HG_temp" / ("SMILES" + str(smiles_list.index(one_smiles)) + "_" + fig_name + ".png")
            output_png_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_png_path)
            image_list.append(Image.open(output_png_path))
        #--------------------------------------------------#
        # UNFINISHED!!
        '''
        for one_img_idx in range(len(image_list)):
            get_concat_multi_h_blank()
        get_concat_multi_v_blank()
        '''

    if separate_img == False :
        img = Draw.MolsToGridImage(r_mol_list, molsPerRow = min(molsPerRow, len(smiles_list)), subImgSize = img_size, returnPNG=False) 
        img.save(fig_folder / ("SMILES_list_" + fig_name + ".png"))

    return r_mol_list


#######################################################################################################################################
#######################################################################################################################################
#
def plot_smiles_annotated(smiles_x, fig_folder = Path("./"), img_size = (900, 900), fig_name = "test"):
    #--------------------------------------------------#
    # Input SMILES
    mol_x = Chem.MolFromSmiles(smiles_x)
    num_atom_x = len([atom for atom in mol_x.GetAtoms()])
    #--------------------------------------------------#
    [mol_x.GetBondWithIdx(bond.GetIdx()).SetProp('bondNote', "B#" + str(bond.GetIdx())) for bond in mol_x.GetBonds()]
    #--------------------------------------------------#
    d = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1]) # or MolDraw2DSVG to get SVGs
    d.drawOptions().addStereoAnnotation = True
    d.drawOptions().addAtomIndices = True
    d.DrawMolecule(mol_x)
    d.FinishDrawing()
    #--------------------------------------------------#
    output_png_path_1 = fig_folder / ("Molecule_annotation_" + fig_name + ".png")
    d.WriteDrawingText(str(output_png_path_1))
    #--------------------------------------------------#
    #Image(filename=output_png_path_1) 
    return 
#============================================================================================================================#
def plot_molecule(smiles_x, fig_folder = Path("./"), img_size = (900, 900), fig_name = "test"):
    return plot_smiles_annotated(smiles_x, fig_folder = Path("./"), img_size = (900, 900), fig_name = "test")


#######################################################################################################################################
#######################################################################################################################################
# 
def plot_Morgan_substructures(smiles_x, radius = 2, level = [0,1], fig_folder = Path("./"), img_size = (120, 120), molsPerRow = 8, fig_name = "test"):
    #============================================================================================================================#
    # Input SMILES
    mol_x = Chem.MolFromSmiles(smiles_x)
    num_atom_x = len([atom for atom in mol_x.GetAtoms()])
    #============================================================================================================================#
    # Get Bit Info - 1
    Morgan_dict = {}
    Morgan_fingerprint = AllChem.GetMorganFingerprint(mol_x, radius = radius, bitInfo = Morgan_dict)
    #[print(one_Morgan_info, Morgan_dict[one_Morgan_info]) for one_Morgan_info in Morgan_dict]
    #============================================================================================================================#
    Morgan_substructures_list = [(mol_x, one_MorganFP, Morgan_dict) for one_MorganFP in Morgan_dict]   # (mol, MorganFP, Morgan_mapping)
    legends = [str(one_MorganFP) + ": " + str(Morgan_dict[one_MorganFP]) for one_MorganFP in Morgan_dict]
    MorganFP_svg = Draw.DrawMorganBits( Morgan_substructures_list, 
                                        molsPerRow = min(molsPerRow, num_atom_x),
                                        subImgSize = img_size, 
                                        legends = legends,
                                        useSVG = True)
    #print(type(MorganFP_svg))
    #--------------------------------------------------#
    # Save SVG.
    output_svg_path = fig_folder / ("MorganFP_" + fig_name + ".svg")
    # !!!!! In Notebook, use MorganFP_svg.data instead of MorganFP_svg. Because of different variable type.
    with open(output_svg_path, 'w') as f_handle:
        f_handle.write(MorganFP_svg)
    svg2png(bytestring = MorganFP_svg, write_to = str(fig_folder / ("MorganFP_unordered_" + fig_name  + ".png")))
    #============================================================================================================================#
    return MorganFP_svg


#######################################################################################################################################
#######################################################################################################################################
# 
def order_Morgan_substructures(mol_x, Morgan_dict, radius, level_x):
    #--------------------------------------------------#
    # Input SMILES
    #[print(one_Morgan_info, Morgan_dict[one_Morgan_info]) for one_Morgan_info in Morgan_dict]
    atom_idx_list_x = [atom.GetIdx() for atom in mol_x.GetAtoms()]
    #--------------------------------------------------#
    Morgan_dict_reverse = {}
    for one_MorganFP in Morgan_dict:
        for one_substructure in Morgan_dict[one_MorganFP]:
            Morgan_dict_reverse[one_substructure] = one_MorganFP
    #print(Morgan_dict_reverse)
    #--------------------------------------------------#
    ordered_Morgan_substructures_list = []
    ordered_legends = []
    
    for one_radius in [level_x, ]:
        for one_idx in atom_idx_list_x:
            one_substructure_mapping = (one_idx, one_radius)
            #print(one_substructure_mapping)
            if one_substructure_mapping not in Morgan_dict_reverse:
                continue

            Morgan_dict_part = {Morgan_dict_reverse[one_substructure_mapping]: (one_substructure_mapping,) , }
            ordered_Morgan_substructures_list.append((mol_x, Morgan_dict_reverse[one_substructure_mapping], Morgan_dict_part ))
            ordered_legends.append(str(Morgan_dict_reverse[one_substructure_mapping]) + ": " + str(one_substructure_mapping))
    #--------------------------------------------------#
    return ordered_Morgan_substructures_list, ordered_legends

#============================================================================================================================#
def plot_Morgan_substructures_ordered(smiles_x, radius = 2, level = [0,], fig_folder = Path("./"), img_size = (120, 120), molsPerRow = 8, fig_name = ""):
    #--------------------------------------------------##--------------------------------------------------#
    # Input SMILES
    mol_x = Chem.MolFromSmiles(smiles_x)
    num_atom_x = len([atom for atom in mol_x.GetAtoms()])
    #--------------------------------------------------##--------------------------------------------------#
    # Get Bit Info - 1
    Morgan_dict = {}
    Morgan_fingerprint = AllChem.GetMorganFingerprint(mol_x, radius = radius, bitInfo = Morgan_dict)
    #[print(one_Morgan_info, Morgan_dict[one_Morgan_info]) for one_Morgan_info in Morgan_dict]
    #--------------------------------------------------##--------------------------------------------------#
    level = range(radius)
    image_list = []
    for level_x in level:
        #--------------------------------------------------#
        ordered_Morgan_substructures_list, ordered_legends = order_Morgan_substructures(mol_x, 
                                                                                        Morgan_dict, 
                                                                                        radius = radius, 
                                                                                        level_x = level_x)
        #--------------------------------------------------#
        # Get SVG for all Bit Info
        Morgan_substructures_list = [(mol_x, one_MorganFP, Morgan_dict) for one_MorganFP in Morgan_dict]   # (mol, MorganFP, Morgan_mapping)
        #[print(x) for x in Morgan_substructures_list]
        #[print(x) for x in order_Morgan_substructures_list]
        legends = [str(one_MorganFP) + ": " + str(Morgan_dict[one_MorganFP]) for one_MorganFP in Morgan_dict]

        MorganFP_svg = Draw.DrawMorganBits(ordered_Morgan_substructures_list, 
                                            molsPerRow = min(molsPerRow, num_atom_x),
                                            subImgSize = img_size, 
                                            legends = ordered_legends,
                                            useSVG = True)
        
        #print(type(MorganFP_svg))
        #--------------------------------------------------#
        # Save SVG.
        output_svg_path = fig_folder / ("HG_temp/MorganFP_" + fig_name + "level_" + str(level_x) + ".svg")
        output_png_path = fig_folder / ("HG_temp/MorganFP_" + fig_name + "level_" + str(level_x) + ".png")
        #--------------------------------------------------#
        # !!!!! In Notebook, use MorganFP_svg.data instead of MorganFP_svg. Because of different variable type.
        with open(output_svg_path, 'w') as f_handle:
            f_handle.write(MorganFP_svg)
        svg2png(bytestring = MorganFP_svg, write_to = str(output_png_path))
        #--------------------------------------------------#
        image_list.append(Image.open(output_png_path))
        #--------------------------------------------------#
    get_concat_multi_v_blank(image_list).save(fig_folder / ("MorganFP_" + fig_name + ".png"))
    #--------------------------------------------------##--------------------------------------------------#
    return MorganFP_svg



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# `7MMM.     ,MMF'     db     `7MMF'`7MN.   `7MF'  `7MM"""YMM`7MMF'   `7MF'`7MN.   `7M ' .g8"""bgd MMP""MM""YMM `7MMF' .g8""8q. `7MN.   `7MF'.M"""bgd  #
#   MMMb    dPMM      ;MM:      MM    MMN.    M      MM    `7  MM       M    MMN.    M .dP'     `M P'   MM   `7   MM .dP'    `YM. MMN.    M ,MI    "Y  #
#   M YM   ,M MM     ,V^MM.     MM    M YMb   M      MM   d    MM       M    M YMb   M dM'       `      MM        MM dM'      `MM M YMb   M `MMb.      #
#   M  Mb  M' MM    ,M  `MM     MM    M  `MN. M      MM""MM    MM       M    M  `MN. M MM               MM        MM MM        MM M  `MN. M   `YMMNq.  #
#   M  YM.P'  MM    AbmmmqMA    MM    M   `MM.M      MM   Y    MM       M    M   `MM.M MM.              MM        MM MM.      ,MP M   `MM.M .     `MM  #
#   M  `YM'   MM   A'     VML   MM    M     YMM      MM        YM.     ,M    M     YMM `Mb.     ,'      MM        MM `Mb.    ,dP' M     YMM Mb     dM  #
# .JML. `'  .JMML.AMA.   .AMMA.JMML..JML.    YM    .JMML.       `bmmmmd"'  .JML.    YM   `"bmmmd'     .JMML.    .JMML. `"bmmd"' .JML.    YM P"Ybmmd"   #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#



#######################################################################################################################################
# `7MMF`7MN.   `7MF`7MMFMMP""MM""YMM `7MMF'     db     `7MMF'      `7MMFMMM"""AMV       db  MMP""MM""YMM `7MMF' .g8""8q. `7MN.   `7MF'#
#   MM   MMN.    M   MM P'   MM   `7   MM      ;MM:      MM          MM M'   AMV       ;MM: P'   MM   `7   MM .dP'    `YM. MMN.    M  #
#   MM   M YMb   M   MM      MM        MM     ,V^MM.     MM          MM '   AMV       ,V^MM.     MM        MM dM'      `MM M YMb   M  #
#   MM   M  `MN. M   MM      MM        MM    ,M  `MM     MM          MM    AMV       ,M  `MM     MM        MM MM        MM M  `MN. M  #
#   MM   M   `MM.M   MM      MM        MM    AbmmmqMA    MM      ,   MM   AMV   ,    AbmmmqMA    MM        MM MM.      ,MP M   `MM.M  #
#   MM   M     YMM   MM      MM        MM   A'     VML   MM     ,M   MM  AMV   ,M   A'     VML   MM        MM `Mb.    ,dP' M     YMM  #
# .JMML.JML.    YM .JMML.  .JMML.    .JMML.AMA.   .AMMA.JMMmmmmMMM .JMMLAMVmmmmMM .AMA.   .AMMA.JMML.    .JMML. `"bmmd"' .JML.    YM  #
#######################################################################################################################################
#
def smiles_to_attributes(smiles_x):
    #--------------------------------------------------#
    # Input SMILES
    mol_x = Chem.MolFromSmiles(smiles_x)
    #============================================================================================================================#
    # Go through all atoms, get all properties and write to a dict.
    atom_idx_list_x = [atom.GetIdx() for atom in mol_x.GetAtoms()]
    atom_dict = {}
    # atom_dict             <-  { atom_idx  : atom_attributes_dict }
    # atom_attributes_dict  <-  { atom_attr : atom_value }
    #--------------------------------------------------#
    # Atoms
    for atom in mol_x.GetAtoms():
        # Add mapping number to SMARTS, mapping number = idx + 1. (mapping number cant be 0).
        atom.SetAtomMapNum(atom.GetIdx()+1)
        # Get All Attributes of Each Atom.
        atom_attributes_dict = {}
        atom_attributes_dict["atom_idx"]            = atom.GetIdx()
        atom_attributes_dict["atom_map_num"]        = atom.GetAtomMapNum()
        atom_attributes_dict["atomic_num"]          = atom.GetAtomicNum()
        atom_attributes_dict["ExplicitValence"]     = atom.GetExplicitValence()
        atom_attributes_dict["FormalCharge"]        = atom.GetFormalCharge()
        atom_attributes_dict["ImplicitValence"]     = atom.GetImplicitValence()
        atom_attributes_dict["IsAromatic"]          = atom.GetIsAromatic()
        atom_attributes_dict["Isotope"]             = atom.GetIsotope()
        atom_attributes_dict["Mass"]                = atom.GetMass()
        atom_attributes_dict["MonomerInfo"]         = atom.GetMonomerInfo()
        atom_attributes_dict["NoImplicit"]          = atom.GetNoImplicit()
        atom_attributes_dict["NumExplicitHs"]       = atom.GetNumExplicitHs()
        atom_attributes_dict["NumImplicitHs"]       = atom.GetNumImplicitHs()
        atom_attributes_dict["NumRadicalElectrons"] = atom.GetNumRadicalElectrons()
        atom_attributes_dict["PDBResidueInfo"]      = atom.GetPDBResidueInfo()
        atom_attributes_dict["Smarts"]              = atom.GetSmarts()
        atom_attributes_dict["Symbol"]              = atom.GetSymbol()
        atom_attributes_dict["TotalDegree"]         = atom.GetTotalDegree() # Degree is defined to be its number of directly-bonded neighbors.
        atom_attributes_dict["TotalNumHs"]          = atom.GetTotalNumHs()
        atom_attributes_dict["TotalValence"]        = atom.GetTotalValence()
        atom_attributes_dict["ChiralTag"]           = atom.GetChiralTag()
        atom_attributes_dict["Neighbors"]           = [atom.GetIdx() for atom in atom.GetNeighbors()]
        atom_attributes_dict["Bonds"]               = [bond.GetIdx() for bond in atom.GetBonds()]
        atom_dict[atom.GetIdx()] = atom_attributes_dict
    #--------------------------------------------------#
    # Get dict and list to generate the dataframe.
    atom_attributes_key_list = list(atom_dict[0].keys())
    atom_attributes_value_dict = {attr : [atom_dict[one_atom][attr] for one_atom in atom_dict]  for attr in atom_attributes_key_list }
    #print(atom_attributes_value_dict)
    #print(atom_attributes_key_list)
    #--------------------------------------------------#
    # Get the Dataframe.
    df_atom_attributes = pd.DataFrame(data = atom_attributes_value_dict, columns = atom_attributes_key_list)
    #============================================================================================================================#
    # Go through all bonds, get all properties and write to a dict.
    bond_idx_list_x = [bond.GetIdx() for bond in mol_x.GetBonds()]
    bond_dict = {}
    # bond_dict             <-  { bond_idx  : bond_attributes_dict }
    # bond_attributes_dict  <-  { bond_attr : bond_value }
    #--------------------------------------------------#
    # Bonds
    for bond in mol_x.GetBonds():
        bond_attributes_dict = {}
        # Get All Attributes of Each Bond.
        bond_attributes_dict["bond_idx"]           =     bond.GetIdx()
        bond_attributes_dict["BondDir"]            =     bond.GetBondDir()
        bond_attributes_dict["BondType"]           =     bond.GetBondType()
        bond_attributes_dict["BondTypeAsDouble"]   =     bond.GetBondTypeAsDouble()
        bond_attributes_dict["IsAromatic"]         =     bond.GetIsAromatic()
        bond_attributes_dict["IsConjugated"]       =     bond.GetIsConjugated()
        bond_attributes_dict["Smarts"]             =     bond.GetSmarts()
        bond_attributes_dict["Stereo"]             =     bond.GetStereo()
        bond_attributes_dict["bond_idx_tuple"]     =     (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        bond_dict[bond.GetIdx()] = bond_attributes_dict
    #--------------------------------------------------#
    # Get dict and list to generate the dataframe.
    bond_attributes_key_list = ["bond_idx", "BondDir", "BondType", "BondTypeAsDouble", "IsAromatic", "IsConjugated", "Smarts", "Stereo", "bond_idx_tuple"] # = list(bond_dict[0].keys())
    bond_attributes_value_dict = {attr : [bond_dict[one_bond][attr] for one_bond in bond_dict]  for attr in bond_attributes_key_list }
    #print(bond_attributes_value_dict)
    #print(bond_attributes_key_list)
    #--------------------------------------------------#
    # Get the Dataframe.
    df_bond_attributes = pd.DataFrame(data = bond_attributes_value_dict, columns = bond_attributes_key_list)

    return df_atom_attributes, df_bond_attributes, atom_dict, bond_dict



#######################################################################################################################################
# `7MM"""Yp,   .g8""8q. `7MN.   `7MF`7MM"""Yb.      `7MM"""YMM `7MM"""YMM        db  MMP""MM""YMM `7MMF'   `7MF`7MM"""Mq. `7MM"""YMM  #
#   MM    Yb .dP'    `YM. MMN.    M   MM    `Yb.      MM    `7   MM    `7       ;MM: P'   MM   `7   MM       M   MM   `MM.  MM    `7  #
#   MM    dP dM'      `MM M YMb   M   MM     `Mb      MM   d     MM   d        ,V^MM.     MM        MM       M   MM   ,M9   MM   d    #
#   MM"""bg. MM        MM M  `MN. M   MM      MM      MM""MM     MMmmMM       ,M  `MM     MM        MM       M   MMmmdM9    MMmmMM    #
#   MM    `Y MM.      ,MP M   `MM.M   MM     ,MP      MM   Y     MM   Y  ,    AbmmmqMA    MM        MM       M   MM  YM.    MM   Y  , #
#   MM    ,9 `Mb.    ,dP' M     YMM   MM    ,dP'      MM         MM     ,M   A'     VML   MM        YM.     ,M   MM   `Mb.  MM     ,M #
# .JMMmmmd9    `"bmmd"' .JML.    YM .JMMmmmdP'      .JMML.     .JMMmmmmMMM .AMA.   .AMMA.JMML.       `bmmmmd"' .JMML. .JMM.JMMmmmmMMM #
#######################################################################################################################################
#from alfabet import model

def smiles_to_bond_bde_df(smiles_x, drop_duplicates = False):
    df_alfabet_pred_BDE = model.predict([smiles_x], drop_duplicates = drop_duplicates) #BDE: bond desociation energy
    return df_alfabet_pred_BDE


#============================================================================================================================#
def smiles_to_bond_bde_dict(smiles_x, drop_duplicates = False):
    #--------------------------------------------------#
    # Input SMILES
    mol_x = Chem.MolFromSmiles(smiles_x)
    bond_idx_list_x = [bond.GetIdx() for bond in mol_x.GetBonds()]
    #--------------------------------------------------#
    # Get bond_index_bde_dict.
    # {idx : (bond_type, bond_bde) }
    try:
        df_alfabet_pred_BDE = model.predict([smiles_x], drop_duplicates = drop_duplicates) #BDE: bond desociation energy
        num_row = len(df_alfabet_pred_BDE)
        bond_index_dict = {one_row: df_alfabet_pred_BDE.loc[one_row, "bond_index"] for one_row in range(num_row)}
        bond_type_dict = {one_row: df_alfabet_pred_BDE.loc[one_row, "bond_type"] for one_row in range(num_row)}
        bond_bde_dict = {one_row: df_alfabet_pred_BDE.loc[one_row, "bde_pred"] for one_row in range(num_row)}
        bond_index_bde_dict = {bond_index_dict[one_row] : (bond_type_dict[one_row], bond_bde_dict[one_row])  for one_row in range(num_row)}
    except:
        bond_index_bde_dict = dict([])
    return bond_index_bde_dict

#============================================================================================================================#
def smiles_to_bond_len_dict(smiles_x):
    #--------------------------------------------------#
    # Input SMILES
    mol_x = Chem.MolFromSmiles(smiles_x)
    bond_idx_list_x = [bond.GetIdx() for bond in mol_x.GetBonds()]
    #--------------------------------------------------#
    # Get bond_index_len_dict.
    print(smiles_x)

    # Must do 
    # (1) remove radicals.
    # (2) remove stereochemistry. 
    # (3) use RandomCoords.
    # (4) use MolToMolBlock.

    # Problem 1: If contains radicals, AllChem.EmbedMolecule returns ERROR.
    try:
        AllChem.EmbedMolecule(mol_x, randomSeed = 42)
    except: 
        print("problem 1.")
        smiles_x = smiles_x.replace("*", "C")
        mol_x    = Chem.MolFromSmiles(smiles_x)

    AllChem.EmbedMolecule(mol_x, randomSeed = 42)
    # print(AllChem.EmbedMolecule(mol_x, randomSeed = 42))

    # Problem 2: If contains weird structure, AllChem.EmbedMolecule returns -1.
    if AllChem.EmbedMolecule(mol_x, randomSeed = 42) == -1:
        print("problem 2.")
        smiles_x = GetUnqSmi.UNQSMI(smiles_x)
        mol_x    = Chem.MolFromSmiles(smiles_x)
        AllChem.EmbedMolecule(mol_x, randomSeed = 42)
        # print(AllChem.EmbedMolecule(mol_x, randomSeed = 42))

    # Problem 3: If contains large structure, AllChem.EmbedMolecule returns -1.
    if AllChem.EmbedMolecule(mol_x, randomSeed = 42) == -1:
        print("problem 3.")
        AllChem.EmbedMolecule(mol_x, useRandomCoords = True)

    # Problem 5: If contains complicated structure, AllChem.EmbedMolecule still returns -1.
    # Last Defense, no more solutions :( , use MolToMolBlock to force getting conformer ID.
    if AllChem.EmbedMolecule(mol_x, useRandomCoords = True) == -1:
        print("problem 4.")
        mol_x = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol_x))


    # Problem 5: If contains complicated structure, AllChem.EmbedMolecule still returns -1.
    # Last Defense, no more solutions :( , use MolToMolBlock to force getting conformer ID.
    try:
        AllChem.MMFFOptimizeMolecule(mol_x)
    except:
        mol_x = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol_x))
        print("problem 5.")


    bond_index_len_dict = {}
    for bond in mol_x.GetBonds():
        bond_length = rdMolTransforms.GetBondLength(mol_x.GetConformer(), bond.GetBeginAtomIdx(),bond.GetEndAtomIdx() )
        bond_index_len_dict[bond.GetIdx()] = bond_length
    #--------------------------------------------------#
    #print(bond_index_len_dict)
    return bond_index_len_dict


#============================================================================================================================#
def smiles_to_bond_info(smiles_x):
    #--------------------------------------------------#
    # Input SMILES.
    mol_x = Chem.MolFromSmiles(smiles_x)
    bond_idx_list_x = [bond.GetIdx() for bond in mol_x.GetBonds()]
    atom_idx_list_x = [atom.GetIdx() for atom in mol_x.GetAtoms()]
    #--------------------------------------------------#
    # Get atom and bond attributes.
    df_atom_attributes, df_bond_attributes, atom_dict, bond_dict = smiles_to_attributes(smiles_x)
    df_bond_info = deepcopy(df_bond_attributes[["bond_idx", "BondTypeAsDouble", "bond_idx_tuple"]])
    #--------------------------------------------------#
    # Get bond energy for bonds with order >= 2.
    bond_type_SMARTS_dict = {1.0 : "-", 2.0 : "=", 3.0 : "#", 1.5 : ":"}

    bde_kJ_dict = { "C=C": 602, "C:C": 518, "C#C": 835, 
                    "C=O": 749, "O=P": 544, "C:N": 508, 
                    "C=N": 615, "C#N": 887, "C=S": 573, 
                    "N=O": 607, "P=S": 335, "C:O": 621,
                    "C:S": 456, 

                    
                    "C-C": 418, "C-O": 470, "C-S": 317, # on ring

                    "X": 420}
    bde_kcal_dict = {one_bond : bde_kJ_dict[one_bond] * 0.239006 for one_bond in bde_kJ_dict}

    bond_index_bde_dict = smiles_to_bond_bde_dict(smiles_x)
    bond_index_len_dict = smiles_to_bond_len_dict(smiles_x)

    #--------------------------------------------------#
    # Get bond string. (e.g., "C-O")
    bond_str_list = []
    bond_idx_str_dict = {}
    for bond_idx in bond_idx_list_x:
        bond_middle = bond_type_SMARTS_dict[bond_dict[bond_idx]["BondTypeAsDouble"]]
        bond_head = atom_dict[bond_dict[bond_idx]["bond_idx_tuple"][0]]["Symbol"]
        bond_tail = atom_dict[bond_dict[bond_idx]["bond_idx_tuple"][1]]["Symbol"]
        one_bond_str = bond_head + bond_middle + bond_tail
        one_bond_str_rev = one_bond_str[::-1]
        one_bond_str = min((one_bond_str, one_bond_str_rev))
        bond_str_list.append(one_bond_str)
        bond_idx_str_dict[bond_idx] = one_bond_str
    df_bond_info["bond_str"] = bond_str_list
    #--------------------------------------------------#
    # Get df_bond_info
    bond_bde_list = []
    bond_len_list = []
    for bond_idx in bond_idx_list_x:
        if bond_idx in bond_index_bde_dict.keys():
            bond_bde_list.append(bond_index_bde_dict[bond_idx][1])
        else:
            if bond_idx_str_dict[bond_idx] in bde_kcal_dict:
                bond_bde_list.append(bde_kcal_dict[bond_idx_str_dict[bond_idx]])
            else:
                print("\nCAUTION: Found Unknown Bond Type !!! #" + str(bond_idx) + ": " + bond_idx_str_dict[bond_idx])
                print(smiles_x, bond_index_bde_dict)
                bond_bde_list.append(bde_kcal_dict["X"])
        bond_len_list.append(bond_index_len_dict[bond_idx])
        
    df_bond_info["bond_bde"] = bond_bde_list
    df_bond_info["bond_len"] = bond_len_list
    #print(df_bond_info)

    return df_bond_info


#============================================================================================================================#
def smiles_to_bond_adjacency_matrix(smiles_x: str) -> numpy.ndarray: # size: (n_node, n_node)
    #--------------------------------------------------#
    # Input SMILES
    mol_x = Chem.MolFromSmiles(smiles_x)
    bond_idx_list_x = [bond.GetIdx() for bond in mol_x.GetBonds()]
    atom_idx_list_x = [atom.GetIdx() for atom in mol_x.GetAtoms()]
    #--------------------------------------------------#
    #
    bond_index_bde_dict = smiles_to_bond_bde_dict(smiles_x)
    df_bond_info = smiles_to_bond_info(smiles_x)
    #--------------------------------------------------#
    # bond_adjacency_matrix
    bond_adjacency_matrix = np.zeros((len(atom_idx_list_x), len(atom_idx_list_x)))
    for one_bond_tuple in df_bond_info.loc[:, "bond_idx_tuple"]:
        bond_adjacency_matrix[one_bond_tuple[0]][one_bond_tuple[1]] = 1
        bond_adjacency_matrix[one_bond_tuple[1]][one_bond_tuple[0]] = 1
    #print (bond_adjacency_matrix)
    return bond_adjacency_matrix


#============================================================================================================================#
def smiles_to_bond_graph_attributes(smiles_x: str) -> numpy.ndarray: # size: (n_node, n_node, dim_attr)
    #--------------------------------------------------#
    # Input SMILES
    mol_x = Chem.MolFromSmiles(smiles_x)
    bond_idx_list_x = [bond.GetIdx() for bond in mol_x.GetBonds()]
    atom_idx_list_x = [atom.GetIdx() for atom in mol_x.GetAtoms()]
    #--------------------------------------------------#
    #
    bond_index_bde_dict = smiles_to_bond_bde_dict(smiles_x)
    df_bond_info = smiles_to_bond_info(smiles_x)
    #--------------------------------------------------#
    # bond_graph_attributes
    bond_graph_attributes = np.zeros((len(atom_idx_list_x), len(atom_idx_list_x), 2))
    for one_bond_tuple in df_bond_info.loc[:, "bond_idx_tuple"]:
        bond_bde = df_bond_info.loc[ df_bond_info["bond_idx_tuple"] == one_bond_tuple, ["bond_bde"]].values
        bond_len = df_bond_info.loc[ df_bond_info["bond_idx_tuple"] == one_bond_tuple, ["bond_len"]].values
        bond_graph_attributes[one_bond_tuple[0]][one_bond_tuple[1]][0] = bond_bde
        bond_graph_attributes[one_bond_tuple[1]][one_bond_tuple[0]][0] = bond_bde
        bond_graph_attributes[one_bond_tuple[0]][one_bond_tuple[1]][1] = bond_len
        bond_graph_attributes[one_bond_tuple[1]][one_bond_tuple[0]][1] = bond_len
    #print (bond_graph_attributes)
    return bond_graph_attributes


#============================================================================================================================#
def smiles_to_bond_matrices(smiles_x: str) -> numpy.ndarray: # size: (n_node, n_node)
    #--------------------------------------------------#
    # Input SMILES
    mol_x = Chem.MolFromSmiles(smiles_x)
    bond_idx_list_x = [bond.GetIdx() for bond in mol_x.GetBonds()]
    atom_idx_list_x = [atom.GetIdx() for atom in mol_x.GetAtoms()]
    #--------------------------------------------------#
    #
    bond_index_bde_dict = smiles_to_bond_bde_dict(smiles_x)
    df_bond_info = smiles_to_bond_info(smiles_x)
    #--------------------------------------------------#
    # bond_adjacency_matrix
    bond_adjacency_matrix = np.zeros((len(atom_idx_list_x), len(atom_idx_list_x)))
    for one_bond_tuple in df_bond_info.loc[:, "bond_idx_tuple"]:
        bond_adjacency_matrix[one_bond_tuple[0]][one_bond_tuple[1]] = 1
        bond_adjacency_matrix[one_bond_tuple[1]][one_bond_tuple[0]] = 1
    #--------------------------------------------------#
    # bond_graph_attributes
    bond_graph_attributes = np.zeros((len(atom_idx_list_x), len(atom_idx_list_x), 2))
    for one_bond_tuple in df_bond_info.loc[:, "bond_idx_tuple"]:
        bond_bde = df_bond_info.loc[ df_bond_info["bond_idx_tuple"] == one_bond_tuple, ["bond_bde"]].values
        bond_len = df_bond_info.loc[ df_bond_info["bond_idx_tuple"] == one_bond_tuple, ["bond_len"]].values
        bond_graph_attributes[one_bond_tuple[0]][one_bond_tuple[1]][0] = bond_bde
        bond_graph_attributes[one_bond_tuple[1]][one_bond_tuple[0]][0] = bond_bde
        bond_graph_attributes[one_bond_tuple[0]][one_bond_tuple[1]][1] = bond_len
        bond_graph_attributes[one_bond_tuple[1]][one_bond_tuple[0]][1] = bond_len
    #--------------------------------------------------#
    # bond_general_info
    bond_general_info = df_bond_info[["BondTypeAsDouble", "bond_bde", "bond_len"]].to_numpy()

    #print (bond_adjacency_matrix)
    #print (bond_graph_attributes)
    return bond_adjacency_matrix, bond_graph_attributes, bond_general_info






#######################################################################################################################################
#      db  MMP""MM""YMM   .g8""8q. `7MMM.     ,MMF'  `7MM"""YMM `7MM"""YMM       db  MMP""MM""YMM`7MMF'   `7MF`7MM"""Mq. `7MM"""YMM   #
#     ;MM: P'   MM   `7 .dP'    `YM. MMMb    dPMM      MM    `7   MM    `7      ;MM: P'   MM   `7  MM       M   MM   `MM.  MM    `7   #
#    ,V^MM.     MM      dM'      `MM M YM   ,M MM      MM   d     MM   d       ,V^MM.     MM       MM       M   MM   ,M9   MM   d     #
#   ,M  `MM     MM      MM        MM M  Mb  M' MM      MM""MM     MMmmMM      ,M  `MM     MM       MM       M   MMmmdM9    MMmmMM     #
#   AbmmmqMA    MM      MM.      ,MP M  YM.P'  MM      MM   Y     MM   Y  ,   AbmmmqMA    MM       MM       M   MM  YM.    MM   Y  ,  #
#  A'     VML   MM      `Mb.    ,dP' M  `YM'   MM      MM         MM     ,M  A'     VML   MM       YM.     ,M   MM   `Mb.  MM     ,M  #
#.AMA.   .AMMA.JMML.      `"bmmd"' .JML. `'  .JMML.  .JMML.     .JMMmmmmMMM.AMA.   .AMMA.JMML.      `bmmmmd"' .JMML. .JMM.JMMmmmmMMM  #
#######################################################################################################################################

# 
def smiles_to_node_Morgan_dict(smiles_x, radius = 3):
    #--------------------------------------------------#
    # Inputs
    mol_x = Chem.MolFromSmiles(smiles_x)
    num_atom_x = len([atom for atom in mol_x.GetAtoms()])
    # Get Morgan dict
    Morgan_dict = {}
    Morgan_fingerprint = AllChem.GetMorganFingerprint(mol_x, radius = radius, bitInfo = Morgan_dict)
    #--------------------------------------------------#
    #[print(one_Morgan_info, Morgan_dict[one_Morgan_info]) for one_Morgan_info in Morgan_dict]
    atom_idx_list_x = [atom.GetIdx() for atom in mol_x.GetAtoms()]
    #--------------------------------------------------#
    Morgan_dict_reverse = {}
    for one_MorganFP in Morgan_dict:
        for one_substructure in Morgan_dict[one_MorganFP]:
            Morgan_dict_reverse[one_substructure] = one_MorganFP
    #print(Morgan_dict_reverse)
    #--------------------------------------------------#
    # Output
    node_Morgan_dict = {}
    for one_idx in atom_idx_list_x:
        node_Morgan_dict[one_idx] = []
    node_Morgan_list = []
    
    for one_idx in atom_idx_list_x:
        for one_radius in range(radius):
            one_substructure_mapping = (one_idx, one_radius)
            #print(one_substructure_mapping)
            if one_substructure_mapping not in Morgan_dict_reverse:
                continue
            node_Morgan_dict[one_idx].append(Morgan_dict_reverse[one_substructure_mapping])
    node_Morgan_list = [[one_node, node_Morgan_dict[one_node]] for one_node in node_Morgan_dict]
    #--------------------------------------------------#
    return node_Morgan_dict, node_Morgan_list



#============================================================================================================================#
# 
def smiles_list_to_all_Morgan_list(smiles_list: List[str], radius: int = 3, duplicates = False) -> List: 
    if duplicates:
        all_Morgan_list = []
        for one_smiles in smiles_list:
            _, node_Morgan_list = smiles_to_node_Morgan_dict(one_smiles, radius = radius)
            for one_node_info in node_Morgan_list:
                all_Morgan_list = all_Morgan_list + list(one_node_info[1])
    else:
        all_Morgan_set = set([])
        for one_smiles in smiles_list:
            _, node_Morgan_list = smiles_to_node_Morgan_dict(one_smiles, radius = radius)
            for one_node_info in node_Morgan_list:
                all_Morgan_set = all_Morgan_set.union((one_node_info[1]))
        all_Morgan_list = list(all_Morgan_set)
    return all_Morgan_list

#============================================================================================================================#
# 
def smiles_to_all_nodes_Morgan_list_dict(smiles_x: str, radius: int = 3) -> Tuple[List, Dict]:
    #--------------------------------------------------#
    # Inputs
    mol_x = Chem.MolFromSmiles(smiles_x)
    num_atom_x = len([atom for atom in mol_x.GetAtoms()])
    atom_idx_list_x = [atom.GetIdx() for atom in mol_x.GetAtoms()]
    #--------------------------------------------------#   
    all_Morgan = set([])
    all_nodes_Morgan_dict = dict([])
    node_Morgan_dict, node_Morgan_list = smiles_to_node_Morgan_dict(smiles_x, radius = radius)
    count_x = 0

    for one_atom_id in atom_idx_list_x:
        #print(count_x," out of ", len(atom_idx_list_x) )
        count_x += 1
        all_Morgan = all_Morgan.union(set(node_Morgan_dict[one_atom_id]))
    all_nodes_Morgan_dict = node_Morgan_dict
    return list(all_Morgan), all_nodes_Morgan_dict

#============================================================================================================================#
# 
def smiles_to_nodes_encodings(smiles_x: str, smiles_list, radius: int = 3, all_Morgan_list = None):
    #--------------------------------------------------#
    # Inputs
    mol_x = Chem.MolFromSmiles(smiles_x)
    num_atom_x = len([atom for atom in mol_x.GetAtoms()])
    atom_idx_list_x = [atom.GetIdx() for atom in mol_x.GetAtoms()]
    #--------------------------------------------------#
    # 
    if all_Morgan_list == None:
        all_Morgan_list = smiles_list_to_all_Morgan_list(smiles_list, radius = radius)
    _, all_nodes_Morgan_dict = smiles_to_all_nodes_Morgan_list_dict(smiles_x, radius = radius)
    node_attrs_dim = len(all_Morgan_list)
    nodes_encodings = []
    for one_node in atom_idx_list_x:
        Xi = [0] * node_attrs_dim
        Xi_Morgan_list = all_nodes_Morgan_dict[one_node]
        for one_Morgan in Xi_Morgan_list:
            Xi[all_Morgan_list.index(one_Morgan)] = Xi_Morgan_list.count(one_Morgan)
        nodes_encodings.append(Xi) # np.zeros((len(atom_idx_list_x), node_attrs_dim))

    nodes_encodings_np = np.array(nodes_encodings)
    print("nodes_encodings_np: ", nodes_encodings_np.shape, ", norm: ", np.linalg.norm(nodes_encodings_np, 1))
    print(nodes_encodings_np)
    return nodes_encodings_np

#============================================================================================================================#
# 
def smiles_to_vertices_attrs(smiles_x: str, smiles_list: List, radius = 3) -> numpy.ndarray:

    return smiles_to_nodes_encodings(smiles_x, smiles_list, radius = radius)









#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#        db     `7MM"""Mq.     `7MM"""YMM `7MMF'   `7MF'`7MN.   `7MF' .g8"""bgd  .M"""bgd    MMP""MM""YMM `7MM"""YMM   .M"""bgd MMP""MM""YMM           #
#       ;MM:      MM   `MM.      MM    `7   MM       M    MMN.    M .dP'     `M ,MI    "Y    P'   MM   `7   MM    `7  ,MI    "Y P'   MM   `7           #
#      ,V^MM.     MM   ,M9       MM   d     MM       M    M YMb   M dM'       ` `MMb.             MM        MM   d    `MMb.          MM                #
#     ,M  `MM     MMmmdM9        MM""MM     MM       M    M  `MN. M MM            `YMMNq.         MM        MMmmMM      `YMMNq.      MM                #
#     AbmmmqMA    MM             MM   Y     MM       M    M   `MM.M MM.         .     `MM         MM        MM   Y  , .     `MM      MM                #
#    A'     VML   MM             MM         YM.     ,M    M     YMM `Mb.     ,' Mb     dM         MM        MM     ,M Mb     dM      MM                #
#  .AMA.   .AMMA.JMML.         .JMML.        `bmmmmd"'  .JML.    YM   `"bmmmd'  P"Ybmmd"        .JMML.    .JMMmmmmMMM P"Ybmmd"     .JMML.              #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

from AP_convert import *
def AP_convert_test():

    # All of a,b and c are CoA.
    a="CC(C)(COP(=O)(O)OP(=O)(O)OCC1C(C(C(O1)N2C=NC3=C2N=CN=C3N)O)OP(=O)(O)O)C(C(=O)NCCC(=O)NCCS)O"
    b="O=C(NCCS)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)[C@H](O)[C@@H]3OP(=O)(O)O"
    c="CC(C)(COP(=O)(O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C2N=CN=C3N)O)OP(=O)(O)O)[C@H](C(=O)NCCC(=O)NCCS)O"
    #d="[H]N=c1c(c([H])nc(n1[H])C([H])([H])[H])C([H])([H])[n+]1cc(CC(N)C(=O)O)c2ccccc21"

    #============================================================================================================================#
    print ("\nTesting unique_canonical_smiles_AP()")
    print (unique_canonical_smiles_AP(a,False))
    print (unique_canonical_smiles_AP(b,False))
    print (unique_canonical_smiles_AP(c,False))
    #print (unique_canonical_smiles_AP("CC(C)CC(=O)C(=O)O"))
    #============================================================================================================================#
    print ("\nTesting canonical_smiles_AP()")
    print (canonical_smiles_AP(a))
    print (canonical_smiles_AP(b))
    print (canonical_smiles_AP(c))
    print (unique_canonical_smiles_AP(canonical_smiles_AP(c)))
    bad_ss_dict=dict([])
    #============================================================================================================================#
    print ("\nTesting MolFromSmiles_AP() and MolToSmiles_AP()")
    print (MolToSmiles_AP(MolFromSmiles_AP(a,bad_ss_dict),bad_ss_dict))
    print (MolToSmiles_AP(MolFromSmiles_AP(b,bad_ss_dict),bad_ss_dict))
    print (MolToSmiles_AP(MolFromSmiles_AP(c,bad_ss_dict),bad_ss_dict))
    #============================================================================================================================#
    print ("\nTesting unique_canonical_smiles_list_AP()")
    [print (x) for x in unique_canonical_smiles_list_AP([a,b,c])]
    #============================================================================================================================#
    print ("\nTesting canonical_smiles_list_AP()")
    bkgd=['O','CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCS','O=P(O)(O)O','O=C=O','N','CCCC(O)CC=O']
    bkgd_cmpd_list = canonical_smiles_list_AP(bkgd)
    print (bkgd_cmpd_list)
    #============================================================================================================================#
    print ("\nTesting pattern_matching_AP()")
    print (pattern_matching_AP("CC(=O)O","[CH3:1][C:2](=[O:3])[OH:4]"))
    print (pattern_matching_AP("CC(=O)OP(O)(O)=O","[C:2][O:6][P:7]([OH:8])([OH:9])=[O:10]"))
    print (pattern_matching_AP("CCO","[CH,CH2,CH3:2][C:4][OH:1]"))
    print (pattern_matching_AP("O","[O:1]-[C:2]"))




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MMF' `YMM'`7MM"""YMM `YMM'   `MM'    `7MM"""YMM `7MMF'   `7MF'`7MN.   `7MF' .g8"""bgd  .M"""bgd    MMP""MM""YMM `7MM"""YMM   .M"""bgd MMP""MM""YMM#
#    MM   .M'    MM    `7   VMA   ,V        MM    `7   MM       M    MMN.    M .dP'     `M ,MI    "Y    P'   MM   `7   MM    `7  ,MI    "Y P'   MM   `7#
#    MM .d"      MM   d      VMA ,V         MM   d     MM       M    M YMb   M dM'       ` `MMb.             MM        MM   d    `MMb.          MM     #
#    MMMMM.      MMmmMM       VMMP          MM""MM     MM       M    M  `MN. M MM            `YMMNq.         MM        MMmmMM      `YMMNq.      MM     #
#    MM  VMA     MM   Y  ,     MM           MM   Y     MM       M    M   `MM.M MM.         .     `MM         MM        MM   Y  , .     `MM      MM     #
#    MM   `MM.   MM     ,M     MM           MM         YM.     ,M    M     YMM `Mb.     ,' Mb     dM         MM        MM     ,M Mb     dM      MM     #
#  .JMML.   MMb.JMMmmmmMMM   .JMML.       .JMML.        `bmmmmd"'  .JML.    YM   `"bmmmd'  P"Ybmmd"        .JMML.    .JMMmmmmMMM P"Ybmmd"     .JMML.   #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#


def molecule_operation_test():

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # 
    unis = unique_canonical_smiles_AP
    #--------------------------------------------------#
    # Directory
    input_folder = Path("X_DataProcessing/")
    output_folder = Path("X_DataProcessing/HG_results/")
    output_temp_folder = Path("X_DataProcessing/HG_results/HG_temp/")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_temp_folder):
        os.makedirs(output_temp_folder)
    #--------------------------------------------------#
    dataset_nme = ["phosphatase", "kinase", "halogenase", "esterase"][0]
    compounds_file = "X00_" + dataset_nme + ".smiles"

    smiles_list_test = get_all_smiles_from_file(input_folder / compounds_file)
    smiles_x = "Cc1ncc(C[n+]2csc(CCOP(=O)([O-])OP(=O)(O)O)c2C)c(N)n1"
    #--------------------------------------------------#
    # Input SMILES
    # smiles_list_test = ["NC(COP(=O)(O)O)Cc1cnc[nH]1",
    #                     "NC(Cc1ccc(OP(=O)(O)O)cc1)C(=O)O",
    #                     "NC1C(OP(=O)(O)O)OC(CO)C(O)C1O",
    #                     "NCCC[P+](=O)O",]
    # smiles_x = smiles_list_test[3]


    #--------------------------------------------------#
    print("\ntest plot_smiles_list()")
    plot_smiles_list(   smiles_list_test,
                        fig_folder = output_folder, 
                        img_size = (520, 520), 
                        molsPerRow = 5, 
                        separate_img = False, 
                        fig_name = "test")


    #--------------------------------------------------#
    print("\ntest plot_smiles_annotated()")
    plot_smiles_annotated(  smiles_x, 
                            fig_folder = output_folder, 
                            img_size = (900, 900), 
                            fig_name = "test")


    #--------------------------------------------------#
    print("\ntest plot_Morgan_substructure()")
    MorganFP_svg = plot_Morgan_substructures(   smiles_x, 
                                                radius = 3, 
                                                fig_folder = output_folder, 
                                                img_size = (250, 250), 
                                                fig_name = "test")


    print("\ntest plot_Morgan_substructure_ordered()")
    MorganFP_svg = plot_Morgan_substructures_ordered(   smiles_x                   , 
                                                        radius = 3                 , 
                                                        fig_folder = output_folder , 
                                                        img_size = (240,180), 
                                                        fig_name = "test")


    #--------------------------------------------------#
    print("\ntest smiles_to_attributes()\n")
    df_atom_attributes, df_bond_attributes, _, _ = smiles_to_attributes(smiles_x)
    beautiful_print(df_atom_attributes, col = 30)
    beautiful_print(df_bond_attributes, col = 9)


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # 
    print("\nTesting smiles_to_bond_bde_dict()")
    bond_index_bde_dict = smiles_to_bond_bde_dict(smiles_x)
    beautiful_print(smiles_to_bond_bde_dict(smiles_x), col = 15)

    #--------------------------------------------------#
    print("\ntest smiles_to_bond_len_dict()")
    print(smiles_to_bond_len_dict(smiles_x))

    #--------------------------------------------------#
    print("\nTesting smiles_to_bond_info()")
    print(smiles_to_bond_info(smiles_x))

    #--------------------------------------------------#
    print("\nTesting smiles_to_bond_adjacency_matrix()")    
    print(smiles_to_bond_adjacency_matrix(smiles_x))

    #--------------------------------------------------#
    print("\nTesting smiles_to_bond_graph_attributes()")   
    print(smiles_to_bond_graph_attributes(smiles_x))


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # 
    print("\ntest smiles_to_node_Morgan_dict()")
    print(smiles_to_node_Morgan_dict(smiles_x, radius = 4))

    #--------------------------------------------------#
    print("\ntest smiles_list_to_all_Morgan()")
    print(smiles_list_to_all_Morgan_list(smiles_list_test))

    #--------------------------------------------------#
    print("\ntest smiles_to_all_nodes_Morgan_list_dict()")
    print(smiles_to_all_nodes_Morgan_list_dict(smiles_x))

    #--------------------------------------------------#
    print("\ntest smiles_to_nodes_encodings()")
    print(smiles_to_nodes_encodings(smiles_x, smiles_list_test, radius = 3))






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    .g8""8q. `7MMF'   `7MF'`7MMF' .g8"""bgd `7MMF' `YMM'     MMP""MM""YMM `7MM"""YMM   .M"""bgd MMP""MM""YMM                                          #
#  .dP'    `YM. MM       M    MM .dP'     `M   MM   .M'       P'   MM   `7   MM    `7  ,MI    "Y P'   MM   `7                                          #
#  dM'      `MM MM       M    MM dM'       `   MM .d"              MM        MM   d    `MMb.          MM                                               #
#  MM        MM MM       M    MM MM            MMMMM.              MM        MMmmMM      `YMMNq.      MM                                               #
#  MM.   Vb ,MP MM       M    MM MM.           MM  VMA             MM        MM   Y  , .     `MM      MM                                               #
#  `Mb.   XbdP' YM.     ,M    MM `Mb.     ,'   MM   `MM.           MM        MM     ,M Mb     dM      MM                                               #
#    `"bmmd"M.   `bmmmmd"'  .JMML. `"bmmmd'  .JMML.   MMb.       .JMML.    .JMMmmmmMMM P"Ybmmd"     .JMML.                                             #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def quick_test_1():
    #--------------------------------------------------#
    # Input SMILES
    smiles_x = "C1[C@@H](O)[C@@H](O)[C@H](On2c3c4nc(F)nc3N3OP(=O)(O3)Oc2n4)[C@@H](O)O1"
    smiles_x = "[2H]N[C@H]([C@H](C=O)O)[C@@H]([C@@H](CO)O)O"
    smiles_x = "CC1=C(CCC(=O)[O-])C2=[N+]3C1=Cc1c(CCC(=O)[O-])c(C)c4n1[Fe-2]31n3c(c(C)c(CCC(=O)[O-])c3=CC3=[N+]1C(=C4)C(C)=C3CCC(=O)[O-])=C2"
    smiles_x = "CC1=C(CCC(=O)[O-])C2=[N+]3C1=Cc1c(CCC(=O)[O-])c(C)c4n1[Fe-2]31n3c(c(C)c(CCC(=O)[O-])c3=CC3=[N+]1C(=C4)C(C)=C3CCC(=O)[O-])=C2"

    smiles_x = "[H]/N=C/[C@@H]1N=C1NC"
    mol_x    = Chem.MolFromSmiles(smiles_x)

    # for atom in mol_x.GetAtoms():
    #     atom.SetFormalCharge(0)
    #mol_x    = Chem.RemoveAllHs(mol_x)

    smiles_x = Chem.MolToSmiles(mol_x)
    print(Chem.MolToSmiles(mol_x))
    bond_idx_list_x = [bond.GetIdx() for bond in mol_x.GetBonds()]
    print(smiles_to_bond_info(smiles_x))
    #--------------------------------------------------#
    # Get bond_index_len_dict.
    print(smiles_x)

    # Problem 1: If contains radicals, AllChem.EmbedMolecule returns ERROR.
    try:
        AllChem.EmbedMolecule(mol_x, randomSeed = 42)
    except: 
        print("problem 1.")
        smiles_x = smiles_x.replace("*", "C")
        mol_x    = Chem.MolFromSmiles(smiles_x)

    AllChem.EmbedMolecule(mol_x, randomSeed = 42)
    # print(AllChem.EmbedMolecule(mol_x, randomSeed = 42))

    # Problem 2: If contains weird structure, AllChem.EmbedMolecule returns -1.
    if AllChem.EmbedMolecule(mol_x, randomSeed = 42) == -1:
        print("problem 2.")
        smiles_x = GetUnqSmi.UNQSMI(smiles_x)
        mol_x    = Chem.MolFromSmiles(smiles_x)
        AllChem.EmbedMolecule(mol_x, randomSeed = 42)
        # print(AllChem.EmbedMolecule(mol_x, randomSeed = 42))

    # Problem 3: If contains large structure, AllChem.EmbedMolecule returns -1.
    if AllChem.EmbedMolecule(mol_x, randomSeed = 42) == -1:
        print("problem 3.")
        AllChem.EmbedMolecule(mol_x, useRandomCoords = True)

    # Problem 4: If contains complicated structure, AllChem.EmbedMolecule still returns -1.
    # Last Defense, no more solutions :( , use MolToMolBlock to force getting conformer ID.
    if AllChem.EmbedMolecule(mol_x, useRandomCoords = True) == -1:
        print("problem 4.")
        mol_x = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol_x))


    # Problem 5: If contains complicated structure, AllChem.EmbedMolecule still returns -1.
    # Last Defense, no more solutions :( , use MolToMolBlock to force getting conformer ID.
    try:
        AllChem.MMFFOptimizeMolecule(mol_x)
    except:
        mol_x = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol_x))
        print("problem 5.")


    bond_index_len_dict = {}
    for bond in mol_x.GetBonds():
        bond_length = rdMolTransforms.GetBondLength(mol_x.GetConformer(), bond.GetBeginAtomIdx(),bond.GetEndAtomIdx() )
        bond_index_len_dict[bond.GetIdx()] = bond_length

    print(bond_index_len_dict)



#######################################################################################################################################
#######################################################################################################################################
if __name__ == '__main__':
    #AP_convert_test()
    #molecule_operation_test()
    quick_test_1()
























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







