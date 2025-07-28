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
import csv
import copy
import pickle
import argparse
import numpy as np
import pandas as pd
#--------------------------------------------------#
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
#--------------------------------------------------#
from AP_convert import *
from AP_convert import Get_Unique_SMILES

# Set up the Unique SMILES parser.
#GetUnqSmi = Get_Unique_SMILES(isomericSmiles = False, canonical = True , SMARTS_bool = False, removeAtomMapNumber = True)
GetUnqSmi  = Get_Unique_SMILES(isomericSmiles = True , canonical = False, SMARTS_bool = False, removeAtomMapNumber = False)

#SMILES_blacklist = ["[H][H]"]
SMILES_blacklist  = [""]



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#              `7MM"""Yb.      db  MMP""MM""YMM  db     `7MM"""YMM `7MM"""Mq.       db     `7MMM.     ,MMF'`7MM"""YMM                           M      #
#   __,          MM    `Yb.   ;MM: P'   MM   `7 ;MM:      MM    `7   MM   `MM.     ;MM:      MMMb    dPMM    MM    `7                           M      #
#  `7MM          MM     `Mb  ,V^MM.     MM     ,V^MM.     MM   d     MM   ,M9     ,V^MM.     M YM   ,M MM    MM   d                             M      #
#    MM          MM      MM ,M  `MM     MM    ,M  `MM     MM""MM     MMmmdM9     ,M  `MM     M  Mb  M' MM    MMmmMM                         `7M'M`MF'  #
#    MM          MM     ,MP AbmmmqMA    MM    AbmmmqMA    MM   Y     MM  YM.     AbmmmqMA    M  YM.P'  MM    MM   Y  ,                        VAM,V    #
#    MM  ,,      MM    ,dP'A'     VML   MM   A'     VML   MM         MM   `Mb.  A'     VML   M  `YM'   MM    MM     ,M                         VVV     #
#  .JMML.db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.JMML.     .JMML. .JMM.AMA.   .AMMA.JML. `'  .JMML..JMMmmmmMMM                          V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#  
# Import data and obtain a DataFrame.
def Get_processed_data_df(data_folder            , 
                          data_file              , 
                          data_file_binary       , 
                          binary_class_bool      , 
                          y_prpty_cls_threshd_3  , 
                          target_col_nme         , 
                          reaction_col_nme       , ):
    #====================================================================================================#
    def validate_reaction(one_reaction_string):
        try:
            rctt_str = one_reaction_string.split(">>")[0]
            prod_str = one_reaction_string.split(">>")[1]
            reaction_valid_bool = GetUnqSmi.ValidSMI(rctt_str) and GetUnqSmi.ValidSMI(prod_str)
            return reaction_valid_bool
        except:
            print( "!!! PROBLEMATIC reaction_string: ", one_reaction_string)
            return False
        
    #====================================================================================================#
    # Process regression data.
    processed_data_df = pd.read_csv(data_folder / data_file, header = 0)
    processed_data_df = processed_data_df.dropna(subset=[reaction_col_nme, target_col_nme])
    processed_data_df.reset_index(drop = True, inplace = True)
    # Remove instances that are NOT valid.
    processed_data_df["reaction_validity"] = processed_data_df[reaction_col_nme].apply(validate_reaction)
    processed_data_df = processed_data_df[processed_data_df.reaction_validity == True]
    processed_data_df.reset_index(drop = True, inplace = True)

    #====================================================================================================#
    # Process binary classification data.
    if binary_class_bool and ((data_folder / data_file_binary).exists()):
        processed_data_df_bi = pd.read_csv(data_folder / data_file_binary, header=0)
    else:
        print("binary classification file does not exist.")
        processed_data_df_bi = pd.read_csv(data_folder / data_file, header=0)
        processed_data_df_bi = processed_data_df_bi.dropna(subset=[reaction_col_nme, target_col_nme])
        processed_data_df_bi.reset_index(drop = True, inplace = True)
        processed_data_df_bi[target_col_nme] = [1 if one_cvsn > y_prpty_cls_threshd_3 else 0 
                                    for one_cvsn in list(processed_data_df_bi[target_col_nme])]
        
    # Remove instances that are NOT valid.
    processed_data_df_bi["reaction_validity"] = processed_data_df_bi[reaction_col_nme].apply(validate_reaction)
    processed_data_df_bi = processed_data_df_bi[processed_data_df_bi.reaction_validity == True]
    processed_data_df_bi.reset_index(drop = True, inplace = True)
    
    return processed_data_df, processed_data_df_bi


###################################################################################################################
###################################################################################################################
# Print the DataFrame obtained.
def beautiful_print(df):
    # Print the dataset in a well-organized format.
    with pd.option_context('display.max_rows'      , 20   , 
                           'display.min_rows'      , 20   , 
                           'display.max_columns'   , 6    , 
                           #"display.max_colwidth" , None ,
                           "display.width"         , None ,
                           "expand_frame_repr"     , True ,
                           "max_seq_items"         , None , ):  # more options can be specified
        # Once the display.max_rows is exceeded, 
        # the display.min_rows options determines 
        # how many rows are shown in the truncated repr.
        print(df)
    return 




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                      .g8""8q. `7MMF'   `7MF'MMP""MM""YMM `7MM"""Mq.`7MMF'   `7MF'MMP""MM""YMM  .M"""bgd                                       M      #
#                    .dP'    `YM. MM       M  P'   MM   `7   MM   `MM. MM       M  P'   MM   `7 ,MI    "Y                                       M      #
#   pd*"*b.          dM'      `MM MM       M       MM        MM   ,M9  MM       M       MM      `MMb.                                           M      #
#  (O)   j8          MM        MM MM       M       MM        MMmmdM9   MM       M       MM        `YMMNq.                                   `7M'M`MF'  #
#      ,;j9          MM.      ,MP MM       M       MM        MM        MM       M       MM      .     `MM                                     VAM,V    #
#   ,-='     ,,      `Mb.    ,dP' YM.     ,M       MM        MM        YM.     ,M       MM      Mb     dM                                      VVV     #
#  Ammmmmmm  db        `"bmmd"'    `bmmmmd"'     .JMML.    .JMML.       `bmmmmd"'     .JMML.    P"Ybmmd"                                        V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Output #2: Write reactions_properties_list
# Obtain a list of reactions_properties.
# [ SUBSTRATE_name, Y_Property_value, Y_Property_Class_#1, Y_Property_Class_#2, SMILES_str ]
# Y_Property_Class_#1: threshold 1e-5 (y_prpty_cls_threshd_2)
# Y_Property_Class_#2: threshold 1e-2 (provided by LEARNING PROTEIN SEQUENCE EMBEDDINGS USING INFORMATION FROM STRUCTURE)
# Get a compound_list including all compounds (compounds here use SMILES representation)
def output_formated_dataset(processed_data_df      , 
                            processed_data_df_bi   , 
                            y_prpty_cls_threshd_2  , 
                            target_col_nme         , 
                            reaction_col_nme       , 
                            data_folder            , 
                            output_folder          , 
                            output_file_1          , ):
    #====================================================================================================#   
    # Get all compound SMILES.
    raw_rxn_list              = [] # Reactions with non-unique SMILES (original SMILES/SMARTS from dataset).
    processed_rxn_list        = [] # Reactions with unique SMILES (processed SMILES/SMARTS from dataset).
    cmpd_smiles_list          = [] # List of Unique SMILES
    original_cmpd_smiles_list = [] # List of Original SMILES
    processed_data_df_row_num = processed_data_df.shape[0]

    # Process all reactions and get a list of unique SMILES.
    raw_rxn_list = list(processed_data_df[reaction_col_nme])
    for i in range(processed_data_df_row_num):
        one_raw_rxn_str = processed_data_df.loc[i, reaction_col_nme]
        rctt_list = one_raw_rxn_str.split(">>")[0].split(".")
        prod_list = one_raw_rxn_str.split(">>")[1].split(".")
        processed_rctt_list = [GetUnqSmi.UNQSMI(one_smiles) for one_smiles in rctt_list]
        processed_prod_list = [GetUnqSmi.UNQSMI(one_smiles) for one_smiles in prod_list]
        # Remove all SMILES_blacklist SMILES.
        processed_rctt_list = [one_smiles for one_smiles in processed_rctt_list if one_smiles not in SMILES_blacklist]
        processed_prod_list = [one_smiles for one_smiles in processed_prod_list if one_smiles not in SMILES_blacklist]
        prcessed_rxn_str    = ".".join(processed_rctt_list) + ">>" + ".".join(processed_prod_list)
        processed_rxn_list.append(prcessed_rxn_str)
        cmpd_smiles_list.extend(processed_rctt_list)
        cmpd_smiles_list.extend(processed_prod_list)
        original_cmpd_smiles_list.extend(rctt_list)
        original_cmpd_smiles_list.extend(prod_list)


    processed_data_df   ["PROCESSED_RXN_SMILES"] = processed_rxn_list
    processed_data_df_bi["PROCESSED_RXN_SMILES"] = processed_rxn_list

    cmpd_smiles_list          = list(set(cmpd_smiles_list))
    original_cmpd_smiles_list = list(set(original_cmpd_smiles_list))
    print("number of cmpd with duplicates removed: ", len(cmpd_smiles_list))
    print("number of original cmpd with duplicates removed: ", len(original_cmpd_smiles_list))

    cmpd_smiles_list_no_duplicates = []
    for cmpd in cmpd_smiles_list:
        if cmpd not in cmpd_smiles_list_no_duplicates:
            cmpd_smiles_list_no_duplicates.append(cmpd)
    cmpd_smiles_list = copy.deepcopy(cmpd_smiles_list_no_duplicates)
    del cmpd_smiles_list_no_duplicates
    #====================================================================================================#
    # 
    y_prpty_reg_list   = [ None for _ in range(len(raw_rxn_list))]
    y_prpty_cls_2_list = [ None for _ in range(len(raw_rxn_list))]

    #====================================================================================================#
    # 
    processed_data_df = processed_data_df.reset_index()  # make sure indexes pair with number of rows
    count_records     = 0 
    unique_rxn_list   = []
    unique_rxn_data   = []
    for index, row in processed_data_df.iterrows():
        rxn  = row['PROCESSED_RXN_SMILES']
        vals = row[target_col_nme]
        
        if vals != None: 
            if rxn not in unique_rxn_list:
                unique_rxn_list.append(rxn)
                unique_rxn_data.append([rxn, vals])
            count_records += 1

        y_prpty_reg_list[index] = vals

    print("len(processed_rxn_list): ", len(processed_rxn_list))
    print("len(unique_rxn_data):    ", len(unique_rxn_data   ))

    #====================================================================================================#
    # 
    processed_data_df_bi = processed_data_df_bi.reset_index()

    for index, row in processed_data_df_bi.iterrows():
        rxn   = row['PROCESSED_RXN_SMILES']
        vals  = row[target_col_nme]

        y_prpty_cls_2_list[index] = vals

    #====================================================================================================#
    # 
    # Get reactions_properties_list
    reactions_properties_list = []
    count_unknown = 0

    for one_rxn_raw, one_rxn_proc in zip(raw_rxn_list, processed_rxn_list):
        #--------------------------------------------------#
        y_prpty_reg   = y_prpty_reg_list[processed_rxn_list.index(one_rxn_proc)]
        y_prpty_cls_2 = y_prpty_cls_2_list[processed_rxn_list.index(one_rxn_proc)]
        y_prpty_cls_1 = None if y_prpty_reg == None else 1 if y_prpty_reg >= y_prpty_cls_threshd_2 else 0
        y_prpty_cls   = y_prpty_cls_1
        #--------------------------------------------------#
        reactions_properties = [one_rxn_raw, y_prpty_reg, y_prpty_cls_1, y_prpty_cls_2, one_rxn_proc]
        reactions_properties_list.append(reactions_properties)

    count_y = 0
    for one_cmpd_properties in reactions_properties_list:
        if one_cmpd_properties[1] != None:
            count_y += 1
    print("Number of Data Points: ", count_y)

    #====================================================================================================#
    # reactions_properties = [one_rxn_raw, y_prpty_reg, y_prpty_cls_1, y_prpty_cls_2, one_rxn_proc]
    # [Raw_Reaction_String, Y_Property_value, Y_Property_Class_#1, Y_Property_Class_#2, Processed_Reaction_String]
    # Y_Property_Class_#1: threshold 1e-5 (y_prpty_cls_threshd_2)
    # Y_Property_Class_#2: threshold 1e-2 (provided by binary classification dataset files if exists)

    #====================================================================================================#
    # Output a pickle file including all reactions_properties_list.
    pickle.dump( reactions_properties_list, open( output_folder / output_file_1, "wb" ) )

    #====================================================================================================#
    # Output a tsv file including all reactions_properties_list.
    output_file_2 = output_file_1.split(".")[0] + ".tsv"
    with open(output_folder / output_file_2, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')

        # Write the header
        writer.writerow(['one_rxn_raw', 'y_prpty_reg', 'y_prpty_cls_1', 'y_prpty_cls_2', 'one_rxn_proc'])

        # Write each row
        for reactions_properties in reactions_properties_list:
            writer.writerow(reactions_properties)


    return original_cmpd_smiles_list, cmpd_smiles_list, reactions_properties_list




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   pd""b.      `7MMF'     A     `7MF'`7MM"""Mq.  `7MMF'MMP""MM""YMM `7MM"""YMM       .M"""bgd `7MMM.     ,MMF'`7MMF'`7MMF'      `7MM"""YMM   .M"""bgd #
#  (O)  `8b       `MA     ,MA     ,V    MM   `MM.   MM  P'   MM   `7   MM    `7      ,MI    "Y   MMMb    dPMM    MM    MM          MM    `7  ,MI    "Y #
#       ,89        VM:   ,VVM:   ,V     MM   ,M9    MM       MM        MM   d        `MMb.       M YM   ,M MM    MM    MM          MM   d    `MMb.     #
#     ""Yb.         MM.  M' MM.  M'     MMmmdM9     MM       MM        MMmmMM          `YMMNq.   M  Mb  M' MM    MM    MM          MMmmMM      `YMMNq. #
#        88         `MM A'  `MM A'      MM  YM.     MM       MM        MM   Y  ,     .     `MM   M  YM.P'  MM    MM    MM      ,   MM   Y  , .     `MM #
#  (O)  .M' ,,       :MM;    :MM;       MM   `Mb.   MM       MM        MM     ,M     Mb     dM   M  `YM'   MM    MM    MM     ,M   MM     ,M Mb     dM #
#   bmmmd'  db        VF      VF      .JMML. .JMM..JMML.   .JMML.    .JMMmmmmMMM     P"Ybmmd"  .JML. `'  .JMML..JMML..JMMmmmmMMM .JMMmmmmMMM P"Ybmmd"  #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Output #3: Write a text file including all smiles strings.
def output_smiles(cmpd_smiles_list, output_folder, output_file_0):

    #====================================================================================================#
    # Output a pickle file including all compound SMILES in the processed reactions.
    pickle.dump(cmpd_smiles_list, open(output_folder / output_file_0, "wb"))
    
    # Also output a text file including all compound SMILES in the processed reactions.
    output_file_0 = output_file_0.split(".")[0] + ".smiles"
    with open(output_folder / output_file_0 , 'w') as f:
        count_x = 0
        for one_smiles in cmpd_smiles_list:
            f.write(one_smiles + "\n")
    return cmpd_smiles_list






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MMM.     ,MMF' .g8""8q. `7MM"""Mq.   .g8"""bgd       db     `7MN.   `7MF'     `7MM"""Yp, `7MMF'MMP""MM""YMM                                M      #
#    MMMb    dPMM .dP'    `YM. MM   `MM..dP'     `M      ;MM:      MMN.    M         MM    Yb   MM  P'   MM   `7                                M      #
#    M YM   ,M MM dM'      `MM MM   ,M9 dM'       `     ,V^MM.     M YMb   M         MM    dP   MM       MM                                     M      #
#    M  Mb  M' MM MM        MM MMmmdM9  MM             ,M  `MM     M  `MN. M         MM"""bg.   MM       MM                                 `7M'M`MF'  #
#    M  YM.P'  MM MM.      ,MP MM  YM.  MM.    `7MMF'  AbmmmqMA    M   `MM.M         MM    `Y   MM       MM                                   VAMAV    #
#    M  `YM'   MM `Mb.    ,dP' MM   `Mb.`Mb.     MM   A'     VML   M     YMM         MM    ,9   MM       MM                                    VVV     #
#  .JML. `'  .JMML. `"bmmd"' .JMML. .JMM. `"bmmmdPY .AMA.   .AMMA.JML.    YM       .JMMmmmd9  .JMML.   .JMML.                                   V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

# Get Morgan FP
#====================================================================================================#
# Morgan Function #1 (Not used here.)
def Get_Morgan_FP_1024(cmpd_smiles_list, output_folder, output_file_2):
    cmpd_SMILES_MorganFP1024_dict=dict([])
    for one_smiles in cmpd_smiles_list:
        rd_mol = Chem.MolFromSmiles(one_smiles)
        MorganFP = AllChem.GetMorganFingerprintAsBitVect(rd_mol, 2, nBits=1024)
        MorganFP_features = np.array(MorganFP)
        cmpd_SMILES_MorganFP1024_dict[one_smiles]=MorganFP_features
    pickle.dump(cmpd_SMILES_MorganFP1024_dict, open(output_folder / output_file_2,"wb"))
    return cmpd_SMILES_MorganFP1024_dict
#====================================================================================================#
# Morgan Function #2 (Not used here.)
def Get_Morgan_FP_2048(cmpd_smiles_list, output_folder, output_file_3):
    cmpd_SMILES_MorganFP2048_dict=dict([])
    for one_smiles in cmpd_smiles_list:
        rd_mol = Chem.MolFromSmiles(one_smiles)
        MorganFP = AllChem.GetMorganFingerprintAsBitVect(rd_mol, 2, nBits=2048)
        MorganFP_features = np.array(MorganFP)
        cmpd_SMILES_MorganFP2048_dict[one_smiles]=MorganFP_features
    pickle.dump(cmpd_SMILES_MorganFP2048_dict, open(output_folder / output_file_3,"wb"))
    return cmpd_SMILES_MorganFP2048_dict
#====================================================================================================#
# Morgan Function #3 (Not used here.)
from rdkit.Chem import Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
def ECFP_from_SMILES(smiles, radius=2, bit_len=1024, scaffold=0, index=None): # Not useful here !
    fps = np.zeros((len(smiles), bit_len))
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        arr = np.zeros((1,))
        try:
            if scaffold == 1:
                mol = MurckoScaffold.GetScaffoldForMol(mol)
            elif scaffold == 2:
                mol = MurckoScaffold.MakeScaffoldGeneric(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps[i, :] = arr
        except:
            print(smile)
            fps[i, :] = [0] * bit_len
    return pd.DataFrame(fps, index=(smiles if index is None else index)) 
#====================================================================================================#
# Morgan Function #4 (Not used here.)
def morgan_fingerprint(smiles: str, radius: int = 2, num_bits: int = 1024, use_counts: bool = False) -> np.ndarray:
    """
    Generates a morgan fingerprint for a smiles string.
    :param smiles: A smiles string for a molecule.
    :param radius: The radius of the fingerprint.
    :param num_bits: The number of bits to use in the fingerprint.
    :param use_counts: Whether to use counts or just a bit vector for the fingerprint
    :return: A 1-D numpy array containing the morgan fingerprint.
    """
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
    if use_counts:
        fp_vect = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    else:
        fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vect, fp)
    return fp 











#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#`7MM"""Yb.      db  MMP""MM""YMM  db       `7MM"""Mq.`7MM"""Mq.   .g8""8q.     .g8"""bgd`7MM"""YMM   .M"""bgd .M"""bgd `7MMF'`7MN.   `7MF' .g8"""bgd  #
#  MM    `Yb.   ;MM: P'   MM   `7 ;MM:        MM   `MM. MM   `MM..dP'    `YM. .dP'     `M  MM    `7  ,MI    "Y,MI    "Y   MM    MMN.    M .dP'     `M  #
#  MM     `Mb  ,V^MM.     MM     ,V^MM.       MM   ,M9  MM   ,M9 dM'      `MM dM'       `  MM   d    `MMb.    `MMb.       MM    M YMb   M dM'       `  #
#  MM      MM ,M  `MM     MM    ,M  `MM       MMmmdM9   MMmmdM9  MM        MM MM           MMmmMM      `YMMNq.  `YMMNq.   MM    M  `MN. M MM           #
#  MM     ,MP AbmmmqMA    MM    AbmmmqMA      MM        MM  YM.  MM.      ,MP MM.          MM   Y  , .     `MM.     `MM   MM    M   `MM.M MM.    `7MMF #
#  MM    ,dP'A'     VML   MM   A'     VML     MM        MM   `Mb.`Mb.    ,dP' `Mb.     ,'  MM     ,M Mb     dMMb     dM   MM    M     YMM `Mb.     MM  #
#.JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA. .JMML.    .JMML. .JMM. `"bmmd"'     `"bmmmd' .JMMmmmmMMM P"Ybmmd" P"Ybmmd"  .JMML..JML.    YM   `"bmmmdPY  #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

def X00_Data_Processing(binary_class_bool          , 
                        data_folder                , 
                        data_file                  , 
                        data_file_binary           , 
                        y_prpty_cls_threshd_2      , 
                        y_prpty_cls_threshd_3      , 
                        output_folder              , 
                        output_file_0              , 
                        output_file_1              , 
                        target_col_nme             , 
                        reaction_col_nme           , 
                        ):

    #--------------------------------------------------#
    # Step #1: Get_processed_data_df
    processed_data_df, processed_data_df_bi = \
        Get_processed_data_df(data_folder           , 
                              data_file             , 
                              data_file_binary      , 
                              binary_class_bool     , 
                              y_prpty_cls_threshd_3 , 
                              target_col_nme        , 
                              reaction_col_nme      , 
                              )
    print("\n\nprocessed_data_df: ")
    beautiful_print(processed_data_df)
    #beautiful_print(processed_data_df_bi)

    #--------------------------------------------------#
    # Step #2: Write reactions_properties_list
    _, cmpd_smiles_list, reactions_properties_list = \
        output_formated_dataset(processed_data_df      , 
                                processed_data_df_bi   , 
                                y_prpty_cls_threshd_2  , 
                                target_col_nme         , 
                                reaction_col_nme       , 
                                data_folder            , 
                                output_folder          , 
                                output_file_1          , 
                                )
    '''
    for one_list in reactions_properties_list:
        print(one_list)
        ''' 

    #--------------------------------------------------#
    # Step #3: Write cmpd_smiles_list
    cmpd_smiles_list = output_smiles(cmpd_smiles_list, output_folder, output_file_0)

    return











#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   `7MMM.     ,MMF'      db      `7MMF'`7MN.   `7MF'                 M             M             M                                                    #
#     MMMb    dPMM       ;MM:       MM    MMN.    M                   M             M             M                                                    #
#     M YM   ,M MM      ,V^MM.      MM    M YMb   M                   M             M             M                                                    #
#     M  Mb  M' MM     ,M  `MM      MM    M  `MN. M               `7M'M`MF'     `7M'M`MF'     `7M'M`MF'                                                #
#     M  YM.P'  MM     AbmmmqMA     MM    M   `MM.M                 VAMAV         VAMAV         VAMAV                                                  #
#     M  `YM'   MM    A'     VML    MM    M     YMM                  VVV           VVV           VVV                                                   #
#   .JML. `'  .JMML..AMA.   .AMMA..JMML..JML.    YM                   V             V             V                                                    #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

if __name__ == "__main__":

    ###################################################################################################################
    ###################################################################################################################
    # Args
    #--------------------------------------------------#
    # Inputs
    Step_code     = "RXN00_"
    dataset_index = 4
    #--------------------------------------------------#
    dataset_nme_list = ["sample_reaction_dataset" ,        # 0
                        "Reaction_Energy"         ,        # 1
                        "Log_RateConstant"        ,        # 2
                        "Phosphatase"             ,        # 3
                        "E2SN2"                   ,        # 4
                        ""                        ,        # 99

                        ]
    
    dataset_nme      = dataset_nme_list[dataset_index]
    #--------------------------------------------------#

    #                  --dataset_nme------------------------value_col-----------reaction_col---------dataset_path-----------------------------
    data_info_dict   = {"sample_reaction_dataset"       : ["ea"              , "AAM"              , "sample_reaction_regression.csv"         ,],  # 0
                        "Reaction_Energy"               : ["dh"              , "AAM"              , "Rad_6_Reaction_Energy.csv"              ,],  # 1
                        "Log_RateConstant"              : ["lograte"         , "AAM"              , "Logk_Rate_Const.csv"                    ,],  # 2
                        "Phosphatase"                   : ["Conversion"      , "AAM"              , "phosphatase.csv"                        ,],  # 3
                        "E2SN2"                         : ["ea"              , "AAM"              , "e2sn2.csv"                              ,],  # 4
                        ""                              : [""                , ""                 , ""                                       ,],  # 99

                       }

    #--------------------------------------------------#
    target_col_nme    = data_info_dict[dataset_nme][0]
    reaction_col_nme  = data_info_dict[dataset_nme][1]
    data_file         = data_info_dict[dataset_nme][2]
    #--------------------------------------------------#
    binary_class_bool = True # Whether there is a file for binary classification.
    #--------------------------------------------------#
    # Inputs
    data_folder = Path("RXN_DataProcessing/RXN00_datasets/")
    data_file_binary = data_file.replace(".csv", "_binary.csv") # Filename (binary classification) to be read.
    #--------------------------------------------------#
    # Settings
    y_prpty_cls_threshd_2   = 5e-1 # Used for type II  screening.
    y_prpty_cls_threshd_3   = 1    # Used for type III screening.
    #--------------------------------------------------#
    # Outputs
    output_folder   = Path("RXN_DataProcessing/") / ("RXN_task_intermediates_" + dataset_nme)
    output_file_0   = Step_code + dataset_nme + ".smiles"
    output_file_1   = Step_code + dataset_nme + "_reactions_properties_list.p"
    os.makedirs(output_folder, exist_ok = True)

    ###################################################################################################################
    ###################################################################################################################
    parser = argparse.ArgumentParser(
        description="Preprocesses the sequence datafile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--binary_class_bool"     ,  type = bool  , default = binary_class_bool     , help = "If there is a file for binary classification."   )
    parser.add_argument("--data_folder"           ,  type = Path  , default = data_folder           , help = "Path to the directory containing your datasets." )
    parser.add_argument("--data_file"             ,  type = str   , default = data_file             , help = "Filename to be read."                            )
    parser.add_argument("--dataset_nme"           ,  type = str   , default = dataset_nme           , help = "dataset_nme."                                    )
    parser.add_argument("--data_file_binary"      ,  type = str   , default = data_file_binary      , help = "Filename (binary classification) to be read."    )
    parser.add_argument("--y_prpty_cls_threshd_2" ,  type = float , default = y_prpty_cls_threshd_2 , help = "y_prpty_cls_threshd_2."                          )
    parser.add_argument("--y_prpty_cls_threshd_3" ,  type = float , default = y_prpty_cls_threshd_3 , help = "y_prpty_cls_threshd_3."                          )
    parser.add_argument("--output_folder"         ,  type = Path  , default = output_folder         , help = "Path to the directory containing output."        )
    parser.add_argument("--output_file_0"         ,  type = str   , default = output_file_0         , help = "Filename of output_file_0_0."                    )
    parser.add_argument("--output_file_1"         ,  type = str   , default = output_file_1         , help = "Filename of output_file_1."                      )
    parser.add_argument("--target_col_nme"        ,  type = str   , default = target_col_nme        , help = "target_col_nme in the data file."                )
    parser.add_argument("--reaction_col_nme"      ,  type = str   , default = reaction_col_nme      , help = "target_col_nme in the data file."                )
    parser.add_argument("--default_setting"       ,  type = str   , default = "y"                   , help = "Whether to use default settings [y/n]."          )
    args = parser.parse_args()

    #====================================================================================================#
    # The following code is to ensure that user can do #3 in the following list of choices for users.
    #   1. run the code directly in IDEs, with preset arguments (saved in data_info_dict).
    #   2. run the code in the command line with user specified arguments.
    #   3. run the code in the command line with just the `dataset_nme` input 
    #      and its corresponding preset arguments saved in data_info_dict.
    vars_dict = vars(args)
    dataset_nme = vars_dict["dataset_nme"]
    if vars_dict["default_setting"] == "y":
        vars_dict["data_file"             ] = data_info_dict[dataset_nme][2]
        vars_dict["output_file_0"         ] = Step_code + dataset_nme + "_compounds_smiles.p"
        vars_dict["output_file_1"         ] = Step_code + dataset_nme + "_reactions_properties_list.p"
        vars_dict["target_col_nme"        ] = data_info_dict[dataset_nme][0]
        vars_dict["reaction_col_nme"      ] = data_info_dict[dataset_nme][1]
    vars_dict.pop('dataset_nme'     , None) # Remove "dataset_nme" from vars_dict
    vars_dict.pop('default_setting' , None) # Remove "default_setting" from vars_dict

    #====================================================================================================#
    # Main
    #--------------------------------------------------#
    # Run Main
    X00_Data_Processing(**vars_dict)
    print("*" * 50)
    print(Step_code + " Done!")
    #====================================================================================================#



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




