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
import sys
import time
import pickle
import argparse
import subprocess
import numpy as np
import pandas as pd

###################################################################################################################
###################################################################################################################
# ECFP from CDK java file
def CDK_ECFP(smiles_str,ecfp_type,iteration_number):    
    try:
        # Use java file CDKImpl class to get ECFP from cmd line
        query_str1='java -cp .;cdk-2.2.jar CDKImpl ' + smiles_str + ' ' + ecfp_type + ' ' + str(iteration_number)
        print(query_str1)
        query_result = subprocess.check_output(query_str1, shell=True)
        query_result = query_result.decode("gb2312")
        query_result = query_result.replace('[', '')
        query_result = query_result.replace(']', '')
        query_result = query_result.replace(' ', '')
        query_result = query_result.replace('\n','')
        query_result = query_result.replace('\r','')
        if query_result!="":
            if query_result[-1]==',':
                query_result=query_result[0:-1]
            list_of_ecfp=query_result.split(",")
        else:
            list_of_ecfp=[]
    except Exception:
        print("problematic")
        list_of_ecfp=[]
    return list_of_ecfp 
#====================================================================================================#
def get_full_ecfp(smiles_str,ecfp_type,iteration_number):   
    # ECFP4 + itr2 or ECFP2 + itr1
    full_ecfp_list=[]
    for i in range(iteration_number+1):
        full_ecfp_list=full_ecfp_list+CDK_ECFP(smiles_str,ecfp_type,i)
    return full_ecfp_list

#====================================================================================================#
def generate_all_ECFPs(list_smiles,ecfp_type="ECFP2",iteration_number=1):
# return a list of ECFPs of all depth for a list of compounds (UNIQUE!!!)
    all_ecfps=set([])
    for smiles_a in list_smiles:
        discriptors = get_full_ecfp(smiles_a,ecfp_type,iteration_number)
        #print(smiles_a)
        all_ecfps = all_ecfps.union(set(discriptors))
    return all_ecfps

#====================================================================================================#
def generate_all_smiles_ecfps_dict(list_smiles,ecfp_type="ECFP2",iteration_number=1):
    all_smiles_ecfps_dict=dict([])
    for smiles_a in list_smiles:
        #print(smiles_a)
        all_smiles_ecfps_dict[smiles_a]=get_full_ecfp(smiles_a,ecfp_type,iteration_number)
    return all_smiles_ecfps_dict

#====================================================================================================#
def generate_all_smiles_ecfps_list_dict(list_smiles,ecfp_type="ECFP2",iteration_number=1):
    all_ecfps=set([])
    all_smiles_ecfps_dict=dict([])
    count_x = 0
    for smiles_a in list_smiles:
        print(count_x," out of ", len(list_smiles) )
        count_x+=1
        discriptors = get_full_ecfp(smiles_a,ecfp_type,iteration_number)
        #print(smiles_a)
        all_smiles_ecfps_dict[smiles_a]=discriptors
        all_ecfps=all_ecfps.union(set(discriptors))
    return list(all_ecfps),all_smiles_ecfps_dict

###################################################################################################################
###################################################################################################################
# Encode Compounds.
#====================================================================================================#
def list_smiles_to_ecfp_through_dict(smiles_list, all_smiles_ecfps_dict):
    ecfp_list=[]
    for one_smiles in smiles_list:
        ecfp_list=ecfp_list + all_smiles_ecfps_dict[one_smiles]
    return ecfp_list
#====================================================================================================#
def smiles_to_ECFP_vec( smiles_x, all_ecfps, all_smiles_ecfps_dict):
    dimension=len(all_ecfps)
    Xi = [0]*dimension
    Xi_ecfp_list = list_smiles_to_ecfp_through_dict( [smiles_x, ] ,all_smiles_ecfps_dict)
    for one_ecfp in Xi_ecfp_list:
        Xi[all_ecfps.index(one_ecfp)]=Xi_ecfp_list.count(one_ecfp)
    return np.array(Xi)
#====================================================================================================#
def Get_ECFPs_encoding(X_cmpd_representations, all_ecfps, all_smiles_ecfps_dict):
    X_cmpd_encodings=[]
    for one_smiles in X_cmpd_representations:
        one_cmpd_encoding = smiles_to_ECFP_vec(one_smiles, all_ecfps, all_smiles_ecfps_dict) # compound_encoding
        X_cmpd_encodings.append(one_cmpd_encoding)
    return X_cmpd_encodings
#====================================================================================================#
def Get_ECFPs_encoding_dict(X_cmpd_smiles_list, all_ecfps, all_smiles_ecfps_dict):
    X_cmpd_encodings_dict=dict([])
    max_encoding_count = 0
    for one_smiles in X_cmpd_smiles_list:
        one_cmpd_encoding = smiles_to_ECFP_vec(one_smiles, all_ecfps, all_smiles_ecfps_dict) # compound_encoding
        X_cmpd_encodings_dict[one_smiles] = one_cmpd_encoding
        max_encoding_count = max(one_cmpd_encoding) if max(one_cmpd_encoding)>max_encoding_count else max_encoding_count
    print("max_encoding_count (used in Cross-Attention): ", max_encoding_count)
    return X_cmpd_encodings_dict, max_encoding_count
#====================================================================================================#
def Get_JTVAE_encoding(X_cmpd_representations, cmpd_SMILES_JTVAE_dict):
    X_cmpd_encodings=[]
    for one_smiles in X_cmpd_representations:
        one_cmpd_encoding = cmpd_SMILES_JTVAE_dict[one_smiles] # compound_encoding
        X_cmpd_encodings.append(one_cmpd_encoding)
    return X_cmpd_encodings
#====================================================================================================#
def Get_JTVAE_encoding_dict(X_cmpd_representations, cmpd_SMILES_JTVAE_dict):
    return cmpd_SMILES_JTVAE_dict
#====================================================================================================#
def Get_Morgan_encoding(X_cmpd_representations, cmpd_SMILES_Morgan1024_dict):
    X_cmpd_encodings=[]
    for one_smiles in X_cmpd_representations:
        one_cmpd_encoding = cmpd_SMILES_Morgan1024_dict[one_smiles] # compound_encoding
        X_cmpd_encodings.append(one_cmpd_encoding)
    return X_cmpd_encodings
#====================================================================================================#
def Get_Morgan_encoding(X_cmpd_representations, cmpd_SMILES_Morgan1024_dict):
    return cmpd_SMILES_Morgan1024_dict

###################################################################################################################
###################################################################################################################
def RXN02A_Get_Cmpd_Encodings(data_folder, 
                            properties_file,
                            cmpd_encodings,
                            data_folder_2,
                            cmpd_JTVAE_file,
                            cmpd_Morgan1024_file,
                            cmpd_Morgan2048_file,
                            i_o_put_file_1,
                            i_o_put_file_2,
                            output_folder,
                            output_file_1,
                            ref_dict_file,
                            ):
    
    #====================================================================================================#
    # Get cmpd_properties_list.
    with open( data_folder / properties_file, 'rb') as cmpd_properties:
        cmpd_properties_list = pickle.load(cmpd_properties) # [[one_compound, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
    #====================================================================================================#
    # Get all SMILES.
    all_smiles_list=[]
    for one_list_prpt in cmpd_properties_list:
        all_smiles_list.append(one_list_prpt[-1])
    #print(all_smiles_list)


    #====================================================================================================#
    # Get all compounds ECFP encoded.

    if cmpd_encodings == "ECFP2" or cmpd_encodings == "ECFP4"  or cmpd_encodings == "ECFP6" :
        if os.path.exists(data_folder / i_o_put_file_1) and os.path.exists(data_folder / i_o_put_file_2) :
            print("Prepared dicts already exist.")
            ref_dict = i_o_put_file_2
            pass
        else:
            ref_dict = Path("RXN02_temp") / ref_dict_file


        if os.path.exists(data_folder / ref_dict):
            with open( data_folder / ref_dict, 'rb') as smiles_ecfps_dict_ref:
                smiles_ecfps_dict_ref = pickle.load(smiles_ecfps_dict_ref)

            all_ecfps_ref = set([]) 
            for one_smiles in all_smiles_list:
                if one_smiles in smiles_ecfps_dict_ref.keys():
                    one_list_descriptors = smiles_ecfps_dict_ref[one_smiles]
                    all_ecfps_ref = all_ecfps_ref.union(set(one_list_descriptors))

            all_smiles_ecfps_dict_ref = dict([])
            for one_smiles in all_smiles_list:
                if one_smiles in smiles_ecfps_dict_ref.keys():
                    all_smiles_ecfps_dict_ref[one_smiles] = smiles_ecfps_dict_ref[one_smiles]

            all_smiles_list_rest = [x for x in all_smiles_list if x not in list(smiles_ecfps_dict_ref.keys())]

            all_ecfps, all_smiles_ecfps_dict = generate_all_smiles_ecfps_list_dict(all_smiles_list_rest, ecfp_type = cmpd_encodings, iteration_number=int(int(ECFP_type)/2))

            all_ecfps = list(set(all_ecfps).union(all_ecfps_ref))

            all_smiles_ecfps_dict = {**all_smiles_ecfps_dict, **all_smiles_ecfps_dict_ref}

        else:
            all_ecfps, all_smiles_ecfps_dict = generate_all_smiles_ecfps_list_dict(all_smiles_list, ecfp_type = cmpd_encodings, iteration_number=int(int(ECFP_type)/2))

        pickle.dump(all_ecfps, open(data_folder / i_o_put_file_1,"wb") )
        pickle.dump(all_smiles_ecfps_dict, open(data_folder / i_o_put_file_2,"wb"))
    

    #====================================================================================================#
    # Encode Compounds.
    # cmpd_encodings_list = ["ECFP2", "ECFP4", "ECFP6", "JTVAE", "MorganFP"]
    if cmpd_encodings in ["ECFP2", "ECFP4", "ECFP6",] :
        with open( data_folder / i_o_put_file_1, 'rb') as all_ecfps:
            all_ecfps = pickle.load(all_ecfps)
        with open( data_folder / i_o_put_file_2, 'rb') as all_smiles_ecfps_dict:
            all_smiles_ecfps_dict = pickle.load(all_smiles_ecfps_dict)
        X_cmpd_encodings_dict, _ = Get_ECFPs_encoding_dict(all_smiles_list, all_ecfps, all_smiles_ecfps_dict)
    if cmpd_encodings == "JTVAE":
        with open( data_folder_2 / cmpd_JTVAE_file, 'rb') as cmpd_JTVAE_info:
            cmpd_SMILES_JTVAE_dict = pickle.load(cmpd_JTVAE_info)
        X_cmpd_encodings_dict = cmpd_SMILES_JTVAE_dict
    if cmpd_encodings == "MorganFP":
        with open( data_folder / cmpd_Morgan1024_file, 'rb') as cmpd_Morgan1024_info:
            cmpd_SMILES_Morgan1024_dict = pickle.load(cmpd_Morgan1024_info)
        X_cmpd_encodings_dict = cmpd_SMILES_Morgan1024_dict
    #====================================================================================================#
    pickle.dump( X_cmpd_encodings_dict, open( output_folder / output_file_1, "wb" ) )
    #====================================================================================================#
    print(Step_code + " Done!")









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
    Step_code = "RXN02A_"
    dataset_nme_list = ["sample_reaction_dataset" ,        # 0
                        ""                        ,        # 99

                        ]

    dataset_nme     = dataset_nme_list[0]
    data_folder     = Path("RXN_DataProcessing/")
    smiles_file     = "RXN00_" + dataset_nme + "_compounds_smiles.smiles"
    properties_file = "RXN00_" + dataset_nme + "_reactions_properties_list.p"
    ref_dict_file   = "RXN02A_" + "[NAME]" + "_all_cmpds_ecfps6_dict.p"
    #--------------------------------------------------#
    # Select compound encodings
    # compound encodings-----[0]------[1]------[2]-----
    cmpd_encodings_list = ["ECFP2", "ECFP4", "ECFP6", ]
    cmpd_encodings      = cmpd_encodings_list[2]
    #---------- ECFP
    ECFP_type = cmpd_encodings[-1] if cmpd_encodings in ["ECFP2", "ECFP4", "ECFP6",] else 0 # 2, 4, 6
    #--------------------------------------------------#
    # In/Outputs
    i_o_put_file_1 = Path( Path("RXN02_temp") / Path(Step_code + dataset_nme + "_all_ecfps" + str(ECFP_type) + ".p"))
    i_o_put_file_2 = Path( Path("RXN02_temp") / Path(Step_code + dataset_nme + "_all_cmpds_ecfps" + str(ECFP_type) + "_dict.p"))
    #--------------------------------------------------#
    # Outputs
    output_folder = Path("RXN_DataProcessing/")
    output_file_1 = Step_code + dataset_nme + "_" + cmpd_encodings  + "_" + "encodings_dict.p"

    ###################################################################################################################
    ###################################################################################################################

    parser = argparse.ArgumentParser(
        description="Preprocesses the sequence datafile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_folder"          , type = Path, default = data_folder           , help = "Path to the directory containing your datasets." ) 
    parser.add_argument("--properties_file"      , type = str , default = properties_file       , help = "properties_file."                                ) 
    parser.add_argument("--cmpd_encodings"       , type = str , default = cmpd_encodings        , help = "cmpd_encodings."                                 ) 
    parser.add_argument("--i_o_put_file_1"       , type = str , default = i_o_put_file_1        , help = "i_o_put_file_1."                                 ) 
    parser.add_argument("--i_o_put_file_2"       , type = str , default = i_o_put_file_2        , help = "i_o_put_file_2."                                 ) 
    parser.add_argument("--output_folder"        , type = str , default = output_folder         , help = "output_folder."                                  ) 
    parser.add_argument("--output_file_1"        , type = str , default = output_file_1         , help = "output_file_1."                                  ) 
    parser.add_argument("--ref_dict_file"        , type = str , default = ref_dict_file         , help = "ref_dict_file."                                  ) 
    parser.add_argument("--dataset_nme"          , type = str , default = dataset_nme           , help = "dataset_nme."                                    ) 
    parser.add_argument("--default_setting"      , type = str , default = "y"                   , help = "Whether to use default settings [y/n]."          ) 
    args = parser.parse_args()
    #====================================================================================================#
    # Simplify the input
    vars_dict = vars(args)
    dataset_nme = vars_dict["dataset_nme"]
    if vars_dict["default_setting"] == "y":
        vars_dict["properties_file"        ] = "RXN00_" + dataset_nme + "_reactions_properties_list.p"
        vars_dict["i_o_put_file_1"         ] = Path( Path("RXN02_temp") / Path(Step_code + dataset_nme + "_all_ecfps" + str(ECFP_type) + ".p"))
        vars_dict["i_o_put_file_2"         ] = Path( Path("RXN02_temp") / Path(Step_code + dataset_nme + "_all_cmpds_ecfps" + str(ECFP_type) + "_dict.p"))
        vars_dict["output_file_1"          ] = Step_code + dataset_nme + "_" + cmpd_encodings  + "_" + "encodings_dict.p"
    vars_dict.pop('dataset_nme'    , None)
    vars_dict.pop('default_setting', None)
    #====================================================================================================#
    # Main
    RXN02A_Get_Cmpd_Encodings(**vars_dict)

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
