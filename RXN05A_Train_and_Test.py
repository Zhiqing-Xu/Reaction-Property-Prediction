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
import copy
import scipy
import torch
import pickle
import random
import argparse
import numpy as np
#--------------------------------------------------#
from torch import nn
#from torchvision import models
#from torchsummary import summary
#--------------------------------------------------#
from pathlib import Path
#--------------------------------------------------#
from ZX03_rxn_mpnn_args import TrainArgs
from ZX05_loss_functions import get_loss_func
#------------------------------
from rxn05_utils import *
from rxn05_run_train import run_train
#------------------------------
from rxn05A_Model_MPNN import *
from rxn05A_RXN_CMPD_Dataset import *
#--------------------------------------------------#








#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                    `7MMF'`7MN.   `7MF'`7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM  .M"""bgd                                                                #
#                      MM    MMN.    M    MM   `MM.  MM       M  P'   MM   `7 ,MI    "Y                                                                #
#   ,pP""Yq.           MM    M YMb   M    MM   ,M9   MM       M       MM      `MMb.                                                                    #
#  6W'    `Wb          MM    M  `MN. M    MMmmdM9    MM       M       MM        `YMMNq.                                                                #
#  8M      M8          MM    M   `MM.M    MM         MM       M       MM      .     `MM                                                                #
#  YA.    ,A9 ,,       MM    M     YMM    MM         YM.     ,M       MM      Mb     dM                                                                #
#   `Ybmmd9'  db     .JMML..JML.    YM  .JMML.        `bmmmmd"'     .JMML.    P"Ybmmd"                                                                 #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 


if __name__ == "__main__":

    # Main Training Func Start Here.
    Step_code        = "RXN05A_"


    #====================================================================================================#
    # Mostly Adjusted settings.

    # Run Train Settings.
    dataset_idx     = 4        # 
    encoding_idx    = 3        # | ECFP6  : 2   | MgFP6        : 3  | (Choose the encoding method.)
    run_type        = 0        # | split  : 0   | train & test : 1  | (Choose the mode of training.)
    split_type      = 0        # | random : 0   | compound     : 3  |（Data split method if applicable.）
    log_value_bool  = bool(0)  # Whether to take log value of the property.
    screen_bool     = bool(0)  # Whether to screen the dataset based on the property value.
    epoch_num       = 100      # 
    batch_size      = 56       # 
    lr_model        = 1e-4     # Default learning rate, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6.
    seed            = 0        # This random seed will be used for data split (if applicable), model initialization and training.

    
    # Model Architecture.
    last_hid   = 1280          # | 1024 |
    dropout    = 0.1           # | 0    |

    hyperparameters_dict = dict([])
    for one_hyperpara in [ "last_hid", "dropout"]:
        hyperparameters_dict[one_hyperpara] = locals()[one_hyperpara]


    # Reaction / Compound related hyperparameters.

    """
    All possible choices, (assuming bond messaging use directed message passing by default)
        1. d-MPNN + bond message + extra atom-level feat + extra molecule-level feat 
        2. u-MPNN + atom message + extra atom-level feat + extra molecule-level feat (u-MPNN-a-m)
        3. d-MPNN + bond message + extra atom-level feat 
        4. u-MPNN + atom message + extra atom-level feat (u-MPNN-a)
        5. d-MPNN + bond message (d-MPNN)
        6. u-MPNN + atom message (u-MPNN)
        7. NOT using MPNN, use only extra molecule-level feat (Mol_Feat)
    """

    MPNN_size     = "2160"
    cmpd_process  = ["u-MPNN-a", "u-MPNN", "d-MPNN", "u-MPNN-a-m", "Mol_Feat"][2]





    #====================================================================================================#
    # The following code is to ensure that user can do #2 in the following list of choices for users.
    #   1. run the code directly in IDEs, with preset arguments (saved in this file).
    # > 2. run the code in the terminal with user specified arguments.
    #   3. run the code in the terminal with just the `dataset_idx` input 
    #      and its corresponding preset arguments saved in this file.

    parser = argparse.ArgumentParser(
        description="Preprocesses the sequence datafile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--dataset_idx"     , type = int   , default = dataset_idx    , help = "dataset_idx"           )
    parser.add_argument("-c", "--encoding_idx"    , type = int   , default = encoding_idx   , help = "encoding_idx"          )
    parser.add_argument("-u", "--run_type"        , type = int   , default = run_type       , help = "run_type"              )
    parser.add_argument("-s", "--split_type"      , type = int   , default = split_type     , help = "split_type"            )
    parser.add_argument("-g", "--log_value_bool"  , type = bool  , default = log_value_bool , help = "log_value_bool"        )
    parser.add_argument("-t", "--screen_bool"     , type = bool  , default = screen_bool    , help = "screen_bool"           )
    parser.add_argument("-n", "--epoch_num"       , type = int   , default = epoch_num      , help = "epoch_num"             )
    parser.add_argument("-b", "--batch_size"      , type = int   , default = batch_size     , help = "batch_size"            )
    parser.add_argument("-l", "--lr_model"        , type = int   , default = lr_model       , help = "lr_model"              )
    parser.add_argument("-r", "--seed"            , type = int   , default = seed           , help = "seed"                  )
    parser.add_argument("-x", "--default_setting" , type = str   , default = "n"            , help = "default_setting [y/n]" )
    args = parser.parse_args()
    run_train_vars_dict = vars(args)

    # The default of default_setting is "n", which means the user will need to:
    #   - 

    #====================================================================================================#
    # The following code is to ensure that user can do #3 in the following list of choices for users.
    #   1. run the code directly in IDEs, with preset arguments (saved in this file).
    #   2. run the code in the terminal with user specified arguments.
    # > 3. run the code in the terminal with just the `dataset_idx` input 
    #      and its corresponding preset arguments saved in this file.

    if run_train_vars_dict["default_setting"] != "y":
        dataset_idx   = run_train_vars_dict["dataset_idx" ]
        encoding_idx  = run_train_vars_dict["encoding_idx"]
        lr_model      = run_train_vars_dict["lr_model"    ]


    #====================================================================================================#
    # Prediction NN settings
    NN_type_list   = ["Reg", "Clf"]
    NN_type        = NN_type_list[0]



        
    #====================================================================================================#
    # Seed.
    seed = seed # 42, 0, 1, 2, 3, 4
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #====================================================================================================#
    # If log_value_bool is True, screen_bool will be changed.
    if log_value_bool==True:
        screen_bool = True

    #--------------------------------------------------#
    # If value is "Clf", log_value_bool will be changed.
    if NN_type=="Clf":
        screen_bool = False # Actually Useless.
        log_value_bool = False

    #--------------------------------------------------#
    if os.name == 'nt' or platform == 'win32':
        pass
    else:
        print("Running on Linux, change epoch number to 200")
        epoch_num = 200
    


    #====================================================================================================#
    # Finalize
    if run_train_vars_dict["default_setting"] != "y":
        seed            = run_train_vars_dict["seed"           ]
        log_value_bool  = run_train_vars_dict["log_value_bool" ]
        split_type      = run_train_vars_dict["split_type"     ]
        epoch_num       = run_train_vars_dict["epoch_num"      ]
        batch_size      = run_train_vars_dict["batch_size"     ]


    print("\n" + "="*80)















    #====================================================================================================#
    # All Files used.

    # Dataset Folders.
    dataset_nme_list = ["sample_reaction_dataset" ,        # 0
                        "Reaction_Energy"         ,        # 1
                        "Log_RateConstant"        ,        # 2
                        "Phosphatase"             ,        # 3
                        "E2SN2"                   ,        # 4
                        ""                        ,        # 99

                        ]

    dataset_nme      = dataset_nme_list[dataset_idx]
    data_folder      = Path("RXN_DataProcessing/") / ("RXN_task_intermediates_" + dataset_nme)

    # Reaction Properties File.
    # properties_file contains a list of properties of the reactions.
    # reactions_properties = [[one_rxn_raw, y_prpty_reg, y_prpty_cls_1, y_prpty_cls_2, one_rxn_proc], ..., ...]
    #   - Raw_Reaction_String       : Raw reaction SMILES/SMARTS string in the original dataset. 
    #   - Y_Property_value          : properties values of interest (float).
    #   - Y_Property_Class_#1       : 
    #   - Y_Property_Class_#2       : 
    #   - Processed_Reaction_String : Processed reaction SMILES/SMARTS string (checked SMILES validity).
    properties_file  = "RXN00_" + dataset_nme + "_reactions_properties_list.p"


    # Encoding Files.
    encoding_file_list = ["RXN02A_" + dataset_nme + "_ECFP2_encodings_dict.p" ,  # 0
                          "RXN02A_" + dataset_nme + "_ECFP4_encodings_dict.p" ,  # 1
                          "RXN02A_" + dataset_nme + "_ECFP6_encodings_dict.p" ,  # 2
                          "RXN02B_" + dataset_nme + "_MgFP6_encodings_dict.p" ,  # 3
                         ]
    encoding_file      = encoding_file_list[encoding_idx]


    # Encoding Files.
    substruct_encoding_file = "RXN02C_" + dataset_nme + "_Cmpd_Graph_Attributes.p"

    
    # Result folder and result files.
    results_folder = Path("RXN_Results/" + Step_code +"intermediate_results/")
    output_file_0 = Step_code + "_all_X_y.p"
    output_file_header = Step_code + "_result_"

    #====================================================================================================#





    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Select properties (Y) of the model 
    screen_bool     = screen_bool
    clf_thrhld_type = 2 # 2: 1e-5, 3: 1e-2
    log_value_bool  = log_value_bool ##### !!!!! If value is True, screen_bool will be changed










    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #                          db                      `7MM"""Mq.                                                                                          #
    #                         ;MM:                       MM   `MM.                                                                                         #
    #   ,pP""Yq.             ,V^MM.   `7Mb,od8 .P"Ybmmm  MM   ,M9 ,6"Yb. `7Mb,od8 ,pP"Ybd  .gP"Ya `7Mb,od8                                                 #
    #  6W'    `Wb           ,M  `MM     MM' "':MI  I8    MMmmdM9 8)   MM   MM' "' 8I   `" ,M'   Yb  MM' "'                                                 #
    #  8M      M8           AbmmmqMA    MM     WmmmP"    MM       ,pm9MM   MM     `YMMMa. 8M""""""  MM                                                     #
    #  YA.    ,A9 ,,       A'     VML   MM    8M         MM      8M   MM   MM     L.   I8 YM.    ,  MM                                                     #
    #   `Ybmmd9'  db     .AMA.   .AMMA.JMML.   YMMMMMb .JMML.    `Moo9^Yo.JMML.   M9mmmP'  `Mbmmd'.JMML.                                                   #
    #                                         6'     dP                                                                                                    #
    #                                         Ybmmmd'                                                                                                      #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#


    #====================================================================================================#
    # Reaction side configuration.
    """
    u-MPNN-a-m : Using all features.            
                    MPNN                          [+]
                    Extra Molecule-level Features [+]
                    Extra Atom-level Features     [+]
                    Using Atom Message            [+] (atom_messages are by their nature undirected)

    u-MPNN-a   : MPNN using substructure encodings as atom features. 
                    MPNN                          [+]
                    Extra Molecule-level Features [-]
                    Extra Atom-level Features     [+]
                    Using Atom Message            [+] (atom_messages are by their nature undirected)

    d-MPNN     : Using directed MPNN, without using substructure encodings as atom features. 
                    MPNN                          [+]
                    Extra Molecule-level Features [-]
                    Extra Atom-level Features     [-]
                    Using Atom Message            [-] (directed message passing is set by default)

    u-MPNN     : Using undirected MPNN, without using substructure encodings as atom features. 
                    MPNN                          [+]
                    Extra Molecule-level Features [-]
                    Extra Atom-level Features     [-]
                    Using Atom Message            [+] (atom_messages are by their nature undirected)

    Mol_Feat   : Using molecule-level features as compound representation ONLY.
                    MPNN                          [-]

    All possible choices, (assuming bond messaging use directed message passing by default)
        1. d-MPNN + bond message + extra atom-level feat + extra molecule-level feat 
        2. u-MPNN + atom message + extra atom-level feat + extra molecule-level feat (u-MPNN-a-m)
        3. d-MPNN + bond message + extra atom-level feat 
        4. u-MPNN + atom message + extra atom-level feat (u-MPNN-a)
        5. d-MPNN + bond message (d-MPNN)
        6. u-MPNN + atom message (u-MPNN)
        7. NOT using MPNN, use only extra molecule-level feat (Mol_Feat)

    """

    # Compound side train arguments.
    """
    '--features_path'
       - Whether to use MPNN only
       - comment out -> only MPNN
       - keep -> DONT EXCLUDE mol-level extra features
    '--features_only'
       - Whether to use molecule-level features only
       - keep -> only mol-level extra features
       - comment out -> DONT EXCLUDE MPNN
    '--undirected'
       - Whether to use directed message passing 
       - (only w/ bond_messages, ERROR w/ atom_messages)
    '--no_atom_descriptor_scaling'
       - CANNOT set directly without providing extra features
    '--no_bond_features_scaling'
       - CANNOT set directly without providing extra features
    """


    arguments = ['--data_path'                                ,    ''                   ,
                 '--dataset_type'                             ,    'regression'         ,
                 '--hidden_size'                              ,    MPNN_size            ,
                 '--depth'                                    ,    '3'                  ,
 
                 '--atom_messages'                            , 
                 '--reaction'                                 , 

                 # DO NOT APPLY standardization on features
                 '--no_features_scaling'                      , 

                 # '--overwrite_default_atom_features'        , 
                 # '--overwrite_default_bond_features'        , 

                ]
    


    # Best Performance
    if   cmpd_process == "u-MPNN-a": 
        pass

    # Best for Testing
    elif cmpd_process == "u-MPNN": 
        substruct_encoding_file = None

    # Default `d-MPNN`
    #   - No extra molecule/atom level features
    #   - directed + bond message
    elif cmpd_process == "d-MPNN":
        arguments.remove('--atom_messages')
        substruct_encoding_file = None

    # u-MPNN + extra molecule-level features
    elif cmpd_process == "u-MPNN-a-m":
        arguments.append('--features_path')

    # Extra molecule-level features ONLY
    elif cmpd_process == "Mol_Feat": 
        arguments.append('--features_path')
        arguments.append('')
        arguments.append('--features_only')
        substruct_encoding_file = None

    # Exceptions
    else:
        raise Exception(f"{cmpd_process} compound encoder type unknown!" )



    # Use the "tap" package here, better than argparse
    X_TrainArgs = TrainArgs().parse_args(arguments)


    #X_TrainArgs.use_input_features = False







    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #               `7MM"""Mq. `7MM"""Mq.  `7MM"""YMM  `7MM"""Mq.       db      `7MM"""Mq.        db      MMP""MM""YMM `7MMF'  .g8""8q.   `7MN.   `7MF'    #
    #   __,           MM   `MM.  MM   `MM.   MM    `7    MM   `MM.     ;MM:       MM   `MM.      ;MM:     P'   MM   `7   MM  .dP'    `YM.   MMN.    M      #
    #  `7MM           MM   ,M9   MM   ,M9    MM   d      MM   ,M9     ,V^MM.      MM   ,M9      ,V^MM.         MM        MM  dM'      `MM   M YMb   M      #
    #    MM           MMmmdM9    MMmmdM9     MMmmMM      MMmmdM9     ,M  `MM      MMmmdM9      ,M  `MM         MM        MM  MM        MM   M  `MN. M      #
    #    MM           MM         MM  YM.     MM   Y  ,   MM          AbmmmqMA     MM  YM.      AbmmmqMA        MM        MM  MM.      ,MP   M   `MM.M      #
    #    MM  ,,       MM         MM   `Mb.   MM     ,M   MM         A'     VML    MM   `Mb.   A'     VML       MM        MM  `Mb.    ,dP'   M     YMM      #
    #  .JMML.db     .JMML.     .JMML. .JMM..JMMmmmmMMM .JMML.     .AMA.   .AMMA..JMML. .JMM..AMA.   .AMMA.   .JMML.    .JMML.  `"bmmd"'   .JML.    YM      #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 
    # Step 1. Create temp folder for results.
    results_sub_folder = Create_Temp_Folder(results_folder    , 
                                            encoding_file     , 
                                            dataset_nme       , 
                                            Step_code         , 
                                            NN_type           , 
                                            split_type        , 
                                            screen_bool       , 
                                            log_value_bool    , 
                                            clf_thrhld_type   , )







    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #                .g8""8q.   `7MMF'   `7MF'MMP""MM""YMM `7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM    `7MM"""Mq. `7MM"""Mq.  `7MMF'`7MN.   `7MF'MMP""MM""YMM #
    #              .dP'    `YM.   MM       M  P'   MM   `7   MM   `MM.  MM       M  P'   MM   `7      MM   `MM.  MM   `MM.   MM    MMN.    M  P'   MM   `7 #
    #  pd*"*b.     dM'      `MM   MM       M       MM        MM   ,M9   MM       M       MM           MM   ,M9   MM   ,M9    MM    M YMb   M       MM      #
    # (O)   j8     MM        MM   MM       M       MM        MMmmdM9    MM       M       MM           MMmmdM9    MMmmdM9     MM    M  `MN. M       MM      #
    #     ,;j9     MM.      ,MP   MM       M       MM        MM         MM       M       MM           MM         MM  YM.     MM    M   `MM.M       MM      #
    #  ,-='    ,,  `Mb.    ,dP'   YM.     ,M       MM        MM         YM.     ,M       MM           MM         MM   `Mb.   MM    M     YMM       MM      #
    # Ammmmmmm db    `"bmmd"'      `bmmmmd"'     .JMML.    .JMML.        `bmmmmd"'     .JMML.       .JMML.     .JMML. .JMM..JMML..JML.    YM     .JMML.    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 
    # Step 2. Output print (to print details of the model including,
    # dimensions of dataset, # of seqs, # of subs and hyperparameters of the model).
    output_print(dataset_nme            ,
                 results_sub_folder     ,
                 encoding_file          ,
                 log_value_bool         ,
                 screen_bool            ,
                 clf_thrhld_type        ,
                 split_type             ,
                 epoch_num              ,
                 batch_size             ,
                 lr_model               ,
                 NN_type                ,
                 seed                   ,
                 hyperparameters_dict   , )

    print("\n\ncmpd_process: "   , cmpd_process)
    print(    "MPNN_size   : "   , MPNN_size   )
    #print(    "X_TrainArgs : \n" , X_TrainArgs )






    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #   pd""b.          `7MM"""Yb.      db  MMP""MM""YMM  db           .M"""bgd `7MM"""Mq.`7MMF'      `7MMF'MMP""MM""YMM                                   #
    #  (O)  `8b           MM    `Yb.   ;MM: P'   MM   `7 ;MM:         ,MI    "Y   MM   `MM. MM          MM  P'   MM   `7                                   #
    #       ,89           MM     `Mb  ,V^MM.     MM     ,V^MM.        `MMb.       MM   ,M9  MM          MM       MM                                        #
    #     ""Yb.           MM      MM ,M  `MM     MM    ,M  `MM          `YMMNq.   MMmmdM9   MM          MM       MM                                        #
    #        88           MM     ,MP AbmmmqMA    MM    AbmmmqMA       .     `MM   MM        MM      ,   MM       MM                                        #
    #  (O)  .M'   ,,      MM    ,dP'A'     VML   MM   A'     VML      Mb     dM   MM        MM     ,M   MM       MM                                        #
    #   bmmmd'    db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.    P"Ybmmd"  .JMML.    .JMMmmmmMMM .JMML.   .JMML.                                      #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 
    # Step 3. Get all dataset for train/test/validate model (data split).
    # If random split, split_type = 0, go to Z05_split_data to adjust split ratio. (Find split_seqs_cmpd_idx_book())
    # If split_type = 1, 2 or 3, go to Z05_utils to adjust split ratio. (Find split_idx())
    X_tr_rxn, y_tr, \
    X_ts_rxn, y_ts, \
    X_va_rxn, y_va, \
    X_cmpd_encodings_dim, y_scalar, \
    X_cmpd_encodings_dict, = \
        tr_ts_va_for_NN(dataset_nme         ,  
                        data_folder         ,  
                        results_folder      ,  
                        results_sub_folder  ,  
                        encoding_file       ,  
                        properties_file     ,  
                        screen_bool         ,  
                        log_value_bool      ,  
                        split_type          ,  
                        NN_type             ,  
                        clf_thrhld_type     ,  
                        seed                , )
    







    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #       ,AM         `7MM"""Yb.      db  MMP""MM""YMM  db       .M"""bgd `7MM"""YMM MMP""MM""YMM                                                        #
    #      AVMM           MM    `Yb.   ;MM: P'   MM   `7 ;MM:     ,MI    "Y   MM    `7 P'   MM   `7                                                        #
    #    ,W' MM           MM     `Mb  ,V^MM.     MM     ,V^MM.    `MMb.       MM   d        MM                                                             #
    #  ,W'   MM           MM      MM ,M  `MM     MM    ,M  `MM      `YMMNq.   MMmmMM        MM                                                             #
    #  AmmmmmMMmm         MM     ,MP AbmmmqMA    MM    AbmmmqMA   .     `MM   MM   Y  ,     MM                                                             #
    #        MM   ,,      MM    ,dP'A'     VML   MM   A'     VML  Mb     dM   MM     ,M     MM                                                             #
    #        MM   db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.P"Ybmmd"  .JMMmmmmMMM   .JMML.                                                           #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#    
    # Step 4. Prepare smiles for molecule processing MPNN.

    X_tr_smiles_dataset, X_ts_smiles_dataset, X_va_smiles_dataset, X_PARAMS = \
        RXN_CMPD_Dataset_Main(X_tr_rxn                                          , 
                              X_ts_rxn                                          , 
                              X_va_rxn                                          , 
                              X_cmpd_encodings_dict                             , 
                              X_TrainArgs                                       , 
                              data_folder             = data_folder             ,  # folder that contains X02C_
                              extra_cmpd_feature_file = substruct_encoding_file ,  # atom/bond feature
                              )










    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #   M******         `7MM"""Yb.      db  MMP""MM""YMM  db     `7MMF'        .g8""8q.      db     `7MM"""Yb. `7MM"""YMM  `7MM"""Mq.                      #
    #  .M                 MM    `Yb.   ;MM: P'   MM   `7 ;MM:      MM        .dP'    `YM.   ;MM:      MM    `Yb. MM    `7    MM   `MM.                     #
    #  |bMMAg.            MM     `Mb  ,V^MM.     MM     ,V^MM.     MM        dM'      `MM  ,V^MM.     MM     `Mb MM   d      MM   ,M9                      #
    #       `Mb           MM      MM ,M  `MM     MM    ,M  `MM     MM        MM        MM ,M  `MM     MM      MM MMmmMM      MMmmdM9                       #
    #        jM           MM     ,MP AbmmmqMA    MM    AbmmmqMA    MM      , MM.      ,MP AbmmmqMA    MM     ,MP MM   Y  ,   MM  YM.                       #
    #  (O)  ,M9   ,,      MM    ,dP'A'     VML   MM   A'     VML   MM     ,M `Mb.    ,dP'A'     VML   MM    ,dP' MM     ,M   MM   `Mb.                     #
    #   6mmm9     db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.JMMmmmmMMM   `"bmmd"'.AMA.   .AMMA.JMMmmmdP' .JMMmmmmMMM .JMML. .JMM.                    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    # Step 5. Call DataLoader.
    train_loader, valid_loader, test_loader = \
        get_MPNN_DataLoader(X_tr_smiles_dataset   , 
                            y_tr                  , 
                            X_va_smiles_dataset   , 
                            y_va                  , 
                            X_ts_smiles_dataset   , 
                            y_ts                  , 
                            batch_size            , 
                            X_TrainArgs           , 
                            )


    loss_func = get_loss_func(X_TrainArgs)
    print("loss_func: ", loss_func)
    print("batch_size:", batch_size)




    # optimizer = build_optimizer(model, X_TrainArgs)
    # Learning rate schedulers
    # scheduler = build_lr_scheduler(optimizer, X_TrainArgs)






    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #     .6*"            .g8"""bgd       db      `7MMF'      `7MMF'          `7MMM.     ,MMF'  .g8""8q.   `7MM"""Yb.   `7MM"""YMM  `7MMF'                 #
    #   ,M'             .dP'     `M      ;MM:       MM          MM              MMMb    dPMM  .dP'    `YM.   MM    `Yb.   MM    `7    MM                   #
    #  ,Mbmmm.          dM'       `     ,V^MM.      MM          MM              M YM   ,M MM  dM'      `MM   MM     `Mb   MM   d      MM                   #
    #  6M'  `Mb.        MM             ,M  `MM      MM          MM              M  Mb  M' MM  MM        MM   MM      MM   MMmmMM      MM                   #
    #  MI     M8        MM.            AbmmmqMA     MM      ,   MM      ,       M  YM.P'  MM  MM.      ,MP   MM     ,MP   MM   Y  ,   MM      ,            #
    #  WM.   ,M9 ,,     `Mb.     ,'   A'     VML    MM     ,M   MM     ,M       M  `YM'   MM  `Mb.    ,dP'   MM    ,dP'   MM     ,M   MM     ,M            #
    #   WMbmmd9  db       `"bmmmd'  .AMA.   .AMMA..JMMmmmmMMM .JMMmmmmMMM     .JML. `'  .JMML.  `"bmmd"'   .JMMmmmdP'   .JMMmmmmMMM .JMMmmmmMMM            #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    # Step 6. Create NN model and Initialize
    print("\n\n\n>>> Initializing the model... ")
    #====================================================================================================#
    # Get the model.
    modelG1 = RXN_CP_MPNN_Model(X_TrainArgs = X_TrainArgs           , 
                                PARAMS      = X_PARAMS              , 
                                cmpd_dim    = X_cmpd_encodings_dim  , 
                                last_hid    = last_hid              , 
                                dropout     = dropout               , 
                                )

    print("Count Model Parameters: ", param_count_all(modelG1))
    #initialize_weights(model)

    #--------------------------------------------------#
    modelG1.double()
    modelG1.cuda()
    #--------------------------------------------------#
    print("\n\n" + "#"*80)
    print(modelG1)
    #model.float()
    #print( summary( model,[(seqs_max_len, NN_input_dim), (X_cmpd_encodings_dim, )] )  )
    #model.double()
    print("#"*80)
    #--------------------------------------------------#
    # Model Hyperparaters
    optimizer1 = torch.optim.Adam(modelG1.parameters(), lr = lr_model)
    criterion1 = nn.MSELoss()






    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #       .6*"         `7MM"""Mq.`7MMF'   `7MF'`7MN.   `7MF'    MMP""MM""YMM `7MM"""Mq.       db     `7MMF'`7MN.   `7MF                                  #
    #     ,M'              MM   `MM. MM       M    MMN.    M      P'   MM   `7   MM   `MM.     ;MM:      MM    MMN.    M                                   #
    #    ,Mbmmm.           MM   ,M9  MM       M    M YMb   M           MM        MM   ,M9     ,V^MM.     MM    M YMb   M                                   #
    #    6M'  `Mb.         MMmmdM9   MM       M    M  `MN. M           MM        MMmmdM9     ,M  `MM     MM    M  `MN. M                                   #
    #    MI     M8         MM  YM.   MM       M    M   `MM.M           MM        MM  YM.     AbmmmqMA    MM    M   `MM.M                                   #
    #    WM.   ,M9 ,,      MM   `Mb. YM.     ,M    M     YMM           MM        MM   `Mb.  A'     VML   MM    M     YMM                                   #
    #     WMbmmd9  db    .JMML. .JMM. `bmmmmd"'  .JML.    YM         .JMML.    .JMML. .JMM.AMA.   .AMMA.JMML..JML.    YM                                   #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    # Step 6. Now, train the model
    print("\n\n\n>>>  Training... ")
    print("="*80)


    # Use one single model here. 
    output_unknown = \
        run_train(model                =  modelG1             , 
                  optimizer            =  optimizer1          , 
                  criterion            =  criterion1          , 
                  epoch_num            =  epoch_num           , 
                  train_loader         =  train_loader        , 
                  valid_loader         =  valid_loader        , 
                  test_loader          =  test_loader         , 
                  y_scalar             =  y_scalar            , 
                  log_value_bool       =  log_value_bool      , 
                  screen_bool          =  screen_bool         , 
                  results_sub_folder   =  results_sub_folder  , 
                  output_file_header   =  output_file_header  , 
                  input_var_names_list =  ["cmpd_dataset", ]  , 
                  target_name          =  "y_property"        , 
                  )


    #########################################################################################################
    #########################################################################################################
    print("="*80)
    print("max_r: ", max(output_unknown))


































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




