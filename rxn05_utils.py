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
import re
import sys
import time
import copy
import scipy
import torch
import pickle
import random
import argparse
import subprocess
import numpy as np
import pandas as pd
#--------------------------------------------------#
from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm
#from torchvision import models
#from torchsummary import summary
#--------------------------------------------------#
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#--------------------------------------------------#
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
#--------------------------------------------------#
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeRegressor
#--------------------------------------------------#
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#--------------------------------------------------#
from Bio import SeqIO
from tqdm import tqdm
from ipywidgets import IntProgress
from pathlib import Path
from copy import deepcopy
#--------------------------------------------------#
from datetime import datetime
#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#
from rxn05_split_data import *

#--------------------------------------------------#

from ZX02_nn_utils import StandardScaler, normalize_targets







#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#               `7MM"""Mq. `7MM"""Mq.  `7MM"""YMM  `7MM"""Mq.       db      `7MM"""Mq.        db      MMP""MM""YMM `7MMF'  .g8""8q.   `7MN.   `7MF'    #
#   __,           MM   `MM.  MM   `MM.   MM    `7    MM   `MM.     ;MM:       MM   `MM.      ;MM:     P'   MM   `7   MM  .dP'    `YM.   MMN.    M      #
#  `7MM           MM   ,M9   MM   ,M9    MM   d      MM   ,M9     ,V^MM.      MM   ,M9      ,V^MM.         MM        MM  dM'      `MM   M YMb   M      #
#    MM           MMmmdM9    MMmmdM9     MMmmMM      MMmmdM9     ,M  `MM      MMmmdM9      ,M  `MM         MM        MM  MM        MM   M  `MN. M      #
#    MM           MM         MM  YM.     MM   Y  ,   MM          AbmmmqMA     MM  YM.      AbmmmqMA        MM        MM  MM.      ,MP   M   `MM.M      #
#    MM  ,,       MM         MM   `Mb.   MM     ,M   MM         A'     VML    MM   `Mb.   A'     VML       MM        MM  `Mb.    ,dP'   M     YMM      #
#  .JMML.db     .JMML.     .JMML. .JMM..JMMmmmmMMM .JMML.     .AMA.   .AMMA..JMML. .JMM..AMA.   .AMMA.   .JMML.    .JMML.  `"bmmd"'   .JML.    YM      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 
## Create Temp Folder for Saving Results
def Create_Temp_Folder(results_folder  , 
                       encoding_file   , 
                       dataset_nme     , 
                       Step_code       , 
                       NN_type         , 
                       split_type      , 
                       screen_bool     , 
                       log_value       , 
                       clf_thrhld_type , ):
    
    print("="*80)
    print("\n\n\n>>> Creating temporary subfolder and clear past empty folders... ")
    print("="*80)
    
    #====================================================================================================#
    # Get Timestamp.
    now         = datetime.now()
    #d_t_string = now.strftime("%Y%m%d_%H%M%S")
    d_t_string  = now.strftime("%m%d-%H%M%S")
    
    #====================================================================================================#
    # Check empty folders in the results_folder (failed runs can lead to lots of empty folders).
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_folder_contents = os.listdir(results_folder)
    count_non_empty_folder = 0
    '''
    for item in results_folder_contents:
        if os.path.isdir(results_folder / item):
            num_files = len(os.listdir(results_folder/item))
            if num_files in [1,2]:
                try:
                    for idx in range(num_files):
                        os.remove(results_folder / item / os.listdir(results_folder/item)[0])
                    os.rmdir(results_folder / item)
                    print("Remove empty folder " + item + "!")
                except:
                    print("Cannot remove empty folder " + item + "!")
            elif num_files == 0:
                try:
                    os.rmdir(results_folder / item)
                    print("Remove empty folder " + item + "!")
                except:
                    print("Cannot remove empty folder " + item + "!")
            else:
                count_non_empty_folder += 1
                '''
    print("Found " + str(count_non_empty_folder) + " non-empty folders: " + "!")
    print("="*80)

    #====================================================================================================#
    # Retrieve encoding info and embedding info.
    encoding_code = encoding_file.replace("RXN02A_" + dataset_nme + "_", "")
    encoding_code = encoding_file.replace("RXN02B_" + dataset_nme + "_", "")
    encoding_code = encoding_code.replace("_encodings_dict.p", "")


    #--------------------------------------------------#
    # Generate Temp Folder Name
    temp_folder_name = Step_code 
    temp_folder_name += d_t_string + "_"
    temp_folder_name += dataset_nme + "_"
    temp_folder_name += encoding_code + "_"
    temp_folder_name += NN_type.upper() + "_"
    temp_folder_name += "splt" + str(split_type) + "_"
    temp_folder_name += "scrn" + str(screen_bool)[0] + "_"
    temp_folder_name += "lg" + str(log_value)[0] + "_"
    temp_folder_name += "thrhld" + str(clf_thrhld_type) + "_"
    
    #--------------------------------------------------#
    # Create Temp Folder
    results_sub_folder=Path("RXN_Results/" + Step_code + "intermediate_results/" + temp_folder_name +"/")
    if not os.path.exists(results_sub_folder):
        os.makedirs(results_sub_folder)
    return results_sub_folder




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                .g8""8q.   `7MMF'   `7MF'MMP""MM""YMM `7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM    `7MM"""Mq. `7MM"""Mq.  `7MMF'`7MN.   `7MF'MMP""MM""YMM #
#              .dP'    `YM.   MM       M  P'   MM   `7   MM   `MM.  MM       M  P'   MM   `7      MM   `MM.  MM   `MM.   MM    MMN.    M  P'   MM   `7 #
#  pd*"*b.     dM'      `MM   MM       M       MM        MM   ,M9   MM       M       MM           MM   ,M9   MM   ,M9    MM    M YMb   M       MM      #
# (O)   j8     MM        MM   MM       M       MM        MMmmdM9    MM       M       MM           MMmmdM9    MMmmdM9     MM    M  `MN. M       MM      #
#     ,;j9     MM.      ,MP   MM       M       MM        MM         MM       M       MM           MM         MM  YM.     MM    M   `MM.M       MM      #
#  ,-='    ,,  `Mb.    ,dP'   YM.     ,M       MM        MM         YM.     ,M       MM           MM         MM   `Mb.   MM    M     YMM       MM      #
# Ammmmmmm db    `"bmmd"'      `bmmmmd"'     .JMML.    .JMML.        `bmmmmd"'     .JMML.       .JMML.     .JMML. .JMM..JMML..JML.    YM     .JMML.    #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

# Modify the print function AND print all info to a log.
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # Make the output to be visible immediately.
    def flush(self) :
        for f in self.files:
            f.flush()

# Print All Info to a text file.
def output_print(dataset_nme          ,   
                 results_sub_folder   ,   
                 encoding_file        ,   
                 log_value            ,   
                 screen_bool          ,   
                 clf_thrhld_type      ,   
                 split_type           ,   
                 epoch_num            ,   
                 batch_size           ,   
                 learning_rate        ,   
                 NN_type              ,   
                 seed                 ,   
                 hyperparameters_dict ,): 
    #====================================================================================================#
    orig_stdout = sys.stdout
    f = open(results_sub_folder / 'print_out.txt', 'w')
    sys.stdout = Tee(sys.stdout, f)
    #--------------------------------------------------#
    print("\n\n\n>>> Initializing hyperparameters and settings... ")
    print("="*80)
    #--------------------------------------------------#
    print("dataset_nme           : ", dataset_nme)
    print("encoding_file         : ", encoding_file)
    #--------------------------------------------------#
    print("log_value             : ", log_value       , " (Whether to use log values of Y.)" )
    print("screen_bool           : ", screen_bool     , " (Whether to remove zeroes.)"       )
    print("clf_thrhld_type       : ", clf_thrhld_type , " (Type 2: 1e-5, Type 3: 1e-2)"      )
    #--------------------------------------------------#
    print("split_type            : ", split_type)
    #--------------------------------------------------#
    print("NN_type               : ", NN_type)
    print("Random Seed           : ", seed)
    print("epoch_num             : ", epoch_num)
    print("batch_size            : ", batch_size)
    print("learning_rate         : ", learning_rate)
    #--------------------------------------------------#
    print("-"*80)
    for one_hyperpara in hyperparameters_dict:
        print(one_hyperpara, " "*(21-len(one_hyperpara)), ": ", hyperparameters_dict[one_hyperpara])
    print("="*80)
    return 





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   pd""b.          `7MM"""Yb.      db  MMP""MM""YMM  db           .M"""bgd `7MM"""Mq.`7MMF'      `7MMF'MMP""MM""YMM                                   #
#  (O)  `8b           MM    `Yb.   ;MM: P'   MM   `7 ;MM:         ,MI    "Y   MM   `MM. MM          MM  P'   MM   `7                                   #
#       ,89           MM     `Mb  ,V^MM.     MM     ,V^MM.        `MMb.       MM   ,M9  MM          MM       MM                                        #
#     ""Yb.           MM      MM ,M  `MM     MM    ,M  `MM          `YMMNq.   MMmmdM9   MM          MM       MM                                        #
#        88           MM     ,MP AbmmmqMA    MM    AbmmmqMA       .     `MM   MM        MM      ,   MM       MM                                        #
#  (O)  .M'   ,,      MM    ,dP'A'     VML   MM   A'     VML      Mb     dM   MM        MM     ,M   MM       MM                                        #
#   bmmmd'    db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.    P"Ybmmd"  .JMML.    .JMMmmmmMMM .JMML.   .JMML.                                      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#  

# Prepare Train/Test/Validation Dataset for NN model.
def tr_ts_va_for_NN(dataset_nme         ,  
                    data_folder         ,  
                    results_folder      ,  
                    results_sub_folder  ,  
                    encoding_file       ,  
                    properties_file     ,  
                    screen_bool         ,  
                    log_value           ,  
                    split_type          ,  
                    NN_type             ,  
                    clf_thrhld_type     ,  
                    seed                , ):



    ###################################################################################################################
    #                               `7MM"""YMM `7MMF'`7MMF'      `7MM"""YMM   .M"""bgd                                #
    #                                 MM    `7   MM    MM          MM    `7  ,MI    "Y                                #
    #                                 MM   d     MM    MM          MM   d    `MMb.                                    #
    #                                 MM""MM     MM    MM          MMmmMM      `YMMNq.                                #
    #                                 MM   Y     MM    MM      ,   MM   Y  , .     `MM                                #
    #                                 MM         MM    MM     ,M   MM     ,M Mb     dM                                #
    #                               .JMML.     .JMML..JMMmmmmMMM .JMMmmmmMMM P"Ybmmd"                                 #
    ###################################################################################################################
    # Get Input files
    print("\n\n\n>>> Load all input files and splitting the data... ")
    print("="*80)

    #====================================================================================================#
    # Get Compound Encodings from RXN02 pickles.
    with open( data_folder / encoding_file, 'rb') as cmpd_encodings:
        cmpd_encodings_dict = pickle.load(cmpd_encodings)

    #====================================================================================================#
    # Get rxn_properties_list.
    # [[one_rxn_raw, y_prpty_reg, y_prpty_cls_1, y_prpty_cls_2, one_rxn_proc], [], [], ...[] ]
    with open( data_folder / properties_file, 'rb') as rxn_properties:
        rxn_properties_list = pickle.load(rxn_properties)


    ###################################################################################################################
    #        `7MMF'`7MN.   `7MF'`7MMF'MMP""MM""YMM `7MMF'     db     `7MMF'      `7MMF'MMM"""AMV `7MM"""YMM           #
    #          MM    MMN.    M    MM  P'   MM   `7   MM      ;MM:      MM          MM  M'   AMV    MM    `7           #
    #          MM    M YMb   M    MM       MM        MM     ,V^MM.     MM          MM  '   AMV     MM   d             #
    #          MM    M  `MN. M    MM       MM        MM    ,M  `MM     MM          MM     AMV      MMmmMM             #
    #          MM    M   `MM.M    MM       MM        MM    AbmmmqMA    MM      ,   MM    AMV   ,   MM   Y  ,          #
    #          MM    M     YMM    MM       MM        MM   A'     VML   MM     ,M   MM   AMV   ,M   MM     ,M          #
    #        .JMML..JML.    YM  .JMML.   .JMML.    .JMML.AMA.   .AMMA.JMMmmmmMMM .JMML.AMVmmmmMM .JMMmmmmMMM          #
    ###################################################################################################################
    # Prepare for initializing X_data and y_data.
    #====================================================================================================#
    # For Regression Model.
    def Get_represented_X_y_data(rxn_properties_list     , 
                                 cmpd_encodings_dict     , 
                                 screen_bool             , 
                                 clf_thrhld_type         , 
                                 NN_type = "Reg"         , ):
        
        # rxn_properties_list: [ [one_rxn_raw, y_prpty_reg, y_prpty_cls_1, y_prpty_cls_2, one_rxn_proc], [], [], ...[] ]

        X_rxn_data            = []
        y_data                = []

        if NN_type == "Reg":
            for i in range(len(rxn_properties_list)):
                one_rxn_proc = rxn_properties_list[i][-1] # reaction string
                if not (screen_bool == True and rxn_properties_list[i][clf_thrhld_type] == False):
                    """
                    clf_thrhld_type                            -> 2: Threshold = 1e-5 (pre-set by user) | 3: Threshold = 1e-2 (Original)
                    rxn_properties_list[i]                     -> [one_rxn_raw, y_prpty_reg, y_prpty_cls_1, y_prpty_cls_2, one_rxn_proc]
                    rxn_properties_list[i][clf_thrhld_type]    -> y_prpty_cls OR y_prpty_cls2
                    """
                    X_rxn_data.append(one_rxn_proc)
                    y_data    .append(rxn_properties_list[i][1])
            
        if NN_type == "Clf":
            for i in range(len(rxn_properties_list)):
                one_rxn_proc = rxn_properties_list[i][-1] # reaction string\
                X_rxn_data.append(one_rxn_proc)
                y_data    .append(rxn_properties_list[i][clf_thrhld_type])


        return X_rxn_data, y_data



    X_rxn_data, y_data = \
        Get_represented_X_y_data(rxn_properties_list    , 
                                 cmpd_encodings_dict    , 
                                 screen_bool            , 
                                 clf_thrhld_type        , )

    #print(X_rxn_data)

    ###################################################################################################################
    #                      `7MM"""Mq. `7MM"""Mq.  `7MMF'`7MN.   `7MF'MMP""MM""YMM  .M"""bgd                           #
    #                        MM   `MM.  MM   `MM.   MM    MMN.    M  P'   MM   `7 ,MI    "Y                           #
    #                        MM   ,M9   MM   ,M9    MM    M YMb   M       MM      `MMb.                               #
    #                        MMmmdM9    MMmmdM9     MM    M  `MN. M       MM        `YMMNq.                           #
    #                        MM         MM  YM.     MM    M   `MM.M       MM      .     `MM                           #
    #                        MM         MM   `Mb.   MM    M     YMM       MM      Mb     dM                           #
    #                      .JMML.     .JMML. .JMM..JMML..JML.    YM     .JMML.    P"Ybmmd"                            #
    ###################################################################################################################
    # Check Data Processing Steps and Intermediates.

    #====================================================================================================#
    # Save seqs_embeddings and cmpd_encodings
    #save_dict = dict([])
    #save_dict["X_seqs_all_hiddens_dict"] = X_seqs_all_hiddens_dict
    #save_dict["X_cmpd_encodings"]        = X_cmpd_encodings
    #save_dict["y_data"]                  = y_data
    #pickle.dump( save_dict , open( results_folder / output_file_0, "wb" ) )
    #print("Done getting X_seqs_all_hiddens_dict, X_cmpd_encodings and y_data!")

    print("len(X_rxn_data): ", len(X_rxn_data), ", len(y_data): ", len(y_data) )

    #====================================================================================================#
    # Get size of some interested parameters.
    X_cmpd_encodings_dim = len(list(cmpd_encodings_dict.values())[0])
    X_rxn_num = len(rxn_properties_list)
    print("cmpd dimensions: ", X_cmpd_encodings_dim)
    print("rxn counts: ", X_rxn_num)


    #====================================================================================================#
    # Print the total number of data points.
    count_y = 0
    for one_rxn_properties in rxn_properties_list:
            if one_rxn_properties[1] != None:
                count_y += 1
    print("Number of Data Points (#y-values): ", count_y)




    ###################################################################################################################
    #                                  .M"""bgd `7MM"""Mq. `7MMF'      `7MMF'MMP""MM""YMM                             #
    #                                 ,MI    "Y   MM   `MM.  MM          MM  P'   MM   `7                             #
    #                                 `MMb.       MM   ,M9   MM          MM       MM                                  #
    #                                   `YMMNq.   MMmmdM9    MM          MM       MM                                  #
    #                                 .     `MM   MM         MM      ,   MM       MM                                  #
    #                                 Mb     dM   MM         MM     ,M   MM       MM                                  #
    #                                 P"Ybmmd"  .JMML.     .JMMmmmmMMM .JMML.   .JMML.                                #
    ###################################################################################################################
    # Get Separate SEQS index and SUBS index.
    """
    - split_type = 0, train/test split completely randomly selected

    
    """

    #====================================================================================================#
    def split_idx(X_num, train_split = 0.8, test_split = 0.1, random_state = 42): 
        # This function is used for split_type = 1, 2, 3, 4, 5. (NOT for random split.)
        # X_seqs_idx = y_seqs_idx = list(range(len(X_seqs_all_hiddens_list)))
        # X_cmpd_idx = y_cmpd_idx = list(range(len(cmpd_properties_list)))
        #--------------------------------------------------#
        print("train_split: ", train_split, ", test_split: ", test_split)
        X_idx = y_idx = list(range(X_num))
        X_tr_idx, X_ts_idx, y_tr_idx, y_ts_idx = train_test_split(X_idx, y_idx, test_size = (1-train_split), random_state = random_state)
        X_va_idx, X_ts_idx, y_va_idx, y_ts_idx = train_test_split(X_ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)) , random_state = random_state)
        return X_tr_idx, X_ts_idx, X_va_idx


    #====================================================================================================#
    # Initialize
    if split_type == 0:
        tr_idx_rxn, ts_idx_rxn, va_idx_rxn = split_idx(X_rxn_num, train_split = 0.8, test_split = 0.1, random_state = seed)

    else:
        raise Exception("WRONG SPLIT TYPE! ")


    #====================================================================================================#
    # Print the Index Pre-Split Results.
    print("-"*50)
    print("Use tr:ts:va = 8:1:1 splitting on both SEQs & SUBs.")
    print("These are only for split_type = 2, 3, 4, 5 : ")
    print("len(tr_idx_rxn): ", len(tr_idx_rxn) if len(tr_idx_rxn) != 0 else "N/A")
    print("len(ts_idx_rxn): ", len(ts_idx_rxn) if len(ts_idx_rxn) != 0 else "N/A")
    print("len(va_idx_rxn): ", len(va_idx_rxn) if len(va_idx_rxn) != 0 else "N/A")



    ###################################################################################################################
    #         `7MMF'`7MM"""Yb.`YMM'   `MP'       mm               `7MM"""Yb.      db  MMP""MM""YMM  db                #
    #           MM    MM    `Yb.VMb.  ,P         MM                 MM    `Yb.   ;MM: P'   MM   `7 ;MM:               #
    #           MM    MM     `Mb `MM.M'        mmMMmm ,pW"Wq.       MM     `Mb  ,V^MM.     MM     ,V^MM.              #
    #           MM    MM      MM   MMb           MM  6W'   `Wb      MM      MM ,M  `MM     MM    ,M  `MM              #
    #           MM    MM     ,MP ,M'`Mb.         MM  8M     M8      MM     ,MP AbmmmqMA    MM    AbmmmqMA             #
    #           MM    MM    ,dP',P   `MM.        MM  YA.   ,A9      MM    ,dP'A'     VML   MM   A'     VML            #
    #         .JMML..JMMmmmdP'.MM:.  .:MMa.      `Mbmo`Ybmd9'     .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.          #
    ###################################################################################################################
    # Get splitted data of the combined dataset using the splitted index.

    def Get_X_y_data_selected(X_idx, X_rxn_data, y_data, log_value):
        X_rxn_selected      = []
        y_data_selected     = []
        for idx in X_idx:
            if y_data[idx] == None: # Skip
                continue
            X_rxn_selected .append(X_rxn_data[idx])
            y_data_selected.append(y_data[idx])

        y_data_selected = np.array(y_data_selected)
        if log_value == True:
            y_data_selected = np.log10(y_data_selected)

        return X_rxn_selected, y_data_selected

    #====================================================================================================#
    # 
    X_tr_rxn, y_tr = Get_X_y_data_selected(tr_idx_rxn, X_rxn_data, y_data, log_value)
    X_ts_rxn, y_ts = Get_X_y_data_selected(ts_idx_rxn, X_rxn_data, y_data, log_value)
    X_va_rxn, y_va = Get_X_y_data_selected(va_idx_rxn, X_rxn_data, y_data, log_value)
    y_scalar = None


    if log_value == False and NN_type == "Reg":
        y_tr, y_scalar = normalize_targets(y_tr)
        y_ts = y_scalar.transform(y_ts)
        y_va = y_scalar.transform(y_va)

        y_tr = np.array(y_tr, dtype = np.float32)
        y_ts = np.array(y_ts, dtype = np.float32)
        y_va = np.array(y_va, dtype = np.float32)


    #====================================================================================================#
    return X_tr_rxn, y_tr, \
           X_ts_rxn, y_ts, \
           X_va_rxn, y_va, \
           X_cmpd_encodings_dim, y_scalar, \
           cmpd_encodings_dict

































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







