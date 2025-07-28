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
import random
#--------------------------------------------------#
import numpy as np
import pandas as pd
import seaborn as sns
#--------------------------------------------------#
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib import cm
#--------------------------------------------------#
import sklearn
from sklearn.model_selection import train_test_split
#--------------------------------------------------#
from AP_funcs import cart_prod, cart_dual_prod_re




###################################################################################################################
#           `7MM"""Mq. `7MM"""Mq.  `7MM"""YMM       .M"""bgd `7MM"""Mq. `7MMF'      `7MMF'MMP""MM""YMM            #
#             MM   `MM.  MM   `MM.   MM    `7      ,MI    "Y   MM   `MM.  MM          MM  P'   MM   `7            #
#             MM   ,M9   MM   ,M9    MM   d        `MMb.       MM   ,M9   MM          MM       MM                 #
#             MMmmdM9    MMmmdM9     MMmmMM          `YMMNq.   MMmmdM9    MM          MM       MM                 #
#             MM         MM  YM.     MM   Y  ,     .     `MM   MM         MM      ,   MM       MM                 #
#             MM         MM   `Mb.   MM     ,M     Mb     dM   MM         MM     ,M   MM       MM                 #
#           .JMML.     .JMML. .JMM..JMMmmmmMMM     P"Ybmmd"  .JMML.     .JMMmmmmMMM .JMML.   .JMML.               #
###################################################################################################################



#====================================================================================================#
def split_seqs_idx_custom(X_num, customized_idx_list, customized_idx_dict, train_split = 0.7, test_split = 0.2, random_state = 42, split_type = 4): 
    # FOR SEQUENCES ONLY !!!
    # customized_idx_list: idx of those SEQUENCES selected by CD-HIT.

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type == 4: 
        # non-customized idx -> the set that contains its representatives.
        # customized idx -> 70% training set, 20% validation set, 10% test set.
        print("len(customized_idx_list): ", len(customized_idx_list))
        print("len(customized_idx_dict): ", len(customized_idx_dict))
        customized_idx_dict_keys = list(customized_idx_dict.keys())
        customized_idx_dict_values = [id2 for id in customized_idx_dict for id2 in customized_idx_dict[id]]
        #--------------------------------------------------#
        # Split the representative sequences.
        X_tr_idx = y_tr_idx = customized_idx_list
        X_tr_idx, X_ts_idx, y_tr_idx, y_ts_idx = train_test_split(X_tr_idx, y_tr_idx, test_size = (1-train_split), random_state = random_state)
        X_va_idx, X_ts_idx, y_va_idx, y_ts_idx = train_test_split(X_ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)) , random_state = random_state)
        #--------------------------------------------------#
        # Split the .

        X_idx = y_idx = list(range(X_num))
        X_tr_idx_extra = [non_rep_idx for rep_idx in X_tr_idx for non_rep_idx in customized_idx_dict[rep_idx]]
        X_ts_idx_extra = [non_rep_idx for rep_idx in X_ts_idx for non_rep_idx in customized_idx_dict[rep_idx]]
        X_va_idx_extra = [non_rep_idx for rep_idx in X_va_idx for non_rep_idx in customized_idx_dict[rep_idx]]

        X_tr_idx_extra_extra = [idx for idx in X_idx if idx not in customized_idx_dict_values and idx not in customized_idx_dict_keys]
        #--------------------------------------------------#
        X_tr_idx = X_tr_idx + X_tr_idx_extra + X_tr_idx_extra_extra
        X_ts_idx = X_ts_idx + X_ts_idx_extra
        X_va_idx = X_va_idx + X_va_idx_extra

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type == 5:
        # non-customized idx -> DUMPED
        # customized idx -> 70% training set, 20% validation set, 10% test set.
        print("len(customized_idx_list): ", len(customized_idx_list))

        X_tr_idx_extra = []
        #--------------------------------------------------#
        X_tr_idx = y_tr_idx = customized_idx_list
        X_tr_idx, X_ts_idx, y_tr_idx, y_ts_idx = train_test_split(X_tr_idx, y_tr_idx, test_size = (1-train_split), random_state = random_state)
        X_va_idx, X_ts_idx, y_va_idx, y_ts_idx = train_test_split(X_ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)) , random_state = random_state)
        #--------------------------------------------------#
        X_tr_idx = X_tr_idx + X_tr_idx_extra

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    return X_tr_idx, X_ts_idx, X_va_idx







###################################################################################################################
#           .M"""bgd `7MM"""Mq. `7MMF'      `7MMF'MMP""MM""YMM           db      `7MMF'      `7MMF'               #
#          ,MI    "Y   MM   `MM.  MM          MM  P'   MM   `7          ;MM:       MM          MM                 #
#          `MMb.       MM   ,M9   MM          MM       MM              ,V^MM.      MM          MM                 #
#            `YMMNq.   MMmmdM9    MM          MM       MM             ,M  `MM      MM          MM                 #
#          .     `MM   MM         MM      ,   MM       MM             AbmmmqMA     MM      ,   MM      ,          #
#          Mb     dM   MM         MM     ,M   MM       MM            A'     VML    MM     ,M   MM     ,M          #
#          P"Ybmmd"  .JMML.     .JMMmmmmMMM .JMML.   .JMML.        .AMA.   .AMMA..JMMmmmmMMM .JMMmmmmMMM          #
###################################################################################################################
def split_seqs_cmpd_idx_book(tr_idx_seqs         , 
                             ts_idx_seqs         , 
                             va_idx_seqs         , 
                             tr_idx_cmpd         , 
                             ts_idx_cmpd         , 
                             va_idx_cmpd         ,
                             X_seqs_num          ,
                             X_cmpd_num          , 
                             y_data              ,
                             customized_idx_list , # representative sequences
                             customized_idx_dict , # clusters dictionary
                             seqs_cmpd_idx_book  , 
                             split_type          , 
                             random_state = 42   , 
                             train_split  = 0.7  , 
                             test_split   = 0.2  ,
                             ):
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    """
    - split_type = 0, train/test split completely randomly selected
    - split_type = 1, train/test split contains different seqs-cmpd pairs
    - split_type = 2, train/test split contains different seqs
    - split_type = 3, train/test split contains different cmpd
    - split_type = 4, train/test split contains different seqs clusters,
                        i.e., train/test split contains different CD-hit-representative seqs, 
                        with the rest seqs data being sent to its representative sequence's split.
    - split_type = 5, train/test split contains different CD-hit-representative seqs,
                        i.e., train/test split contains different CD-hit-representative seqs, 
                        with non-CD-hit-representative seqs data being left out.
    - split_type = 6, train/test split randomly selected from data with CD-hit-representative seqs, 
                        with non-CD-hit-representative seqs data being sent to training set.
                            (this is MEANINGLESS since representative 
                            sequences can be in both train and test, 
                            being paired with different compounds.
                            All non-representatives are in training. 
                            This shall make the prediction even better.)
    - split_type = 7, train/test split randomly selected from data with CD-hit-representative seqs,  
                        with non-CD-hit-representative seqs data being left out.
    """
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    tr_idx, ts_idx, va_idx = [], [], []
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type == 0: 
        # In this case, split_idx outputs are not used at all. split ratio is defined HERE in the func, not from Z05_utils
        dataset_size = len(seqs_cmpd_idx_book)
        X_data_idx = np.array(list(range(dataset_size)))
        tr_idx, ts_idx, y_train, y_test = train_test_split(X_data_idx, y_data, test_size = (1-train_split), random_state = random_state)             # tr : ts & va 
        va_idx, ts_idx, y_valid, y_test = train_test_split(ts_idx, y_test, test_size = (test_split/(1.0-train_split)), random_state = random_state)  # va : ts      
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type in [1, 2, 3, 4, 5, ]:
        all_idx_cmpd = list(range(X_cmpd_num))
        all_idx_seqs = list(range(X_seqs_num))
        #--------------------------------------------------#
        if split_type == 1:
            tr_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([tr_idx_seqs, tr_idx_cmpd])]
            ts_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([ts_idx_seqs, ts_idx_cmpd])]
            va_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([va_idx_seqs, va_idx_cmpd])]
            seqs_cmpd_idx_book_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        #--------------------------------------------------#
        if split_type == 2: 
            tr_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([tr_idx_seqs, all_idx_cmpd])]
            ts_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([ts_idx_seqs, all_idx_cmpd])]
            va_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([va_idx_seqs, all_idx_cmpd])]
            seqs_cmpd_idx_book_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        #--------------------------------------------------#
        if split_type == 3:
            tr_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([all_idx_seqs, tr_idx_cmpd])]
            ts_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([all_idx_seqs, ts_idx_cmpd])]
            va_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([all_idx_seqs, va_idx_cmpd])]
            seqs_cmpd_idx_book_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        #--------------------------------------------------#
        if split_type == 4 or split_type == 5: # Exactly same as TYPE 2.
            tr_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([tr_idx_seqs, all_idx_cmpd])]
            ts_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([ts_idx_seqs, all_idx_cmpd])]
            va_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([va_idx_seqs, all_idx_cmpd])]
            seqs_cmpd_idx_book_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        #--------------------------------------------------#
        tr_idx = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, tr_idx_pairs_encrypt))[0]))
        ts_idx = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, ts_idx_pairs_encrypt))[0]))
        va_idx = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, va_idx_pairs_encrypt))[0]))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type == 6: # Meaningless
        # non-customized idx -> training set
        # customized idx -> 70% training set, 20% validation set, 10% test set.
        all_idx_cmpd = list(range(X_cmpd_num))

        #--------------------------------------------------#
        n_CD_hit_seqs_idx_list      = [idx for idx in list(range(X_seqs_num)) if (idx not in customized_idx_list)]
        dataset_n_CD_hit            = cart_prod([n_CD_hit_seqs_idx_list, all_idx_cmpd])
        dataset_n_CD_hit_encrypt    = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in dataset_n_CD_hit]
        seqs_cmpd_idx_book_encrypt  = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        dataset_n_CD_hit_idx        = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, dataset_n_CD_hit_encrypt))[0]))
        #--------------------------------------------------#
        y_CD_hit_seqs_idx_list      = customized_idx_list
        dataset_y_CD_hit            = cart_prod([y_CD_hit_seqs_idx_list, all_idx_cmpd])
        dataset_y_CD_hit_encrypt    = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in dataset_y_CD_hit]
        seqs_cmpd_idx_book_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        dataset_y_CD_hit_idx        = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, dataset_y_CD_hit_encrypt))[0]))
        #--------------------------------------------------#
        tr_idx, ts_idx, y_tr_idx, y_ts_idx = train_test_split(dataset_y_CD_hit_idx, dataset_y_CD_hit_idx, test_size = (1-train_split), random_state = random_state) # tr : ts & va 
        va_idx, ts_idx, y_va_idx, y_ts_idx = train_test_split(ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)), random_state = random_state)        
        #--------------------------------------------------#
        tr_idx = list(tr_idx) + list(dataset_n_CD_hit_idx)
        ts_idx = list(ts_idx)
        va_idx = list(va_idx)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type == 7: # Try to predict seqs-cmpd pairs with no help from sequences similarities.
        # non-customized idx -> training set
        # customized idx -> 70% training set, 20% validation set, 10% test set.
        all_idx_cmpd = list(range(X_cmpd_num))
        #--------------------------------------------------#
        y_CD_hit_seqs_idx_list      = customized_idx_list
        dataset_y_CD_hit            = cart_prod([y_CD_hit_seqs_idx_list, all_idx_cmpd])
        dataset_y_CD_hit_encrypt    = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in dataset_y_CD_hit]
        seqs_cmpd_idx_book_encrypt  = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        dataset_y_CD_hit_idx        = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, dataset_y_CD_hit_encrypt))[0]))
        #--------------------------------------------------#
        tr_idx, ts_idx, y_tr_idx, y_ts_idx = train_test_split(dataset_y_CD_hit_idx, dataset_y_CD_hit_idx, test_size = (1-train_split), random_state = random_state) # tr : ts & va 
        va_idx, ts_idx, y_va_idx, y_ts_idx = train_test_split(ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)), random_state = random_state) 
        #--------------------------------------------------#
        tr_idx = list(tr_idx)
        ts_idx = list(ts_idx)
        va_idx = list(va_idx)


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    tr_seqs_idx_verify = list(set([seqs_cmpd_idx_book[idx][0] for idx in tr_idx]))
    ts_seqs_idx_verify = list(set([seqs_cmpd_idx_book[idx][0] for idx in ts_idx]))
    va_seqs_idx_verify = list(set([seqs_cmpd_idx_book[idx][0] for idx in va_idx]))
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #print(tr_seqs_idx_verify)
    print("-"*50)
    print("Verify the number of sequences in each split: ")
    print("len(tr_seqs_idx_verify): ", len(tr_seqs_idx_verify))
    print("len(ts_seqs_idx_verify): ", len(ts_seqs_idx_verify))
    print("len(va_seqs_idx_verify): ", len(va_seqs_idx_verify))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    return tr_idx, ts_idx, va_idx





###################################################################################################################
#         `7MMF'`7MM"""Yb.`YMM'   `MP'       mm               `7MM"""Yb.      db  MMP""MM""YMM  db                #
#           MM    MM    `Yb.VMb.  ,P         MM                 MM    `Yb.   ;MM: P'   MM   `7 ;MM:               #
#           MM    MM     `Mb `MM.M'        mmMMmm ,pW"Wq.       MM     `Mb  ,V^MM.     MM     ,V^MM.              #
#           MM    MM      MM   MMb           MM  6W'   `Wb      MM      MM ,M  `MM     MM    ,M  `MM              #
#           MM    MM     ,MP ,M'`Mb.         MM  8M     M8      MM     ,MP AbmmmqMA    MM    AbmmmqMA             #
#           MM    MM    ,dP',P   `MM.        MM  YA.   ,A9      MM    ,dP'A'     VML   MM   A'     VML            #
#         .JMML..JMMmmmdP'.MM:.  .:MMa.      `Mbmo`Ybmd9'     .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.          #
###################################################################################################################
def Get_X_y_data_selected(X_idx, X_all_seqs, X_cmpd_encodings, X_cmpd_smiles, y_data, log_value):
    #X_seqs_emb_selected = [] # Previously store large embeddings.
    X_seqs_selected     = []
    X_cmpd_selected     = []
    X_smls_selected     = []
    y_data_selected     = []
    for idx in X_idx:
        #print(y_data[idx])
        if y_data[idx] == None or X_cmpd_smiles[idx] == "[HH]": # Skip
            continue
        #print(y_data[idx])

        #X_seqs_emb_selected.append(X_seqs_all_hiddens[idx])
        X_seqs_selected.append(X_all_seqs[idx])
        X_cmpd_selected.append(X_cmpd_encodings[idx])
        X_smls_selected.append(X_cmpd_smiles[idx])
        y_data_selected.append(y_data[idx])


    y_data_selected = np.array(y_data_selected)
    if log_value == True:
        y_data_selected = np.log10(y_data_selected)

    return X_seqs_selected, X_cmpd_selected, X_smls_selected, y_data_selected







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












