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
import copy
import time
import math
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
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#--------------------------------------------------#
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#--------------------------------------------------#
from datetime import datetime

#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection

#--------------------------------------------------#
from ZX01_PLOT import *

from rxn05A_Model_MPNN import RXN_CP_MPNN_Model



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#         .M"""bgd `7MMF'`7MN.   `7MF' .g8"""bgd `7MMF'      `7MM"""YMM           `7MMM.     ,MMF' .g8""8q. `7MM"""Yb. `7MM"""YMM  `7MMF'              #
#        ,MI    "Y   MM    MMN.    M .dP'     `M   MM          MM    `7             MMMb    dPMM .dP'    `YM. MM    `Yb. MM    `7    MM                #
#        `MMb.       MM    M YMb   M dM'       `   MM          MM   d               M YM   ,M MM dM'      `MM MM     `Mb MM   d      MM                #
#          `YMMNq.   MM    M  `MN. M MM            MM          MMmmMM               M  Mb  M' MM MM        MM MM      MM MMmmMM      MM                #
#        .     `MM   MM    M   `MM.M MM.    `7MMF' MM      ,   MM   Y  ,            M  YM.P'  MM MM.      ,MP MM     ,MP MM   Y  ,   MM      ,         #
#        Mb     dM   MM    M     YMM `Mb.     MM   MM     ,M   MM     ,M            M  `YM'   MM `Mb.    ,dP' MM    ,dP' MM     ,M   MM     ,M         #
#        P"Ybmmd"  .JMML..JML.    YM   `"bmmmdPY .JMMmmmmMMM .JMMmmmmMMM          .JML. `'  .JMML. `"bmmd"' .JMMmmmdP' .JMMmmmmMMM .JMMmmmmMMM         #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def run_train(model                                 , 
              optimizer                             , 
              criterion                             , 
              epoch_num                             , 
              train_loader                          , 
              valid_loader                          , 
              test_loader                           , 
              y_scalar                              , 
              log_value_bool                        , 
              screen_bool                           , 
              results_sub_folder                    , 
              output_file_header                    , 
              input_var_names_list                  , 
              batch_size           = 96             , 
              target_name          = "y_property"   , 
              ):


    predictions_all = []
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [50, 100], gamma = 0.5)

    for epoch in range(epoch_num): 
        begin_time = time.time()
        #====================================================================================================#
        #                     MMP""MM""YMM `7MM"""Mq.       db     `7MMF'`7MN.   `7MF'                       #
        #                     P'   MM   `7   MM   `MM.     ;MM:      MM    MMN.    M                         #
        #                          MM        MM   ,M9     ,V^MM.     MM    M YMb   M                         #
        #                          MM        MMmmdM9     ,M  `MM     MM    M  `MN. M                         #
        #                          MM        MM  YM.     AbmmmqMA    MM    M   `MM.M                         #
        #                          MM        MM   `Mb.  A'     VML   MM    M     YMM                         #
        #                        .JMML.    .JMML. .JMM.AMA.   .AMMA.JMML..JML.    YM                         #
        #====================================================================================================#
        model.train()
        count_x = 0
        #====================================================================================================#
        for one_rxn_cmpd_y_group in train_loader:

            print("len(one_rxn_cmpd_y_group[cmpd_dataset]): ", len(one_rxn_cmpd_y_group["cmpd_dataset"]))
            print("len(one_rxn_cmpd_y_group[y_property])  : ", len(one_rxn_cmpd_y_group["y_property"])  )
            len_train_loader = len(train_loader)
            count_x += 1

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # Format Print. 
            if count_x == 20 :
                print(" " * 12, end = " ") 
            if ((count_x) % 160) == 0:
                print( str(count_x) + "/" + str(len_train_loader) + "->" + "\n" + " " * 12, end=" ")
            elif ((count_x) % 20) == 0:
                print( str(count_x) + "/" + str(len_train_loader) + "->", end=" ")

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # Process the input_vars and prepare to send them to model. 
            input_vars = [one_rxn_cmpd_y_group[one_var] for one_var in input_var_names_list]

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # X05A, X05V.
            if  type(model) == RXN_CP_MPNN_Model: # A
                input_vars = [input_vars[0], ]


            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # Output

            output, _ = model(*input_vars)
            print("\nlen_output: ", len(output))
            print("\nlen_input_vars: ", len(input_vars[0]))

            target = one_rxn_cmpd_y_group[target_name]
            target = target.double().cuda()

            loss = criterion(output, target.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        scheduler.step()

        #====================================================================================================#
        #      `7MMF'   `7MF' db     `7MMF'      `7MMF'`7MM"""Yb.      db  MMP""MM""YMM `7MM"""YMM           #
        #        `MA     ,V  ;MM:      MM          MM    MM    `Yb.   ;MM: P'   MM   `7   MM    `7           #
        #         VM:   ,V  ,V^MM.     MM          MM    MM     `Mb  ,V^MM.     MM        MM   d             #
        #          MM.  M' ,M  `MM     MM          MM    MM      MM ,M  `MM     MM        MMmmMM             #
        #          `MM A'  AbmmmqMA    MM      ,   MM    MM     ,MP AbmmmqMA    MM        MM   Y  ,          #
        #           :MM;  A'     VML   MM     ,M   MM    MM    ,dP'A'     VML   MM        MM     ,M          #
        #            VF .AMA.   .AMMA.JMMmmmmMMM .JMML..JMMmmmdP'.AMA.   .AMMA.JMML.    .JMMmmmmMMM          #
        #====================================================================================================#
        model.eval()
        y_pred_valid = []
        y_real_valid = []
        #====================================================================================================#
        for one_rxn_cmpd_y_group in valid_loader:

            len_valid_loader = batch_size

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # Process the input_vars and prepare to send them to model. 
            input_vars = [one_rxn_cmpd_y_group[one_var] for one_var in input_var_names_list]

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # X05A, X05V
            if  type(model) == RXN_CP_MPNN_Model: # A
                input_vars = [input_vars[0], ]


            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # Output
            output, _ = model(*input_vars)
            output = output.cpu().detach().numpy().reshape(-1)

            target = one_rxn_cmpd_y_group[target_name]
            target = target.numpy()

            y_pred_valid.append(output)
            y_real_valid.append(target)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        # Get Correlation Coefficient.
        y_pred_valid = np.concatenate(y_pred_valid)
        y_real_valid = np.concatenate(y_real_valid)

        #--------------------------------------------------#
        # if log_value_bool == False and screen_bool == True:
        #      y_pred_valid[y_pred_valid<0] = 0  # For test use only.
        #--------------------------------------------------#

        slope, intercept, r_value_va, p_value, std_err = scipy.stats.linregress(y_pred_valid, y_real_valid)

        #====================================================================================================#
        #                          MMP""MM""YMM `7MM"""YMM   .M"""bgd MMP""MM""YMM                           #
        #                          P'   MM   `7   MM    `7  ,MI    "Y P'   MM   `7                           #
        #                               MM        MM   d    `MMb.          MM                                #
        #                               MM        MMmmMM      `YMMNq.      MM                                #
        #                               MM        MM   Y  , .     `MM      MM                                #
        #                               MM        MM     ,M Mb     dM      MM                                #
        #                             .JMML.    .JMMmmmmMMM P"Ybmmd"     .JMML.                              #
        #====================================================================================================#
        y_pred = []
        y_real = []
        #--------------------------------------------------#
        for one_rxn_cmpd_y_group in test_loader:

            len_test_loader = batch_size

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # Process the input_vars and prepare to send them to model.             
            input_vars = [one_rxn_cmpd_y_group[one_var] for one_var in input_var_names_list]

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # X05A, X05V
            if  type(model) == RXN_CP_MPNN_Model: # A
                input_vars = [input_vars[0], ]


            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # Output

            output, _ = model(*input_vars)
            output = output.cpu().detach().numpy().reshape(-1)
            
            target = one_rxn_cmpd_y_group[target_name]
            target = target.numpy()
            
            y_pred.append(output)
            y_real.append(target)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        # Get Correlation Coefficient.
        y_pred = np.concatenate(y_pred)
        y_real = np.concatenate(y_real)
        

        #--------------------------------------------------#
        # if log_value_bool == False and screen_bool == True:
        #      y_pred[y_pred<0] = 0  # For test use only.
        #--------------------------------------------------#

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_real)

        #====================================================================================================#
        #                       `7MM"""Mq.`7MM"""Mq.  `7MMF'`7MN.   `7MF'MMP""MM""YMM                        #
        #                         MM   `MM. MM   `MM.   MM    MMN.    M  P'   MM   `7                        #
        #                         MM   ,M9  MM   ,M9    MM    M YMb   M       MM                             #
        #                         MMmmdM9   MMmmdM9     MM    M  `MN. M       MM                             #
        #                         MM        MM  YM.     MM    M   `MM.M       MM                             #
        #                         MM        MM   `Mb.   MM    M     YMM       MM                             #
        #                       .JMML.    .JMML. .JMM..JMML..JML.    YM     .JMML.                           #
        #====================================================================================================#
        loss_copy = copy.copy(loss)
        print("\n" + "_" * 101, end = " ")
        print("\nepoch: {} | time_elapsed: {:5.4f} | train_loss: {:5.4f} | vali_R_VALUE: {:5.4f} | test_R_VALUE: {:5.4f} ".format( 
             str((epoch+1)+1000).replace("1","",1), 

             np.round((time.time()-begin_time), 5),
             np.round(loss_copy.cpu().detach().numpy(), 5), 
             np.round(r_value_va, 5), 
             np.round(r_value, 5),
             )
             )

        r_value, r_value_va = r_value, r_value_va 

        va_MAE  = np.round(mean_absolute_error(y_pred_valid, y_real_valid), 4)
        va_MSE  = np.round(mean_squared_error (y_pred_valid, y_real_valid), 4)
        va_RMSE = np.round(math.sqrt(va_MSE), 4)
        va_R2   = np.round(r2_score(y_real_valid, y_pred_valid), 4)
        va_rho  = np.round(scipy.stats.spearmanr(y_pred_valid, y_real_valid)[0], 4)
        
        
        ts_MAE  = np.round(mean_absolute_error(y_pred, y_real), 4)
        ts_MSE  = np.round(mean_squared_error (y_pred, y_real), 4)
        ts_RMSE = np.round(math.sqrt(ts_MSE), 4) 
        ts_R2   = np.round(r2_score(y_real, y_pred), 4)
        ts_rho  = np.round(scipy.stats.spearmanr(y_pred, y_real)[0], 4)


        print("           | va_MAE: {:4.3f} | va_MSE: {:4.3f} | va_RMSE: {:4.3f} | va_R2: {:4.3f} | va_rho: {:4.3f} ".format( 
             va_MAE, 
             va_MSE,
             va_RMSE, 
             va_R2, 
             va_rho,
             )
             )

        print("           | ts_MAE: {:4.3f} | ts_MSE: {:4.3f} | ts_RMSE: {:4.3f} | ts_R2: {:4.3f} | ts_rho: {:4.3f} ".format( 
             ts_MAE, 
             ts_MSE,
             ts_RMSE, 
             ts_R2, 
             ts_rho,
             )
             )

        y_pred_all = np.concatenate([y_pred, y_pred_valid], axis = None)
        y_real_all = np.concatenate([y_real, y_real_valid], axis = None)

        all_rval = np.round(scipy.stats.pearsonr(y_pred_all, y_real_all), 5)
        all_MAE  = np.round(mean_absolute_error(y_pred_all, y_real_all), 4)
        all_MSE  = np.round(mean_squared_error (y_pred_all, y_real_all), 4)
        all_RMSE = np.round(math.sqrt(ts_MSE), 4) 
        all_R2   = np.round(r2_score(y_real_all, y_pred_all), 4)
        all_rho  = np.round(scipy.stats.spearmanr(y_pred_all, y_real_all)[0], 4)

        print("           | tv_MAE: {:4.3f} | tv_MSE: {:4.3f} | tv_RMSE: {:4.3f} | tv_R2: {:4.3f} | tv_rho: {:4.3f} ".format( 
             all_MAE ,
             all_MSE ,
             all_RMSE,
             all_R2  ,
             all_rho ,
             )
             )
        print("           | tv_R_VALUE:", all_rval)

        print("_" * 101)

        #====================================================================================================#
        #                        `7MM"""Mq.`7MMF'        .g8""8q.  MMP""MM""YMM                              #
        #                          MM   `MM. MM        .dP'    `YM.P'   MM   `7                              #
        #                          MM   ,M9  MM        dM'      `MM     MM                                   #
        #                          MMmmdM9   MM        MM        MM     MM                                   #
        #                          MM        MM      , MM.      ,MP     MM                                   #
        #                          MM        MM     ,M `Mb.    ,dP'     MM                                   #
        #                        .JMML.    .JMMmmmmMMM   `"bmmd"'     .JMML.                                 #
        #====================================================================================================#
        if ((epoch+1) % 1) == 0:
            if log_value_bool == False:
                y_pred = y_scalar.inverse_transform(y_pred)
                y_real = y_scalar.inverse_transform(y_real)

            if log_value_bool == False and screen_bool == True:
                  y_pred[y_pred<0] = 0  # For test use only.

            _, _, r_value, _ , _ = scipy.stats.linregress(y_pred, y_real)

            reg_scatter_distn_plot(y_pred,
                                   y_real,
                                   fig_size        =  (10,8),
                                   marker_size     =  35,
                                   fit_line_color  =  "brown",
                                   distn_color_1   =  "gold",
                                   distn_color_2   =  "lightpink",
                                    # title        =  "Predictions vs. Actual Values\n R = " + \
                                    #                        str(round(r_value,3)) + \
                                    #                        ", Epoch: " + str(epoch+1) ,
                                    title          =  "",
                                    plot_title     =  "R = " + str(round(r_value,3)) + \
                                                       "\n" + r'$\rho$' + " = " + str(round(ts_rho,3)) + \
                                                            "\nEpoch: " + str(epoch+1) ,
                                   x_label         =  "Actual Values",
                                   y_label         =  "Predictions",
                                   cmap            =  None,
                                   cbaxes          =  (0.425, 0.055, 0.525, 0.015),
                                   font_size       =  18,
                                   result_folder   =  results_sub_folder,
                                   file_name       =  output_file_header + "_TS_" + "epoch_" + str(epoch+1),
                                   ) #For checking predictions fittings.



            if log_value_bool == False:
                y_pred_valid = y_scalar.inverse_transform(y_pred_valid)
                y_real_valid = y_scalar.inverse_transform(y_real_valid)

            if log_value_bool == False and screen_bool == True:
                y_pred_valid[y_pred_valid<0] = 0  # For test use only.


            _, _, r_value, _ , _ = scipy.stats.linregress(y_pred_valid, y_real_valid)                       
            reg_scatter_distn_plot(y_pred_valid,
                                   y_real_valid,
                                   fig_size        =  (10,8),
                                   marker_size     =  35,
                                   fit_line_color  =  "brown",
                                   distn_color_1   =  "gold",
                                   distn_color_2   =  "lightpink",
                                    # title        =  "Predictions vs. Actual Values\n R = " + \
                                    #                        str(round(r_value,3)) + \
                                    #                        ", Epoch: " + str(epoch+1) ,
                                    title          =  "",
                                    plot_title     =  "R = " + str(round(r_value,3)) + \
                                                       "\n" + r'$\rho$' + " = " + str(round(va_rho,3)) + \
                                                            "\nEpoch: " + str(epoch+1) ,
                                   x_label         =  "Actual Values",
                                   y_label         =  "Predictions",
                                   cmap            =  None,
                                   cbaxes          =  (0.425, 0.055, 0.525, 0.015),
                                   font_size       =  18,
                                   result_folder   =  results_sub_folder,
                                   file_name       =  output_file_header + "_VA_" + "epoch_" + str(epoch+1),
                                   ) #For checking predictions fittings.

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            # TS, VA plot done. Now plot the fittings for log values (for test set only).
            if log_value_bool == False and screen_bool == True:

                y_real = np.delete(y_real, np.where(y_pred == 0.0))
                y_pred = np.delete(y_pred, np.where(y_pred == 0.0))

                y_real = np.delete(y_real, np.where(y_real == 0.0))
                y_pred = np.delete(y_pred, np.where(y_real == 0.0))

                y_real = np.log10(y_real)
                y_pred = np.log10(y_pred)

                reg_scatter_distn_plot(y_pred,
                                       y_real,
                                       fig_size       = (10,8),
                                       marker_size    = 20,
                                       fit_line_color = "brown",
                                       distn_color_1  = "gold",
                                       distn_color_2  = "lightpink",
                                       # title        =  "Predictions vs. Actual Values\n R = " + \
                                       #                        str(round(r_value,3)) + \
                                       #                        ", Epoch: " + str(epoch+1) ,
                                       title          =  "",
                                       plot_title     =  "R = " + str(round(r_value,3)) + \
                                                          "\n" + r'$\rho$' + " = " + str(round(ts_rho,3)) + \
                                                               "\nEpoch: " + str(epoch+1) ,
                                       x_label        = "Actual Values",
                                       y_label        = "Predictions",
                                       cmap           = None,
                                       font_size      = 18,
                                       result_folder  = results_sub_folder,
                                       file_name      = output_file_header + "_logplot" + "epoch_" + str(epoch+1),
                                       ) #For checking predictions fittings.
    ##############################################################################################################
    ##############################################################################################################

    return predictions_all
























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












