
#!/usr/bin/env python
# coding: utf-8

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
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


###################################################################################################################
###################################################################################################################


from simplify_chemprop_train import cross_validate, run_training
from simplify_chemprop_args import TrainArgs


def main():

    """
    Parses Chemprop training arguments and trains (cross-validates) a Chemprop model.
    This is the entry point for the command line command :code:`chemprop_train`.
    """

    cmpd_process = ["u-MPNN-a", "u-MPNN", "d-MPNN", "u-MPNN-a-m", "Mol_Feat"][2]
    
    MPNN_size = "2160"

    arguments = ['--data_path'                                ,    './Logk_Rate_Const.csv' ,
                 '--dataset_type'                             ,    'regression'            ,
                 '--save_dir'                                 ,    'test_checkpoints_reg'  ,

                 '--epochs'                                   ,    '10'                    ,
                 '--hidden_size'                              ,    MPNN_size               ,
                 '--depth'                                    ,    '3'                     ,

                 '--save_smiles_splits'                       ,
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



    
    args = TrainArgs().parse_args(arguments)

    mean_score, std_score = cross_validate(args = args, train_func=run_training)

    return mean_score, std_score


if __name__ == "__main__":
    main()