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
###################################################################################################################
###################################################################################################################
# Imports
import pickle
import numpy as np
from pathlib import Path

def load_and_print(file_path):
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File {file_path} does not exist.")
        return

    try:
        if file_path.suffix == '.pickle' or file_path.suffix == '.p':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif file_path.suffix == '.npz':
            data = np.load(file_path)
            for key in data.files:
                print("Key: ", key)
                print("Shape: ", data[key].shape)
                print("Data: ", data[key][:100])
                if data[key].size > 100:
                    print("Only the first 100 items are printed due to large content.")
                print()
            return
        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return

        # Attempt to print the type of data
        print("Type of data: ", type(data).__name__)
        
        # Attempt to print the data
        if isinstance(data, dict):
            for i, (key, value) in enumerate(data.items()):
                if i < 100:
                    print(f"Key: {key}, Value: {value}")
                else:
                    break
            if len(data) > 100:
                print("Only the first 100 key-value pairs are printed due to large content.")
        elif isinstance(data, (list, set, tuple, np.ndarray)):
            print("Data: ", data[:100])
            if len(data) > 100:
                print("Only the first 100 items are printed due to large content.")
        else:
            print("Data: ", data)

        # Attempt to print the shape of the data if it's a numpy array
        if isinstance(data, np.ndarray):
            print("Shape: ", data.shape)
    except Exception as e:
        print(f"An error occurred while loading or printing the file: {e}")


if __name__ == '__main__':
    # Call the function with your file path
    load_and_print(Path("./RXN_DataProcessing") / "RXN02B_Reaction_Energy_MgFP6_encodings_dict.p")


































