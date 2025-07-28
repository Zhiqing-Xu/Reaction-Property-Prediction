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
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
#--------------------------------------------------#
import re
import gzip
#--------------------------------------------------#
from AP_funcs import *


#====================================================================================================#
# global

global bkgd_cmpd_list; bkgd_cmpd_list=['O','[H]O[H]', 'O=P(O)(O)O', 'O=C=O', 'N']

def bkgd_cmpd_list_func():
    return bkgd_cmpd_list # For test only

# For test only
global CoA_cmpd_list; CoA_cmpd_list=["Acetyl-CoA","Malonyl-CoA", "Succinyl-CoA"] 
global CoA_cmpd_dict; CoA_cmpd_dict={ "CC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O"       : "Acetyl-CoA"  ,
                                      "CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCSC(=O)CC(=O)O" : "Malonyl-CoA" ,
                                      "CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCSC(=O)CCC(=O)O": "Succinyl-CoA", 
                                     } 

# For test only
def CoA_cmpd_list_func():
    #return CoA_cmpd_list
    return [] # For test only
def CoA_cmpd_list_func():
    #return CoA_cmpd_list
    return [] # For test only


global map_compound2graph          ; map_compound2graph         = {}
global map_hash2compound           ; map_hash2compound          = {}
global map_hash2compound_unchiral  ; map_hash2compound_unchiral = {}




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#     .g8"""bgd `7MM"""YMM  `7MN.   `7MF'`7MM"""YMM  `7MM"""Mq.       db     `7MMF'         `7MM"""YMM `7MMF'   `7MF'`7MN.   `7MF' .g8"""bgd           #
#   .dP'     `M   MM    `7    MMN.    M    MM    `7    MM   `MM.     ;MM:      MM             MM    `7   MM       M    MMN.    M .dP'     `M           #
#   dM'       `   MM   d      M YMb   M    MM   d      MM   ,M9     ,V^MM.     MM             MM   d     MM       M    M YMb   M dM'       `           #
#   MM            MMmmMM      M  `MN. M    MMmmMM      MMmmdM9     ,M  `MM     MM             MM""MM     MM       M    M  `MN. M MM                    #
#   MM.    `7MMF' MM   Y  ,   M   `MM.M    MM   Y  ,   MM  YM.     AbmmmqMA    MM      ,      MM   Y     MM       M    M   `MM.M MM.                   #
#   `Mb.     MM   MM     ,M   M     YMM    MM     ,M   MM   `Mb.  A'     VML   MM     ,M      MM         YM.     ,M    M     YMM `Mb.     ,'           #
#     `"bmmmdPY .JMMmmmmMMM .JML.    YM  .JMMmmmmMMM .JMML. .JMM.AMA.   .AMMA.JMMmmmmMMM    .JMML.        `bmmmmd"'  .JML.    YM   `"bmmmd'            #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

# General Functions for reading SMILES
class Get_Unique_SMILES:
    # The purpose of this class is to help detect SMILES that represent same chemical compound.
    # All SMILES can be converted to a "unique" string with pre-defined settings.
    # SMILES that represent the same compound are expected to be converted to one unique string.
    # degree of uniqueness is determined by settings (i.e., isomericSmiles, SMARTS_bool, etc.)
    # Uniqueness Ranking:
    # 1. smiles -> mol -> smiles, with isomericSmiles = True  and SMARTS_bool = False
    # 2. smiles -> mol -> smiles, with isomericSmiles = False and SMARTS_bool = False
    # 3. smiles -> mol -> smiles, with isomericSmiles = False and SMARTS_bool = True
    # #1 can be used to parse SMILES string.
    # #2 is normally used to identify same compounds.
    # #3 can be used to deal with SMARTS-like strings output by reaction simulation.
    # #3 strings can be problematic when using RDKIT functions, so USE #3 VERY CAREFULLY.
    #--------------------------------------------------#
    def __init__(self, isomericSmiles = True, kekuleSmiles = False, canonical = True, SMARTS_bool = False, removeAtomMapNumber = False):
        self.isomericSmiles      = isomericSmiles
        self.kekuleSmiles        = kekuleSmiles 
        self.canonical           = canonical
        self.SMARTS_bool         = SMARTS_bool
        self.removeAtomMapNumber = removeAtomMapNumber

    #--------------------------------------------------#
    def UniS(self, smiles_x): 
        #------------------------------
        if self.SMARTS_bool == True:
            # old_name: UniS(), UniSS()
            # SMILES -> R-mol -> SMARTS -> R-mol -> SMILES 
            SMILES_bool = True
            mol_x=Chem.MolFromSmiles(smiles_x) # MolFromSmiles will NOT return ERROR, no matter what string is input.
            try:
                mol_x_unique=Chem.MolFromSmarts(Chem.MolToSmarts(mol_x))
                unique_smiles=Chem.MolToSmiles(mol_x_unique, 
                                               isomericSmiles = self.isomericSmiles ,
                                               kekuleSmiles   = self.kekuleSmiles   , 
                                               canonical      = self.canonical       )
            except Exception:
                print ("!!!!! Problematic SMILES (UniSS): ", smiles_x)
                unique_smiles=smiles_x
                SMILES_bool = False
            
        #------------------------------
        if self.SMARTS_bool == False:
            # old_name: CanS()
            # SMILES -> R-mol -> SMILES
            SMILES_bool = True
            #print(smiles_x)
            mol_x = Chem.MolFromSmiles(smiles_x) # MolFromSmiles will NOT return ERROR, no matter what string is input.
            try:
                unique_smiles=Chem.MolToSmiles(mol_x, 
                                               isomericSmiles = self.isomericSmiles,
                                               kekuleSmiles   = self.kekuleSmiles, 
                                               canonical      = self.canonical)
            except Exception:
                print ("!!!!! Problematic SMILES (UniS): ", smiles_x)
                unique_smiles=smiles_x
                SMILES_bool = False
        #--------------------------------
        # Whether to remove atom map number.
        # TODO: Check if the following is necessary
        if self.removeAtomMapNumber:
            #unique_smiles = re.sub(r":\d+", "", unique_smiles)
            mol_x=Chem.MolFromSmiles(smiles_x)
            for atom in mol_x.GetAtoms():
                atom.SetAtomMapNum(mapno = 0)
            unique_smiles=Chem.MolToSmiles(mol_x, 
                                            isomericSmiles = self.isomericSmiles,
                                            kekuleSmiles   = self.kekuleSmiles, 
                                            canonical      = self.canonical)

        
        return unique_smiles, SMILES_bool



    #--------------------------------------------------#
    def UNQSMI(self, smiles_x):
        # Return Unique SMILES
        return self.UniS(smiles_x)[0]

    #--------------------------------------------------#
    def ValidSMI(self, smiles_x):
        # Return False if input is NOT identified as SMILES
        return self.UniS(smiles_x)[1]





#====================================================================================================#
# 
def unique_canonical_smiles_AP(smiles_x, isomericSmiles = True, kekuleSmiles = False, canonical = True):
    # SMILES -> R-mol -> SMARTS -> R-mol -> SMILES
    GetUnqSmi = Get_Unique_SMILES(isomericSmiles = isomericSmiles, kekuleSmiles = kekuleSmiles, canonical = canonical, SMARTS_bool = True)
    return GetUnqSmi.UniS(smiles_x)[0]


def unique_canonical_smiles_list_AP(list_x, isomericSmiles = True, kekuleSmiles = False, canonical = True):
    # SMILES -> R-mol -> SMARTS -> R-mol -> SMILES
    return [unique_canonical_smiles_AP(one_smiles, isomericSmiles) for one_smiles in list_x]


def canonical_smiles_AP(smiles_x, isomericSmiles = True, kekuleSmiles = False, canonical = True):
    # SMILES -> R-mol -> SMILES
    GetUnqSmi = Get_Unique_SMILES(isomericSmiles = isomericSmiles, kekuleSmiles = kekuleSmiles, canonical = canonical, SMARTS_bool = False)
    return GetUnqSmi.UniS(smiles_x)[0]


def canonical_smiles_list_AP(list_x, isomericSmiles = True, kekuleSmiles = False, canonical = True):
    # SMILES -> R-mol -> SMILES
    return [canonical_smiles_AP(one_smiles, isomericSmiles) for one_smiles in list_x]








#====================================================================================================#
# 
def MolFromSmiles_ZX(smiles_x, bad_ss_dict = {}, 
                     isomericSmiles        = True  ,
                     kekuleSmiles          = False ,  
                     canonical             = True  ,
                     SMARTS_bool           = False , ): 
    # SMILES -> R-mol
    mol_x = Chem.MolFromSmiles(smiles_x)
    try: 
        Chem.MolToSmiles(mol_x, isomericSmiles = isomericSmiles)
    except:
        bad_ss_dict[smiles_x] = mol_x
        print ("!!!!! Problematic SMILES (MolFromSmiles_ZX): ", smiles_x)
    return mol_x



def MolToSmiles_ZX(mol_x, bad_ss_dict = {}, 
                   isomericSmiles     = True  ,
                   kekuleSmiles       = False ,  
                   canonical          = True  ,
                   SMARTS_bool        = False , ): 
    # R-mol -> SMARTS -> R-mol -> SMILES
    try: 
        if SMARTS_bool:
            mol_x_unique = Chem.MolFromSmarts(Chem.MolToSmarts(mol_x))
            smiles_x = Chem.MolToSmiles(mol_x_unique, isomericSmiles = isomericSmiles)
        else:
            smiles_x = Chem.MolToSmiles(mol_x, isomericSmiles = isomericSmiles)

    except Exception:
        try:
            smarts_x = Chem.MolToSmarts(mol_x)
            bad_ss_dict[smarts_x] = mol_x
            smiles_x = smarts_x
            print ("!!!!! Problematic SMILES (MolToSmiles_ZX): ", mol_x)
            print ("!!!!! TRY using SMARTS: ", smarts_x)
        except Exception:
            print ("!!!!! Problematic SMILES (MolToSmiles_ZX): ", mol_x)
            smiles_x = ""
    return smiles_x






#====================================================================================================#
# 
def pattern_matching_AP(cmpd_smiles, substructure_smarts, bad_ss_dict={}):
    #####----------return boolean variable (if the compound match the subgraph)
    mol_x = MolFromSmiles_AP(cmpd_smiles, bad_ss_dict)
    try:
        pattern_matching = mol_x.HasSubstructMatch(Chem.MolFromSmarts(substructure_smarts))
    except Exception:
        pattern_matching = False
    return pattern_matching







#====================================================================================================#
# 
def custom_smiles_cleaner(smiles):
    # Remove @TB followed by any number
    smiles = re.sub(r'@TB\d+', '', smiles)
    
    # Remove @SP followed by any number
    smiles = re.sub(r'@SP\d+', '', smiles)
    
    # Handle radicals
    smiles = re.sub(r'\(\*\)', '', smiles)  # Remove (*) 
    smiles = smiles.replace('*', '')        # Remove * 
    
    # Handle * in the middle, beginning, or end of a reaction string
    smiles = re.sub(r'\.\*\.', '.', smiles)  # Middle
    smiles = re.sub(r'^\*\.', '', smiles)    # Beginning
    smiles = re.sub(r'\.\*$', '', smiles)    # End

    # Further handle cases with double dots
    smiles = smiles.replace('..', '.')

    # Further handle cases with starting or ending dots
    smiles = smiles.strip('.')
    
    return smiles







#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# `7MM"""YMM                      db     `7MN.   `7MF'`7MN.   `7MF'`7MM"""YMM       db     `7MMF'        `7MM"""Mq.   db  MMP""MM""YMM `7MMF'  `7MMF'  #
#   MM    `7                     ;MM:      MMN.    M    MMN.    M    MM    `7      ;MM:      MM            MM   `MM. ;MM: P'   MM   `7   MM      MM    #
#   MM   d  ,pW"Wq.`7Mb,od8     ,V^MM.     M YMb   M    M YMb   M    MM   d       ,V^MM.     MM            MM   ,M9 ,V^MM.     MM        MM      MM    #
#   MM""MM 6W'   `Wb MM' "'    ,M  `MM     M  `MN. M    M  `MN. M    MMmmMM      ,M  `MM     MM            MMmmdM9 ,M  `MM     MM        MMmmmmmmMM    #
#   MM   Y 8M     M8 MM        AbmmmqMA    M   `MM.M    M   `MM.M    MM   Y  ,   AbmmmqMA    MM      ,     MM      AbmmmqMA    MM        MM      MM    #
#   MM     YA.   ,A9 MM       A'     VML   M     YMM    M     YMM    MM     ,M  A'     VML   MM     ,M     MM     A'     VML   MM        MM      MM    #
# .JMML.    `Ybmd9'.JMML.   .AMA.   .AMMA.JML.    YM  .JML.    YM  .JMMmmmmMMM.AMA.   .AMMA.JMMmmmmMMM   .JMML. .AMA.   .AMMA.JMML.    .JMML.  .JMML.  #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Functions for AP only.


# MolFromSmiles and MolToSmiles are rewrite to take into account very rare situations when simulate reactions.
# The following two functions are used for Anneal Path only and shall NOT be used for processing SMILES strings.
# Note that isomericSmiles is set to False here!!
def MolFromSmiles_AP(smiles_x, bad_ss_dict={}, isomericSmiles = False):
    mol_x=Chem.MolFromSmiles(smiles_x)
    try: 
        Chem.MolToSmiles(mol_x, isomericSmiles = isomericSmiles)
    except:
        mol_x=Chem.MolFromSmarts(bad_ss_dict[smiles_x])
        #print ("!!!!! Problematic SMILES (MolFromSmiles): ", smiles_x)
    return mol_x


# MolFromSmiles and MolToSmiles are rewrite to take into account very rare situations when simulate reactions.
# The following two functions are used for Anneal Path only and shall NOT be used for processing SMILES strings.
# Note that isomericSmiles is set to False here!!
def MolToSmiles_AP(mol_x, bad_ss_dict={}, isomericSmiles = False):
    # !!!!!: Converting to smarts and back to rdkit-format to ensure uniqueness before converting to smiles
    mol_x_unique=Chem.MolFromSmarts(Chem.MolToSmarts(mol_x))
    smiles_x=Chem.MolToSmiles(mol_x_unique, isomericSmiles = isomericSmiles)
    try:
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles_x), isomericSmiles = isomericSmiles)
    except Exception:
        bad_ss_dict[smiles_x]=Chem.MolToSmarts(mol_x_unique)
        #print ("!!!!! Problematic SMILES (MolToSmiles): ", mol_x)
    return smiles_x



#====================================================================================================#
# Similarity
def similarity_metric_select(fp_a,fp_b,parameter_1,parameter=2):
    if (parameter_1=="top"):
        similarity=DataStructs.FingerprintSimilarity(fp_a,fp_b)
    elif (parameter_1=="MACCS"):
        similarity=DataStructs.FingerprintSimilarity(fp_a,fp_b)
    elif (parameter_1=="atom_pairs"):
        similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    elif (parameter_1=="vec_pairs"):
        similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    elif (parameter_1=="torsions"):
        similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    elif (parameter_1=="FCFP"):
        similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    else: # ECFP
        similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    return similarity




#====================================================================================================#
def generate_fingerprint(smiles_a, parameter_1, parameter_2=2):
    try: 
        cmpd_a=Chem.MolFromSmiles(str(smiles_a))
        if (parameter_1=="top"):
            fp_a=FingerprintMols.FingerprintMol(cmpd_a)
        elif (parameter_1=="MACCS"):
            fp_a=MACCSkeys.GenMACCSKeys(cmpd_a)
        elif (parameter_1=="atom_pairs"):
            fp_a=Pairs.GetAtomPairFingerprint(cmpd_a)
        elif (parameter_1=="vec_pairs"):
            fp_a=Pairs.GetAtomPairFingerprintAsBitVect(cmpd_a)
        elif (parameter_1=="torsions"):
            fp_a=Torsions.GetTopologicalTorsionFingerprintAsIntVect(cmpd_a)
        elif (parameter_1=="FCFP"):
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2,useFeatures=True)
        else: #ECFP
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2)
    except Exception:
        #print ("Rdkit ERROR: generate fingerprint, ", smiles_a)
        cmpd_a=Chem.MolFromSmiles(str('O'))
        if (parameter_1=="top"):
            fp_a=FingerprintMols.FingerprintMol(cmpd_a)
        elif (parameter_1=="MACCS"):
            fp_a=MACCSkeys.GenMACCSKeys(cmpd_a)
        elif (parameter_1=="atom_pairs"):
            fp_a=Pairs.GetAtomPairFingerprint(cmpd_a)
        elif (parameter_1=="vec_pairs"):
            fp_a=Pairs.GetAtomPairFingerprintAsBitVect(cmpd_a)
        elif (parameter_1=="torsions"):
            fp_a=Torsions.GetTopologicalTorsionFingerprintAsIntVect(cmpd_a)
        elif (parameter_1=="FCFP"):
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2,useFeatures=True)
        else: #ECFP
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2)
    return fp_a




#====================================================================================================#
def similarity_score(smiles_a, smiles_b, parameter_1="ECFP", parameter_2=2): # Return the similarity of two compounds
    try:
        # parameter_1 is similarity metric selected
        cmpd_a=Chem.MolFromSmiles(str(smiles_a))
        cmpd_b=Chem.MolFromSmiles(str(smiles_b))
        if (parameter_1=="top"):
            fp_a=FingerprintMols.FingerprintMol(cmpd_a)
            fp_b=FingerprintMols.FingerprintMol(cmpd_b)  
            similarity=DataStructs.FingerprintSimilarity(fp_a,fp_b)
        elif (parameter_1=="MACCS"):
            fp_a=MACCSkeys.GenMACCSKeys(cmpd_a)
            fp_b=MACCSkeys.GenMACCSKeys(cmpd_b)
            similarity=DataStructs.FingerprintSimilarity(fp_a,fp_b)
        elif (parameter_1=="atom_pairs"):
            fp_a=Pairs.GetAtomPairFingerprint(cmpd_a)
            fp_b=Pairs.GetAtomPairFingerprint(cmpd_b)
            similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
        elif (parameter_1=="vec_pairs"):
            fp_a=Pairs.GetAtomPairFingerprintAsBitVect(cmpd_a)
            fp_b=Pairs.GetAtomPairFingerprintAsBitVect(cmpd_b)
            similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
        elif (parameter_1=="torsions"):
            fp_a=Torsions.GetTopologicalTorsionFingerprintAsIntVect(cmpd_a)
            fp_b=Torsions.GetTopologicalTorsionFingerprintAsIntVect(cmpd_b)
            similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
        elif (parameter_1=="FCFP"):
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2,useFeatures=True)
            fp_b=AllChem.GetMorganFingerprint(cmpd_b,parameter_2,useFeatures=True)
            similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
        else: #ECFP
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2)
            fp_b=AllChem.GetMorganFingerprint(cmpd_b,parameter_2)
            similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    except Exception:
        if smiles_a.find("CoA")==-1 and smiles_b.find("CoA")==-1:
            similarity=0
            #print ("Rdkit ERROR: similarity score, ", smiles_a, smiles_b)
        else:
            similarity=1
    return similarity





#====================================================================================================#
def similarity_dict(list_tb_elmnt, list_tb_cmp, parameter_1, num_cmpds=10): 
    # Compare "list to be eliminated" with "list of SMILES to be compared" and return top "num cmpds" compounds
    # Inputs: a list of hashes, a list of SMILES
    # 0. Initialize
    taofactors_list=[] #
    # 1. if using MNA
    if (type(parameter_1)==int):
        for hash_a in list_tb_elmnt:
            taofactor=[]
            for hash_b in list_tb_cmp:
                taofactor.append(similarity_score(hash_a, hash_b, parameter_1))
            taofactors_list.append((hash_a,max(taofactor)))
    # 2. if using SimIndex's fingerprints
    if (type(parameter_1)==str):
        # 2.1. Convert "list to be compared" to "fp2" (fp2 is a list of fingerprints)
        fp2=[]
        for smiles_a in list_tb_cmp:
            # (hash -> SMILES str -> molecule -> fingerprint)
            fp2.append(generate_fingerprint(smiles_a,parameter_1))
        # 2.2. Convert COMPOUNDS in "list to be eliminated" to "fp1" and obtain maximum taofactors for all compounds

        for smiles_a in list_tb_elmnt:
            # (hash -> SMILES str -> molecule -> fingerprint)
            fp1=generate_fingerprint(smiles_a,parameter_1)
            taofactor=[]
            for k in range(len(fp2)):
                taofactor.append(similarity_metric_select(fp1,fp2[k],parameter_1))
            taofactors_list.append((smiles_a,max(taofactor)))
    # 3. Sort the taofactors for all compounds and return top ranked compunds
    taofactors_dict={}
    for (hash,tao) in taofactors_list:
        if hash not in bkgd_cmpd_list:
            taofactors_dict[hash]=tao
    return taofactors_dict






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  MMP""MM""YMM `7MM"""YMM   .M"""bgd MMP""MM""YMM   MMP""MM""YMM `7MM"""YMM   .M"""bgd MMP""MM""YMM   MMP""MM""YMM `7MM"""YMM   .M"""bgd MMP""MM""YMM #
#  P'   MM   `7   MM    `7  ,MI    "Y P'   MM   `7   P'   MM   `7   MM    `7  ,MI    "Y P'   MM   `7   P'   MM   `7   MM    `7  ,MI    "Y P'   MM   `7 #
#       MM        MM   d    `MMb.          MM             MM        MM   d    `MMb.          MM             MM        MM   d    `MMb.          MM      #
#       MM        MMmmMM      `YMMNq.      MM             MM        MMmmMM      `YMMNq.      MM             MM        MMmmMM      `YMMNq.      MM      #
#       MM        MM   Y  , .     `MM      MM             MM        MM   Y  , .     `MM      MM             MM        MM   Y  , .     `MM      MM      #
#       MM        MM     ,M Mb     dM      MM             MM        MM     ,M Mb     dM      MM             MM        MM     ,M Mb     dM      MM      #
#     .JMML.    .JMMmmmmMMM P"Ybmmd"     .JMML.         .JMML.    .JMMmmmmMMM P"Ybmmd"     .JMML.         .JMML.    .JMMmmmmMMM P"Ybmmd"     .JMML.    #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def AP_convert_test1():
    # All of a,b and c are CoA.
    a="CC(C)(COP(=O)(O)OP(=O)(O)OCC1C(C(C(O1)N2C=NC3=C2N=CN=C3N)O)OP(=O)(O)O)C(C(=O)NCCC(=O)NCCS)O"
    b="O=C(NCCS)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)[C@H](O)[C@@H]3OP(=O)(O)O"
    c="CC(C)(COP(=O)(O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C2N=CN=C3N)O)OP(=O)(O)O)[C@H](C(=O)NCCC(=O)NCCS)O"
    d="[H]N=c1c(c([H])nc(n1[H])C([H])([H])[H])C([H])([H])[n+]1cc(CC(N)C(=O)O)c2ccccc21"

    #====================================================================================================#
    print ("\nTesting unique_canonical_smiles_AP()")
    print (unique_canonical_smiles_AP(a,False))
    print (unique_canonical_smiles_AP(b,False))
    print (unique_canonical_smiles_AP(c,False))
    print (unique_canonical_smiles_AP(a,True))
    print (unique_canonical_smiles_AP(b,True))
    print (unique_canonical_smiles_AP(c,True))
    #print (unique_canonical_smiles_AP("CC(C)CC(=O)C(=O)O"))
    #====================================================================================================#
    print ("\nTesting canonical_smiles_AP()")
    print (canonical_smiles_AP(a, False))
    print (canonical_smiles_AP(b, False))
    print (canonical_smiles_AP(c, False))
    print (canonical_smiles_AP(a, True))
    print (canonical_smiles_AP(b, True))
    print (canonical_smiles_AP(c, True))

    print ("\nTesting canonical_smiles_AP() combined with unique_canonical_smiles_AP()")
    print (unique_canonical_smiles_AP(canonical_smiles_AP(c)))
    #====================================================================================================#
    print ("\nTesting GUSmi()")
    GetUnqSmi = Get_Unique_SMILES(isomericSmiles = False, kekuleSmiles = False, canonical = True, SMARTS_bool = False)
    print (GetUnqSmi.UNQSMI(a))
    print (GetUnqSmi.UNQSMI(b))
    print (GetUnqSmi.UNQSMI(c))

    print (GetUnqSmi.UNQSMI("CC1C2C(N(CN2C3=C(N1)N=C(NC3=O)N)C4=CC=C(C=C4)CC(C(C(COC5C(C(C(O5)COP(=O)(O)OC(CCC(=O)O)C(=O)O)O)O)O)O)O)C"))
    print (GetUnqSmi.UNQSMI("CC1Nc2nc(N)[nH]c(=O)c2N2CN(c3ccc(CC(O)C(O)C(O)COC4OC(COP(=O)(O)OC(CCC(=O)O)C(=O)O)C(O)C4O)cc3)C(C)C12"))
    print (GetUnqSmi.UNQSMI("O=C[C@@H](O)[C@@H](O)[C@@H](O)CO"))

    #====================================================================================================#
    print ("\nTesting MolFromSmiles_AP() and MolToSmiles_AP()")
    bad_ss_dict = []
    print (MolToSmiles_AP(MolFromSmiles_AP(a)))
    print (MolToSmiles_AP(MolFromSmiles_AP(b)))
    print (MolToSmiles_AP(MolFromSmiles_AP(c)))
    #====================================================================================================#
    print ("\nTesting unique_canonical_smiles_list_AP()")
    [print (x) for x in unique_canonical_smiles_list_AP([a,b,c])]
    #====================================================================================================#
    print ("\nTesting canonical_smiles_list_AP()")
    bkgd=['O','CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCS','O=P(O)(O)O','O=C=O','N','CCCC(O)CC=O']
    bkgd_cmpd_list = canonical_smiles_list_AP(bkgd)
    print (bkgd_cmpd_list)
    #====================================================================================================#
    print ("\nTesting pattern_matching_AP()")
    print (pattern_matching_AP("CC(=O)O","[CH3:1][C:2](=[O:3])[OH:4]"))
    print (pattern_matching_AP("CC(=O)OP(O)(O)=O","[C:2][O:6][P:7]([OH:8])([OH:9])=[O:10]"))
    print (pattern_matching_AP("CCO","[CH,CH2,CH3:2][C:4][OH:1]"))
    print (pattern_matching_AP("O","[O:1]-[C:2]"))
    






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
if __name__ == '__main__':
    AP_convert_test1()









#------------------------------
#--------------------------------------------------#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#====================================================================================================#
###################################################################################################################
###################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#




