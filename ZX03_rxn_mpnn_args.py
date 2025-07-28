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
import json
import random
import pickle
#--------------------------------------------------#
import math
import numpy as np

from tap import Tap  
from warnings import warn
from packaging import version
from tempfile import TemporaryDirectory
#--------------------------------------------------#
# pip install typed-argument-parser 
# (https://github.com/swansonk14/typed-argument-parser)
# (https://pypi.org/project/typed-argument-parser/)
from typing import List, Union
from typing import List, Optional
from typing_extensions import Literal
#--------------------------------------------------#
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Globals

Metric = Literal['auc'                  ,
                 'prc-auc'              ,
                 'rmse'                 ,
                 'mae'                  ,
                 'mse'                  ,
                 'r2'                   ,
                 'accuracy'             ,
                 'cross_entropy'        ,
                 'binary_cross_entropy' ,
                 'sid'                  ,
                 'wasserstein'          ,
                 'f1'                   ,
                 'mcc'                  ,
                 'bounded_rmse'         ,
                 'bounded_mae'          ,
                 'bounded_mse'          , ]




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#        db          `7MM      `7MM   db   mm     db                             `7MM      `7MM"""YMM                                                  #
#       ;MM:           MM        MM        MM                                      MM        MM    `7                                                  #
#      ,V^MM.     ,M""bMM   ,M""bMM `7MM mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.   ,6"Yb.   MM        MM   d `7MM  `7MM `7MMpMMMb.  ,p6"bo                      #
#     ,M  `MM   ,AP    MM ,AP    MM   MM   MM     MM 6W'   `Wb MM    MM  8)   MM   MM        MM""MM   MM    MM   MM    MM 6M'  OO                      #
#     AbmmmqMA  8MI    MM 8MI    MM   MM   MM     MM 8M     M8 MM    MM   ,pm9MM   MM        MM   Y   MM    MM   MM    MM 8M                           #
#    A'     VML `Mb    MM `Mb    MM   MM   MM     MM YA.   ,A9 MM    MM  8M   MM   MM        MM       MM    MM   MM    MM YM.    ,                     #
#  .AMA.   .AMMA.`Wbmd"MML.`Wbmd"MML.JMML. `Mbmo.JMML.`Ybmd9'.JMML  JMML.`Moo9^Yo.JMML.    .JMML.     `Mbod"YML.JMML  JMML.YMbmd'                      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# checkpoint

def get_checkpoint_paths(checkpoint_path: Optional[str] = None,
                         checkpoint_paths: Optional[List[str]] = None,
                         checkpoint_dir: Optional[str] = None,
                         ext: str = '.pt') -> Optional[List[str]]:

    # Gets a list of checkpoint paths either from a single checkpoint path or from a directory of checkpoints.

    # If CODE `checkpoint_path` is provided, only collects that one checkpoint.
    # If CODE `checkpoint_paths` is provided, collects all of the provided checkpoints.
    # If CODE `checkpoint_dir` is provided, walks the directory and collects all checkpoints.
    # A checkpoint is any file ending in the extension ext.

    # param checkpoint_path: Path to a checkpoint.
    # param checkpoint_paths: List of paths to checkpoints.
    # param checkpoint_dir: Path to a directory containing checkpoints.
    # param ext: The extension which defines a checkpoint file.
    # return A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.

    if sum(var is not None for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]) > 1:
        raise ValueError('Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths')

    if checkpoint_path is not None:
        return [checkpoint_path]

    if checkpoint_paths is not None:
        return checkpoint_paths

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"')
        
        print("checkpoint_paths: ", checkpoint_paths)
        return checkpoint_paths

    return None



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#       .g8"""bgd   .g8""8q. `7MMM.     ,MMF'`7MMM.     ,MMF' .g8""8q. `7MN.   `7MF'
#     .dP'     `M .dP'    `YM. MMMb    dPMM    MMMb    dPMM .dP'    `YM. MMN.    M  
#     dM'       ` dM'      `MM M YM   ,M MM    M YM   ,M MM dM'      `MM M YMb   M  
#     MM          MM        MM M  Mb  M' MM    M  Mb  M' MM MM        MM M  `MN. M  
#     MM.         MM.      ,MP M  YM.P'  MM    M  YM.P'  MM MM.      ,MP M   `MM.M  
#     `Mb.     ,' `Mb.    ,dP' M  `YM'   MM    M  `YM'   MM `Mb.    ,dP' M     YMM  
#       `"bmmmd'    `"bmmd"' .JML. `'  .JMML..JML. `'  .JMML. `"bmmd"' .JML.    YM  
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# class CommonArgs contains arguments that are used in both class TrainArgs and class PredictArgs.
class CommonArgs(Tap):

    # List of names of the columns containing SMILES strings.
    # By default, uses the first CODE `number_of_molecules` columns.
    smiles_columns: List[str] = None


    # Number of molecules in each input to the model.
    # This must equal the length of CODE `smiles_columns` (if not CODE `None`).
    number_of_molecules: int = 1


    # Directory from which to load model checkpoints (walks directory and ensembles all models that are found).
    checkpoint_dir: str = None


    # Path to model checkpoint ( CODE `.pt` file).
    checkpoint_path: str = None


    # List of paths to model checkpoints ( CODE `.pt` files).
    checkpoint_paths: List[str] = None


    # Turn off cuda (i.e., use CPU instead of GPU).
    no_cuda: bool = False


    # Which GPU to use.
    gpu: int = None


    # Method(s) of generating additional features.
    features_generator: List[str] = None


    # Path(s) to features to use in FNN (instead of features_generator).
    features_path: List[str] = None


    # Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype.
    phase_features_path: str = None


    # Turn off scaling of features.
    no_features_scaling: bool = False


    # Maximum number of data points to load.    
    max_data_size: int = None


    # Number of workers for the parallel data loading (0 means sequential).
    num_workers: int = 8


    # Batch size.
    batch_size: int = 50


    # Custom extra atom descriptors.
    # CODE `feature`   : used as atom features to featurize a given molecule.
    # CODE `descriptor`: used as descriptor and concatenated to the machine learned atomic representation.
    atom_descriptors: Literal['feature', 'descriptor'] = None


    # Path to the extra atom descriptors.
    atom_descriptors_path: str = None


    # Custom extra bond descriptors.
    # CODE `feature`   : used as bond features to featurize a given molecule.
    # CODE `descriptor`: used as descriptor and concatenated to the machine learned bond representation.
    bond_descriptors: Literal['feature', 'descriptor'] = None


    # Path to the extra bond descriptors that will be used as bond features to featurize a given molecule.
    bond_descriptors_path: str = None


    # Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).
    no_cache_mol: bool = False


    # Whether to empty all caches before training or predicting. This is necessary if multiple jobs are run within a single script and the atom or bond features change.
    empty_cache: bool = False


    # Path to constraints applied to atomic/bond properties prediction. (ZX_USELESS ???)
    constraints_path: str = None



    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)
        self._atom_features_size = 0
        self._bond_features_size = 0
        self._atom_descriptors_size = 0
        self._bond_descriptors_size = 0
        self._atom_constraints = []
        self._bond_constraints = []


    @property
    def device(self) -> torch.device:
        # The CODE `torch.device` on which to load and process data and models.
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)


    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index


    @property
    def cuda(self) -> bool:
        # Whether to use CUDA (i.e., GPUs) or not.
        return not self.no_cuda and torch.cuda.is_available()


    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda


    @property
    def features_scaling(self) -> bool:
        # Whether to apply normalization with a  CLASS `~chemprop.data.scaler.StandardScaler`
        # to the additional molecule-level features.
        return not self.no_features_scaling


    @features_scaling.setter
    def features_scaling(self, features_scaling: bool) -> None:
        self.no_features_scaling = not features_scaling


    @property
    def atom_features_size(self) -> int:
        # The size of the atom features.
        return self._atom_features_size


    @atom_features_size.setter
    def atom_features_size(self, atom_features_size: int) -> None:
        self._atom_features_size = atom_features_size


    @property
    def atom_descriptors_size(self) -> int:
        # The size of the atom descriptors.
        return self._atom_descriptors_size


    @atom_descriptors_size.setter
    def atom_descriptors_size(self, atom_descriptors_size: int) -> None:
        self._atom_descriptors_size = atom_descriptors_size


    @property
    def bond_features_size(self) -> int:
        # The size of the atom features.
        return self._bond_features_size


    @bond_features_size.setter
    def bond_features_size(self, bond_features_size: int) -> None:
        self._bond_features_size = bond_features_size


    @property
    def bond_descriptors_size(self) -> int:
        """The size of the bond descriptors."""
        return self._bond_descriptors_size


    @bond_descriptors_size.setter
    def bond_descriptors_size(self, bond_descriptors_size: int) -> None:
        self._bond_descriptors_size = bond_descriptors_size


    def configure(self) -> None:
        self.add_argument('--gpu', choices=list(range(torch.cuda.device_count())))
        #self.add_argument('--features_generator', choices=get_available_features_generators())


    def process_args(self) -> None:
        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path   = self.checkpoint_path  ,
            checkpoint_paths  = self.checkpoint_paths ,
            checkpoint_dir    = self.checkpoint_dir   , )

        # Validate features
        if self.features_generator is not None and 'rdkit_2d_normalized' in self.features_generator and self.features_scaling:
            raise ValueError('When using rdkit_2d_normalized features, --no_features_scaling must be specified.')

        # Validate atom descriptors
        if (self.atom_descriptors is None) != (self.atom_descriptors_path is None):
            raise ValueError('If atom_descriptors is specified, then an atom_descriptors_path must be provided '
                             'and vice versa.')

        if self.atom_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                      'per input (i.e., number_of_molecules = 1).')

        # Validate bond descriptors
        if (self.bond_descriptors is None) != (self.bond_descriptors_path is None):
            raise ValueError('If bond_descriptors is specified, then an bond_descriptors_path must be provided '
                             'and vice versa.')

        if self.bond_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError('Bond descriptors are currently only supported with one molecule '
                                      'per input (i.e., number_of_molecules = 1).')






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    MMP""MM""YMM `7MM"""Mq.       db     `7MMF'`7MN.   `7MF'            db     `7MM"""Mq.   .g8"""bgd          
#    P'   MM   `7   MM   `MM.     ;MM:      MM    MMN.    M             ;MM:      MM   `MM..dP'     `M          
#         MM        MM   ,M9     ,V^MM.     MM    M YMb   M            ,V^MM.     MM   ,M9 dM'       `  ,pP"Ybd 
#         MM        MMmmdM9     ,M  `MM     MM    M  `MN. M           ,M  `MM     MMmmdM9  MM           8I   `" 
#         MM        MM  YM.     AbmmmqMA    MM    M   `MM.M           AbmmmqMA    MM  YM.  MM.    `7MMF'`YMMMa. 
#         MM        MM   `Mb.  A'     VML   MM    M     YMM          A'     VML   MM   `Mb.`Mb.     MM  L.   I8 
#       .JMML.    .JMML. .JMM.AMA.   .AMMA.JMML..JML.    YM        .AMA.   .AMMA.JMML. .JMM. `"bmmmdPY  M9mmmP' 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# TrainArgs includes  class CommonArgs along with additional arguments used for training a Chemprop model.

class TrainArgs(CommonArgs):

    ###################################################################################################################
    #   __                .g8"""bgd `7MM"""YMM  `7MN.   `7MF'`7MM"""YMM  `7MM"""Mq.       db     `7MMF'               #
    #   MM              .dP'     `M   MM    `7    MMN.    M    MM    `7    MM   `MM.     ;MM:      MM                 #
    #   MM   `MM.       dM'       `   MM   d      M YMb   M    MM   d      MM   ,M9     ,V^MM.     MM                 #
    #   MM     `Mb.     MM            MMmmMM      M  `MN. M    MMmmMM      MMmmdM9     ,M  `MM     MM                 #
    #   MMMMMMMMMMMMD   MM.    `7MMF' MM   Y  ,   M   `MM.M    MM   Y  ,   MM  YM.     AbmmmqMA    MM      ,          #
    #           ,M'     `Mb.     MM   MM     ,M   M     YMM    MM     ,M   MM   `Mb.  A'     VML   MM     ,M          #
    #         .M'         `"bmmmdPY .JMMmmmmMMM .JML.    YM  .JMMmmmmMMM .JMML. .JMM.AMA.   .AMMA.JMMmmmmMMM          #
    ###################################################################################################################
    # General arguments

    # Path to data CSV file.#     
    data_path: str

    # Name of the columns containing target values.
    # By default, uses all columns except the SMILES column and the CODE `ignore_columns`.
    target_columns: List[str] = None

    # Name of the columns to ignore when  CODE `target_columns` is not provided.
    ignore_columns: List[str] = None


    # Type of dataset. This determines the default loss function used during training.
    dataset_type: Literal['regression', 'classification', 'multiclass', 'spectra']


    # Choice of loss function. Loss functions are limited to compatible dataset types.
    loss_function: Literal['mse', 'bounded_mse', 'binary_cross_entropy', 'cross_entropy', 'mcc', 'sid', 'wasserstein', 'mve', 'evidential', 'dirichlet'] = None


    # Number of classes when running multiclass classification.
    multiclass_num_classes: int = 3


    # Path to separate val set, optional.
    separate_val_path: str = None


    # Path to separate test set, optional.
    separate_test_path: str = None


    # Path to a file containing a phase mask array, used for excluding particular regions in spectra predictions.
    spectra_phase_mask_path: str = None



    # Path to weights for each molecule in the training data, affecting the relative weight of molecules in the loss function.    
    data_weights_path: str = None


    # Weights associated with each target, affecting the relative weight of targets in the loss function. Must match the number of target columns.
    target_weights: List[float] = None


    # Method of splitting the data into train/val/test.
    split_type: Literal['random', 'scaffold_balanced', 'predetermined', 'crossval', 'cv', 'cv-no-test', 'index_predetermined', 'random_with_repeated_smiles'] = 'random'


    # Split proportions for train/validation/test sets.
    split_sizes: List[float] = None


    # The index of the key molecule used for splitting when multiple molecules are present and 
    # constrained split_type is used, like scaffold_balanced or random_with_repeated_smiles.
    # Note that this index begins with zero for the first molecule.
    split_key_molecule: int = 0



    # Number of folds when performing cross validation.    
    num_folds: int = 1


    # Optional file of fold labels.
    folds_file: str = None


    # Which fold to use as val for leave-one-out cross val.
    val_fold_index: int = None


    # Which fold to use as test for leave-one-out cross val.
    test_fold_index: int = None


    # Directory in which to find cross validation index files.
    crossval_index_dir: str = None


    # Indices of files to use as train/val/test. Overrides  CODE `--num_folds` and  CODE `--seed`.
    crossval_index_file: str = None



    # Random seed to use when splitting data into train/val/test sets.
    # When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.    
    seed: int = 0


    # Seed for PyTorch randomness (e.g., random initial weights).
    pytorch_seed: int = 0


    # Metric to use during evaluation. It is also used with the validation set for early stopping.
    # Defaults to "auc" for classification, "rmse" for regression, and "sid" for spectra.
    metric: Metric = None



    # Additional metrics to use to evaluate the model. Not used for early stopping.
    extra_metrics: List[Metric] = []


    # Directory where model checkpoints will be saved.
    save_dir: str = None


    # Path to model checkpoint file to be loaded for overwriting and freezing weights.
    checkpoint_frzn: str = None


    # Save smiles for each train/val/test splits for prediction convenience later.    
    save_smiles_splits: bool = False


    # Whether to skip training and only test the model.
    test: bool = False


    # Skip non-essential print statements.
    quiet: bool = False


    # The number of batches between each logging of the training loss.
    log_frequency: int = 10


    # Show all scores for individual targets, not just average, at the end.
    show_individual_scores: bool = False


    # Maximum number of molecules in dataset to allow caching.
    # Below this number, caching is used and data loading is sequential.
    # Above this number, caching is not used and data loading is parallel.
    # Use "inf" to always cache.
    cache_cutoff: float = 10000



    # Whether to save test split predictions during training.
    save_preds: bool = False


    # Whether to resume the experiment.
    # Loads test results from any folds that have already been completed and skips training those folds.
    resume_experiment: bool = False



    ###################################################################################################################
    #   __                `7MMM.     ,MMF' .g8""8q. `7MM"""Yb. `7MM"""YMM  `7MMF'                                     #
    #   MM                  MMMb    dPMM .dP'    `YM. MM    `Yb. MM    `7    MM                                       #
    #   MM   `MM.           M YM   ,M MM dM'      `MM MM     `Mb MM   d      MM                                       #
    #   MM     `Mb.         M  Mb  M' MM MM        MM MM      MM MMmmMM      MM                                       #
    #   MMMMMMMMMMMMD       M  YM.P'  MM MM.      ,MP MM     ,MP MM   Y  ,   MM      ,                                #
    #           ,M'         M  `YM'   MM `Mb.    ,dP' MM    ,dP' MM     ,M   MM     ,M                                #
    #         .M'         .JML. `'  .JMML. `"bmmd"' .JMMmmmdP' .JMMmmmmMMM .JMMmmmmMMM                                #
    ###################################################################################################################
    # Model arguments

    # Whether to add bias to linear layers.
    bias: bool = False



    # Dimensionality of hidden layers in MPN.    
    hidden_size: int = 300


    # Number of message passing steps.
    depth: int = 3


    # Whether to add bias to linear layers for solvent MPN if  CODE `reaction_solvent` is True.
    bias_solvent: bool = False


    # Dimensionality of hidden layers in solvent MPN if  CODE `reaction_solvent` is True.
    hidden_size_solvent: int = 300


    # Number of message passing steps for solvent if  CODE `reaction_solvent` is True.
    depth_solvent: int = 3


    # Whether to use the same message passing neural network for all input molecules
    # Only relevant if CODE `number_of_molecules > 1`
    mpn_shared: bool = False


    # Dropout probability.
    dropout: float = 0.0


    # Activation function.
    activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'


    # Centers messages on atoms instead of on bonds.
    atom_messages: bool = False


    # Undirected edges (always sum the two relevant bond vectors).
    undirected: bool = False


    # Hidden dim for higher-capacity FFN (defaults to hidden_size).
    ffn_hidden_size: int = None


    # Number of layers in FFN after MPN encoding.
    ffn_num_layers: int = 2


    # Use only the additional features in an FFN, no graph network.
    features_only: bool = False
    


    # Path to file with features for separate val set.
    separate_val_features_path: List[str] = None


    # Path to file with features for separate test set.
    separate_test_features_path: List[str] = None


    # Path to file with phase features for separate val set.
    separate_val_phase_features_path: str = None


    # Path to file with phase features for separate test set.
    separate_test_phase_features_path: str = None


    # Path to file with extra atom descriptors for separate val set.
    separate_val_atom_descriptors_path: str = None


    # Path to file with extra atom descriptors for separate test set.
    separate_test_atom_descriptors_path: str = None


    # Path to file with extra atom descriptors for separate val set.
    separate_val_bond_descriptors_path: str = None


    # Path to file with extra atom descriptors for separate test set.
    separate_test_bond_descriptors_path: str = None


    # Path to file with constraints for separate val set.
    separate_val_constraints_path: str = None


    # Path to file with constraints for separate test set.
    separate_test_constraints_path: str = None
    


    # Path to a  CODE `.json` file containing arguments. Any arguments present in the config file
    # will override arguments specified via the command line or by the defaults.
    config_path: str = None



    # Number of models in ensemble.
    ensemble_size: int = 1


    # Aggregation scheme for atomic vectors into molecular vectors
    aggregation: Literal['mean', 'sum', 'norm'] = 'mean'


    # For norm aggregation, number by which to divide summed up atomic features
    aggregation_norm: int = 100


    # Whether to adjust MPNN layer to take reactions as input instead of molecules.
    reaction: bool = False



    # Choices for construction of atom and bond features for reactions
    # CODE `reac_prod`        : concatenates the reactants feature with the products feature.
    # CODE `reac_diff`        : concatenates the reactants feature with the difference in features between reactants and products.
    # CODE `prod_diff`        : concatenates the products feature with the difference in features between reactants and products.
    # CODE `reac_prod_balance`: concatenates the reactants feature with the products feature, balances imbalanced reactions.
    # CODE `reac_diff_balance`: concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions.
    # CODE `prod_diff_balance`: concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions.
    reaction_mode: Literal['reac_prod', 'reac_diff', 'prod_diff', 'reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance'] = 'reac_diff'



    # Whether to adjust the MPNN layer to take as input a reaction and a molecule, and to encode them with separate MPNNs.
    reaction_solvent: bool = False


    # Whether H are explicitly specified in input (and should be kept this way). This option is intended to be used
    # with the  CODE `reaction` or  CODE `reaction_solvent` options, and applies only to the reaction part.
    explicit_h: bool = False



    # Whether RDKit molecules will be constructed with adding the Hs to them. This option is intended to be used
    # with Chemprop's default molecule or multi-molecule encoders, or in  CODE `reaction_solvent` mode where it applies to the solvent only.
    adding_h: bool = False




    is_atom_bond_targets: bool = False
    """
    whether this is atomic/bond properties prediction.
    """


    keeping_atom_map: bool = False
    """
    Whether RDKit molecules keep the original atom mapping. This option is intended to be used when providing atom-mapped SMILES with
    the :code:`is_atom_bond_targets`.
    """


    no_shared_atom_bond_ffn: bool = False
    """
    Whether the FFN weights for atom and bond targets should be independent between tasks.
    """


    weights_ffn_num_layers: int = 2
    """
    Number of layers in FFN for determining weights used in constrained targets.
    """



    no_adding_bond_types: bool = False
    """
    Whether the bond types determined by RDKit molecules added to the output of bond targets. This option is intended to be used
    with the :code:`is_atom_bond_targets`.
    """






    ###################################################################################################################
    #   __                  MMP""MM""YMM `7MM"""Mq.       db     `7MMF'`7MN.   `7MF'                                  #
    #   MM                  P'   MM   `7   MM   `MM.     ;MM:      MM    MMN.    M                                    #
    #   MM   `MM.                MM        MM   ,M9     ,V^MM.     MM    M YMb   M                                    #
    #   MM     `Mb.              MM        MMmmdM9     ,M  `MM     MM    M  `MN. M                                    #
    #   MMMMMMMMMMMMD            MM        MM  YM.     AbmmmqMA    MM    M   `MM.M                                    #
    #           ,M'              MM        MM   `Mb.  A'     VML   MM    M     YMM                                    #
    #         .M'              .JMML.    .JMML. .JMM.AMA.   .AMMA.JMML..JML.    YM                                    #
    ###################################################################################################################
    # Training arguments

    # Number of epochs to run.
    epochs: int = 30


    # Number of epochs during which learning rate increases linearly from  CODE `init_lr` to  CODE `max_lr`.
    # Afterwards, learning rate decreases exponentially from  CODE `max_lr` to  CODE `final_lr`.
    warmup_epochs: float = 2.0


    # Initial learning rate.
    init_lr: float = 1e-4


    # Maximum learning rate.
    max_lr: float = 1e-3


    # Final learning rate.
    final_lr: float = 1e-4


    # Maximum magnitude of gradient during training.
    grad_clip: float = None


    # Trains with an equal number of positives and negatives in each batch.
    class_balance: bool = False


    # Indicates which function to use in dataset_type spectra training to constrain outputs to be positive.
    spectra_activation: Literal['exp', 'softplus'] = 'exp'


    # Values in targets for dataset type spectra are replaced with this value, intended to be a small positive number used to enforce positive values.
    spectra_target_floor: float = 1e-8


    # Value used in regularization for evidential loss function. Value used in literature was 1.
    evidential_regularization: float = 0


    # Overwrites the default atom descriptors with the new ones instead of concatenating them.
    # Can only be used if atom_descriptors are used as a feature.
    overwrite_default_atom_features: bool = False


    # Turn off atom feature scaling.
    no_atom_descriptor_scaling: bool = False


    # Overwrites the default bond descriptors with the new ones instead of concatenating them.
    # Can only be used if bond_descriptors are used as a feature.
    overwrite_default_bond_features: bool = False


    # Turn off bond feature scaling.
    no_bond_descriptor_scaling: bool = False


    # Overwrites weights for the first n layers of the ffn from checkpoint model (specified checkpoint_frzn),
    # where n is specified in the input.
    # Automatically also freezes mpnn weights.
    frzn_ffn_layers: int = 0


    # Determines whether or not to use checkpoint_frzn for just the first encoder.
    # Default (False) is to use the checkpoint to freeze all encoders.
    # (only relevant for number_of_molecules > 1, where checkpoint model has number_of_molecules = 1)
    freeze_first_only: bool = False


    # Extra_Feature_Dim (ZX NEWLY ADDED !!)
    Extra_Atom_Feature_Dim: int = 0
    Extra_Bond_Feature_Dim: int = 0


    ###################################################################################################################
    #   __                    .M"""bgd `7MM"""YMM  `7MMF'      `7MM"""YMM                                             #
    #   MM                   ,MI    "Y   MM    `7    MM          MM    `7                                             #
    #   MM   `MM.            `MMb.       MM   d      MM          MM   d                                               #
    #   MM     `Mb.            `YMMNq.   MMmmMM      MM          MM""MM                                               #
    #   MMMMMMMMMMMMD        .     `MM   MM   Y  ,   MM      ,   MM   Y                                               #
    #           ,M'          Mb     dM   MM     ,M   MM     ,M   MM                                                   #
    #         .M'            P"Ybmmd"  .JMMmmmmMMM .JMMmmmmMMM .JMML.                                                 #
    ###################################################################################################################

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names          = None
        self._crossval_index_sets = None
        self._task_names          = None
        self._num_tasks           = None
        self._features_size       = None
        self._train_data_size     = None

    @property
    def metrics(self) -> List[str]:
        # The list of metrics used for evaluation. Only the first is used for early stopping.# 
        return [self.metric] + self.extra_metrics

    @property
    def minimize_score(self) -> bool:
        # Whether the model should try to minimize the score metric or maximize it.# 
        return self.metric in {'rmse', 'mae', 'mse', 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein', 'bounded_mse', 'bounded_mae', 'bounded_rmse'}

    @property
    def use_input_features(self) -> bool:
        # Whether the model is using additional molecule-level features.# 
        return self.features_generator is not None or self.features_path is not None or self.phase_features_path is not None

    @property
    def num_lrs(self) -> int:
        # The number of learning rates to use (currently hard-coded to 1).# 
        return 1

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        # Index sets used for splitting data into train/validation/test during cross-validation# 
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        # A list of names of the tasks being trained on.# 
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        # The number of tasks being trained on.# 
        return len(self.task_names) if self.task_names is not None else 0

    @property
    def features_size(self) -> int:
        # The dimensionality of the additional molecule-level features.# 
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        # The size of the training data set.# 
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size

    @property
    def atom_descriptor_scaling(self) -> bool:
        # Whether to apply normalization with a  CLASS `~chemprop.data.scaler.StandardScaler`
        # to the additional atom features."
        return not self.no_atom_descriptor_scaling

    @property
    def bond_descriptor_scaling(self) -> bool:
        # Whether to apply normalization with a  CLASS `~chemprop.data.scaler.StandardScaler`
        # to the additional bond features."
        return not self.no_bond_descriptor_scaling

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()

        global temp_save_dir  # Prevents the temporary directory from being deleted upon function return

        # Adapt the number of molecules for reaction_solvent mode
        if self.reaction_solvent is True and self.number_of_molecules != 2:
            raise ValueError('In reaction_solvent mode, --number_of_molecules 2 must be specified.')


        # Load config file
        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)

        # Check whether the number of input columns is two for the reaction_solvent mode
        if self.reaction_solvent is True and len(self.smiles_columns) != 2:
            raise ValueError(f'In reaction_solvent mode, exactly two smiles column must be provided (one for reactions, and one for molecules)')

        # Validate reaction/reaction_solvent mode
        if self.reaction is True and self.reaction_solvent is True:
            raise ValueError('Only reaction or reaction_solvent mode can be used, not both.')
        
        # Create temporary directory as save directory if not provided
        if self.save_dir is None:
            temp_save_dir = TemporaryDirectory()
            self.save_dir = temp_save_dir.name

        # Fix ensemble size if loading checkpoints
        if self.checkpoint_paths is not None and len(self.checkpoint_paths) > 0:
            self.ensemble_size = len(self.checkpoint_paths)

        # Process and validate metric and loss function
        if self.metric is None:
            if self.dataset_type == 'classification':
                self.metric = 'auc'
            elif self.dataset_type == 'multiclass':
                self.metric = 'cross_entropy'
            elif self.dataset_type == 'spectra':
                self.metric = 'sid'
            elif self.dataset_type == 'regression' and self.loss_function == 'bounded_mse':
                self.metric = 'bounded_mse'
            elif self.dataset_type == 'regression':
                self.metric = 'rmse'
            else:
                raise ValueError(f'Dataset type {self.dataset_type} is not supported.')

        if self.metric in self.extra_metrics:
            raise ValueError(f'Metric {self.metric} is both the metric and is in extra_metrics. '
                             f'Please only include it once.')

        for metric in self.metrics:
            if not any([(self.dataset_type == 'classification' and metric in ['auc', 'prc-auc', 'accuracy', 'binary_cross_entropy', 'f1', 'mcc']), 
                    (self.dataset_type == 'regression' and metric in ['rmse', 'mae', 'mse', 'r2', 'bounded_rmse', 'bounded_mae', 'bounded_mse']), 
                    (self.dataset_type == 'multiclass' and metric in ['cross_entropy', 'accuracy', 'f1', 'mcc']),
                    (self.dataset_type == 'spectra' and metric in ['sid','wasserstein'])]):
                raise ValueError(f'Metric "{metric}" invalid for dataset type "{self.dataset_type}".')
        
        if self.loss_function is None:
            if self.dataset_type == 'classification':
                self.loss_function = 'binary_cross_entropy'
            elif self.dataset_type == 'multiclass':
                self.loss_function = 'cross_entropy'
            elif self.dataset_type == 'spectra':
                self.loss_function = 'sid'
            elif self.dataset_type == 'regression':
                self.loss_function = 'mse'
            else:
                raise ValueError(f'Default loss function not configured for dataset type {self.dataset_type}.')

        if self.loss_function != 'bounded_mse' and any(metric in ['bounded_mse', 'bounded_rmse', 'bounded_mae'] for metric in self.metrics):
            raise ValueError('Bounded metrics can only be used in conjunction with the regression loss function bounded_mse.')

        # Validate class balance
        if self.class_balance and self.dataset_type != 'classification':
            raise ValueError('Class balance can only be applied if the dataset type is classification.')

        # Validate features
        if self.features_only and not (self.features_generator or self.features_path):
            raise ValueError('When using features_only, a features_generator or features_path must be provided.')

        # Handle FFN hidden size
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size

        # Handle MPN variants
        if self.atom_messages and self.undirected:
            raise ValueError('Undirected is unnecessary when using atom_messages '
                             'since atom_messages are by their nature undirected.')

        # Validate split type settings
        if not (self.split_type == 'predetermined') == (self.folds_file is not None) == (self.test_fold_index is not None):
            raise ValueError('When using predetermined split type, must provide folds_file and test_fold_index.')

        if not (self.split_type == 'crossval') == (self.crossval_index_dir is not None):
            raise ValueError('When using crossval split type, must provide crossval_index_dir.')

        if not (self.split_type in ['crossval', 'index_predetermined']) == (self.crossval_index_file is not None):
            raise ValueError('When using crossval or index_predetermined split type, must provide crossval_index_file.')

        if self.split_type in ['crossval', 'index_predetermined']:
            with open(self.crossval_index_file, 'rb') as rf:
                self._crossval_index_sets = pickle.load(rf)
            self.num_folds = len(self.crossval_index_sets)
            self.seed = 0
        
        # Validate split size entry and set default values
        if self.split_sizes is None:
            if self.separate_val_path is None and self.separate_test_path is None: # separate data paths are not provided
                self.split_sizes = [0.8, 0.1, 0.1]
            elif self.separate_val_path is not None and self.separate_test_path is None: # separate val path only
                self.split_sizes = [0.8, 0., 0.2]
            elif self.separate_val_path is None and self.separate_test_path is not None: # separate test path only
                self.split_sizes = [0.8, 0.2, 0.]
            else: # both separate data paths are provided
                self.split_sizes = [1., 0., 0.]

        else:
            if not np.isclose(sum(self.split_sizes), 1):
                raise ValueError(f'Provided split sizes of {self.split_sizes} do not sum to 1.')
            if any([size < 0 for size in self.split_sizes]):
                raise ValueError(f'Split sizes must be non-negative. Received split sizes: {self.split_sizes}')


            if len(self.split_sizes) not in [2,3]:
                raise ValueError(f'Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s).')

            if self.separate_val_path is None and self.separate_test_path is None: # separate data paths are not provided
                if len(self.split_sizes) != 3:
                    raise ValueError(f'Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s).')
                if self.split_sizes[0] == 0.:
                    raise ValueError(f'Provided split size for train split must be nonzero. Received split size {self.split_sizes[0]}')
                if self.split_sizes[1] == 0.:
                    raise ValueError(f'Provided split size for validation split must be nonzero. Received split size {self.split_sizes[1]}')

            elif self.separate_val_path is not None and self.separate_test_path is None: # separate val path only
                if len(self.split_sizes) == 2: # allow input of just 2 values
                    self.split_sizes = [self.split_sizes[0], 0., self.split_sizes[1]]
                if self.split_sizes[0] == 0.:
                    raise ValueError('Provided split size for train split must be nonzero.')
                if self.split_sizes[1] != 0.:
                    raise ValueError(f'Provided split size for validation split must be 0 because validation set is provided separately. Received split size {self.split_sizes[1]}')

            elif self.separate_val_path is None and self.separate_test_path is not None: # separate test path only
                if len(self.split_sizes) == 2: # allow input of just 2 values
                    self.split_sizes = [self.split_sizes[0], self.split_sizes[1], 0.]
                if self.split_sizes[0] == 0.:
                    raise ValueError('Provided split size for train split must be nonzero.')
                if self.split_sizes[1] == 0.:
                    raise ValueError('Provided split size for validation split must be nonzero.')
                if self.split_sizes[2] != 0.:
                    raise ValueError(f'Provided split size for test split must be 0 because test set is provided separately. Received split size {self.split_sizes[2]}')


            else: # both separate data paths are provided
                if self.split_sizes != [1., 0., 0.]:
                    raise ValueError(f'Separate data paths were provided for val and test splits. Split sizes should not also be provided. Received split sizes: {self.split_sizes}')

        # Test settings
        if self.test:
            self.epochs = 0

        # Validate features are provided for separate validation or test set for each of the kinds of additional features
        for (features_argument, base_features_path, val_features_path, test_features_path) in [
            ('`--features_path`', self.features_path, self.separate_val_features_path, self.separate_test_features_path),
            ('`--phase_features_path`', self.phase_features_path, self.separate_val_phase_features_path, self.separate_test_phase_features_path),
            ('`--atom_descriptors_path`', self.atom_descriptors_path, self.separate_val_atom_descriptors_path, self.separate_test_atom_descriptors_path),
            ('`--bond_descriptors_path`', self.bond_descriptors_path, self.separate_val_bond_descriptors_path, self.separate_test_bond_descriptors_path),
            ('`--constraints_path`', self.constraints_path, self.separate_val_constraints_path, self.separate_test_constraints_path)
        ]:
            if base_features_path is not None:
                if self.separate_val_path is not None and val_features_path is None:
                    raise ValueError(f'Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the separate validation set.')
                if self.separate_test_path is not None and test_features_path is None:
                    raise ValueError(f'Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the separate test set.')
                


        # validate extra atom descriptor options
        if self.overwrite_default_atom_features and self.atom_descriptors != 'feature':
            raise NotImplementedError('Overwriting of the default atom descriptors can only be used if the'
                                      'provided atom descriptors are features.')

        if not self.atom_descriptor_scaling and self.atom_descriptors is None:
            raise ValueError('Atom descriptor scaling is only possible if additional atom features are provided.')



        # validate extra bond descriptor options
        if self.overwrite_default_bond_features and self.bond_descriptors != 'feature':
            raise NotImplementedError('Overwriting of the default bond descriptors can only be used if the'
                                      'provided bond descriptors are features.')

        if not self.bond_descriptor_scaling and self.bond_descriptors is None:
            raise ValueError('Bond descriptor scaling is only possible if additional bond features are provided.')

        if self.bond_descriptors == 'descriptor' and not self.is_atom_bond_targets:
            raise NotImplementedError('Bond descriptors as descriptor can only be used with `--is_atom_bond_targets`.')



        # normalize target weights
        if self.target_weights is not None:
            avg_weight = sum(self.target_weights)/len(self.target_weights)
            self.target_weights = [w/avg_weight for w in self.target_weights]
            if min(self.target_weights) < 0:
                raise ValueError('Provided target weights must be non-negative.')

        # check if key molecule index is outside of the number of molecules
        if self.split_key_molecule >= self.number_of_molecules:
            raise ValueError('The index provided with the argument `--split_key_molecule` must be less than the number of molecules. Note that this index begins with 0 for the first molecule. ')




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   `7MM"""Mq.`7MM"""Mq. `7MM"""YMM  `7MM"""Yb. `7MMF' .g8"""bgd MMP""MM""YMM          db     `7MM"""Mq.   .g8"""bgd          
#     MM   `MM. MM   `MM.  MM    `7    MM    `Yb. MM .dP'     `M P'   MM   `7         ;MM:      MM   `MM..dP'     `M          
#     MM   ,M9  MM   ,M9   MM   d      MM     `Mb MM dM'       `      MM             ,V^MM.     MM   ,M9 dM'       `  ,pP"Ybd 
#     MMmmdM9   MMmmdM9    MMmmMM      MM      MM MM MM               MM            ,M  `MM     MMmmdM9  MM           8I   `" 
#     MM        MM  YM.    MM   Y  ,   MM     ,MP MM MM.              MM            AbmmmqMA    MM  YM.  MM.    `7MMF'`YMMMa. 
#     MM        MM   `Mb.  MM     ,M   MM    ,dP' MM `Mb.     ,'      MM           A'     VML   MM   `Mb.`Mb.     MM  L.   I8 
#   .JMML.    .JMML. .JMM.JMMmmmmMMM .JMMmmmdP' .JMML. `"bmmmd'     .JMML.       .AMA.   .AMMA.JMML. .JMM. `"bmmmdPY  M9mmmP' 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# class `PredictArgs` includes  CLASS `CommonArgs` along with additional arguments used for predicting with a Chemprop model.

class PredictArgs(CommonArgs):

    test_path: str
    # Path to CSV file containing testing data for which predictions will be made.

    preds_path: str
    # Path to CSV file where predictions will be saved.

    drop_extra_columns: bool = False
    # Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns.

    ensemble_variance: bool = False
    # Deprecated. Whether to calculate the variance of ensembles as a measure of epistemic uncertainty. If True, the variance is saved as an additional column for each target in the preds_path.

    individual_ensemble_predictions: bool = False
    # Whether to return the predictions made by each of the individual models rather than the average of the ensemble"""
    # Uncertainty arguments
    uncertainty_method: Literal[
        'mve'                  ,
        'ensemble'             ,
        'evidential_epistemic' ,
        'evidential_aleatoric' ,
        'evidential_total'     ,
        'classification'       ,
        'dropout'              ,
        'spectra_roundrobin',
    ] = None
    # The method of calculating uncertainty.

    calibration_method: Literal['zscaling', 'tscaling', 'zelikman_interval', 'mve_weighting', 'platt', 'isotonic'] = None
    # Methods used for calibrating the uncertainty calculated with uncertainty method.

    evaluation_methods: List[str] = None
    # The methods used for evaluating the uncertainty performance if the test data provided includes targets.
    # Available methods are [nll, miscalibration_area, ence, spearman] or any available classification or multiclass metric.
    evaluation_scores_path: str = None
    # Location to save the results of uncertainty evaluations.

    uncertainty_dropout_p: float = 0.1
    # The probability to use for Monte Carlo dropout uncertainty estimation.

    dropout_sampling_size: int = 10
    # The number of samples to use for Monte Carlo dropout uncertainty estimation. Distinct from the dropout used during training.

    calibration_interval_percentile: float = 95
    # Sets the percentile used in the calibration methods. Must be in the range (1,100).

    regression_calibrator_metric: Literal['stdev', 'interval'] = None
    # Regression calibrators can output either a stdev or an inverval. # 
    calibration_path: str = None
    # Path to data file to be used for uncertainty calibration.

    calibration_features_path: str = None
    # Path to features data to be used with the uncertainty calibration dataset.

    calibration_phase_features_path: str = None
    #  # 
    calibration_atom_descriptors_path: str = None
    # Path to the extra atom descriptors.

    calibration_bond_descriptors_path: str = None
    # Path to the extra bond descriptors that will be used as bond features to featurize a given molecule.


    @property
    def ensemble_size(self) -> int:
        # The number of models in the ensemble.

        return len(self.checkpoint_paths)

    def process_args(self) -> None:
        super(PredictArgs, self).process_args()

        if self.regression_calibrator_metric is None:
            if self.calibration_method == 'zelikman_interval':
                self.regression_calibrator_metric = 'interval'
            else:
                self.regression_calibrator_metric = 'stdev'

        if self.uncertainty_method == 'dropout' and version.parse(torch.__version__) < version.parse('1.9.0'):
            raise ValueError('Dropout uncertainty is only supported for pytorch versions >= 1.9.0')


        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')

        if self.ensemble_variance == True:
            if self.uncertainty_method in ['ensemble', None]:
                warn(
                    'The `--ensemble_variance` argument is deprecated and should \
                        be replaced with `--uncertainty_method ensemble`.',
                    DeprecationWarning,
                )
                self.uncertainty_method = 'ensemble'
            else:
                raise ValueError(
                    f'Only one uncertainty method can be used at a time. \
                        The arguement `--ensemble_variance` was provided along \
                        with the uncertainty method {self.uncertainty_method}. The `--ensemble_variance` \
                        argument is deprecated and should be replaced with `--uncertainty_method ensemble`.'
                )

        if self.calibration_interval_percentile <= 1 or self.calibration_interval_percentile >= 100:
            raise ValueError('The calibration interval must be a percentile value in the range (1,100).')

        if self.uncertainty_dropout_p < 0 or self.uncertainty_dropout_p > 1:
            raise ValueError('The dropout probability must be in the range (0,1).')

        if self.dropout_sampling_size <= 1:
            raise ValueError('The argument `--dropout_sampling_size` must be an integer greater than 1.')

        # Validate that features provided for the prediction test set are also provided for the calibration set
        for (features_argument, base_features_path, cal_features_path) in [
            ('`--features_path`', self.features_path, self.calibration_features_path),
            ('`--phase_features_path`', self.phase_features_path, self.calibration_phase_features_path),
            ('`--atom_descriptors_path`', self.atom_descriptors_path, self.calibration_atom_descriptors_path),
            ('`--bond_descriptors_path`', self.bond_descriptors_path, self.calibration_bond_descriptors_path)
        ]:
            if base_features_path is not None and self.calibration_path is not None and cal_features_path is None:
                raise ValueError(f'Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the calibration dataset.')






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MMF'`7MN.   `7MF'MMP""MM""YMM `7MM"""YMM  `7MM"""Mq. `7MM"""Mq.`7MM"""Mq. `7MM"""YMM MMP""MM""YMM        db     `7MM"""Mq.   .g8"""bgd            #
#    MM    MMN.    M  P'   MM   `7   MM    `7    MM   `MM.  MM   `MM. MM   `MM.  MM    `7 P'   MM   `7       ;MM:      MM   `MM..dP'     `M            #
#    MM    M YMb   M       MM        MM   d      MM   ,M9   MM   ,M9  MM   ,M9   MM   d        MM           ,V^MM.     MM   ,M9 dM'       `  ,pP"Ybd   #
#    MM    M  `MN. M       MM        MMmmMM      MMmmdM9    MMmmdM9   MMmmdM9    MMmmMM        MM          ,M  `MM     MMmmdM9  MM           8I   `"   #
#    MM    M   `MM.M       MM        MM   Y  ,   MM  YM.    MM        MM  YM.    MM   Y  ,     MM          AbmmmqMA    MM  YM.  MM.    `7MMF'`YMMMa.   #
#    MM    M     YMM       MM        MM     ,M   MM   `Mb.  MM        MM   `Mb.  MM     ,M     MM         A'     VML   MM   `Mb.`Mb.     MM  L.   I8   #
#  .JMML..JML.    YM     .JMML.    .JMMmmmmMMM .JMML. .JMM.JMML.    .JMML. .JMM.JMMmmmmMMM   .JMML.     .AMA.   .AMMA.JMML. .JMM. `"bmmmdPY  M9mmmP'   #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  CLASS `InterpretArgs` includes  CLASS `CommonArgs` along with additional arguments used for interpreting a trained Chemprop model.

class InterpretArgs(CommonArgs):

    # Path to data CSV file.
    data_path: str

    # Batch size.
    batch_size: int = 500

    # Index of the property of interest in the trained model.
    property_id: int = 1

    # Number of rollout steps.
    rollout: int = 20

    # Constant factor in MCTS.
    c_puct: float = 10.0


    # Maximum number of atoms in rationale.    
    max_atoms: int = 20

    # Minimum number of atoms in rationale.
    min_atoms: int = 8

    # Minimum score to count as positive.
    prop_delta: float = 0.5



    def process_args(self) -> None:
        super(InterpretArgs, self).process_args()

        if self.features_path is not None:
            raise ValueError('Cannot use --features_path <path> for interpretation since features '
                             'need to be computed dynamically for molecular substructures. '
                             'Please specify --features_generator <generator>.')

        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')











#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MM"""YMM db                                                           db             mm              db     `7MM"""Mq.   .g8"""bgd                #
#    MM    `7                                                                             MM             ;MM:      MM   `MM..dP'     `M                #
#    MM   d `7MM `7MMpMMMb.  .P"Ybmmm .gP"Ya `7Mb,od8 `7MMpdMAo.`7Mb,od8 `7MM `7MMpMMMb.mmMMmm          ,V^MM.     MM   ,M9 dM'       `  ,pP"Ybd       #
#    MM""MM   MM   MM    MM :MI  I8  ,M'   Yb  MM' "'   MM   `Wb  MM' "'   MM   MM    MM  MM           ,M  `MM     MMmmdM9  MM           8I   `"       #
#    MM   Y   MM   MM    MM  WmmmP"  8M""""""  MM       MM    M8  MM       MM   MM    MM  MM           AbmmmqMA    MM  YM.  MM.    `7MMF'`YMMMa.       #
#    MM       MM   MM    MM 8M       YM.    ,  MM       MM   ,AP  MM       MM   MM    MM  MM          A'     VML   MM   `Mb.`Mb.     MM  L.   I8       #
#  .JMML.   .JMML.JMML  JMML.YMMMMMb  `Mbmmd'.JMML.     MMbmmd' .JMML.   .JMML.JMML  JMML.`Mbmo     .AMA.   .AMMA.JMML. .JMM. `"bmmmdPY  M9mmmP'       #
#                           6'     dP                   MM                                                                                             #
#                           Ybmmmd'                   .JMML.                                                                                           #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  CLASS `FingerprintArgs` includes  CLASS `PredictArgs` with additional arguments for the generation of latent fingerprint vectors.

class FingerprintArgs(PredictArgs):

    # Choice of which type of latent fingerprint vector to use. Default is the output of the MPNN, excluding molecular features
    fingerprint_type: Literal['MPN', 'last_FFN'] = 'MPN'






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MMF'  `7MMF'                                                         mm            db     `7MM"""Mq.   .g8"""bgd            
#    MM      MM                                                           MM           ;MM:      MM   `MM..dP'     `M            
#    MM      MM `7M'   `MF'`7MMpdMAo.  .gP"Ya `7Mb,od8 ,pW"Wq.`7MMpdMAo.mmMMmm        ,V^MM.     MM   ,M9 dM'       `  ,pP"Ybd   
#    MMmmmmmmMM   VA   ,V    MM   `Wb ,M'   Yb  MM' "'6W'   `Wb MM   `Wb  MM         ,M  `MM     MMmmdM9  MM           8I   `"   
#    MM      MM    VA ,V     MM    M8 8M""""""  MM    8M     M8 MM    M8  MM         AbmmmqMA    MM  YM.  MM.    `7MMF'`YMMMa.   
#    MM      MM     VVV      MM   ,AP YM.    ,  MM    YA.   ,A9 MM   ,AP  MM        A'     VML   MM   `Mb.`Mb.     MM  L.   I8   
#  .JMML.  .JMML.   ,V       MMbmmd'   `Mbmmd'.JMML.   `Ybmd9'  MMbmmd'   `Mbmo   .AMA.   .AMMA.JMML. .JMM. `"bmmmdPY  M9mmmP'   
#                  ,V        MM                                 MM                                                               
#               OOb"       .JMML.                             .JMML.                                                             
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  CLASS `HyperoptArgs` includes  CLASS `TrainArgs` along with additional arguments used for optimizing Chemprop hyperparameters.
class HyperoptArgs(TrainArgs):


    # Number of hyperparameter choices to try.
    num_iters: int = 20

    # Path to  CODE `.json` file where best hyperparameter settings will be written.
    config_save_path: str

    # (Optional) Path to a directory where all results of the hyperparameter optimization will be written.
    log_dir: str = None

    # Path to a directory where hyperopt completed trial data is stored. Hyperopt job will include these trials if restarted.
    # Can also be used to run multiple instances in parallel if they share the same checkpoint directory.
    hyperopt_checkpoint_dir: str = None

    # The initial number of trials that will be randomly specified before TPE algorithm is used to select the rest.
    # By default will be half the total number of trials.
    startup_random_iters: int = None

    # Paths to save directories for manually trained models in the same search space as the hyperparameter search.
    # Results will be considered as part of the trial history of the hyperparameter search.
    manual_trial_dirs: List[str] = None

    # The model parameters over which to search for an optimal hyperparameter configuration.
    # Some options are bundles of parameters or otherwise special parameter operations.
    search_parameter_keywords: List[str] = ["basic"]


    # Special keywords:
    #     basic - the default set of hyperparameters for search: depth, ffn_num_layers, dropout, and linked_hidden_size.
    #     linked_hidden_size - search for hidden_size and ffn_hidden_size, but constrained for them to have the same value.
    #         If either of the component words are entered in separately, both are searched independently.
    #     learning_rate - search for max_lr, init_lr, final_lr, and warmup_epochs. The search for init_lr and final_lr values
    #         are defined as fractions of the max_lr value. The search for warmup_epochs is as a fraction of the total epochs used.
    #     all - include search for all 13 inidividual keyword options

    # Individual supported parameters:
    #     activation, aggregation, aggregation_norm, batch_size, depth,
    #     dropout, ffn_hidden_size, ffn_num_layers, final_lr, hidden_size,
    #     init_lr, max_lr, warmup_epochs


    def process_args(self) -> None:
        super(HyperoptArgs, self).process_args()

        # Assign log and checkpoint directories if none provided
        if self.log_dir is None:
            self.log_dir = self.save_dir
        if self.hyperopt_checkpoint_dir is None:
            self.hyperopt_checkpoint_dir = self.log_dir
        
        # Set number of startup random trials
        if self.startup_random_iters is None:
            self.startup_random_iters = self.num_iters // 2

        # Construct set of search parameters
        supported_keywords = [
            "basic", "learning_rate", "linked_hidden_size", "all",
            "activation", "aggregation", "aggregation_norm", "batch_size", "depth",
            "dropout", "ffn_hidden_size", "ffn_num_layers", "final_lr", "hidden_size",
            "init_lr", "max_lr", "warmup_epochs"
        ]
        supported_parameters = [
            "activation", "aggregation", "aggregation_norm", "batch_size", "depth",
            "dropout", "ffn_hidden_size", "ffn_num_layers", "final_lr_ratio", "hidden_size",
            "init_lr_ratio", "linked_hidden_size", "max_lr", "warmup_epochs"
        ]
        unsupported_keywords = set(self.search_parameter_keywords) - set(supported_keywords)
        if len(unsupported_keywords) != 0:
            raise NotImplementedError(
                f"Keywords for what hyperparameters to include in the search are designated \
                    with the argument `--search_parameter_keywords`. The following unsupported\
                    keywords were received: {unsupported_keywords}. The available supported\
                    keywords are: {supported_keywords}"
            )
        search_parameters = set()
        if "all" in self.search_parameter_keywords:
            search_parameters.update(supported_parameters)
        if "basic" in self.search_parameter_keywords:
            search_parameters.update(["depth", "ffn_num_layers", "dropout", "linked_hidden_size"])
        if "learning_rate" in self.search_parameter_keywords:
            search_parameters.update(["max_lr", "init_lr_ratio", "final_lr_ratio", "warmup_epochs"])
        for kw in self.search_parameter_keywords:
            if kw in supported_parameters:
                search_parameters.add(kw)
        if "init_lr" in self.search_parameter_keywords:
            search_parameters.add("init_lr_ratio")
        if "final_lr" in self.search_parameter_keywords:
            search_parameters.add("final_lr_ratio")
        if "linked_hidden_size" in search_parameters and ("hidden_size" in search_parameters or "ffn_hidden_size" in search_parameters):
            search_parameters.remove("linked_hidden_size")
            search_parameters.update(["hidden_size", "ffn_hidden_size"])
        self.search_parameters = list(search_parameters)











#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    .M"""bgd `7MM     `7MM                                                db     `7MM"""Mq.   .g8"""bgd                                               #
#   ,MI    "Y   MM       MM                                               ;MM:      MM   `MM..dP'     `M                                               #
#   `MMb.       MM  ,MP' MM  .gP"Ya   ,6"Yb. `7Mb,od8 `7MMpMMMb.         ,V^MM.     MM   ,M9 dM'       `  ,pP"Ybd                                      #
#     `YMMNq.   MM ;Y    MM ,M'   Yb 8)   MM   MM' "'   MM    MM        ,M  `MM     MMmmdM9  MM           8I   `"                                      #
#   .     `MM   MM;Mm    MM 8M""""""  ,pm9MM   MM       MM    MM        AbmmmqMA    MM  YM.  MM.    `7MMF'`YMMMa.                                      #
#   Mb     dM   MM `Mb.  MM YM.    , 8M   MM   MM       MM    MM       A'     VML   MM   `Mb.`Mb.     MM  L.   I8                                      #
#   P"Ybmmd"  .JMML. YA.JMML.`Mbmmd' `Moo9^Yo.JMML.   .JMML  JMML.   .AMA.   .AMMA.JMML. .JMM. `"bmmmdPY  M9mmmP'                                      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# The CLASS `SklearnTrainArgs` includes  CLASS `TrainArgs` along with additional arguments for training a scikit-learn model.

class SklearnTrainArgs(TrainArgs):

    # scikit-learn model to use.
    model_type: Literal['random_forest', 'svm']

    # How to weight classes (None means no class balance).
    class_weight: Literal['balanced'] = None

    # Whether to run each task separately (needed when dataset has null entries).
    single_task: bool = False

    # Morgan fingerprint radius.
    radius: int = 2

    # Number of bits in morgan fingerprint.
    num_bits: int = 2048

    # Number of random forest trees.
    num_trees: int = 500

    # How to impute missing data (None means no imputation).
    impute_mode: Literal['single_task', 'median', 'mean', 'linear','frequent'] = None






###################################################################################################################
###################################################################################################################
class SklearnPredictArgs(Tap):
    #  CLASS `SklearnPredictArgs` contains arguments used for predicting with a trained scikit-learn model.

    # Path to CSV file containing testing data for which predictions will be made.
    test_path: str


    # List of names of the columns containing SMILES strings.
    # By default, uses the first  CODE `number_of_molecules` columns.    
    smiles_columns: List[str] = None

    # Number of molecules in each input to the model.
    # This must equal the length of  CODE `smiles_columns` (if not  CODE `None`).
    number_of_molecules: int = 1

    # Path to CSV file where predictions will be saved.
    preds_path: str

    # Path to directory containing model checkpoints ( CODE `.pkl` file).
    checkpoint_dir: str = None

    # Path to model checkpoint ( CODE `.pkl` file)
    checkpoint_path: str = None

    # List of paths to model checkpoints ( CODE `.pkl` files).
    checkpoint_paths: List[str] = None


    def process_args(self) -> None:

        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(checkpoint_path    =  self.checkpoint_path,
                                                     checkpoint_paths   =  self.checkpoint_paths,
                                                     checkpoint_dir     =  self.checkpoint_dir,
                                                     ext                =  '.pkl'
        )



















