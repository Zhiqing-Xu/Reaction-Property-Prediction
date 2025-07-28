# from args
import json
import os
from tempfile import TemporaryDirectory
import pickle
from typing import Dict, Iterator, List, Optional, Union, Tuple
from typing_extensions import Literal
from packaging import version
from warnings import warn
import torch
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)
import numpy as np
from rdkit import Chem

import csv
from simplify_chemprop_features import get_available_features_generators,MolGraph

Metric = Literal['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein', 'f1', 'mcc', 'bounded_rmse', 'bounded_mae', 'bounded_mse']

# Cache of graph featurizations
CACHE_GRAPH = True
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}

# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]] = {}


def get_mixed_task_names(path: str,
                         smiles_columns: Union[str, List[str]] = None,
                         target_columns: List[str] = None,
                         ignore_columns: List[str] = None,
                         keep_h: bool = None,
                         add_h: bool = None,
                         keep_atom_map: bool = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Gets the task names for atomic, bond, and molecule targets separately from a data CSV file.

    If :code:`target_columns` is provided, returned lists based off `target_columns`.
    Otherwise, returned lists based off all columns except the :code:`smiles_columns`
    (or the first column, if the :code:`smiles_columns` is None) and
    the :code:`ignore_columns`.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_columns` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h: Boolean whether to add hydrogens to the input smiles.
    :param keep_atom_map: Boolean whether to keep the original atom mapping.
    :return: A tuple containing the task names of atomic, bond, and molecule properties separately.
    """
    columns = get_header(path)


    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    ignore_columns = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))

    if target_columns is not None:
        target_names =  target_columns
    else:
        target_names = [column for column in columns if column not in ignore_columns]

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            atom_target_names, bond_target_names, molecule_target_names = [], [], []
            smiles = [row[c] for c in smiles_columns]
            mol = make_mol(smiles[0], keep_h, add_h, keep_atom_map)
            for column in target_names:
                value = row[column]
                value = value.replace('None', 'null')
                target = np.array(json.loads(value))

                is_atom_target, is_bond_target, is_molecule_target = False, False, False
                if len(target.shape) == 0:
                    is_molecule_target = True
                elif len(target.shape) == 1:
                    if len(mol.GetAtoms()) == len(mol.GetBonds()):
                        break
                    elif len(target) == len(mol.GetAtoms()):  # Atom targets saved as 1D list
                        is_atom_target = True
                    elif len(target) == len(mol.GetBonds()):  # Bond targets saved as 1D list
                        is_bond_target = True
                elif len(target.shape) == 2:  # Bond targets saved as 2D list
                    is_bond_target = True
                else:
                    raise ValueError('Unrecognized targets of column {column} in {path}.')
                
                if is_atom_target:
                    atom_target_names.append(column)
                elif is_bond_target:
                    bond_target_names.append(column)
                elif is_molecule_target:
                    molecule_target_names.append(column)
            if len(atom_target_names) + len(bond_target_names) + len(molecule_target_names) == len(target_names):
                break

    return atom_target_names, bond_target_names, molecule_target_names


def set_cache_mol(cache_mol: bool) -> None:
    r"""Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol


def preprocess_smiles_columns(path: str,
                              smiles_columns: Union[str, List[str]] = None,
                              number_of_molecules: int = 1) -> List[str]:
    """
    Preprocesses the :code:`smiles_columns` variable to ensure that it is a list of column
    headings corresponding to the columns in the data file holding SMILES. Assumes file has a header.
    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules with associated SMILES for each
                           data point.
    :return: The preprocessed version of :code:`smiles_columns` which is guaranteed to be a list.
    """

    if smiles_columns is None:
        if os.path.isfile(path):
            columns = get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None]*number_of_molecules
    else:
        if isinstance(smiles_columns, str):
            smiles_columns = [smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError('Length of smiles_columns must match number_of_molecules.')
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError('Provided smiles_columns do not match the header of data file.')

    return smiles_columns


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.
    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


## from arg.py
def set_cache_graph(cache_graph: bool) -> None:
    r"""Sets whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph


def empty_cache():
    r"""Empties the cache of :class:`~chemprop.features.MolGraph` and RDKit molecules."""
    SMILES_TO_GRAPH.clear()
    SMILES_TO_MOL.clear()


def get_checkpoint_paths(checkpoint_path: Optional[str] = None,
                         checkpoint_paths: Optional[List[str]] = None,
                         checkpoint_dir: Optional[str] = None,
                         ext: str = '.pt') -> Optional[List[str]]:
    """
    Gets a list of checkpoint paths either from a single checkpoint path or from a directory of checkpoints.

    If :code:`checkpoint_path` is provided, only collects that one checkpoint.
    If :code:`checkpoint_paths` is provided, collects all of the provided checkpoints.
    If :code:`checkpoint_dir` is provided, walks the directory and collects all checkpoints.
    A checkpoint is any file ending in the extension ext.

    :param checkpoint_path: Path to a checkpoint.
    :param checkpoint_paths: List of paths to checkpoints.
    :param checkpoint_dir: Path to a directory containing checkpoints.
    :param ext: The extension which defines a checkpoint file.
    :return: A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.
    """
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

        return checkpoint_paths

    return None

class CommonArgs(Tap):
    """:class:`CommonArgs` contains arguments that are used in both :class:`TrainArgs` and :class:`PredictArgs`."""

    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    number_of_molecules: int = 1
    """Number of molecules in each input to the model.
    This must equal the length of :code:`smiles_columns` (if not :code:`None`)."""
    checkpoint_dir: str = None
    """Directory from which to load model checkpoints (walks directory and ensembles all models that are found)."""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pt` files)."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    features_generator: List[str] = None
    """Method(s) of generating additional features."""
    features_path: List[str] = None
    """Path(s) to features to use in FNN (instead of features_generator)."""
    phase_features_path: str = None
    """Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype."""
    no_features_scaling: bool = False
    """Turn off scaling of features."""
    max_data_size: int = None
    """Maximum number of data points to load."""
    num_workers: int = 8
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 50
    """Batch size."""
    atom_descriptors: Literal['feature', 'descriptor'] = None
    """
    Custom extra atom descriptors.
    :code:`feature`: used as atom features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned atomic representation.
    """
    atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    bond_descriptors: Literal['feature', 'descriptor'] = None
    """
    Custom extra bond descriptors.
    :code:`feature`: used as bond features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned bond representation.
    """
    bond_descriptors_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""
    no_cache_mol: bool = False
    """
    Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).
    """
    empty_cache: bool = False
    """
    Whether to empty all caches before training or predicting. This is necessary if multiple jobs are run within a single script and the atom or bond features change.
    """


    constraints_path: str = None
    """
    Path to constraints applied to atomic/bond properties prediction.
    """

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
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    @property
    def features_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional molecule-level features.
        """
        return not self.no_features_scaling

    @features_scaling.setter
    def features_scaling(self, features_scaling: bool) -> None:
        self.no_features_scaling = not features_scaling

    @property
    def atom_features_size(self) -> int:
        """The size of the atom features."""
        return self._atom_features_size

    @atom_features_size.setter
    def atom_features_size(self, atom_features_size: int) -> None:
        self._atom_features_size = atom_features_size

    @property
    def atom_descriptors_size(self) -> int:
        """The size of the atom descriptors."""
        return self._atom_descriptors_size

    @atom_descriptors_size.setter
    def atom_descriptors_size(self, atom_descriptors_size: int) -> None:
        self._atom_descriptors_size = atom_descriptors_size

    @property
    def bond_features_size(self) -> int:
        """The size of the atom features."""
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
        self.add_argument('--features_generator', choices=get_available_features_generators())

    def process_args(self) -> None:
        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
        )

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

        set_cache_mol(not self.no_cache_mol)

        if self.empty_cache:
            empty_cache()







class TrainArgs(CommonArgs):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""


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
    data_path: str
    """Path to data CSV file."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    By default, uses all columns except the SMILES column and the :code:`ignore_columns`.
    """
    ignore_columns: List[str] = None
    """Name of the columns to ignore when :code:`target_columns` is not provided."""
    dataset_type: Literal['regression', 'classification', 'multiclass', 'spectra']
    """Type of dataset. This determines the default loss function used during training."""
    loss_function: Literal['mse', 'bounded_mse', 'binary_cross_entropy', 'cross_entropy', 'mcc', 'sid', 'wasserstein', 'mve', 'evidential', 'dirichlet'] = None
    """Choice of loss function. Loss functions are limited to compatible dataset types."""
    multiclass_num_classes: int = 3
    """Number of classes when running multiclass classification."""
    separate_val_path: str = None
    """Path to separate val set, optional."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    spectra_phase_mask_path: str = None
    """Path to a file containing a phase mask array, used for excluding particular regions in spectra predictions."""
    data_weights_path: str = None
    """Path to weights for each molecule in the training data, affecting the relative weight of molecules in the loss function"""
    target_weights: List[float] = None
    """Weights associated with each target, affecting the relative weight of targets in the loss function. Must match the number of target columns."""
    split_type: Literal['random', 'scaffold_balanced', 'predetermined', 'crossval', 'cv', 'cv-no-test', 'index_predetermined', 'random_with_repeated_smiles'] = 'random'
    """Method of splitting the data into train/val/test."""
    split_sizes: List[float] = None
    """Split proportions for train/validation/test sets."""
    split_key_molecule: int = 0
    """The index of the key molecule used for splitting when multiple molecules are present and constrained split_type is used, like scaffold_balanced or random_with_repeated_smiles.
       Note that this index begins with zero for the first molecule."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    folds_file: str = None
    """Optional file of fold labels."""
    val_fold_index: int = None
    """Which fold to use as val for leave-one-out cross val."""
    test_fold_index: int = None
    """Which fold to use as test for leave-one-out cross val."""
    crossval_index_dir: str = None
    """Directory in which to find cross validation index files."""
    crossval_index_file: str = None
    """Indices of files to use as train/val/test. Overrides :code:`--num_folds` and :code:`--seed`."""
    seed: int = 0
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    pytorch_seed: int = 0
    """Seed for PyTorch randomness (e.g., random initial weights)."""
    metric: Metric = None
    """
    Metric to use during evaluation. It is also used with the validation set for early stopping.
    Defaults to "auc" for classification, "rmse" for regression, and "sid" for spectra.
    """
    extra_metrics: List[Metric] = []
    """Additional metrics to use to evaluate the model. Not used for early stopping."""
    save_dir: str = None
    """Directory where model checkpoints will be saved."""
    checkpoint_frzn: str = None
    """Path to model checkpoint file to be loaded for overwriting and freezing weights."""
    save_smiles_splits: bool = False
    """Save smiles for each train/val/test splits for prediction convenience later."""
    test: bool = False
    """Whether to skip training and only test the model."""
    quiet: bool = False
    """Skip non-essential print statements."""
    log_frequency: int = 10
    """The number of batches between each logging of the training loss."""
    show_individual_scores: bool = False
    """Show all scores for individual targets, not just average, at the end."""
    cache_cutoff: float = 10000
    """
    Maximum number of molecules in dataset to allow caching.
    Below this number, caching is used and data loading is sequential.
    Above this number, caching is not used and data loading is parallel.
    Use "inf" to always cache.
    """
    save_preds: bool = False
    """Whether to save test split predictions during training."""
    resume_experiment: bool = False
    """
    Whether to resume the experiment.
    Loads test results from any folds that have already been completed and skips training those folds.
    """

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
    bias: bool = False
    """Whether to add bias to linear layers."""
    hidden_size: int = 300
    """Dimensionality of hidden layers in MPN."""
    depth: int = 3
    """Number of message passing steps."""
    bias_solvent: bool = False
    """Whether to add bias to linear layers for solvent MPN if :code:`reaction_solvent` is True."""
    hidden_size_solvent: int = 300
    """Dimensionality of hidden layers in solvent MPN if :code:`reaction_solvent` is True."""
    depth_solvent: int = 3
    """Number of message passing steps for solvent if :code:`reaction_solvent` is True."""
    mpn_shared: bool = False
    """Whether to use the same message passing neural network for all input molecules
    Only relevant if :code:`number_of_molecules > 1`"""
    dropout: float = 0.0
    """Dropout probability."""
    activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'
    """Activation function."""
    atom_messages: bool = False
    """Centers messages on atoms instead of on bonds."""
    undirected: bool = False
    """Undirected edges (always sum the two relevant bond vectors)."""
    ffn_hidden_size: int = None
    """Hidden dim for higher-capacity FFN (defaults to hidden_size)."""
    ffn_num_layers: int = 2
    """Number of layers in FFN after MPN encoding."""
    features_only: bool = False
    """Use only the additional features in an FFN, no graph network."""
    separate_val_features_path: List[str] = None
    """Path to file with features for separate val set."""
    separate_test_features_path: List[str] = None
    """Path to file with features for separate test set."""
    separate_val_phase_features_path: str = None
    """Path to file with phase features for separate val set."""
    separate_test_phase_features_path: str = None
    """Path to file with phase features for separate test set."""
    separate_val_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    separate_val_bond_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_bond_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    separate_val_constraints_path: str = None
    """Path to file with constraints for separate val set."""
    separate_test_constraints_path: str = None
    """Path to file with constraints for separate test set."""
    config_path: str = None
    """
    Path to a :code:`.json` file containing arguments. Any arguments present in the config file
    will override arguments specified via the command line or by the defaults.
    """
    ensemble_size: int = 1
    """Number of models in ensemble."""
    aggregation: Literal['mean', 'sum', 'norm'] = 'mean'
    """Aggregation scheme for atomic vectors into molecular vectors"""
    aggregation_norm: int = 100
    """For norm aggregation, number by which to divide summed up atomic features"""
    reaction: bool = False
    """
    Whether to adjust MPNN layer to take reactions as input instead of molecules.
    """
    reaction_mode: Literal['reac_prod', 'reac_diff', 'prod_diff', 'reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance'] = 'reac_diff'
    """
    Choices for construction of atom and bond features for reactions
    :code:`reac_prod`: concatenates the reactants feature with the products feature.
    :code:`reac_diff`: concatenates the reactants feature with the difference in features between reactants and products.
    :code:`prod_diff`: concatenates the products feature with the difference in features between reactants and products.
    :code:`reac_prod_balance`: concatenates the reactants feature with the products feature, balances imbalanced reactions.
    :code:`reac_diff_balance`: concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions.
    :code:`prod_diff_balance`: concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions.
    """
    reaction_solvent: bool = False
    """
    Whether to adjust the MPNN layer to take as input a reaction and a molecule, and to encode them with separate MPNNs.
    """
    explicit_h: bool = False
    """
    Whether H are explicitly specified in input (and should be kept this way). This option is intended to be used
    with the :code:`reaction` or :code:`reaction_solvent` options, and applies only to the reaction part.
    """
    adding_h: bool = False
    """
    Whether RDKit molecules will be constructed with adding the Hs to them. This option is intended to be used
    with Chemprop's default molecule or multi-molecule encoders, or in :code:`reaction_solvent` mode where it applies to the solvent only.
    """
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
    epochs: int = 30
    """Number of epochs to run."""
    warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 1e-4
    """Initial learning rate."""
    max_lr: float = 1e-3
    """Maximum learning rate."""
    final_lr: float = 1e-4
    """Final learning rate."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    class_balance: bool = False
    """Trains with an equal number of positives and negatives in each batch."""
    spectra_activation: Literal['exp', 'softplus'] = 'exp'
    """Indicates which function to use in dataset_type spectra training to constrain outputs to be positive."""
    spectra_target_floor: float = 1e-8
    """Values in targets for dataset type spectra are replaced with this value, intended to be a small positive number used to enforce positive values."""
    evidential_regularization: float = 0
    """Value used in regularization for evidential loss function. The default value recommended by Soleimany et al.(2021) is 0.2. 
    Optimal value is dataset-dependent; it is recommended that users test different values to find the best value for their model."""
    overwrite_default_atom_features: bool = False
    """
    Overwrites the default atom descriptors with the new ones instead of concatenating them.
    Can only be used if atom_descriptors are used as a feature.
    """
    no_atom_descriptor_scaling: bool = False
    """Turn off atom feature scaling."""
    overwrite_default_bond_features: bool = False
    """
    Overwrites the default bond descriptors with the new ones instead of concatenating them.
    Can only be used if bond_descriptors are used as a feature.
    """
    no_bond_descriptor_scaling: bool = False
    """Turn off bond feature scaling."""
    frzn_ffn_layers: int = 0
    """
    Overwrites weights for the first n layers of the ffn from checkpoint model (specified checkpoint_frzn),
    where n is specified in the input.
    Automatically also freezes mpnn weights.
    """
    freeze_first_only: bool = False
    """
    Determines whether or not to use checkpoint_frzn for just the first encoder.
    Default (False) is to use the checkpoint to freeze all encoders.
    (only relevant for number_of_molecules > 1, where checkpoint model has number_of_molecules = 1)
    """


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
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None

    @property
    def metrics(self) -> List[str]:
        """The list of metrics used for evaluation. Only the first is used for early stopping."""
        return [self.metric] + self.extra_metrics

    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {'rmse', 'mae', 'mse', 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein', 'bounded_mse', 'bounded_mae', 'bounded_rmse'}

    @property
    def use_input_features(self) -> bool:
        """Whether the model is using additional molecule-level features."""
        return self.features_generator is not None or self.features_path is not None or self.phase_features_path is not None

    @property
    def num_lrs(self) -> int:
        """The number of learning rates to use (currently hard-coded to 1)."""
        return 1

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        """Index sets used for splitting data into train/validation/test during cross-validation"""
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        """A list of names of the tasks being trained on."""
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0

    @property
    def features_size(self) -> int:
        """The dimensionality of the additional molecule-level features."""
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        """The size of the training data set."""
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size

    @property
    def atom_descriptor_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional atom features."
        """
        return not self.no_atom_descriptor_scaling

    @property
    def bond_descriptor_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional bond features."
        """
        return not self.no_bond_descriptor_scaling
    
    @property
    def shared_atom_bond_ffn(self) -> bool:
        """
        Whether the FFN weights for atom and bond targets should be shared between tasks.
        """
        return not self.no_shared_atom_bond_ffn

    @property
    def adding_bond_types(self) -> bool:
        """
        Whether the bond types determined by RDKit molecules should be added to the output of bond targets.
        """
        return not self.no_adding_bond_types

    @property
    def atom_constraints(self) -> List[bool]:
        """
        A list of booleans indicating whether constraints applied to output of atomic properties.
        """
        if self.is_atom_bond_targets and self.constraints_path:
            if not self._atom_constraints:
                header = get_header(self.constraints_path)
                self._atom_constraints = [target in header for target in self.atom_targets]
        else:
            self._atom_constraints = [False] * len(self.atom_targets)
        return self._atom_constraints

    @property
    def bond_constraints(self) -> List[bool]:
        """
        A list of booleans indicating whether constraints applied to output of bond properties.
        """
        if self.is_atom_bond_targets and self.constraints_path:
            if not self._bond_constraints:
                header = get_header(self.constraints_path)
                self._bond_constraints = [target in header for target in self.bond_targets]
        else:
            self._bond_constraints = [False] * len(self.bond_targets)
        return self._bond_constraints

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()

        global temp_save_dir  # Prevents the temporary directory from being deleted upon function return

        # Adapt the number of molecules for reaction_solvent mode
        if self.reaction_solvent is True and self.number_of_molecules != 2:
            raise ValueError('In reaction_solvent mode, --number_of_molecules 2 must be specified.')

        # Process SMILES columns
        self.smiles_columns = preprocess_smiles_columns(
            path=self.data_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        # Load config file
        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)

        # Determine the target_columns when training atomic and bond targets
        if self.is_atom_bond_targets:
            self.atom_targets, self.bond_targets, self.molecule_targets = get_mixed_task_names(
                path=self.data_path,
                smiles_columns=self.smiles_columns,
                target_columns=self.target_columns,
                ignore_columns=self.ignore_columns,
                keep_h=self.explicit_h,
                add_h=self.adding_h,
                keep_atom_map=self.keeping_atom_map,
            )
            self.target_columns = self.atom_targets + self.bond_targets
            # self.target_columns = self.atom_targets + self.bond_targets + self.molecule_targets  # TODO: Support mixed targets
        else:
            self.atom_targets, self.bond_targets = [], []

        # Check whether atomic/bond constraints have been applied on the correct dataset_type
        if self.constraints_path:
            if not self.is_atom_bond_targets:
                raise ValueError('Constraints on atomic/bond targets can only be used in atomic/bond properties prediction.')
            if self.dataset_type != 'regression':
                raise ValueError(f'In atomic/bond properties prediction, atomic/bond constraints are not supported for {self.dataset_type}.')

        # Check whether the number of input columns is one for the atomic/bond mode
        if self.is_atom_bond_targets:
            if self.number_of_molecules != 1:
                raise ValueError('In atomic/bond properties prediction, exactly one smiles column must be provided.')

        # Check whether the number of input columns is two for the reaction_solvent mode
        if self.reaction_solvent is True and len(self.smiles_columns) != 2:
            raise ValueError('In reaction_solvent mode, exactly two smiles column must be provided (one for reactions, and one for molecules)')

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
                        (self.dataset_type == 'spectra' and metric in ['sid', 'wasserstein'])]):
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


            if len(self.split_sizes) not in [2, 3]:
                raise ValueError(f'Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s).')

            if self.separate_val_path is None and self.separate_test_path is None:  # separate data paths are not provided
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

class PredictArgs(CommonArgs):
    """:class:`PredictArgs` includes :class:`CommonArgs` along with additional arguments used for predicting with a Chemprop model."""

    test_path: str
    """Path to CSV file containing testing data for which predictions will be made."""
    preds_path: str
    """Path to CSV or PICKLE file where predictions will be saved."""
    drop_extra_columns: bool = False
    """Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns."""
    ensemble_variance: bool = False
    """Deprecated. Whether to calculate the variance of ensembles as a measure of epistemic uncertainty. If True, the variance is saved as an additional column for each target in the preds_path."""
    individual_ensemble_predictions: bool = False
    """Whether to return the predictions made by each of the individual models rather than the average of the ensemble"""
    # Uncertainty arguments
    uncertainty_method: Literal[
        'mve',
        'ensemble',
        'evidential_epistemic',
        'evidential_aleatoric',
        'evidential_total',
        'classification',
        'dropout',
        'spectra_roundrobin',
    ] = None
    """The method of calculating uncertainty."""
    calibration_method: Literal['zscaling', 'tscaling', 'zelikman_interval', 'mve_weighting', 'platt', 'isotonic'] = None
    """Methods used for calibrating the uncertainty calculated with uncertainty method."""
    evaluation_methods: List[str] = None
    """The methods used for evaluating the uncertainty performance if the test data provided includes targets.
    Available methods are [nll, miscalibration_area, ence, spearman] or any available classification or multiclass metric."""
    evaluation_scores_path: str = None
    """Location to save the results of uncertainty evaluations."""
    uncertainty_dropout_p: float = 0.1
    """The probability to use for Monte Carlo dropout uncertainty estimation."""
    dropout_sampling_size: int = 10
    """The number of samples to use for Monte Carlo dropout uncertainty estimation. Distinct from the dropout used during training."""
    calibration_interval_percentile: float = 95
    """Sets the percentile used in the calibration methods. Must be in the range (1,100)."""
    regression_calibrator_metric: Literal['stdev', 'interval'] = None
    """Regression calibrators can output either a stdev or an inverval. """
    calibration_path: str = None
    """Path to data file to be used for uncertainty calibration."""
    calibration_features_path: List[str] = None
    """Path to features data to be used with the uncertainty calibration dataset."""
    calibration_phase_features_path: str = None
    """ """
    calibration_atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    calibration_bond_descriptors_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""

    @property
    def ensemble_size(self) -> int:
        """The number of models in the ensemble."""
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

        self.smiles_columns = preprocess_smiles_columns(
            path=self.test_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')

        if self.ensemble_variance:
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

class FingerprintArgs(PredictArgs):
    """:class:`FingerprintArgs` includes :class:`PredictArgs` with additional arguments for the generation of latent fingerprint vectors."""

    fingerprint_type: Literal['MPN', 'last_FFN'] = 'MPN'
    """Choice of which type of latent fingerprint vector to use. Default is the output of the MPNN, excluding molecular features"""

