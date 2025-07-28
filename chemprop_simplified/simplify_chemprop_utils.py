## from spectra_utils
import csv
from tqdm import trange
from logging import Logger
## from utils.py
from argparse import Namespace
from functools import wraps
import logging
import os
import pickle
import re
from time import time
from typing import Any, Callable, List, Tuple,Union
import collections
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from scipy.stats.mstats import gmean
from random import Random
from simplify_chemprop_args import PredictArgs, TrainArgs
from simplify_constants import StandardScaler, AtomBondScaler
from simplify_chemprop_data import MoleculeDataset,get_header,MoleculeDatapoint
from simplify_chemprop_model import MoleculeModel

##from nn_utils.py
import math

def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.

    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')

def activate_dropout(module: nn.Module, dropout_prob: float):
    """
    Set p of dropout layers and set to train mode during inference for uncertainty estimation.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param dropout_prob: A float on (0,1) indicating the dropout probability.
    """
    if isinstance(module, nn.Dropout):
        module.p = dropout_prob
        module.train()

def compute_pnorm(model: nn.Module) -> float:
    """
    Computes the norm of the parameters of a model.

    :param model: A PyTorch model.
    :return: The norm of the parameters of the model.
    """
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))

def compute_gnorm(model: nn.Module) -> float:
    """
    Computes the norm of the gradients of a model.

    :param model: A PyTorch model.
    :return: The norm of the gradients of the model.
    """
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))

class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        if not (
            len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs)
            == len(init_lr) == len(max_lr) == len(final_lr)
        ):
            raise ValueError(
                "Number of param groups must match the number of epochs and learning rates! "
                f"got: len(optimizer.param_groups)= {len(optimizer.param_groups)}, "
                f"len(warmup_epochs)= {len(warmup_epochs)}, "
                f"len(total_epochs)= {len(total_epochs)}, "
                f"len(init_lr)= {len(init_lr)}, "
                f"len(max_lr)= {len(max_lr)}, "
                f"len(final_lr)= {len(final_lr)}"
            )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.

        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]

def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An PyTorch model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def multitask_mean(
    scores: np.ndarray,
    metric: str,
    axis: int = None,
) -> float:
    """
    A function for combining the metric scores across different
    model tasks into a single score. When the metric being used
    is one that varies with the magnitude of the task (such as RMSE),
    a geometric mean is used, otherwise a more typical arithmetic mean
    is used. This prevents a task with a larger magnitude from dominating
    over one with a smaller magnitude (e.g., temperature and pressure).

    :param scores: The scores from different tasks for a single metric.
    :param metric: The metric used to generate the scores.
    :axis: The axis along which to take the mean.
    :return: The combined score across the tasks.
    """
    
    scale_dependent_metrics = ["rmse", "mae", "mse", "bounded_rmse", "bounded_mae", "bounded_mse"]
    nonscale_dependent_metrics = [
        "auc", "prc-auc", "r2", "accuracy", "cross_entropy",
        "binary_cross_entropy", "sid", "wasserstein", "f1", "mcc",
    ]

    if metric in scale_dependent_metrics:
        return gmean(scores, axis=axis)
    elif metric in nonscale_dependent_metrics:
        return np.mean(scores, axis=axis)
    else:
        raise NotImplementedError(
            f"The metric used, {metric}, has not been added to the list of\
                metrics that are scale-dependent or not scale-dependent.\
                This metric must be added to the appropriate list in the multitask_mean\
                function in `chemprop/utils.py` in order to be used."
        )

def param_count_all(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An PyTorch model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.numel() for param in model.parameters())

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.

    :param source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    :param index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    :return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target

## from utils.py
def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :return: A :class:`~chemprop.data.MoleculeDataset` with only the valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in tqdm(data)
                            if all(s != '' for s in datapoint.smiles) and all(m is not None for m in datapoint.mol)
                            and all(m.GetNumHeavyAtoms() > 0 for m in datapoint.mol if not isinstance(m, tuple))
                            and all(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() > 0 for m in datapoint.mol if isinstance(m, tuple))])

def build_optimizer(model: nn.Module, args: TrainArgs) -> Optimizer:
    """
    Builds a PyTorch Optimizer.

    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    params = [{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}]

    return Adam(params)

def build_lr_scheduler(
    optimizer: Optimizer, args: TrainArgs, total_epochs: List[int] = None
) -> _LRScheduler:
    """
    Builds a PyTorch learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=total_epochs or [args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr],
    )

def load_checkpoint(
    path: str, device: torch.device = None, logger: logging.Logger = None
) -> MoleculeModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param device: Device where the model will be moved.
    :param logger: A logger for recording output.
    :return: The loaded :class:`~chemprop.models.model.MoleculeModel`.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args = TrainArgs()
    args.from_dict(vars(state["args"]), skip_unsettable=True)
    loaded_state_dict = state["state_dict"]

    if device is not None:
        args.device = device

    # Build model
    model = MoleculeModel(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        # Backward compatibility for parameter names
        if re.match(r"(encoder\.encoder\.)([Wc])", loaded_param_name) and not args.reaction_solvent:
            param_name = loaded_param_name.replace("encoder.encoder", "encoder..0")
        elif re.match(r"(^ffn)", loaded_param_name):
            param_name = loaded_param_name.replace("ffn", "readout")
        else:
            param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters
        if param_name not in model_state_dict:
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.'
            )
        elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" '
                f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
                f"model parameter of shape {model_state_dict[param_name].shape}."
            )
        else:
            debug(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug("Moving model to cuda")
    model = model.to(args.device)

    return model

def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)

def save_checkpoint(
    path: str,
    model: MoleculeModel,
    scaler: StandardScaler = None,
    features_scaler: StandardScaler = None,
    atom_descriptor_scaler: StandardScaler = None,
    bond_descriptor_scaler: StandardScaler = None,
    atom_bond_scaler: AtomBondScaler = None,
    args: TrainArgs = None,
) -> None:
    """
    Saves a model checkpoint.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the data.
    :param features_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the features.
    :param atom_descriptor_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the atom descriptors.
    :param bond_descriptor_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the bond descriptors.
    :param atom_bond_scaler: A :class:`~chemprop.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param args: The :class:`~chemprop.args.TrainArgs` object containing the arguments the model was trained with.
    :param path: Path where checkpoint will be saved.
    """
    # Convert args to namespace for backwards compatibility
    if args is not None:
        args = Namespace(**args.as_dict())

    data_scaler = {"means": scaler.means, "stds": scaler.stds} if scaler is not None else None
    if atom_bond_scaler is not None:
        atom_bond_scaler = {"means": atom_bond_scaler.means, "stds": atom_bond_scaler.stds}
    if features_scaler is not None:
        features_scaler = {"means": features_scaler.means, "stds": features_scaler.stds}
    if atom_descriptor_scaler is not None:
        atom_descriptor_scaler = {
            "means": atom_descriptor_scaler.means,
            "stds": atom_descriptor_scaler.stds,
        }
    if bond_descriptor_scaler is not None:
        bond_descriptor_scaler = {"means": bond_descriptor_scaler.means, "stds": bond_descriptor_scaler.stds}

    state = {
        "args": args,
        "state_dict": model.state_dict(),
        "data_scaler": data_scaler,
        "features_scaler": features_scaler,
        "atom_descriptor_scaler": atom_descriptor_scaler,
        "bond_descriptor_scaler": bond_descriptor_scaler,
        "atom_bond_scaler": atom_bond_scaler,
    }
    torch.save(state, path)

def save_smiles_splits(
    data_path: str,
    save_dir: str,
    task_names: List[str] = None,
    features_path: List[str] = None,
    constraints_path: str = None,
    train_data: MoleculeDataset = None,
    val_data: MoleculeDataset = None,
    test_data: MoleculeDataset = None,
    logger: logging.Logger = None,
    smiles_columns: List[str] = None,
) -> None:
    """
    Saves a csv file with train/val/test splits of target data and additional features.
    Also saves indices of train/val/test split as a pickle file. Pickle file does not support repeated entries
    with the same SMILES or entries entered from a path other than the main data path, such as a separate test path.

    :param data_path: Path to data CSV file.
    :param save_dir: Path where pickle files will be saved.
    :param task_names: List of target names for the model as from the function get_task_names().
        If not provided, will use datafile header entries.
    :param features_path: List of path(s) to files with additional molecule features.
    :param constraints_path: Path to constraints applied to atomic/bond properties prediction.
    :param train_data: Train :class:`~chemprop.data.data.MoleculeDataset`.
    :param val_data: Validation :class:`~chemprop.data.data.MoleculeDataset`.
    :param test_data: Test :class:`~chemprop.data.data.MoleculeDataset`.
    :param smiles_columns: The name of the column containing SMILES. By default, uses the first column.
    :param logger: A logger for recording output.
    """
    makedirs(save_dir)

    info = logger.info if logger is not None else print
    save_split_indices = True

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=data_path, smiles_columns=smiles_columns)

    with open(data_path) as f:
        f = open(data_path)
        reader = csv.DictReader(f)

        indices_by_smiles = {}
        for i, row in enumerate(tqdm(reader)):
            smiles = tuple([row[column] for column in smiles_columns])
            if smiles in indices_by_smiles:
                save_split_indices = False
                info(
                    "Warning: Repeated SMILES found in data, pickle file of split indices cannot distinguish entries and will not be generated."
                )
                break
            indices_by_smiles[smiles] = i

    if task_names is None:
        task_names = get_task_names(path=data_path, smiles_columns=smiles_columns)

    features_header = []
    if features_path is not None:
        extension_sets = set([os.path.splitext(feat_path)[1] for feat_path in features_path])
        if extension_sets == {'.csv'}:
            for feat_path in features_path:
                with open(feat_path, "r") as f:
                    reader = csv.reader(f)
                    feat_header = next(reader)
                    features_header.extend(feat_header)

    if constraints_path is not None:
        with open(constraints_path, "r") as f:
            reader = csv.reader(f)
            constraints_header = next(reader)

    all_split_indices = []
    for dataset, name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        if dataset is None:
            continue

        with open(os.path.join(save_dir, f"{name}_smiles.csv"), "w") as f:
            writer = csv.writer(f)
            if smiles_columns[0] == "":
                writer.writerow(["smiles"])
            else:
                writer.writerow(smiles_columns)
            for smiles in dataset.smiles():
                writer.writerow(smiles)

        with open(os.path.join(save_dir, f"{name}_full.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(smiles_columns + task_names)
            dataset_targets = dataset.targets()
            for i, smiles in enumerate(dataset.smiles()):
                targets = [x.tolist() if isinstance(x, np.ndarray) else x for x in dataset_targets[i]]
                writer.writerow(smiles + targets)

        if features_path is not None:
            dataset_features = dataset.features()
            if extension_sets == {'.csv'}:
                with open(os.path.join(save_dir, f"{name}_features.csv"), "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(features_header)
                    writer.writerows(dataset_features)
            else:
                np.save(os.path.join(save_dir, f"{name}_features.npy"), dataset_features)

        if constraints_path is not None:
            dataset_constraints = [d.raw_constraints for d in dataset._data]
            with open(os.path.join(save_dir, f"{name}_constraints.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(constraints_header)
                writer.writerows(dataset_constraints)

        if save_split_indices:
            split_indices = []
            for smiles in dataset.smiles():
                index = indices_by_smiles.get(tuple(smiles))
                if index is None:
                    save_split_indices = False
                    info(
                        f"Warning: SMILES string in {name} could not be found in data file, and "
                        "likely came from a secondary data file. The pickle file of split indices "
                        "can only indicate indices for a single file and will not be generated."
                    )
                    break
                split_indices.append(index)
            else:
                split_indices.sort()
                all_split_indices.append(split_indices)

        if name == "train":
            data_weights = dataset.data_weights()
            if any([w != 1 for w in data_weights]):
                with open(os.path.join(save_dir, f"{name}_weights.csv"), "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["data weights"])
                    for weight in data_weights:
                        writer.writerow([weight])

    if save_split_indices:
        with open(os.path.join(save_dir, "split_indices.pckl"), "wb") as f:
            pickle.dump(all_split_indices, f)

def load_frzn_model(
    model: torch.nn,
    path: str,
    current_args: Namespace = None,
    cuda: bool = None,
    logger: logging.Logger = None,
) -> MoleculeModel:
    """
    Loads a model checkpoint.
    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """
    debug = logger.debug if logger is not None else print

    loaded_mpnn_model = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state_dict = loaded_mpnn_model["state_dict"]
    loaded_args = loaded_mpnn_model["args"]

    # Backward compatibility for parameter names
    loaded_state_dict_keys = list(loaded_state_dict.keys())
    for loaded_param_name in loaded_state_dict_keys:
        if re.match(r"(^ffn)", loaded_param_name):
            param_name = loaded_param_name.replace("ffn", "readout")
            loaded_state_dict[param_name] = loaded_state_dict.pop(loaded_param_name)

    model_state_dict = model.state_dict()

    if loaded_args.number_of_molecules == 1 and current_args.number_of_molecules == 1:
        encoder_param_names = [
            "encoder.encoder.0.W_i.weight",
            "encoder.encoder.0.W_h.weight",
            "encoder.encoder.0.W_o.weight",
            "encoder.encoder.0.W_o.bias",
            "encoder.encoder.0.W_o_b.weight",
            "encoder.encoder.0.W_o_b.bias",
        ]
        if current_args.checkpoint_frzn is not None:
            # Freeze the MPNN
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers > 0:
            if isinstance(model.readout, nn.Sequential):  # Molecule properties
                ffn_param_names = [
                    [f"readout.{i*3+1}.weight", f"readout.{i*3+1}.bias"]
                    for i in range(current_args.frzn_ffn_layers)
                ]
        
            ffn_param_names = [item for sublist in ffn_param_names for item in sublist]

            # Freeze MPNN and FFN layers
            for param_name in encoder_param_names + ffn_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.freeze_first_only:
            debug(
                "WARNING: --freeze_first_only flag cannot be used with number_of_molecules=1 (flag is ignored)"
            )

    elif loaded_args.number_of_molecules == 1 and current_args.number_of_molecules > 1:
        # TODO(degraff): these two `if`-blocks can be condensed into one
        if (
            current_args.checkpoint_frzn is not None
            and current_args.freeze_first_only
            and current_args.frzn_ffn_layers <= 0
        ):  # Only freeze first MPNN
            encoder_param_names = [
                "encoder.encoder.0.W_i.weight",
                "encoder.encoder.0.W_h.weight",
                "encoder.encoder.0.W_o.weight",
                "encoder.encoder.0.W_o.bias",
            ]
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )
        if (
            current_args.checkpoint_frzn is not None
            and not current_args.freeze_first_only
            and current_args.frzn_ffn_layers <= 0
        ):  # Duplicate encoder from frozen checkpoint and overwrite all encoders
            loaded_encoder_param_names = [
                "encoder.encoder.0.W_i.weight",
                "encoder.encoder.0.W_h.weight",
                "encoder.encoder.0.W_o.weight",
                "encoder.encoder.0.W_o.bias",
            ] * current_args.number_of_molecules

            model_encoder_param_names = [
                [
                    (
                        f"encoder.encoder.{mol_num}.W_i.weight",
                        f"encoder.encoder.{mol_num}.W_h.weight",
                        f"encoder.encoder.{mol_num}.W_o.weight",
                        f"encoder.encoder.{mol_num}.W_o.bias",
                    )
                ]
                for mol_num in range(current_args.number_of_molecules)
            ]
            model_encoder_param_names = [
                item for sublist in model_encoder_param_names for item in sublist
            ]

            for loaded_param_name, model_param_name in zip(
                loaded_encoder_param_names, model_encoder_param_names
            ):
                model_state_dict = overwrite_state_dict(
                    loaded_param_name, model_param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers > 0:
            raise ValueError(
                f"Number of molecules from checkpoint_frzn ({loaded_args.number_of_molecules}) "
                f"must equal current number of molecules ({current_args.number_of_molecules})!"
            )

    elif loaded_args.number_of_molecules > 1 and current_args.number_of_molecules > 1:
        if loaded_args.number_of_molecules != current_args.number_of_molecules:
            raise ValueError(
                f"Number of molecules in checkpoint_frzn ({loaded_args.number_of_molecules}) "
                f"must either match current model ({current_args.number_of_molecules}) or equal 1."
            )

        if current_args.freeze_first_only:
            raise ValueError(
                f"Number of molecules in checkpoint_frzn ({loaded_args.number_of_molecules}) "
                "must be equal to 1 for freeze_first_only to be used!"
            )

        if (current_args.checkpoint_frzn is not None) & (not (current_args.frzn_ffn_layers > 0)):
            encoder_param_names = [
                [
                    (
                        f"encoder.encoder.{mol_num}.W_i.weight",
                        f"encoder.encoder.{mol_num}.W_h.weight",
                        f"encoder.encoder.{mol_num}.W_o.weight",
                        f"encoder.encoder.{mol_num}.W_o.bias",
                    )
                ]
                for mol_num in range(current_args.number_of_molecules)
            ]
            encoder_param_names = [item for sublist in encoder_param_names for item in sublist]

            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers > 0:
            encoder_param_names = [
                [
                    (
                        f"encoder.encoder.{mol_num}.W_i.weight",
                        f"encoder.encoder.{mol_num}.W_h.weight",
                        f"encoder.encoder.{mol_num}.W_o.weight",
                        f"encoder.encoder.{mol_num}.W_o.bias",
                    )
                ]
                for mol_num in range(current_args.number_of_molecules)
            ]
            encoder_param_names = [item for sublist in encoder_param_names for item in sublist]
            ffn_param_names = [
                [f"readout.{i*3+1}.weight", f"readout.{i*3+1}.bias"]
                for i in range(current_args.frzn_ffn_layers)
            ]
            ffn_param_names = [item for sublist in ffn_param_names for item in sublist]

            for param_name in encoder_param_names + ffn_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers >= current_args.ffn_num_layers:
            raise ValueError(
                f"Number of frozen FFN layers ({current_args.frzn_ffn_layers}) "
                f"must be less than the number of FFN layers ({current_args.ffn_num_layers})!"
            )

    # Load pretrained weights
    model.load_state_dict(model_state_dict)

    return model

def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    If a logger with that name already exists, simply returns that logger.
    Otherwise, creates a new logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of :code:`quiet`.
    One file handler (:code:`verbose.log`) saves all logs, the other (:code:`quiet.log`) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e., print only important info).
    :return: The logger.
    """

    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, "verbose.log"))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, "quiet.log"))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger

def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')

def get_task_names(path: str,
                   smiles_columns: Union[str, List[str]] = None,
                   target_columns: List[str] = None,
                   ignore_columns: List[str] = None) -> List[str]:
    """
    Gets the task names from a data CSV file.
    If :code:`target_columns` is provided, returns `target_columns`.
    Otherwise, returns all columns except the :code:`smiles_columns`
    (or the first column, if the :code:`smiles_columns` is None) and
    the :code:`ignore_columns`.
    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_columns` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :return: A list of task names.
    """
    if target_columns is not None:
        return target_columns

    columns = get_header(path)

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    ignore_columns = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))

    target_names = [column for column in columns if column not in ignore_columns]

    return target_names

def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               key_molecule_index: int = 0,
               seed: int = 0,
               num_folds: int = 1,
               args: TrainArgs = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    r"""
    Splits data into training, validation, and test splits.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param key_molecule_index: For data with multiple molecules, this sets which molecule will be considered during splitting.
    :param seed: The random seed to use before shuffling data.
    :param num_folds: Number of folds to create (only needed for "cv" split type).
    :param args: A :class:`~chemprop.args.TrainArgs` object.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Split sizes do not sum to 1. Received train/val/test splits: {sizes}")
    if any([size < 0 for size in sizes]):
        raise ValueError(f"Split sizes must be non-negative. Received train/val/test splits: {sizes}")

    random = Random(seed)

    if args is not None:
        folds_file, val_fold_index, test_fold_index = \
            args.folds_file, args.val_fold_index, args.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None

    if split_type == 'crossval':
        index_set = args.crossval_index_sets[args.seed]
        data_split = []
        for split in range(3):
            split_indices = []
            for index in index_set[split]:
                with open(os.path.join(args.crossval_index_dir, f'{index}.pkl'), 'rb') as rf:
                    split_indices.extend(pickle.load(rf))
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type in {'cv', 'cv-no-test'}:
        if num_folds <= 1 or num_folds > len(data):
            raise ValueError(f'Number of folds for cross-validation must be between 2 and the number of valid datapoints ({len(data)}), inclusive.')

        random = Random(0)

        indices = np.tile(np.arange(num_folds), 1 + len(data) // num_folds)[:len(data)]
        random.shuffle(indices)
        test_index = seed % num_folds
        val_index = (seed + 1) % num_folds

        train, val, test = [], [], []
        for d, index in zip(data, indices):
            if index == test_index and split_type != 'cv-no-test':
                test.append(d)
            elif index == val_index:
                val.append(d)
            else:
                train.append(d)

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'index_predetermined':
        split_indices = args.crossval_index_sets[args.seed]

        if len(split_indices) != 3:
            raise ValueError('Split indices must have three splits: train, validation, and test')

        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'predetermined':
        if not val_fold_index and sizes[2] != 0:
            raise ValueError('Test size must be zero since test set is created separately '
                             'and we want to put all other data in train and validation')

        if folds_file is None:
            raise ValueError('arg "folds_file" can not be None!')
        if test_fold_index is None:
            raise ValueError('arg "test_fold_index" can not be None!')

        try:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f, encoding='latin1')  # in case we're loading indices from python2

        log_scaffold_stats(data, all_fold_indices, logger=logger)

        folds = [[data[i] for i in fold_indices] for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != test_fold_index and (val_fold_index is None or i != val_fold_index):
                train_val.extend(folds[i])

        if val_fold_index is not None:
            train = train_val
        else:
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, key_molecule_index=key_molecule_index, seed=seed, logger=logger)

    elif split_type == 'random_with_repeated_smiles':  # Use to constrain data with the same smiles go in the same split.
        smiles_dict = defaultdict(set)
        for i, smiles in enumerate(data.smiles()):
            smiles_dict[smiles[key_molecule_index]].add(i)
        index_sets = list(smiles_dict.values())
        random.seed(seed)
        random.shuffle(index_sets)
        train, val, test = [], [], []
        train_size = int(sizes[0] * len(data))
        val_size = int(sizes[1] * len(data))
        for index_set in index_sets:
            if len(train)+len(index_set) <= train_size:
                train += index_set
            elif len(val) + len(index_set) <= val_size:
                val += index_set
            else:
                test += index_set
        train = [data[i] for i in train]
        val = [data[i] for i in val]
        test = [data[i] for i in test]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'random':
        indices = list(range(len(data)))
        random.shuffle(indices)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = [data[i] for i in indices[:train_size]]
        val = [data[i] for i in indices[train_size:train_val_size]]
        test = [data[i] for i in indices[train_val_size:]]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')

def get_data(path: str,
             smiles_columns: Union[str, List[str]] = None,
             target_columns: List[str] = None,
             ignore_columns: List[str] = None,
             skip_invalid_smiles: bool = True,
             args: Union[TrainArgs, PredictArgs] = None,
             data_weights_path: str = None,
             features_path: List[str] = None,
             features_generator: List[str] = None,
             phase_features_path: str = None,
             atom_descriptors_path: str = None,
             bond_descriptors_path: str = None,
             constraints_path: str = None,
             max_data_size: int = None,
             store_row: bool = False,
             logger: Logger = None,
             loss_function: str = None,
             skip_none_targets: bool = False) -> MoleculeDataset:
    """
    Gets SMILES and target values from a CSV file.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_column` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles using :func:`filter_invalid_smiles`.
    :param args: Arguments, either :class:`~chemprop.args.TrainArgs` or :class:`~chemprop.args.PredictArgs`.
    :param data_weights_path: A path to a file containing weights for each molecule in the loss function.
    :param features_path: A list of paths to files containing features. If provided, it is used
                          in place of :code:`args.features_path`.
    :param features_generator: A list of features generators to use. If provided, it is used
                               in place of :code:`args.features_generator`.
    :param phase_features_path: A path to a file containing phase features as applicable to spectra.
    :param atom_descriptors_path: The path to the file containing the custom atom descriptors.
    :param bond_descriptors_path: The path to the file containing the custom bond descriptors.
    :param constraints_path: The path to the file containing constraints applied to different atomic/bond properties.
    :param max_data_size: The maximum number of data points to load.
    :param logger: A logger for recording output.
    :param store_row: Whether to store the raw CSV row in each :class:`~chemprop.data.data.MoleculeDatapoint`.
    :param skip_none_targets: Whether to skip targets that are all 'None'. This is mostly relevant when --target_columns
                              are passed in, so only a subset of tasks are examined.
    :param loss_function: The loss function to be used in training.
    :return: A :class:`~chemprop.data.MoleculeDataset` containing SMILES and target values along
             with other info such as additional features when desired.
    """
    debug = logger.debug if logger is not None else print

    if args is not None:
        # Prefer explicit function arguments but default to args if not provided
        smiles_columns = smiles_columns if smiles_columns is not None else args.smiles_columns
        target_columns = target_columns if target_columns is not None else args.target_columns
        ignore_columns = ignore_columns if ignore_columns is not None else args.ignore_columns
        features_path = features_path if features_path is not None else args.features_path
        features_generator = features_generator if features_generator is not None else args.features_generator
        phase_features_path = phase_features_path if phase_features_path is not None else args.phase_features_path
        atom_descriptors_path = atom_descriptors_path if atom_descriptors_path is not None \
            else args.atom_descriptors_path
        bond_descriptors_path = bond_descriptors_path if bond_descriptors_path is not None \
            else args.bond_descriptors_path
        constraints_path = constraints_path if constraints_path is not None else args.constraints_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        loss_function = loss_function if loss_function is not None else args.loss_function

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    max_data_size = max_data_size or float('inf')

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    if phase_features_path is not None:
        phase_features = load_features(phase_features_path)
        for d_phase in phase_features:
            if not (d_phase.sum() == 1 and np.count_nonzero(d_phase) == 1):
                raise ValueError('Phase features must be one-hot encoded.')
        if features_data is not None:
            features_data = np.concatenate((features_data, phase_features), axis=1)
        else:  # if there are no other molecular features, phase features become the only molecular features
            features_data = np.array(phase_features)
    else:
        phase_features = None

    # Load constraints
    if constraints_path is not None:
        constraints_data, raw_constraints_data = get_constraints(
            path=constraints_path,
            target_columns=args.target_columns,
            save_raw_data=args.save_smiles_splits
        )
    else:
        constraints_data = None
        raw_constraints_data = None

    # Load data weights
    if data_weights_path is not None:
        data_weights = get_data_weights(data_weights_path)
    else:
        data_weights = None

    # By default, the targets columns are all the columns except the SMILES column
    if target_columns is None:
        target_columns = get_task_names(
            path=path,
            smiles_columns=smiles_columns,
            target_columns=target_columns,
            ignore_columns=ignore_columns,
        )

    # Find targets provided as inequalities
    if loss_function == 'bounded_mse':
        gt_targets, lt_targets = get_inequality_targets(path=path, target_columns=target_columns)
    else:
        gt_targets, lt_targets = None, None

    # Load data
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if any([c not in fieldnames for c in smiles_columns]):
            raise ValueError(f'Data file did not contain all provided smiles columns: {smiles_columns}. Data file field names are: {fieldnames}')
        if any([c not in fieldnames for c in target_columns]):
            raise ValueError(f'Data file did not contain all provided target columns: {target_columns}. Data file field names are: {fieldnames}')

        all_smiles, all_targets, all_atom_targets, all_bond_targets, all_rows, all_features, all_phase_features, all_constraints_data, all_raw_constraints_data, all_weights, all_gt, all_lt = [], [], [], [], [], [], [], [], [], [], [], []
        for i, row in enumerate(tqdm(reader)):
            smiles = [row[c] for c in smiles_columns]

            targets, atom_targets, bond_targets = [], [], []
            for column in target_columns:
                value = row[column]
                if value in ['', 'nan']:
                    targets.append(None)
                elif '>' in value or '<' in value:
                    if loss_function == 'bounded_mse':
                        targets.append(float(value.strip('<>')))
                    else:
                        raise ValueError('Inequality found in target data. To use inequality targets (> or <), the regression loss function bounded_mse must be used.')
                elif '[' in value or ']' in value:
                    value = value.replace('None', 'null')
                    target = np.array(json.loads(value))
                    if len(target.shape) == 1 and column in args.atom_targets:  # Atom targets saved as 1D list
                        atom_targets.append(target)
                        targets.append(target)
                    elif len(target.shape) == 1 and column in args.bond_targets:  # Bond targets saved as 1D list
                        bond_targets.append(target)
                        targets.append(target)
                    elif len(target.shape) == 2:  # Bond targets saved as 2D list
                        bond_target_arranged = []
                        mol = make_mol(smiles[0], args.explicit_h, args.adding_h, args.keeping_atom_map)
                        for bond in mol.GetBonds():
                            bond_target_arranged.append(target[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                        bond_targets.append(np.array(bond_target_arranged))
                        targets.append(np.array(bond_target_arranged))
                    else:
                        raise ValueError(f'Unrecognized targets of column {column} in {path}.')
                else:
                    targets.append(float(value))

            # Check whether all targets are None and skip if so
            if skip_none_targets and all(x is None for x in targets):
                continue

            all_smiles.append(smiles)
            all_targets.append(targets)
            all_atom_targets.append(atom_targets)
            all_bond_targets.append(bond_targets)

            if features_data is not None:
                all_features.append(features_data[i])

            if phase_features is not None:
                all_phase_features.append(phase_features[i])

            if constraints_data is not None:
                all_constraints_data.append(constraints_data[i])

            if raw_constraints_data is not None:
                all_raw_constraints_data.append(raw_constraints_data[i])

            if data_weights is not None:
                all_weights.append(data_weights[i])

            if gt_targets is not None:
                all_gt.append(gt_targets[i])

            if lt_targets is not None:
                all_lt.append(lt_targets[i])

            if store_row:
                all_rows.append(row)

            if len(all_smiles) >= max_data_size:
                break

        atom_features = None
        atom_descriptors = None
        if args is not None and args.atom_descriptors is not None:
            try:
                descriptors = load_valid_atom_or_bond_features(atom_descriptors_path, [x[0] for x in all_smiles])
            except Exception as e:
                raise ValueError(f'Failed to load or validate custom atomic descriptors or features: {e}')

            if args.atom_descriptors == 'feature':
                atom_features = descriptors
            elif args.atom_descriptors == 'descriptor':
                atom_descriptors = descriptors

        bond_features = None
        bond_descriptors = None
        if args is not None and args.bond_descriptors is not None:
            try:
                descriptors = load_valid_atom_or_bond_features(bond_descriptors_path, [x[0] for x in all_smiles])
            except Exception as e:
                raise ValueError(f'Failed to load or validate custom bond descriptors or features: {e}')

            if args.bond_descriptors == 'feature':
                bond_features = descriptors
            elif args.bond_descriptors == 'descriptor':
                bond_descriptors = descriptors

        data = MoleculeDataset([
            MoleculeDatapoint(
                smiles=smiles,
                targets=targets,
                atom_targets=all_atom_targets[i] if atom_targets else None,
                bond_targets=all_bond_targets[i] if bond_targets else None,
                row=all_rows[i] if store_row else None,
                data_weight=all_weights[i] if data_weights is not None else None,
                gt_targets=all_gt[i] if gt_targets is not None else None,
                lt_targets=all_lt[i] if lt_targets is not None else None,
                features_generator=features_generator,
                features=all_features[i] if features_data is not None else None,
                phase_features=all_phase_features[i] if phase_features is not None else None,
                atom_features=atom_features[i] if atom_features is not None else None,
                atom_descriptors=atom_descriptors[i] if atom_descriptors is not None else None,
                bond_features=bond_features[i] if bond_features is not None else None,
                bond_descriptors=bond_descriptors[i] if bond_descriptors is not None else None,
                constraints=all_constraints_data[i] if constraints_data is not None else None,
                raw_constraints=all_raw_constraints_data[i] if raw_constraints_data is not None else None,
                overwrite_default_atom_features=args.overwrite_default_atom_features if args is not None else False,
                overwrite_default_bond_features=args.overwrite_default_bond_features if args is not None else False
            ) for i, (smiles, targets) in tqdm(enumerate(zip(all_smiles, all_targets)),
                                            total=len(all_smiles))
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data

def get_task_names(path: str,
                   smiles_columns: Union[str, List[str]] = None,
                   target_columns: List[str] = None,
                   ignore_columns: List[str] = None) -> List[str]:
    """
    Gets the task names from a data CSV file.
    If :code:`target_columns` is provided, returns `target_columns`.
    Otherwise, returns all columns except the :code:`smiles_columns`
    (or the first column, if the :code:`smiles_columns` is None) and
    the :code:`ignore_columns`.
    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_columns` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :return: A list of task names.
    """
    if target_columns is not None:
        return target_columns

    columns = get_header(path)

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    ignore_columns = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))

    target_names = [column for column in columns if column not in ignore_columns]

    return target_names

def get_class_sizes(data: MoleculeDataset, proportion: bool = True) -> List[List[float]]:
    """
    Determines the proportions of the different classes in a classification dataset.

    :param data: A classification :class:`~chemprop.data.MoleculeDataset`.
    :param proportion: Choice of whether to return proportions for class size or counts.
    :return: A list of lists of class proportions. Each inner list contains the class proportions for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if data.is_atom_bond_targets:
                for target in targets[i][task_num]:
                    if targets[i][task_num] is not None:
                        valid_targets[task_num].append(target)
            else:
                if targets[i][task_num] is not None:
                    valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        if set(np.unique(task_targets)) > {0, 1}:
            raise ValueError('Classification dataset must only contains 0s and 1s.')
        if proportion:
            try:
                ones = np.count_nonzero(task_targets) / len(task_targets)
            except ZeroDivisionError:
                ones = float('nan')
                print('Warning: class has no targets')
            class_sizes.append([1 - ones, ones])
        else:  # counts
            ones = np.count_nonzero(task_targets)
            class_sizes.append([len(task_targets) - ones, ones])

    return class_sizes


