from collections import defaultdict
import csv
import json
from logging import Logger
import os
import sys
from typing import Callable, Dict, List, Tuple
import subprocess
import pandas as pd
from typing import List
import numpy as np
import torch
from tqdm import tqdm,trange
import warnings
from typing import List, Callable, Union


# from metrics
def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    r"""
    Gets the metric function corresponding to a given metric name.

    Supports:

    * :code:`auc`: Area under the receiver operating characteristic curve
    * :code:`prc-auc`: Area under the precision recall curve
    * :code:`rmse`: Root mean squared error
    * :code:`mse`: Mean squared error
    * :code:`mae`: Mean absolute error
    * :code:`r2`: Coefficient of determination R\ :superscript:`2`
    * :code:`accuracy`: Accuracy (using a threshold to binarize predictions)
    * :code:`cross_entropy`: Cross entropy
    * :code:`binary_cross_entropy`: Binary cross entropy
    * :code:`sid`: Spectral information divergence
    * :code:`wasserstein`: Wasserstein loss for spectra

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mse':
        return mean_squared_error

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'bounded_rmse':
        return bounded_rmse

    if metric == 'bounded_mse':
        return bounded_mse

    if metric == 'bounded_mae':
        return bounded_mae

    if metric == 'r2':
        return r2_score

    if metric == 'accuracy':
        return accuracy

    if metric == 'cross_entropy':
        return log_loss
    
    if metric == 'f1':
        return f1_metric

    if metric == 'mcc':
        return mcc_metric

    if metric == 'binary_cross_entropy':
        return bce
    
    if metric == 'sid':
        return sid_metric
    
    if metric == 'wasserstein':
        return wasserstein_metric

    raise ValueError(f'Metric "{metric}" not supported.')


def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def bce(targets: List[int], preds: List[float]) -> float:
    """
    Computes the binary cross entropy loss.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed binary cross entropy.
    """
    # Don't use logits because the sigmoid is added in all places except training itself
    bce_func = nn.BCELoss(reduction='mean')
    loss = bce_func(target=torch.Tensor(targets), input=torch.Tensor(preds)).item()

    return loss


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return mean_squared_error(targets, preds, squared=False)


def bounded_rmse(targets: List[float], preds: List[float], gt_targets: List[bool] = None, lt_targets: List[bool] = None) -> float:
    """
    Computes the root mean squared error, considering targets with inequalities.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :param gt_targets: A list of booleans indicating whether the target is a >target inequality.
    :param lt_targets: A list of booleans indicating whether the target is a <target inequality.
    :return: The computed rmse.
    """
    # When the target is a greater-than-inequality and the prediction is greater than the target,
    # replace the prediction with the target. Analogous for less-than-inequalities.
    preds = np.where(
        np.logical_and(np.greater(preds, targets),gt_targets),
        targets,
        preds,
    )
    preds = np.where(
        np.logical_and(np.less(preds, targets),lt_targets),
        targets,
        preds,
    )
    return mean_squared_error(targets, preds, squared=False)


def bounded_mse(targets: List[float], preds: List[float], gt_targets: List[bool] = None, lt_targets: List[bool] = None) -> float:
    """
    Computes the mean squared error, considering targets with inequalities.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :param gt_targets: A list of booleans indicating whether the target is a >target inequality.
    :param lt_targets: A list of booleans indicating whether the target is a <target inequality.
    :return: The computed mse.
    """
    # When the target is a greater-than-inequality and the prediction is greater than the target,
    # replace the prediction with the target. Analogous for less-than-inequalities.
    preds = np.where(
        np.logical_and(np.greater(preds, targets),gt_targets),
        targets,
        preds,
    )
    preds = np.where(
        np.logical_and(np.less(preds, targets),lt_targets),
        targets,
        preds,
    )
    return mean_squared_error(targets, preds, squared=True)


def bounded_mae(targets: List[float], preds: List[float], gt_targets: List[bool] = None, lt_targets: List[bool] = None) -> float:
    """
    Computes the mean absolute error, considering targets with inequalities.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :param gt_targets: A list of booleans indicating whether the target is a >target inequality.
    :param lt_targets: A list of booleans indicating whether the target is a <target inequality.
    :return: The computed mse.
    """
    # When the target is a greater-than-inequality and the prediction is greater than the target,
    # replace the prediction with the target. Analogous for less-than-inequalities.
    preds = np.where(
        np.logical_and(np.greater(preds, targets),gt_targets),
        targets,
        preds,
    )
    preds = np.where(
        np.logical_and(np.less(preds, targets),lt_targets),
        targets,
        preds,
    )
    return mean_absolute_error(targets, preds)


def accuracy(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed accuracy.
    """
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction

    return accuracy_score(targets, hard_preds)


def f1_metric(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    Computes the f1 score of a binary prediction task using a given threshold for generating hard predictions.

    Will calculate for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed f1 score.
    """
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
        score = f1_score(targets, hard_preds, average='micro')
    else: # binary prediction
        hard_preds = [1 if p > threshold else 0 for p in preds]  
        score = f1_score(targets, hard_preds)

    return score


# from .run_training
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
from tensorboardX import SummaryWriter
from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR

## from evaluate.py
from collections import defaultdict
import logging
from typing import Dict, List

from simplify_chemprop_utils import create_logger, makedirs,multitask_mean,activate_dropout,compute_gnorm, compute_pnorm, NoamLR, param_count, param_count_all,get_data,split_data, get_task_names,get_class_sizes,save_smiles_splits,save_checkpoint,build_optimizer,build_lr_scheduler,load_checkpoint
from simplify_constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME,MODEL_FILE_NAME,AtomBondScaler,StandardScaler
from simplify_chemprop_features import set_extra_atom_fdim, set_extra_bond_fdim, set_explicit_h, set_adding_hs, set_keeping_atom_map, set_reaction, reset_featurization_parameters
from simplify_chemprop_model import MoleculeModel
from simplify_chemprop_data import  MoleculeDataset,MoleculeDataLoader, set_cache_graph
from simplify_chemprop_args import TrainArgs

from typing import List, Callable, Union
import torch.nn as nn

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss, f1_score, matthews_corrcoef
from typing import Any, List, Optional
import numpy as np



# from metrics.py
def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    r"""
    Gets the metric function corresponding to a given metric name.

    Supports:

    * :code:`auc`: Area under the receiver operating characteristic curve
    * :code:`prc-auc`: Area under the precision recall curve
    * :code:`rmse`: Root mean squared error
    * :code:`mse`: Mean squared error
    * :code:`mae`: Mean absolute error
    * :code:`r2`: Coefficient of determination R\ :superscript:`2`
    * :code:`accuracy`: Accuracy (using a threshold to binarize predictions)
    * :code:`cross_entropy`: Cross entropy
    * :code:`binary_cross_entropy`: Binary cross entropy
    * :code:`sid`: Spectral information divergence
    * :code:`wasserstein`: Wasserstein loss for spectra

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mse':
        return mean_squared_error

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'bounded_rmse':
        return bounded_rmse

    if metric == 'bounded_mse':
        return bounded_mse

    if metric == 'bounded_mae':
        return bounded_mae

    if metric == 'r2':
        return r2_score

    if metric == 'accuracy':
        return accuracy

    if metric == 'cross_entropy':
        return log_loss
    
    if metric == 'f1':
        return f1_metric

    if metric == 'mcc':
        return mcc_metric

    if metric == 'binary_cross_entropy':
        return bce
    
    if metric == 'sid':
        return sid_metric
    
    if metric == 'wasserstein':
        return wasserstein_metric

    raise ValueError(f'Metric "{metric}" not supported.')



import logging
from typing import Callable
# from train.py
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def train(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    loss_func: Callable,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    args: TrainArgs,
    n_iter: int = 0,
    atom_bond_scaler: AtomBondScaler = None,
    logger: logging.Logger = None,
    writer: SummaryWriter = None,
) -> int:
    """
    Trains a model for an epoch.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param atom_bond_scaler: A :class:`~chemprop.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param logger: A logger for recording output.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    if model.is_atom_bond_targets:
        loss_sum, iter_count = [0]*(len(args.atom_targets) + len(args.bond_targets)), 0
    else:
        loss_sum = iter_count = 0

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch, mask_batch, atom_descriptors_batch, atom_features_batch, bond_descriptors_batch, bond_features_batch, constraints_batch, data_weights_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.mask(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_descriptors(), batch.bond_features(), batch.constraints(), batch.data_weights()

        if model.is_atom_bond_targets:
            targets = []
            for dt in zip(*target_batch):
                dt = np.concatenate(dt)
                targets.append(torch.tensor([0 if x is None else x for x in dt], dtype=torch.float))
            masks = [torch.tensor(mask, dtype=torch.bool) for mask in mask_batch]
            if args.target_weights is not None:
                target_weights = [torch.ones(1, 1) * i for i in args.target_weights]  # shape(tasks, 1)
            else:
                target_weights = [torch.ones(1, 1) for i in targets]
            data_weights = batch.atom_bond_data_weights()
            data_weights = [torch.tensor(x).unsqueeze(1) for x in data_weights]

            natoms, nbonds = batch.number_of_atoms, batch.number_of_bonds
            natoms, nbonds = np.array(natoms).flatten(), np.array(nbonds).flatten()
            constraints_batch = np.transpose(constraints_batch).tolist()
            ind = 0
            for i in range(len(args.atom_targets)):
                if not args.atom_constraints[i]:
                    constraints_batch[ind] = None
                else:
                    mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                    for j, natom in enumerate(natoms):
                        constraints_batch[ind][j] = (constraints_batch[ind][j] - natom * mean) / std
                    constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(args.device)
                ind += 1
            for i in range(len(args.bond_targets)):
                if not args.bond_constraints[i]:
                    constraints_batch[ind] = None
                else:
                    mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                    for j, nbond in enumerate(nbonds):
                        constraints_batch[ind][j] = (constraints_batch[ind][j] - nbond * mean) / std
                    constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(args.device)
                ind += 1
            bond_types_batch = []
            for i in range(len(args.atom_targets)):
                bond_types_batch.append(None)
            for i in range(len(args.bond_targets)):
                if args.adding_bond_types and atom_bond_scaler is not None:
                    mean, std = atom_bond_scaler.means[i+len(args.atom_targets)][0], atom_bond_scaler.stds[i+len(args.atom_targets)][0]
                    bond_types = [(b.GetBondTypeAsDouble() - mean) / std for d in batch for b in d.mol[0].GetBonds()]
                    bond_types = torch.FloatTensor(bond_types).to(args.device)
                    bond_types_batch.append(bond_types)
                else:
                    bond_types_batch.append(None)
        else:
            mask_batch = np.transpose(mask_batch).tolist()
            masks = torch.tensor(mask_batch, dtype=torch.bool)  # shape(batch, tasks)
            targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch])  # shape(batch, tasks)

            if args.target_weights is not None:
                target_weights = torch.tensor(args.target_weights).unsqueeze(0)  # shape(1,tasks)
            else:
                target_weights = torch.ones(targets.shape[1]).unsqueeze(0)
            data_weights = torch.tensor(data_weights_batch).unsqueeze(1)  # shape(batch,1)

            constraints_batch = None
            bond_types_batch = None

            if args.loss_function == "bounded_mse":
                lt_target_batch = batch.lt_targets()  # shape(batch, tasks)
                gt_target_batch = batch.gt_targets()  # shape(batch, tasks)
                lt_target_batch = torch.tensor(lt_target_batch)
                gt_target_batch = torch.tensor(gt_target_batch)

        # Run model
        model.zero_grad()
        preds = model(
            mol_batch,
            features_batch,
            atom_descriptors_batch,
            atom_features_batch,
            bond_descriptors_batch,
            bond_features_batch,
            constraints_batch,
            bond_types_batch,
        )

        # Move tensors to correct device
        torch_device = args.device
        if model.is_atom_bond_targets:
            masks = [x.to(torch_device) for x in masks]
            masks = [x.reshape([-1, 1]) for x in masks]
            targets = [x.to(torch_device) for x in targets]
            targets = [x.reshape([-1, 1]) for x in targets]
            target_weights = [x.to(torch_device) for x in target_weights]
            data_weights = [x.to(torch_device) for x in data_weights]
        else:
            masks = masks.to(torch_device)
            targets = targets.to(torch_device)
            target_weights = target_weights.to(torch_device)
            data_weights = data_weights.to(torch_device)
            if args.loss_function == "bounded_mse":
                lt_target_batch = lt_target_batch.to(torch_device)
                gt_target_batch = gt_target_batch.to(torch_device)

        # Calculate losses
        if model.is_atom_bond_targets:
            loss_multi_task = []
            for target, pred, target_weight, data_weight, mask in zip(targets, preds, target_weights, data_weights, masks):
                if args.loss_function == "mcc" and args.dataset_type == "classification":
                    loss = loss_func(pred, target, data_weight, mask) * target_weight.squeeze(0)
                elif args.loss_function == "bounded_mse":
                    raise ValueError(f'Loss function "{args.loss_function}" is not supported with dataset type {args.dataset_type} in atomic/bond properties prediction.')
                elif args.loss_function in ["binary_cross_entropy", "mse", "mve"]:
                    loss = loss_func(pred, target) * target_weight * data_weight * mask
                elif args.loss_function == "evidential":
                    loss = loss_func(pred, target, args.evidential_regularization) * target_weight * data_weight * mask
                elif args.loss_function == "dirichlet" and args.dataset_type == "classification":
                    loss = loss_func(pred, target, args.evidential_regularization) * target_weight * data_weight * mask
                else:
                    raise ValueError(f'Dataset type "{args.dataset_type}" is not supported.')
                loss = loss.sum() / mask.sum()
                loss_multi_task.append(loss)

            loss_sum = [x + y for x, y in zip(loss_sum, loss_multi_task)]
            iter_count += 1

            sum(loss_multi_task).backward()
        else:
            if args.loss_function == "mcc" and args.dataset_type == "classification":
                loss = loss_func(preds, targets, data_weights, masks) * target_weights.squeeze(0)
            elif args.loss_function == "mcc":  # multiclass dataset type
                targets = targets.long()
                target_losses = []
                for target_index in range(preds.size(1)):
                    target_loss = loss_func(preds[:, target_index, :], targets[:, target_index], data_weights, masks[:, target_index]).unsqueeze(0)
                    target_losses.append(target_loss)
                loss = torch.cat(target_losses) * target_weights.squeeze(0)
            elif args.dataset_type == "multiclass":
                targets = targets.long()
                if args.loss_function == "dirichlet":
                    loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * masks
                else:
                    target_losses = []
                    for target_index in range(preds.size(1)):
                        target_loss = loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1)
                        target_losses.append(target_loss)
                    loss = torch.cat(target_losses, dim=1).to(torch_device) * target_weights * data_weights * masks
            elif args.dataset_type == "spectra":
                loss = loss_func(preds, targets, masks) * target_weights * data_weights * masks
            elif args.loss_function == "bounded_mse":
                loss = loss_func(preds, targets, lt_target_batch, gt_target_batch) * target_weights * data_weights * masks
            elif args.loss_function == "evidential":
                loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * masks
            elif args.loss_function == "dirichlet":  # classification
                loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * masks
            else:
                loss = loss_func(preds, targets) * target_weights * data_weights * masks

            if args.loss_function == "mcc":
                loss = loss.mean()
            else:
                loss = loss.sum() / masks.sum()

            loss_sum += loss.item()
            iter_count += 1

            loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            if model.is_atom_bond_targets:
                loss_avg = sum(loss_sum) / iter_count
                loss_sum, iter_count = [0]*(len(args.atom_targets) + len(args.bond_targets)), 0
            else:
                loss_avg = loss_sum / iter_count
                loss_sum = iter_count = 0

            lrs_str = ", ".join(f"lr_{i} = {lr:.4e}" for i, lr in enumerate(lrs))
            debug(f"Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}")

            if writer is not None:
                writer.add_scalar("train_loss", loss_avg, n_iter)
                writer.add_scalar("param_norm", pnorm, n_iter)
                writer.add_scalar("gradient_norm", gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f"learning_rate_{i}", lr, n_iter)

    return n_iter
## from loss_functions
def get_loss_func(args: TrainArgs) -> Callable:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Arguments containing the dataset type ("classification", "regression", or "multiclass").
    :return: A PyTorch loss function.
    """

    # Nested dictionary of the form {dataset_type: {loss_function: loss_function callable}}
    supported_loss_functions = {
        "regression": {
            "mse": nn.MSELoss(reduction="none"),
            "bounded_mse": bounded_mse_loss,
            "mve": normal_mve,
            "evidential": evidential_loss,
        },
        "classification": {
            "binary_cross_entropy": nn.BCEWithLogitsLoss(reduction="none"),
            "mcc": mcc_class_loss,
            "dirichlet": dirichlet_class_loss,
        },
        "multiclass": {
            "cross_entropy": nn.CrossEntropyLoss(reduction="none"),
            "mcc": mcc_multiclass_loss,
            "dirichlet": dirichlet_multiclass_loss,
        },
        "spectra": {
            "sid": sid_loss,
            "wasserstein": wasserstein_loss,
        },
    }

    # Error if no loss function supported
    if args.dataset_type not in supported_loss_functions.keys():
        raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

    # Return loss function if it is represented in the supported_loss_functions dictionary
    loss_function_choices = supported_loss_functions.get(args.dataset_type, dict())
    loss_function = loss_function_choices.get(args.loss_function)

    if loss_function is not None:
        return loss_function

    else:
        raise ValueError(
            f'Loss function "{args.loss_function}" not supported with dataset type {args.dataset_type}. \
            Available options for that dataset type are {loss_function_choices.keys()}.'
        )
    

## from evaluate.py
from collections import defaultdict
import logging
from typing import Dict, List

# from loss_functions.py
def bounded_mse_loss(
    predictions: torch.tensor,
    targets: torch.tensor,
    less_than_target: torch.tensor,
    greater_than_target: torch.tensor,
) -> torch.tensor:
    """
    Loss function for use with regression when some targets are presented as inequalities.

    :param predictions: Model predictions with shape(batch_size, tasks).
    :param targets: Target values with shape(batch_size, tasks).
    :param less_than_target: A tensor with boolean values indicating whether the target is a less-than inequality.
    :param greater_than_target: A tensor with boolean values indicating whether the target is a greater-than inequality.
    :return: A tensor containing loss values of shape(batch_size, tasks).
    """
    predictions = torch.where(torch.logical_and(predictions < targets, less_than_target), targets, predictions)

    predictions = torch.where(
        torch.logical_and(predictions > targets, greater_than_target),
        targets,
        predictions,
    )

    return nn.functional.mse_loss(predictions, targets, reduction="none")

def normal_mve(pred_values, targets):
    """
    Use the negative log likelihood function of a normal distribution as a loss function used for making
    simultaneous predictions of the mean and error distribution variance simultaneously.

    :param pred_values: Combined predictions of means and variances of shape(data, tasks*2).
                        Means are first in dimension 1, followed by variances.
    :return: A tensor loss value.
    """
    # Unpack combined prediction values
    pred_means, pred_var = torch.split(pred_values, pred_values.shape[1] // 2, dim=1)

    return torch.log(2 * np.pi * pred_var) / 2 + (pred_means - targets) ** 2 / (2 * pred_var)

def mcc_class_loss(
    predictions: torch.tensor,
    targets: torch.tensor,
    data_weights: torch.tensor,
    mask: torch.tensor,
) -> torch.tensor:
    """
    A classification loss using a soft version of the Matthews Correlation Coefficient.

    :param predictions: Model predictions with shape(batch_size, tasks).
    :param targets: Target values with shape(batch_size, tasks).
    :param data_weights: A tensor with float values indicating how heavily to weight each datapoint in training with shape(batch_size, 1)
    :param mask: A tensor with boolean values indicating whether the loss for this prediction is considered in the gradient descent with shape(batch_size, tasks).
    :return: A tensor containing loss values of shape(tasks).
    """
    # shape(batch, tasks)
    # (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    TP = torch.sum(targets * predictions * data_weights * mask, axis=0)
    FP = torch.sum((1 - targets) * predictions * data_weights * mask, axis=0)
    FN = torch.sum(targets * (1 - predictions) * data_weights * mask, axis=0)
    TN = torch.sum((1 - targets) * (1 - predictions) * data_weights * mask, axis=0)
    loss = 1 - ((TP * TN - FP * FN) / torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    return loss


def mcc_multiclass_loss(
    predictions: torch.tensor,
    targets: torch.tensor,
    data_weights: torch.tensor,
    mask: torch.tensor,
) -> torch.tensor:
    """
    A multiclass loss using a soft version of the Matthews Correlation Coefficient. Multiclass definition follows the version in sklearn documentation (https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-correlation-coefficient).

    :param predictions: Model predictions with shape(batch_size, classes).
    :param targets: Target values with shape(batch_size).
    :param data_weights: A tensor with float values indicating how heavily to weight each datapoint in training with shape(batch_size, 1)
    :param mask: A tensor with boolean values indicating whether the loss for this prediction is considered in the gradient descent with shape(batch_size).
    :return: A tensor value for the loss.
    """
    torch_device = predictions.device
    mask = mask.unsqueeze(1)

    bin_targets = torch.zeros_like(predictions, device=torch_device)
    bin_targets[torch.arange(predictions.shape[0]), targets] = 1

    pred_classes = predictions.argmax(dim=1)
    bin_preds = torch.zeros_like(predictions, device=torch_device)
    bin_preds[torch.arange(predictions.shape[0]), pred_classes] = 1

    masked_data_weights = data_weights * mask

    t_sum = torch.sum(bin_targets * masked_data_weights, axis=0)  # number of times each class truly occurred
    p_sum = torch.sum(bin_preds * masked_data_weights, axis=0)  # number of times each class was predicted

    n_correct = torch.sum(bin_preds * bin_targets * masked_data_weights)  # total number of samples correctly predicted
    n_samples = torch.sum(predictions * masked_data_weights)  # total number of samples

    cov_ytyp = n_correct * n_samples - torch.dot(p_sum, t_sum)
    cov_ypyp = n_samples**2 - torch.dot(p_sum, p_sum)
    cov_ytyt = n_samples**2 - torch.dot(t_sum, t_sum)

    if cov_ypyp * cov_ytyt == 0:
        loss = torch.tensor(1.0, device=torch_device)
    else:
        mcc = cov_ytyp / torch.sqrt(cov_ytyt * cov_ypyp)
        loss = 1 - mcc

    return loss


def sid_loss(
    model_spectra: torch.tensor,
    target_spectra: torch.tensor,
    mask: torch.tensor,
    threshold: float = None,
) -> torch.tensor:
    """
    Loss function for use with spectra data type.

    :param model_spectra: The predicted spectra output from a model with shape (batch_size,spectrum_length).
    :param target_spectra: The target spectra with shape (batch_size,spectrum_length). Values must be normalized so that each spectrum sums to 1.
    :param mask: Tensor with boolean indications of where the spectrum output should not be excluded with shape (batch_size,spectrum_length).
    :param threshold: Loss function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :return: A tensor containing loss values for the batch with shape (batch_size,spectrum_length).
    """
    # Move new tensors to torch device
    torch_device = model_spectra.device

    # Normalize the model spectra before comparison
    zero_sub = torch.zeros_like(model_spectra, device=torch_device)
    one_sub = torch.ones_like(model_spectra, device=torch_device)
    if threshold is not None:
        threshold_sub = torch.full(model_spectra.shape, threshold, device=torch_device)
        model_spectra = torch.where(model_spectra < threshold, threshold_sub, model_spectra)
    model_spectra = torch.where(mask, model_spectra, zero_sub)
    sum_model_spectra = torch.sum(model_spectra, axis=1, keepdim=True)
    model_spectra = torch.div(model_spectra, sum_model_spectra)

    # Calculate loss value
    target_spectra = torch.where(mask, target_spectra, one_sub)
    model_spectra = torch.where(mask, model_spectra, one_sub)  # losses in excluded regions will be zero because log(1/1) = 0.
    loss = torch.mul(torch.log(torch.div(model_spectra, target_spectra)), model_spectra) + torch.mul(
        torch.log(torch.div(target_spectra, model_spectra)), target_spectra
    )

    return loss

## from predict.py
def predict(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    disable_progress_bar: bool = False,
    scaler: StandardScaler = None,
    atom_bond_scaler: AtomBondScaler = None,
    return_unc_parameters: bool = False,
    dropout_prob: float = 0.0,
) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param atom_bond_scaler: A :class:`~chemprop.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param return_unc_parameters: A bool indicating whether additional uncertainty parameters would be returned alongside the mean predictions.
    :param dropout_prob: For use during uncertainty prediction only. The propout probability used in generating a dropout ensemble.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks. If returning uncertainty parameters as well,
        it is a tuple of lists of lists, of a length depending on how many uncertainty parameters are appropriate for the loss function.
    """
    model.eval()

    # Activate dropout layers to work during inference for uncertainty estimation
    if dropout_prob > 0.0:

        def activate_dropout_(model):
            return activate_dropout(model, dropout_prob)

        model.apply(activate_dropout_)

    preds = []

    var, lambdas, alphas, betas = [], [], [], []  # only used if returning uncertainty parameters

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch = batch.batch_graph()
        features_batch = batch.features()
        atom_descriptors_batch = batch.atom_descriptors()
        atom_features_batch = batch.atom_features()
        bond_descriptors_batch = batch.bond_descriptors()
        bond_features_batch = batch.bond_features()
        constraints_batch = batch.constraints()

        if model.is_atom_bond_targets:
            natoms, nbonds = batch.number_of_atoms, batch.number_of_bonds
            natoms, nbonds = np.array(natoms).flatten(), np.array(nbonds).flatten()
            constraints_batch = np.transpose(constraints_batch).tolist()
            device = next(model.parameters()).device

            # If the path to constraints is not given, the constraints matrix needs to be reformatted.
            if constraints_batch == []:
                for _ in batch._data:
                    natom_targets = len(model.atom_targets)
                    nbond_targets = len(model.bond_targets)
                    ntargets = natom_targets + nbond_targets
                    constraints_batch.append([None] * ntargets)

            ind = 0
            for i in range(len(model.atom_targets)):
                if not model.atom_constraints[i]:
                    constraints_batch[ind] = None
                else:
                    mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                    for j, natom in enumerate(natoms):
                        constraints_batch[ind][j] = (constraints_batch[ind][j] - natom * mean) / std
                    constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(device)
                ind += 1
            for i in range(len(model.bond_targets)):
                if not model.bond_constraints[i]:
                    constraints_batch[ind] = None
                else:
                    mean, std = atom_bond_scaler.means[ind][0], atom_bond_scaler.stds[ind][0]
                    for j, nbond in enumerate(nbonds):
                        constraints_batch[ind][j] = (constraints_batch[ind][j] - nbond * mean) / std
                    constraints_batch[ind] = torch.tensor(constraints_batch[ind]).to(device)
                ind += 1
            bond_types_batch = []
            for i in range(len(model.atom_targets)):
                bond_types_batch.append(None)
            for i in range(len(model.bond_targets)):
                if model.adding_bond_types and atom_bond_scaler is not None:
                    mean, std = atom_bond_scaler.means[i+len(model.atom_targets)][0], atom_bond_scaler.stds[i+len(model.atom_targets)][0]
                    bond_types = [(b.GetBondTypeAsDouble() - mean) / std for d in batch for b in d.mol[0].GetBonds()]
                    bond_types = torch.FloatTensor(bond_types).to(device)
                    bond_types_batch.append(bond_types)
                else:
                    bond_types_batch.append(None)
        else:
            bond_types_batch = None

        # Make predictions
        with torch.no_grad():
            batch_preds = model(
                mol_batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
                constraints_batch,
                bond_types_batch,
            )

        if model.is_atom_bond_targets:
            batch_preds = [x.data.cpu().numpy() for x in batch_preds]
            batch_vars, batch_lambdas, batch_alphas, batch_betas = [], [], [], []

            for i, batch_pred in enumerate(batch_preds):
                if model.loss_function == "mve":
                    batch_pred, batch_var = np.split(batch_pred, 2, axis=1)
                    batch_vars.append(batch_var)
                elif model.loss_function == "dirichlet":
                    if model.classification:
                        batch_alpha = np.reshape(
                            batch_pred,
                            [batch_pred.shape[0], batch_pred.shape[1] // 2, 2],
                        )
                        batch_pred = batch_alpha[:, :, 1] / np.sum(
                            batch_alpha, axis=2
                        )  # shape(data, tasks, 2)
                        batch_alphas.append(batch_alpha)
                    elif model.multiclass:
                        raise ValueError(
                            f"In atomic/bond properties prediction, {model.multiclass} is not supported."
                        )
                elif model.loss_function == "evidential":  # regression
                    batch_pred, batch_lambda, batch_alpha, batch_beta = np.split(
                        batch_pred, 4, axis=1
                    )
                    batch_alphas.append(batch_alpha)
                    batch_lambdas.append(batch_lambda)
                    batch_betas.append(batch_beta)
                batch_preds[i] = batch_pred

            # Inverse scale for each atom/bond target if regression
            if atom_bond_scaler is not None:
                batch_preds = atom_bond_scaler.inverse_transform(batch_preds)
                for i, stds in enumerate(atom_bond_scaler.stds):
                    if model.loss_function == "mve":
                        batch_vars[i] = batch_vars[i] * stds ** 2
                    elif model.loss_function == "evidential":
                        batch_betas[i] = batch_betas[i] * stds ** 2

            # Collect vectors
            preds.append(batch_preds)
            if model.loss_function == "mve":
                var.append(batch_vars)
            elif model.loss_function == "dirichlet" and model.classification:
                alphas.append(batch_alphas)
            elif model.loss_function == "evidential":  # regression
                lambdas.append(batch_lambdas)
                alphas.append(batch_alphas)
                betas.append(batch_betas)
        else:
            batch_preds = batch_preds.data.cpu().numpy()

            if model.loss_function == "mve":
                batch_preds, batch_var = np.split(batch_preds, 2, axis=1)
            elif model.loss_function == "dirichlet":
                if model.classification:
                    batch_alphas = np.reshape(
                        batch_preds,
                        [batch_preds.shape[0], batch_preds.shape[1] // 2, 2],
                    )
                    batch_preds = batch_alphas[:, :, 1] / np.sum(
                        batch_alphas, axis=2
                    )  # shape(data, tasks, 2)
                elif model.multiclass:
                    batch_alphas = batch_preds
                    batch_preds = batch_preds / np.sum(
                        batch_alphas, axis=2, keepdims=True
                    )  # shape(data, tasks, num_classes)
            elif model.loss_function == "evidential":  # regression
                batch_preds, batch_lambdas, batch_alphas, batch_betas = np.split(
                    batch_preds, 4, axis=1
                )

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
                if model.loss_function == "mve":
                    batch_var = batch_var * scaler.stds**2
                elif model.loss_function == "evidential":
                    batch_betas = batch_betas * scaler.stds**2

            # Collect vectors
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)
            if model.loss_function == "mve":
                var.extend(batch_var.tolist())
            elif model.loss_function == "dirichlet" and model.classification:
                alphas.extend(batch_alphas.tolist())
            elif model.loss_function == "evidential":  # regression
                lambdas.extend(batch_lambdas.tolist())
                alphas.extend(batch_alphas.tolist())
                betas.extend(batch_betas.tolist())

    if model.is_atom_bond_targets:
        preds = [np.concatenate(x) for x in zip(*preds)]
        var = [np.concatenate(x) for x in zip(*var)]
        alphas = [np.concatenate(x) for x in zip(*alphas)]
        betas = [np.concatenate(x) for x in zip(*betas)]
        lambdas = [np.concatenate(x) for x in zip(*lambdas)]

    if return_unc_parameters:
        if model.loss_function == "mve":
            return preds, var
        elif model.loss_function == "dirichlet":
            return preds, alphas
        elif model.loss_function == "evidential":
            return preds, lambdas, alphas, betas

    return preds

# updated evidential regression loss (evidential_loss_new from Amini repo)
def evidential_loss(pred_values, targets, lam: float = 0, epsilon: float = 1e-8, v_min: float = 1e-5):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :param pred_values: Combined prediction values for mu, v, alpha, and beta parameters in shape(data, tasks*4).
                        Order in dimension 1 is mu, v, alpha, beta.
    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :param targets: Outputs to predict
    :param lam: regularization coefficient
    :param v_min: clamp any v below this value to prevent Inf from division

    :return: Loss
    """
    # Unpack combined prediction values
    mu, v, alpha, beta = torch.split(pred_values, pred_values.shape[1] // 4, dim=1)

    # Calculate NLL loss
    v = torch.clamp(v, v_min)
    twoBlambda = 2 * beta * (1 + v)
    nll = (
        0.5 * torch.log(np.pi / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    L_NLL = nll  # torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg  # torch.mean(reg, dim=-1)

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return loss

# evidential classification
def dirichlet_class_loss(alphas, target_labels, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al in classification datasets.
    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, tasks*2).
                   Negative class first then positive class in dimension 1.
    :param target_labels: Digital labels to predict in shape(datapoints, tasks).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    torch_device = alphas.device
    num_tasks = target_labels.shape[1]
    num_classes = 2
    alphas = torch.reshape(alphas, (alphas.shape[0], num_tasks, num_classes))

    y_one_hot = torch.eye(num_classes, device=torch_device)[target_labels.long()]

    return dirichlet_common_loss(alphas=alphas, y_one_hot=y_one_hot, lam=lam)


def dirichlet_multiclass_loss(alphas, target_labels, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al for multiclass datasets.
    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, task, classes).
    :param target_labels: Digital labels to predict in shape(datapoints, tasks).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    torch_device = alphas.device
    num_classes = alphas.shape[2]

    y_one_hot = torch.eye(num_classes, device=torch_device)[target_labels.long()]

    return dirichlet_common_loss(alphas=alphas, y_one_hot=y_one_hot, lam=lam)


def dirichlet_common_loss(alphas, y_one_hot, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al. This function follows
    after the classification and multiclass specific functions that reshape the
    alpha inputs and create one-hot targets.

    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, task, classes).
    :param y_one_hot: Digital labels to predict in shape(datapoints, tasks, classes).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    # SOS term
    S = torch.sum(alphas, dim=-1, keepdim=True)
    p = alphas / S
    A = torch.sum((y_one_hot - p) ** 2, dim=-1, keepdim=True)
    B = torch.sum((p * (1 - p)) / (S + 1), dim=-1, keepdim=True)
    SOS = A + B

    alpha_hat = y_one_hot + (1 - y_one_hot) * alphas

    beta = torch.ones_like(alpha_hat)
    S_alpha = torch.sum(alpha_hat, dim=-1, keepdim=True)
    S_beta = torch.sum(beta, dim=-1, keepdim=True)

    ln_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha_hat), dim=-1, keepdim=True)
    ln_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)

    # digamma terms
    dg_alpha = torch.digamma(alpha_hat)
    dg_S_alpha = torch.digamma(S_alpha)

    # KL
    KL = ln_alpha + ln_beta + torch.sum((alpha_hat - beta) * (dg_alpha - dg_S_alpha), dim=-1, keepdim=True)

    KL = lam * KL

    # loss = torch.mean(SOS + KL)
    loss = SOS + KL
    loss = torch.mean(loss, dim=-1)
    return loss

def wasserstein_loss(
    model_spectra: torch.tensor,
    target_spectra: torch.tensor,
    mask: torch.tensor,
    threshold: float = None,
) -> torch.tensor:
    """
    Loss function for use with spectra data type. This loss assumes that values are evenly spaced.

    :param model_spectra: The predicted spectra output from a model with shape (batch_size,spectrum_length).
    :param target_spectra: The target spectra with shape (batch_size,spectrum_length). Values must be normalized so that each spectrum sums to 1.
    :param mask: Tensor with boolian indications of where the spectrum output should not be excluded with shape (batch_size,spectrum_length).
    :param threshold: Loss function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :return: A tensor containing loss values for the batch with shape (batch_size,spectrum_length).
    """
    # Move new tensors to torch device
    torch_device = model_spectra.device

    # Normalize the model spectra before comparison
    zero_sub = torch.zeros_like(model_spectra, device=torch_device)
    if threshold is not None:
        threshold_sub = torch.full(model_spectra.shape, threshold, device=torch_device)
        model_spectra = torch.where(model_spectra < threshold, threshold_sub, model_spectra)
    model_spectra = torch.where(mask, model_spectra, zero_sub)
    sum_model_spectra = torch.sum(model_spectra, axis=1, keepdim=True)
    model_spectra = torch.div(model_spectra, sum_model_spectra)

    # Calculate loss value
    target_cum = torch.cumsum(target_spectra, axis=1)
    model_cum = torch.cumsum(model_spectra, axis=1)
    loss = torch.abs(target_cum - model_cum)

    return loss

logger_name=TRAIN_LOGGER_NAME

## from evaluate
def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metrics: List[str],
                         dataset_type: str,
                         is_atom_bond_targets: bool = False,
                         gt_targets: List[List[bool]] = None,
                         lt_targets: List[List[bool]] = None,
                         logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates predictions using a metric function after filtering out invalid targets.

    :param preds: A list of lists of shape :code:`(data_size, num_tasks)` with model predictions.
    :param targets: A list of lists of shape :code:`(data_size, num_tasks)` with targets.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param is_atom_bond_targets: Boolean whether this is atomic/bond properties prediction.
    :param gt_targets: A list of lists of booleans indicating whether the target is an inequality rather than a single value.
    :param lt_targets: A list of lists of booleans indicating whether the target is an inequality rather than a single value.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    info = logger.info if logger is not None else print

    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    if len(preds) == 0:
        return {metric: [float('nan')] * num_tasks for metric in metrics}
    
    if is_atom_bond_targets:
        targets = [np.concatenate(x).reshape([-1, 1]) for x in zip(*targets)]

    # Filter out empty targets for most data types, excluding dataset_type spectra
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    if dataset_type != 'spectra':
        valid_preds = [[] for _ in range(num_tasks)]
        valid_targets = [[] for _ in range(num_tasks)]
        for i in range(num_tasks):
            if is_atom_bond_targets:
                for j in range(len(preds[i])):
                    if targets[i][j][0] is not None:  # Skip those without targets
                        valid_preds[i].append(list(preds[i][j]))
                        valid_targets[i].append(list(targets[i][j]))
            else:
                for j in range(len(preds)):
                    if targets[j][i] is not None:  # Skip those without targets
                        valid_preds[i].append(preds[j][i])
                        valid_targets[i].append(targets[j][i])

    # Compute metric. Spectra loss calculated for all tasks together, others calculated for tasks individually.
    results = defaultdict(list)
    if dataset_type == 'spectra':
        for metric, metric_func in metric_to_func.items():
            results[metric].append(metric_func(preds, targets))
    elif is_atom_bond_targets:
        for metric, metric_func in metric_to_func.items():
            for valid_target, valid_pred in zip(valid_targets, valid_preds):
                results[metric].append(metric_func(valid_target, valid_pred))
    else:
        for i in range(num_tasks):
            # # Skip if all targets or preds are identical, otherwise we'll crash during classification
            if dataset_type == 'classification':
                nan = False
                if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                    nan = True
                    info('Warning: Found a task with targets all 0s or all 1s')
                if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                    nan = True
                    info('Warning: Found a task with predictions all 0s or all 1s')

                if nan:
                    for metric in metrics:
                        results[metric].append(float('nan'))
                    continue

            if len(valid_targets[i]) == 0:
                continue

            for metric, metric_func in metric_to_func.items():
                if dataset_type == 'multiclass' and metric == 'cross_entropy':
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i],
                                                    labels=list(range(len(valid_preds[i][0])))))
                elif metric in ['bounded_rmse', 'bounded_mse', 'bounded_mae']:
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i], gt_targets[i], lt_targets[i]))
                else:
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i]))

    results = dict(results)

    return results


def evaluate(model: MoleculeModel,
             data_loader: MoleculeDataLoader,
             num_tasks: int,
             metrics: List[str],
             dataset_type: str,
             scaler: StandardScaler = None,
             atom_bond_scaler: AtomBondScaler = None,
             logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates an ensemble of models on a dataset by making predictions and then evaluating the predictions.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param atom_bond_scaler: A :class:`~chemprop.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.

    """
    # Inequality targets only need for evaluation of certain regression metrics
    if any(m in metrics for m in ['bounded_rmse', 'bounded_mse', 'bounded_mae']):
        gt_targets = data_loader.gt_targets
        lt_targets = data_loader.lt_targets
    else:
        gt_targets = None
        lt_targets = None

    preds = predict(
        model=model,
        data_loader=data_loader,
        scaler=scaler,
        atom_bond_scaler=atom_bond_scaler,
    )

    results = evaluate_predictions(
        preds=preds,
        targets=data_loader.targets,
        num_tasks=num_tasks,
        metrics=metrics,
        dataset_type=dataset_type,
        is_atom_bond_targets=model.is_atom_bond_targets,
        logger=logger,
        gt_targets=gt_targets,
        lt_targets=lt_targets,
    )

    return results


## from run_training.py
def run_training(args: TrainArgs,
                 data: MoleculeDataset,
                 logger: Logger = None) -> Dict[str, List[float]]:
    """
    Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path,
                             args=args,
                             features_path=args.separate_test_features_path,
                             atom_descriptors_path=args.separate_test_atom_descriptors_path,
                             bond_descriptors_path=args.separate_test_bond_descriptors_path,
                             phase_features_path=args.separate_test_phase_features_path,
                             constraints_path=args.separate_test_constraints_path,
                             smiles_columns=args.smiles_columns,
                             loss_function=args.loss_function,
                             logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path,
                            args=args,
                            features_path=args.separate_val_features_path,
                            atom_descriptors_path=args.separate_val_atom_descriptors_path,
                            bond_descriptors_path=args.separate_val_bond_descriptors_path,
                            phase_features_path=args.separate_val_phase_features_path,
                            constraints_path=args.separate_val_constraints_path,
                            smiles_columns=args.smiles_columns,
                            loss_function=args.loss_function,
                            logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data,
                                              split_type=args.split_type,
                                              sizes=args.split_sizes,
                                              key_molecule_index=args.split_key_molecule,
                                              seed=args.seed,
                                              num_folds=args.num_folds,
                                              args=args,
                                              logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data,
                                             split_type=args.split_type,
                                             sizes=args.split_sizes,
                                             key_molecule_index=args.split_key_molecule,
                                             seed=args.seed,
                                             num_folds=args.num_folds,
                                             args=args,
                                             logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data,
                                                     split_type=args.split_type,
                                                     sizes=args.split_sizes,
                                                     key_molecule_index=args.split_key_molecule,
                                                     seed=args.seed,
                                                     num_folds=args.num_folds,
                                                     args=args,
                                                     logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
        train_class_sizes = get_class_sizes(train_data, proportion=False)
        args.train_class_sizes = train_class_sizes

    if args.save_smiles_splits:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            constraints_path=args.constraints_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            logger=logger,
        )

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
        val_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None

    if args.bond_descriptor_scaling and args.bond_descriptors is not None:
        bond_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_bond_descriptors=True)
        val_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
        test_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
    else:
        bond_descriptor_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    if len(val_data) == 0:
        raise ValueError('The validation data split is empty. During normal chemprop training (non-sklearn functions), \
            a validation set is required to conduct early stopping according to the selected evaluation metric. This \
            may have occurred because validation data provided with `--separate_val_path` was empty or contained only invalid molecules.')

    if len(test_data) == 0:
        debug('The test data split is empty. This may be either because splitting with no test set was selected, \
            such as with `cv-no-test`, or because test data provided with `--separate_test_path` was empty or contained only invalid molecules. \
            Performance on the test set will not be evaluated and metric scores will return `nan` for each task.')
        empty_test_set = True
    else:
        empty_test_set = False


    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        if args.is_atom_bond_targets:
            scaler = None
            atom_bond_scaler = train_data.normalize_atom_bond_targets()
        else:
            scaler = train_data.normalize_targets()
            atom_bond_scaler = None
        args.spectra_phase_mask = None
    else:
        args.spectra_phase_mask = None
        debug('Normalizing spectra and excluding spectra regions based on phase')
        scaler = None
        atom_bond_scaler = None

    # Get loss function
    loss_func = get_loss_func(args)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    elif args.is_atom_bond_targets:
        sum_test_preds = []
        for tb in zip(*test_data.targets()):
            tb = np.concatenate(tb)
            sum_test_preds.append(np.zeros((tb.shape[0], 1)))
        sum_test_preds = np.array(sum_test_preds, dtype=object)
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Automatically determine whether to cache
    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    # Create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )

    if args.class_balance:
        debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = MoleculeModel(args)

        # Optionally, overwrite weights:
        if args.checkpoint_frzn is not None:
            debug(f'Loading and freezing parameters from {args.checkpoint_frzn}.')
            model = load_frzn_model(model=model, path=args.checkpoint_frzn, current_args=args, logger=logger)

        debug(model)

        if args.checkpoint_frzn is not None:
            debug(f'Number of unfrozen parameters = {param_count(model):,}')
            debug(f'Total number of parameters = {param_count_all(model):,}')
        else:
            debug(f'Number of parameters = {param_count_all(model):,}')

        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler,
                        features_scaler, atom_descriptor_scaler, bond_descriptor_scaler,
                        atom_bond_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')
            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                atom_bond_scaler=atom_bond_scaler,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                logger=logger
            )

            for metric, scores in val_scores.items():
                # Average validation score\
                mean_val_score = multitask_mean(scores, metric=metric)
                debug(f'Validation {metric} = {mean_val_score:.6f}')
                writer.add_scalar(f'validation_{metric}', mean_val_score, n_iter)

                if args.show_individual_scores:
                    # Individual validation scores
                    for task_name, val_score in zip(args.task_names, scores):
                        debug(f'Validation {task_name} {metric} = {val_score:.6f}')
                        writer.add_scalar(f'validation_{task_name}_{metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score
            mean_val_score = multitask_mean(val_scores[args.metric], metric=args.metric)
            if args.minimize_score and mean_val_score < best_score or \
                    not args.minimize_score and mean_val_score > best_score:
                best_score, best_epoch = mean_val_score, epoch
                save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, features_scaler,
                                atom_descriptor_scaler, bond_descriptor_scaler, atom_bond_scaler, args)

        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)

        if empty_test_set:
            info(f'Model {model_idx} provided with no test set, no metric evaluation will be performed.')
        else:
            test_preds = predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                is_atom_bond_targets=args.is_atom_bond_targets,
                gt_targets=test_data.gt_targets(),
                lt_targets=test_data.lt_targets(),
                logger=logger
            )

            if len(test_preds) != 0:
                if args.is_atom_bond_targets:
                    sum_test_preds += np.array(test_preds, dtype=object)
                else:
                    sum_test_preds += np.array(test_preds)

            # Average test score
            for metric, scores in test_scores.items():
                avg_test_score = np.nanmean(scores)
                info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
                writer.add_scalar(f'test_{metric}', avg_test_score, 0)

                if args.show_individual_scores and args.dataset_type != 'spectra':
                    # Individual test scores
                    for task_name, test_score in zip(args.task_names, scores):
                        info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
                        writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
        writer.close()

    # Evaluate ensemble on test set
    if empty_test_set:
        ensemble_scores = {
            metric: [np.nan for task in args.task_names] for metric in args.metrics
        }
    else:
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            is_atom_bond_targets=args.is_atom_bond_targets,
            gt_targets=test_data.gt_targets(),
            lt_targets=test_data.lt_targets(),
            logger=logger
        )

    for metric, scores in ensemble_scores.items():
        # Average ensemble score
        mean_ensemble_test_score = multitask_mean(scores, metric=metric)
        info(f'Ensemble test {metric} = {mean_ensemble_test_score:.6f}')

        # Individual ensemble scores
        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, scores):
                info(f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')

    # Save scores
    with open(os.path.join(args.save_dir, 'test_scores.json'), 'w') as f:
        json.dump(ensemble_scores, f, indent=4, sort_keys=True)

    # Optionally save test preds
    if args.save_preds and not empty_test_set:
        test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})

        if args.is_atom_bond_targets:
            n_atoms, n_bonds = test_data.number_of_atoms, test_data.number_of_bonds

            for i, atom_target in enumerate(args.atom_targets):
                values = np.split(np.array(avg_test_preds[i]).flatten(), np.cumsum(np.array(n_atoms)))[:-1]
                values = [list(v) for v in values]
                test_preds_dataframe[atom_target] = values
            for i, bond_target in enumerate(args.bond_targets):
                values = np.split(np.array(avg_test_preds[i+len(args.atom_targets)]).flatten(), np.cumsum(np.array(n_bonds)))[:-1]
                values = [list(v) for v in values]
                test_preds_dataframe[bond_target] = values
        else:
            for i, task_name in enumerate(args.task_names):
                test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]

        test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    return ensemble_scores

def cross_validate(args: TrainArgs,
                   train_func: Callable[[TrainArgs, MoleculeDataset, Logger], Dict[str, List[float]]]
                   ) -> Tuple[float, float]:
    """
    Runs k-fold cross-validation.

    For each of k splits (folds) of the data, trains and tests a model on that split
    and aggregates the performance across folds.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param train_func: Function which runs training.
    :return: A tuple containing the mean and standard deviation performance across folds.
    """
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    # Save args
    makedirs(args.save_dir)
    try:
        args.save(os.path.join(args.save_dir, 'args.json'))
    except subprocess.CalledProcessError:
        debug('Could not write the reproducibility section of the arguments to file, thus omitting this section.')
        args.save(os.path.join(args.save_dir, 'args.json'), with_reproducibility=False)

    # set explicit H option and reaction option
    reset_featurization_parameters(logger=logger)
    set_explicit_h(args.explicit_h)
    set_adding_hs(args.adding_h)
    set_keeping_atom_map(args.keeping_atom_map)
    if args.reaction:
        set_reaction(args.reaction, args.reaction_mode)
    elif args.reaction_solvent:
        set_reaction(True, args.reaction_mode)
    
    # Get data
    debug('Loading data')
    data = get_data(
        path=args.data_path,
        args=args,
        logger=logger,
        skip_none_targets=True,
        data_weights_path=args.data_weights_path
    )
   
    args.features_size = data.features_size()

    if args.atom_descriptors == 'descriptor':
        args.atom_descriptors_size = data.atom_descriptors_size()
    elif args.atom_descriptors == 'feature':
        args.atom_features_size = data.atom_features_size()
        set_extra_atom_fdim(args.atom_features_size)
        
    if args.bond_descriptors == 'descriptor':
        args.bond_descriptors_size = data.bond_descriptors_size()
    elif args.bond_descriptors == 'feature':
        args.bond_features_size = data.bond_features_size()
        set_extra_bond_fdim(args.bond_features_size)

    debug(f'Number of tasks = {args.num_tasks}')

    if args.target_weights is not None and len(args.target_weights) != args.num_tasks:
        raise ValueError('The number of provided target weights must match the number and order of the prediction tasks')

    # Run training on different random seeds for each fold
    all_scores = defaultdict(list)
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        data.reset_features_and_targets()

        # If resuming experiment, load results from trained models
        test_scores_path = os.path.join(args.save_dir, 'test_scores.json')
        if args.resume_experiment and os.path.exists(test_scores_path):
            print('Loading scores')
            with open(test_scores_path) as f:
                model_scores = json.load(f)
        # Otherwise, train the models
        else:
            model_scores = train_func(args, data, logger)

        for metric, scores in model_scores.items():
            all_scores[metric].append(scores)
    all_scores = dict(all_scores)

    # Convert scores to numpy arrays
    for metric, scores in all_scores.items():
        all_scores[metric] = np.array(scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    contains_nan_scores = False
    
    # Report scores across folds
    for metric, scores in all_scores.items():
        avg_scores = multitask_mean(scores, axis=1, metric=metric)  # average score for each model across tasks
        mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)
        info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(args.task_names):
                info(f'\tOverall test {task_name} {metric} = '
                     f'{np.mean(scores[:, task_num]):.6f} +/- {np.std(scores[:, task_num]):.6f}')

    if contains_nan_scores:
        info("The metric scores observed for some fold test splits contain 'nan' values. \
            This can occur when the test set does not meet the requirements \
            for a particular metric, such as having no valid instances of one \
            task in the test set or not having positive examples for some classification metrics. \
            Before v1.5.1, the default behavior was to ignore nan values in individual folds or tasks \
            and still return an overall average for the remaining folds or tasks. The behavior now \
            is to include them in the average, converting overall average metrics to 'nan' as well.")

    # Save scores
    with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)

        header = ['Task']
        for metric in args.metrics:
            header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                      [f'Fold {i} {metric}' for i in range(args.num_folds)]
        writer.writerow(header)

        if args.dataset_type == 'spectra': # spectra data type has only one score to report
            row = ['spectra']
            for metric, scores in all_scores.items():
                task_scores = scores[:,0]
                mean, std = np.mean(task_scores), np.std(task_scores)
                row += [mean, std] + task_scores.tolist()
            writer.writerow(row)
        else: # all other data types, separate scores by task
            for task_num, task_name in enumerate(args.task_names):
                row = [task_name]
                for metric, scores in all_scores.items():
                    task_scores = scores[:, task_num]
                    mean, std = np.mean(task_scores), np.std(task_scores)
                    row += [mean, std] + task_scores.tolist()
                writer.writerow(row)

    # Determine mean and std score of main metric
    avg_scores = multitask_mean(all_scores[args.metric], metric=args.metric, axis=1)
    mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)

    # Optionally merge and save test preds
    if args.save_preds:
        all_preds = pd.concat([pd.read_csv(os.path.join(save_dir, f'fold_{fold_num}', 'test_preds.csv'))
                                  for fold_num in range(args.num_folds)])
        all_preds.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

    return mean_score, std_score
