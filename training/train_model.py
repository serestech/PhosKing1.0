#! /usr/bin/env python3
'''
Main script for the training loop of protein phosphorylation (& kinase) predictor.
Right now this is used for kinase only.
'''
import argparse
import sys, os, os.path
import copy
from pprint import pformat

parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description='Train a phosphorilation prediction model with PyTorch using ESM Embeddings')

FILE_PATH = os.path.abspath(__file__)
HERE = os.path.dirname(FILE_PATH)

parser.add_argument('-m', '--model_file', action='store',
                    dest='model_file', required=True,
                    help='Model file (python file with PyTorch model)')
parser.add_argument('-n', '--model_name', action='store',
                    dest='model_name', required=True,
                    help='Model name (class name in the model file)')
parser.add_argument('-md', '--mode', action='store',
                    dest='mode', default='phospho', choices=['phospho', 'kinase'],
                    help='Prediction mode ("phospho" or "kinase")')
parser.add_argument('-f', '--fasta', action='store', dest='fasta_path',
                    default=os.path.abspath(f'{HERE}/../data/database_dumps/temp_seqs.fasta'),
                    help='Fasta file')
parser.add_argument('-ft', '--features', action='store', dest='features_path',
                    default=os.path.abspath(f'{HERE}/../data/database_dumps/temp_feats.tsv'),
                    help='Features table (matching fasta file)')
parser.add_argument('-emb', '--embeddings_dir', action='store', dest='embeddings_dir',
                    default=os.path.abspath(f'{HERE}/../data/embeddings/embeddings_1280'),
                    help='Folder with embeddings pickles')
parser.add_argument('-a', '--model_args', action='store',
                    dest='model_args', default='',
                    help='Comma separated ints to pass to the model constructor (e.g. "1280,2560,1")')
parser.add_argument('-c', '--force_cpu', action='store_true',
                    dest='force_cpu',
                    help='Force CPU training')
parser.add_argument('-l', '--loss_fn', action='store',
                    dest='loss_fn', default='BCE',
                    help='Loss function')
parser.add_argument('-o', '--optimizer', action='store',
                    dest='optimizer', default='Adam',
                    help='Optimizer to use')
parser.add_argument('-lr', '--learning_rate', action='store',
                    dest='lr', default='1e-5',
                    help='Learning rate')
parser.add_argument('-b', '--batch_size', action='store',
                    dest='batch_size', default='128',
                    help='Batch size')
parser.add_argument('-e', '--num_epochs', action='store',
                    dest='n_epochs', default='20',
                    help='Number of epochs')
parser.add_argument('-w', '--weight_decay', action='store',
                    dest='wd', default='1e-5',
                    help='Weight decay (only compatible optimizers)')
parser.add_argument('-es', '--early_stopping', action='store_true',
                    dest='early_stopping',
                    help='Do early stopping')
parser.add_argument('-nzg', '--no_zero_grad', action='store_true',
                    dest='no_zero_grad',
                    help='Wether to reset the gradients or not on each batch')
parser.add_argument('-fp', '--frac_phos', action='store',
                    dest='frac_phos', default='0.5',
                    help='Fraction of phosphorilated amino acids in dataset')
parser.add_argument('-auc', '--auc_type', action='store',
                    dest='auc_type', default='mean', choices=['mean', 'median'],
                    help='Use "mean" or "median" in multiclass AUC (only meaningful for kinase model)')
parser.add_argument('-aaw', '--aa_window', action='store',
                    dest='aa_window', default='0',
                    help='Amino acid window for the tensors (concatenated tensor of the 5 amino acids)')
parser.add_argument('-fw', '--flatten_window', action='store_true',
                    dest='flatten_window',
                    help='Wether to flatten or not the amino acid window (only if -aaw > 0)')
parser.add_argument('-ad', '--add_dim', action='store_true',
                    dest='add_dim',
                    help='Wether to add an extra dimension to the tensor (needed for CNN)')
parser.add_argument('-sd', '--small_data', action='store_true',
                    dest='small_data',
                    help='Dataset only loads 1 embedding pickle.')
parser.add_argument('-cv', '--cross_validation', action='store_true',
                    dest='cross_validation',
                    help='Cross-validation mode (all data goes into training)')
parser.add_argument('-as', '--auto_save', action='store_true',
                    dest='auto_save',
                    help='Save best model when it is discovered instead of just once at the end')
parser.add_argument('-s', '--splits', action='store',
                    dest='splits',
                    help='Graphpart output file to do the splits')
parser.add_argument('-ss', '--split_sequences', action='store_true',
                    dest='split_sequences',
                    help='Perform the train-test splits based on sequences (as opposed to phosphosites)')


args = parser.parse_args()

print(f'Using python env {sys.executable}')

import torch
print(f'Using torch version {torch.__version__}')
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from importlib import import_module
from dataset_new import ESM_Embeddings
from math import sqrt
from sklearn.model_selection import train_test_split
import time as t
from sklearn.metrics import roc_auc_score
import numpy as np
from numpy import ndarray
from statistics import mean, median, StatisticsError

if args.mode == 'kinase':
    from kinase_mapping import kinase_mapping as mapping
    from kinase_mapping import kinase_mapping_reverse as reverse_mapping

def perf_scores(predictions, targets):
    '''
    Calculates performance scores from torch tensors usin torch's vectorized operations
    '''
    batch_size, n_classes = predictions.size()
    total_predictions = batch_size * n_classes
    
    prediction_positives = torch.sum(predictions).item()
    prediction_negatives = total_predictions - prediction_positives
    
    target_positives = torch.sum(targets).item()
    target_negatives = total_predictions - target_positives
    
    mistakes = torch.sum(torch.logical_xor(predictions, targets)).item()
    corrects = total_predictions - mistakes
    
    true_positives = torch.sum(torch.logical_and(predictions, targets)).item()
    false_positives = prediction_positives - true_positives
    true_negatives = total_predictions - torch.sum(torch.logical_or(predictions, targets)).item()
    false_negatives = prediction_negatives - true_negatives
    
    accuracy = corrects / total_predictions
    try:
        sensitivity = true_positives / target_positives
    except ZeroDivisionError as err:
        print(f'WARNING: {err}. Found 0 target positives. {target_positives=} {true_positives=} {targets=}', file=sys.stderr)
        sensitivity = 0
        
    try:
        specificity = true_negatives / target_negatives
    except ZeroDivisionError as err:
        print(f'WARNING: {err}. Found 0 target negatives. {target_negatives=} {true_negatives=} {targets=}', file=sys.stderr)
        specificity = 0
    
    try:
        mcc = (true_positives * true_negatives - false_positives * false_negatives)/(sqrt((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives)))
    except ZeroDivisionError:
        mcc = 0
        
    try:
        precision = true_positives / prediction_positives
    except ZeroDivisionError as err:
        # print(f'WARNING: {err} (usually because kinase model gave all false predictions)', file=sys.stderr)
        precision = 0
    
    return accuracy, precision, sensitivity, specificity, mcc

def save_model(state_dict, model_type='final', **kwargs):
    '''
    Saves model state dict alongside an info file detailing information on the model.
    '''
    # TODO: Add AUC data (different for phos and kinase)

    out_filename = f'{out_name}_{model_type}'
    
    print(f"Saving torch state dict '{out_filename}.pth'")
    torch.save(state_dict, f'{out_filename}.pth')
    
    print(f"    Generating info file '{out_filename}.info'")
    info_contents = []
    
    info_contents.append('%RUN')
    if 'LSB_JOBID' in os.environ.keys() and os.environ['LSB_JOBID']:
        info_contents.append(f'jobid: {os.environ["LSB_JOBID"]}')
    
    info_contents.append('%MODEL')
    
    with open(args.model_file, 'r') as model_file:
        info_contents.append(model_file.read())
        
    info_contents.append('%NAMESPACE')
    for var,val in args.__dict__.items():
        info_contents.append(f'{var}: {val}')
        
    info_contents.append('%PERFORMANCE')
    for var,val in kwargs.items():
        info_contents.append(f'{var}: {val:8.5f}')
    
    def get_tensor_kinases(label_tensor: torch.Tensor) -> list[str]:
        return [reverse_mapping[i] for i, val in enumerate(label_tensor) if val.item() == 1]

    for sub_ds,header in ((train_dataset, '%TRAINING_SET'), (validation_dataset, '%VALIDATION_SET'), (test_dataset, '%TEST_SET')):
        info_contents.append(header)
        for i in sub_ds.indices:
            if args.mode == 'kinase':
                seq_id, pos, label_tensor = dataset.data[i]
                kinases = get_tensor_kinases(label_tensor)
                aminoacid = (seq_id, pos, ','.join(kinases))
            else:
                aminoacid = dataset.data[i]
            info_contents.append(' '.join([str(elem) for elem in aminoacid]))
    
    with open(f'{out_filename}.info', 'w') as info_file:
        info_file.write('\n'.join(info_contents))
    
    training_record_file = f'{HERE}/models/summary_{args.mode}.tsv'
    if not os.path.exists(training_record_file):
        columns = ['Name', 'Model', 'Loss function', 'Optimizer', 'Learning rate',
                   'Batch size', 'Epochs', 'Weight decay', 'Fraction phos', 'aa window',
                   'Validation accuracy', 'Validation AUC',
                   'Test accuracy', 'Test AUC', 'Call']
        with open(training_record_file, 'w') as f:
            f.write('\t'.join(columns) + '\n')

    # Write the updated data to the .info file
    with open(training_record_file, 'a') as f:        
        # Join the values into a tab-separated string
        row = '\t'.join([out_filename, args.model_name, args.loss_fn, args.optimizer, args.lr,
                         args.batch_size, args.n_epochs, args.wd, args.frac_phos, args.aa_window,
                         str(kwargs.get('val_acc')), str(kwargs.get('val_auc')),
                         str(kwargs.get('test_acc')), str(kwargs.get('test_auc')), ' '.join(sys.argv)])
        f.write(row + '\n')

    print(f'    Saved info file')


def kinase_auc(targets: ndarray, predictions: ndarray) -> tuple[float, float, dict]:
    '''
    Computes AUC for each kinase class (treats each kinase as a binary classifications). Returns the
    mean of all kinases, the median of all kinases, and a dick containing the AUC for each kinase.
    '''
    results = {}
    for i, kinase in reverse_mapping.items():
        y_true = targets[:,i]
        y_pred = predictions[:,i]
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError as err:
            # This happens when the batch has a small size (last few elements in the epoch) and a kinase has
            # no phosphorylations in that batch
            # print(f'WARNING: Failed to compute AUC with error "{err}" {y_true.shape=} {y_pred.shape=}')
            auc = None
        results[kinase] = auc
    
    results_nums = [result for result in results.values() if result is not None]
    
    try:
        mean_auc = mean(results_nums)
        median_auc = median(results_nums)
    except StatisticsError as err:
        # This happens in very rare occasions, when the batch has size 1 (last phosphorylation of 
        # its epoch) and no AUC can be computed above
        print(f'WARNING: Failed to compute AUC mean and median with error "{err}" {targets.shape=} {predictions.shape=}')
        mean_auc, median_auc = 0.75, 0.75

    return mean_auc, median_auc, results

def parse_num(num: str):
    '''
    Convert a number from string to int or float (decides best type)
    '''
    return float(num) if '.' in num else int(num) 

def parse_splits(splits_path: str):
    with open(splits_path, 'r') as file:
        lines = [line.strip() for line in file]
    
    seq_splits = {}
    for line in lines[1:]:
        seq_id, split = line.split('\t')
        seq_id = seq_id.strip()
        split = split.strip()
        seq_splits[seq_id] = split
    
    return seq_splits

# Hacky thing to import the model while knowing file and class name at runtime
model_dir = os.path.dirname(args.model_file)
sys.path.append(model_dir)
model_module_name = os.path.basename(args.model_file)[:-3]
model_module = import_module(model_module_name)
model_class = getattr(model_module, args.model_name)

device = torch.device('cuda' if  not args.force_cpu and torch.cuda.is_available() else 'cpu')
print(f'Using torch device of type {device.type}{": " + torch.cuda.get_device_name(device) if device.type == "cuda" else ""}')

if args.model_args is None:
    model: torch.nn.Module = model_class()
else:
    model: torch.nn.Module = model_class(*[parse_num(arg) for arg in args.model_args.split(',')])
model = model.to(device)

dataset = ESM_Embeddings(fasta=args.fasta_path,
                         features=args.features_path,
                         embeddings_dir=args.embeddings_dir,
                         phos_fract=float(args.frac_phos),
                         aa_window=int(args.aa_window),
                         flatten_window=args.flatten_window,
                         small_data=args.small_data,
                         add_dim=args.add_dim,
                         mode=args.mode,
                         verbose=True)

if args.splits is None and not args.split_sequences:
    # Train on 80% of the data. Split the other 20% between test and validation
    print(f'Assigning train-test splits based on phosphosites')
    indices = list(range(len(dataset)))
    train_idx, rest_idx = train_test_split(indices, train_size=0.8, test_size=0.2, shuffle=True)
    test_idx, validation_idx = train_test_split(rest_idx, train_size=0.5, test_size=0.5, shuffle=True)
    assert set(train_idx).isdisjoint(set(rest_idx)) and set(test_idx).isdisjoint(set(validation_idx))  # Sanity check
elif args.splits is None and args.split_sequences:
    print(f'Assigning train-test splits based on sequences')
    # Get all sequences
    sequences = set()
    for seq_id, _, _ in dataset.data:
        sequences.add(seq_id)

    # Split sequences with sklearn
    sequences = list(sequences)
    train_seqs, rest_seqs = train_test_split(sequences, train_size=0.8, test_size=0.2, shuffle=True)
    test_seqs, validation_seqs = train_test_split(rest_seqs, train_size=0.5, test_size=0.5, shuffle=True)

    # Make idx subsets
    train_idx, test_idx, validation_idx = [], [], []
    train_seqs, test_seqs, validation_seqs = set(train_seqs), set(test_seqs), set(validation_seqs)
    n_missing = 0
    for i, (seq_id, _, _) in enumerate(dataset.data):
        if seq_id in train_seqs:
            train_idx.append(i)
        elif seq_id in test_seqs:
            test_idx.append(i)
        elif seq_id in validation_seqs:
            validation_idx.append(i)
        else:
            n_missing += 1

    assert set(train_idx).isdisjoint(set(test_idx)) and \
        set(train_idx).isdisjoint(set(validation_idx)) and \
        set(test_idx).isdisjoint(set(validation_idx))  # Sanity check
    print(f'{n_missing} observations missing from cluster splits (of {len(dataset.data)} total observations)') 
elif args.splits:
    train_idx, test_idx, validation_idx = [], [], []
    splits = parse_splits(args.splits)
    n_missing = 0
    print(f'Assigning train-test splits based on {args.splits}')
    for i, (seq_id, _, _) in enumerate(dataset.data):
        try:
            split = splits[seq_id]
        except KeyError:
            n_missing += 1
            continue
        
        if split == 'train':
            train_idx.append(i)
        elif split == 'test':
            test_idx.append(i)
        elif split == 'valid':
            validation_idx.append(i)
        else:
            raise RuntimeError(f'Bad split: {split}') 
    assert set(train_idx).isdisjoint(set(test_idx)) and \
           set(train_idx).isdisjoint(set(validation_idx)) and \
           set(test_idx).isdisjoint(set(validation_idx))  # Sanity check
    print(f'{n_missing} observations missing from cluster splits (of {len(dataset.data)} total observations)')
else:
    raise ValueError('No method for data splitting specificied')

print(f'Number of observations:\n   Training {len(train_idx)}  Validation: {len(validation_idx)}  Test: {len(test_idx)}')

# TODO: Show how many train, test, valid observations per-kinase

# Create subsets of the data
train_dataset = data.Subset(dataset, train_idx)
test_dataset = data.Subset(dataset, test_idx)
validation_dataset = data.Subset(dataset, validation_idx)

batch_size = int(args.batch_size)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)
val_loader = data.DataLoader(dataset=validation_dataset,
                             batch_size=batch_size,
                             shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)

if args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))
else:
    raise NotImplementedError(f'Optimizer {args.optimizer} not implemented')

if args.loss_fn == 'BCE':
    loss_fn = nn.BCELoss()
elif args.loss_fn == 'CEL':
    loss_fn = nn.CrossEntropyLoss()
else:
    raise NotImplementedError(f'Didn\'t recognize loss function {args.loss_fn}')

num_epochs = int(args.n_epochs)
validation_every_steps = 100

is_training_kinase = args.mode == 'kinase'
early_stop = args.early_stopping

date = t.strftime("%m_%d_%H_%M_%S", t.localtime())
model_mode = 'phos' if args.mode == 'phospho' else 'kin'
out_name = os.path.join(HERE, 'models', f'{args.model_name}_{date}_{model_mode}')

val_auc_best = 0
val_sens_best = 0
val_prec_best = 0
step = 0
model.train()
train_accs = []
val_accs = []
train_precs = []
val_precs = []
train_senss = []
val_senss = []
train_specs = []
val_specs = []
train_aucs = []
val_aucs = []
train_mccs = []
val_mccs = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1} of {num_epochs}')
    
    train_accs_batch  = []
    train_precs_batch  = []
    train_senss_batch = []
    train_specs_batch = []
    train_aucs_batch = []
    train_mccs_batch = []
    for inputs, targets in iter(train_loader):
        inputs: torch.Tensor = inputs.to(device)
        targets: torch.Tensor = targets.to(device)
        
        if not args.no_zero_grad:
            optimizer.zero_grad()
        
        outputs: torch.Tensor = model(inputs)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        optimizer.step()
        step += 1
        
        predictions = torch.round(outputs)
  
        train_acc, train_prec, train_sens, train_spec, train_mcc = perf_scores(predictions, targets)
        if not is_training_kinase:
            try:
                train_auc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            except ValueError:
                train_auc = None

        else:
            avg_auc, median_auc, _ = kinase_auc(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            train_auc = avg_auc if args.auc_type == 'mean' else median_auc

        train_accs_batch.append(train_acc)
        train_precs_batch.append(train_prec)
        train_senss_batch.append(train_sens)
        train_specs_batch.append(train_spec)
        if train_auc is not None:
            train_aucs_batch.append(train_auc)
        train_mccs_batch.append(train_mcc)

        # Validation time!
        if step % validation_every_steps == 0:
            train_loss = float(loss)
            
            train_accs.append(sum(train_accs_batch) / len(train_accs_batch))
            train_precs.append(sum(train_precs_batch) / len(train_precs_batch))
            train_senss.append(sum(train_senss_batch) / len(train_senss_batch))
            train_specs.append(sum(train_specs_batch) / len(train_specs_batch))
            train_aucs.append(sum(train_aucs_batch) / len(train_aucs_batch))
            train_mccs.append(sum(train_mccs_batch) / len(train_mccs_batch))
            train_accs_batch = []
            train_precs_batch = []
            train_senss_batch = []
            train_specs_batch = []
            train_aucs_batch = []
            train_mccs_batch = []

            val_accs_batch = []
            val_precs_batch = []
            val_senss_batch = []
            val_specs_batch = []
            val_aucs_batch = []
            val_mccs_batch = []
            with torch.no_grad():
                model.eval()
                for inputs, targets in iter(val_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    val_loss = float(loss_fn(outputs, targets))
                    predictions = torch.round(outputs)
                    
                    val_acc, val_prec, val_sens, val_spec, val_mcc = perf_scores(predictions, targets)
                    if not is_training_kinase:
                        try:
                            val_auc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                        except ValueError:
                            val_auc = None
                    else:
                        avg_auc, median_auc, _ = kinase_auc(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                        val_auc = avg_auc if args.auc_type == 'mean' else median_auc
                        
                    val_accs_batch.append(val_acc)
                    val_precs_batch.append(val_prec)
                    val_senss_batch.append(val_sens)
                    val_specs_batch.append(val_spec)
                    if val_auc is not None:
                        val_aucs_batch.append(val_auc)
                    val_mccs_batch.append(val_mcc)
                    
                model.train()
            
            val_accs.append(sum(val_accs_batch) / len(val_accs_batch))
            val_precs.append(sum(val_precs_batch) / len(val_precs_batch))
            val_senss.append(sum(val_senss_batch) / len(val_senss_batch))
            val_specs.append(sum(val_specs_batch) / len(val_specs_batch))
            val_aucs.append(sum(val_aucs_batch) / len(val_aucs_batch))
            val_mccs.append(sum(val_mccs_batch) / len(val_mccs_batch))
            
            print(f'  Step {step}')
            print(f'    Training AUC:   {train_aucs[-1]:8.3f}  Training loss:   {train_loss:8.3f}')
            print(f'    Validation AUC: {val_aucs[-1]:8.3f}  Validation loss: {val_loss:8.3f}')

            # Store best model if early stopping, save it if auto-save
            if early_stop and val_aucs[-1] > val_auc_best:
                train_loss_best = train_loss
                train_acc_best = train_accs[-1]
                train_prec_best = train_precs[-1]
                train_sens_best = train_senss[-1]
                train_spec_best = train_specs[-1]
                train_auc_best = train_aucs[-1]
                train_mcc_best = train_mccs[-1]

                val_loss_best = val_loss
                val_acc_best = val_accs[-1]
                val_prec_best = val_precs[-1]
                val_sens_best = val_senss[-1]
                val_spec_best = val_specs[-1]
                val_auc_best = val_aucs[-1]
                val_mcc_best = val_mccs[-1]
                
                epoch_best = epoch
                step_best = step
                state_dict_best = copy.deepcopy(model.state_dict())

                if args.auto_save and step > 250:
                    test_accs_batch = []
                    test_precs_batch = []
                    test_senss_batch = []
                    test_specs_batch = []
                    test_aucs_batch = []
                    test_mccs_batch = []
                    with torch.no_grad():
                        model.eval()
                        for inputs, targets in iter(test_loader):
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            outputs = model(inputs)
                            predictions = torch.round(outputs)
                            
                            test_acc, test_prec, test_sens, test_spec, test_mcc = perf_scores(predictions, targets)
                            if not is_training_kinase:
                                try:
                                    test_auc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                                except ValueError:
                                    test_auc = None
                            else:
                                avg_auc, median_auc, _ = kinase_auc(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                                test_auc = avg_auc if args.auc_type == 'mean' else median_auc
                            test_accs_batch.append(test_acc)
                            test_precs_batch.append(test_prec)
                            test_senss_batch.append(test_sens)
                            test_specs_batch.append(test_spec)
                            if test_auc is not None:
                                test_aucs_batch.append(test_auc)
                            test_mccs_batch.append(test_mcc)
                                
                        model.train()

                    test_acc_best = sum(test_accs_batch) / len(test_accs_batch)
                    test_prec_best = sum(test_precs_batch) / len(test_precs_batch)
                    test_sens_best = sum(test_senss_batch) / len(test_senss_batch)
                    test_spec_best = sum(test_specs_batch) / len(test_specs_batch)
                    test_auc_best = sum(test_aucs_batch) / len(test_aucs_batch)
                    test_mcc_best = sum(test_mccs_batch) / len(test_mccs_batch)

                    save_model(state_dict_best, model_type='best',
                            train_acc=train_acc_best, train_auc=train_auc_best, train_loss=train_loss_best,
                            val_acc=val_acc_best, val_auc=val_auc_best, val_loss=val_loss_best,
                            test_acc=test_acc_best, test_spec=test_spec_best, test_prec=test_prec_best,
                            test_sens=test_sens_best, test_auc=test_auc_best, test_mcc=test_mcc_best,
                            epoch=epoch_best, step=step_best)


# Final evaluation, test and save final model
model.eval()
val_accs_batch = []
val_precs_batch = []
val_senss_batch = []
val_specs_batch = []
val_aucs_batch = []
val_mccs_batch = []
with torch.no_grad():
    for inputs, targets in iter(val_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        predictions = torch.round(outputs)

        val_acc, val_prec, val_sens, val_spec, val_mcc = perf_scores(predictions, targets)
        if not is_training_kinase:
            try:
                val_auc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            except ValueError:
                val_auc = None
        else:
            avg_auc, median_auc, _ = kinase_auc(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            val_auc = avg_auc if args.auc_type == 'mean' else median_auc
        val_accs_batch.append(val_acc)
        val_precs_batch.append(val_prec)
        val_senss_batch.append(val_sens)
        val_specs_batch.append(val_spec)
        if val_auc is not None:
            val_aucs_batch.append(val_auc)
        val_mccs_batch.append(val_mcc)
                    
val_accs.append(sum(val_accs_batch) / len(val_accs_batch))
val_precs.append(sum(val_precs_batch) / len(val_precs_batch))
val_senss.append(sum(val_senss_batch) / len(val_senss_batch))
val_specs.append(sum(val_specs_batch) / len(val_specs_batch))
val_aucs.append(sum(val_aucs_batch) / len(val_aucs_batch))
val_mccs.append(sum(val_mccs_batch) / len(val_mccs_batch))

test_accs_batch = []
test_precs_batch = []
test_senss_batch = []
test_specs_batch = []
test_mccs_batch = []
test_aucs_batch = []
with torch.no_grad():
    for inputs, targets in iter(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        predictions = torch.round(outputs)

        test_acc, test_prec, test_sens, test_spec, test_mcc = perf_scores(predictions, targets)
        if not is_training_kinase:
            try:
                test_auc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            except ValueError:
                test_auc = None
        else:
            avg_auc, median_auc, _ = kinase_auc(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            test_auc = avg_auc if args.auc_type == 'mean' else median_auc
        
        test_accs_batch.append(test_acc)
        test_precs_batch.append(test_prec)
        test_senss_batch.append(test_sens)
        test_specs_batch.append(test_spec)
        if test_auc is not None:
            test_aucs_batch.append(test_auc)
        test_mccs_batch.append(test_mcc)


test_acc = sum(test_accs_batch) / len(test_accs_batch)
test_prec = sum(test_precs_batch) / len(test_precs_batch)
test_sens = sum(test_senss_batch) / len(test_senss_batch)
test_spec = sum(test_specs_batch) / len(test_specs_batch)
test_auc = sum(test_aucs_batch) / len(test_aucs_batch)
test_mcc = sum(test_mccs_batch) / len(test_mccs_batch)

save_model(model.state_dict(), model_type='final',
           train_acc=train_accs[-1], train_auc=train_aucs[-1], train_loss=train_loss,
           val_acc=val_accs[-1], val_auc=val_aucs[-1], val_loss=val_loss,
           test_acc=test_acc, test_spec=test_spec, test_prec=test_prec,
           test_sens=test_sens, test_auc=test_auc, test_mcc=test_mcc,
           epoch=epoch, step=step)


print('AUCs of the last step:')
print(f'  Train {train_aucs[-1]:8.5f}   Valid {val_aucs[-1]:8.5f}   Test {test_auc:8.5f}')


# Test and save best model if early stopping (and if not saved during training)
if early_stop and not args.auto_save:
    model.load_state_dict(state_dict_best)
    test_accs_batch = []
    test_precs_batch = []
    test_senss_batch = []
    test_specs_batch = []
    test_aucs_batch = []
    test_mccs_batch = []
    test_kinase_predictions = []
    with torch.no_grad():
        model.eval()
        for inputs, targets in iter(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            test_kinase_predictions.append((targets.cpu().detach().numpy(), outputs.cpu().detach().numpy()))
            predictions = torch.round(outputs)
            
            test_acc, test_prec, test_sens, test_spec, test_mcc = perf_scores(predictions, targets)
            if not is_training_kinase:
                try:
                    test_auc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                except ValueError:
                    test_auc = None
            else:
                avg_auc, median_auc, kinase_results = kinase_auc(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                test_auc = avg_auc if args.auc_type == 'mean' else median_auc
            test_accs_batch.append(test_acc)
            test_precs_batch.append(test_prec)
            test_senss_batch.append(test_sens)
            test_specs_batch.append(test_spec)
            if test_auc is not None:
                test_aucs_batch.append(test_auc)
            test_mccs_batch.append(test_mcc)

    test_acc_best = sum(test_accs_batch) / len(test_accs_batch)
    test_prec_best = sum(test_precs_batch) / len(test_precs_batch)
    test_sens_best = sum(test_senss_batch) / len(test_senss_batch)
    test_spec_best = sum(test_specs_batch) / len(test_specs_batch)
    test_auc_best = sum(test_aucs_batch) / len(test_aucs_batch)
    test_mcc_best = sum(test_mccs_batch) / len(test_mccs_batch)

    save_model(state_dict_best, model_type='best',
               train_acc=train_acc_best, train_auc=train_auc_best, train_loss=train_loss_best,
               val_acc=val_acc_best, val_auc=val_auc_best, val_loss=val_loss_best,
               test_acc=test_acc_best, test_spec=test_spec_best, test_prec=test_prec_best,
               test_sens=test_sens_best, test_auc=test_auc_best, test_mcc=test_mcc_best,
               epoch=epoch_best, step=step_best)

    if is_training_kinase:
        print('Saving kinase test AUC results')
        _, _, results = kinase_auc(np.vstack([preds[0] for preds in test_kinase_predictions]), 
                                   np.vstack([preds[1] for preds in test_kinase_predictions]))
        with open(f'{out_name}.kinase_results', 'w') as kinase_results_file:
            kinase_results_file.write(pformat(results))

    
    print('AUCs of the best run:')
    print(f'  Train {train_auc_best:8.5f}   Valid {val_auc_best:8.5f}   Test {test_auc_best:8.5f}')

