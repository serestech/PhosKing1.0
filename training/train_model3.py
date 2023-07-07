#! /usr/bin/env python3
'''
Main script for the training loop of protein phosphorylation predictor.
From train_model2.py, recycle False data-points and sample them between epochs. (ONLY PHOSPHO MODE)
'''
import argparse
import sys, os, os.path
import copy
import torch
print(f'Using torch version {torch.__version__}')
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from importlib import import_module
from dataset3 import ESM_Embeddings
import time as t
from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np

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
                    dest='wd', default='0',
                    help='Weight decay (only compatible optimizers)')
parser.add_argument('-es', '--early_stopping', action='store_true',
                    dest='early_stopping',
                    help='Do early stopping')
parser.add_argument('-nzg', '--no_zero_grad', action='store_true',
                    dest='no_zero_grad',
                    help='Wether to reset the gradients or not on each batch')
parser.add_argument('-fp', '--frac_phos', action='store',
                    dest='frac_phos', default='0.5', type=float,
                    help='Fraction of phosphorilated amino acids in dataset')
parser.add_argument('-aaw', '--aa_window', action='store',
                    dest='aa_window', default='0', type=int,
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


def perf_scores(preds, targets):
    '''
    Calculates performance scores from torch tensors usin torch's vectorized operations
    '''
    batch_size, n_classes = preds.size()
    total_preds = batch_size * n_classes
    
    pred_pos = torch.sum(preds).item()
    pred_neg = total_preds - pred_pos
    
    target_pos = torch.sum(targets).item()
    target_neg = total_preds - target_pos
    
    mistakes = torch.sum(torch.logical_xor(preds, targets)).item()
    corrects = total_preds - mistakes
    
    true_pos = torch.sum(torch.logical_and(preds, targets)).item()
    false_pos = pred_pos - true_pos
    true_neg = total_preds - torch.sum(torch.logical_or(preds, targets)).item()
    false_neg = pred_neg - true_neg
    
    accuracy = corrects / total_preds
    try:
        sensitivity = true_pos / target_pos
    except ZeroDivisionError as err:
        print(f'WARNING: {err} in sensitivity; {true_pos=}', file=sys.stderr)
        sensitivity = 0
        
    try:
        specificity = true_neg / target_neg
    except ZeroDivisionError as err:
        print(f'WARNING: {err} in specificity; {target_neg=}', file=sys.stderr)
        specificity = 0

    try:
        precision = true_pos / pred_pos
    except ZeroDivisionError as err:
        print(f'WARNING: {err} in precision; {pred_pos=}', file=sys.stderr)
        precision = 0
    
    # try:
    #     mcc = (true_pos*true_neg - false_pos*false_neg)/(np.sqrt((true_pos+false_pos) * (true_pos+false_neg) * (true_neg+false_pos) * (true_neg+false_neg)))
    # except ZeroDivisionError as err:
    #     print(f'WARNING: {err} in MCC', file=sys.stderr)
    #     mcc = 0
        
    prec_, rec_, thr_ = precision_recall_curve(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())
    auprc = 0
    for i in range(1, len(rec_)):
        auprc += abs(rec_[i-1] - rec_[i]) * (prec_[i] + prec_[i-1]) / 2
    
    return accuracy, precision, sensitivity, specificity, auprc


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

    for IDs,header in ((dataset.train_IDs, '%TRAINING_SET'), (dataset.validation_IDs, '%VALIDATION_SET'), (dataset.test_IDs, '%TEST_SET')):
        info_contents.append(header)
        info_contents.extend(IDs)
    
    with open(f'{out_filename}.info', 'w') as info_file:
        info_file.write('\n'.join(info_contents))
    
    training_record_file = f'{HERE}/models/summary_phospho.tsv'
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
                         args.batch_size, args.n_epochs, args.wd, str(args.frac_phos), str(args.aa_window),
                         str(kwargs.get('val_acc')), str(kwargs.get('val_auc')),
                         str(kwargs.get('test_acc')), str(kwargs.get('test_auc')), ' '.join(sys.argv)])
        f.write(row + '\n')

    print(f'    Saved info file')


def parse_num(num: str):
    '''
    Convert a number from string to int or float (decides best type)
    '''
    return float(num) if '.' in num else int(num) 


args = parser.parse_args()

print(f'Using python env {sys.executable}')

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
                         frac_phos=args.frac_phos,
                         aa_window=args.aa_window,
                         flatten_window=args.flatten_window,
                         small_data=args.small_data,
                         add_dim=args.add_dim,
                         verbose=True)

# Train on 80% of the data. Split the other 20% between test and validation

print(f'Number of sequences:\n   Training {len(dataset.train_IDs)}  Validation: {len(dataset.validation_IDs)}  Test: {len(dataset.test_IDs)}')
print(f'Number of observations:\n   Training {len(dataset.train_idx)}  Validation: {len(dataset.validation_idx)}  Test: {len(dataset.test_idx)}')

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
validations_per_epoch = 20
early_stop = args.early_stopping

date = t.strftime("%m_%d_%H_%M_%S", t.localtime())
out_name = os.path.join(HERE, 'models', f'{args.model_name}_{date}_phos')

val_auc_best = 0
step = 0
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

batch_size = int(args.batch_size)
val_loader = data.DataLoader(dataset=dataset.validation_dataset,
                            batch_size=batch_size,
                            shuffle=True)
test_loader = data.DataLoader(dataset=dataset.test_dataset,
                            batch_size=batch_size,
                            shuffle=True)

validation_every_steps = 500
model.train()
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1} of {num_epochs}')
    train_subset = dataset.train_subset(epoch)
    train_loader = data.DataLoader(dataset=train_subset,
                                   batch_size=batch_size,
                                   shuffle=True)

    # validation_every_steps = int(int(len(train_subset)/batch_size)/validations_per_epoch)
    train_targets = torch.empty((0, 1)).to(device)
    train_outputs = torch.empty((0, 1)).to(device)
    for inputs, targets in iter(train_loader):
        inputs: torch.Tensor = inputs.to(device)
        targets: torch.Tensor = targets.to(device)
        if len(torch.nonzero(targets)) == 0:
            continue
        
        if not args.no_zero_grad:
            optimizer.zero_grad()
        
        outputs: torch.Tensor = model(inputs)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        optimizer.step()
        step += 1
        
        train_targets = torch.cat((train_targets, targets))
        train_outputs = torch.cat((train_outputs, outputs))

        # Validation time!
        if step % validation_every_steps == 0:
            train_predictions = torch.round(train_outputs)
            train_acc, train_prec, train_sens, train_spec, train_mcc = perf_scores(train_predictions, train_targets)
            try:
                train_auc = roc_auc_score(train_targets.cpu().detach().numpy(), train_outputs.cpu().detach().numpy())
            except ValueError as err:
                print(err)
                train_auc = 0

            train_loss = float(loss)
            train_accs.append(train_acc)
            train_precs.append(train_prec)
            train_senss.append(train_sens)
            train_specs.append(train_spec)
            train_aucs.append(train_auc)
            train_mccs.append(train_mcc)
            
            train_targets = torch.empty((0, 1)).to(device)
            train_outputs = torch.empty((0, 1)).to(device)

            val_targets = torch.empty((0, 1)).to(device)
            val_outputs = torch.empty((0, 1)).to(device)
            with torch.no_grad():
                model.eval()
                for inputs, targets in iter(val_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    val_loss = float(loss_fn(outputs, targets))

                    val_targets = torch.cat((val_targets, targets))
                    val_outputs = torch.cat((val_outputs, outputs))
                    
                model.train()

            val_predictions = torch.round(val_outputs)
            val_acc, val_prec, val_sens, val_spec, val_mcc = perf_scores(val_predictions, val_targets)
            try:
                val_auc = roc_auc_score(val_targets.cpu().detach().numpy(), val_outputs.cpu().detach().numpy())
            except ValueError:
                val_auc = 0

            val_accs.append(val_acc)
            val_precs.append(val_prec)
            val_senss.append(val_sens)
            val_specs.append(val_spec)
            val_aucs.append(val_auc)
            val_mccs.append(val_mcc)
            
            print(f'  Step {step}')
            print(f'    Train accuracy: {train_acc:8.5f}   Train loss:        {train_loss:8.5f}')
            print(f'    Valid accuracy: {val_acc:8.5f}   Valid loss:        {val_loss:8.5f}')
            print(f'    Train AUC:      {train_auc:8.5f}   Train sensitivity: {train_sens:8.5f}   Train specificity: {train_spec:8.5f}   Train precision: {train_prec:8.5f}')
            print(f'    Valid AUC:      {val_auc:8.5f}   Valid sensitivity: {val_sens:8.5f}   Valid specificity: {val_spec:8.5f}   Valid precision: {val_prec:8.5f}')

            # Store best model if early stopping, save it if auto-save
            if early_stop and val_auc > val_auc_best:
                train_loss_best = train_loss
                train_acc_best = train_acc
                train_prec_best = train_prec
                train_sens_best = train_sens
                train_spec_best = train_spec
                train_auc_best = train_auc
                train_mcc_best = train_mcc

                val_loss_best = val_loss
                val_acc_best = val_acc
                val_prec_best = val_prec
                val_sens_best = val_sens
                val_spec_best = val_spec
                val_auc_best = val_auc
                val_mcc_best = val_mcc
                
                epoch_best = epoch
                step_best = step
                state_dict_best = copy.deepcopy(model.state_dict())

                if args.auto_save and step > 250:
                    test_targets = torch.empty((0, 1)).to(device)
                    test_outputs = torch.empty((0, 1)).to(device)
                    with torch.no_grad():
                        model.eval()
                        for inputs, targets in iter(test_loader):
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            
                            outputs = model(inputs)
                            test_loss = float(loss_fn(outputs, targets))

                            test_targets = torch.cat((test_targets, targets))
                            test_outputs = torch.cat((test_outputs, outputs))
                            
                        model.train()

                    test_predictions = torch.round(test_outputs)
                    test_acc_best, test_prec_best, test_sens_best, test_spec_best, test_mcc_best = perf_scores(test_predictions, test_targets)
                    try:
                        test_auc_best = roc_auc_score(test_targets.cpu().detach().numpy(), test_outputs.cpu().detach().numpy())
                    except ValueError:
                        test_auc_best = 0

                    save_model(state_dict_best, model_type='best',
                            train_acc=train_acc_best, train_auc=train_auc_best, train_loss=train_loss_best,
                            val_acc=val_acc_best, val_auc=val_auc_best, val_loss=val_loss_best,
                            test_acc=test_acc_best, test_spec=test_spec_best, test_prec=test_prec_best,
                            test_sens=test_sens_best, test_auc=test_auc_best, test_mcc=test_mcc_best,
                            epoch=epoch_best, step=step_best)


# Final evaluation, test and save final model
model.eval()
val_targets = torch.empty((0, 1)).to(device)
val_outputs = torch.empty((0, 1)).to(device)
with torch.no_grad():
    model.eval()
    for inputs, targets in iter(val_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        val_loss = float(loss_fn(outputs, targets))

        val_targets = torch.cat((val_targets, targets))
        val_outputs = torch.cat((val_outputs, outputs))
        
    model.train()

val_predictions = torch.round(val_outputs)
val_acc, val_prec, val_sens, val_spec, val_mcc = perf_scores(val_predictions, val_targets)
try:
    val_auc = roc_auc_score(val_targets.cpu().detach().numpy(), val_outputs.cpu().detach().numpy())
except ValueError:
    val_auc = 0

val_accs.append(val_acc)
val_precs.append(val_prec)
val_senss.append(val_sens)
val_specs.append(val_spec)
val_aucs.append(val_auc)
val_mccs.append(val_mcc)

test_targets = torch.empty((0, 1)).to(device)
test_outputs = torch.empty((0, 1)).to(device)
with torch.no_grad():
    model.eval()
    for inputs, targets in iter(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        test_loss = float(loss_fn(outputs, targets))

        test_targets = torch.cat((test_targets, targets))
        test_outputs = torch.cat((test_outputs, outputs))
        
    model.train()

test_predictions = torch.round(test_outputs)
test_acc, test_prec, test_sens, test_spec, test_mcc = perf_scores(test_predictions, test_targets)
try:
    test_auc = roc_auc_score(test_targets.cpu().detach().numpy(), test_outputs.cpu().detach().numpy())
except ValueError:
    test_auc = 0

save_model(model.state_dict(), model_type='final',
           train_acc=train_acc, train_auc=train_auc, train_loss=train_loss,
           val_acc=val_acc, val_auc=val_auc, val_loss=val_loss,
           test_acc=test_acc, test_spec=test_spec, test_prec=test_prec,
           test_sens=test_sens, test_auc=test_auc, test_mcc=test_mcc,
           epoch=epoch, step=step)

print('Performance of the last run:')
print(f'  Accuracies:     Train {train_acc:8.5f}   Valid {val_acc:8.5f}   Test {test_acc:8.5f}')
print(f'  Precisions:     Train {train_prec:8.5f}   Valid {val_prec:8.5f}   Test {test_prec:8.5f}')
print(f'  Sensitivities:  Train {train_sens:8.5f}   Valid {val_sens:8.5f}   Test {test_sens:8.5f}')
print(f'  Specificities:  Train {train_spec:8.5f}   Valid {val_spec:8.5f}   Test {test_spec:8.5f}')
print(f'  AUCs:           Train {train_auc:8.5f}   Valid {val_auc:8.5f}   Test {test_auc:8.5f}')
print(f'  MCCs:           Train {train_mcc:8.5f}   Valid {val_mcc:8.5f}   Test {test_mcc:8.5f}')

# Test and save best model if early stopping (and if not saved during training)
if early_stop and not args.auto_save:
    model.load_state_dict(state_dict_best)
    test_targets = torch.empty((0, 1)).to(device)
    test_outputs = torch.empty((0, 1)).to(device)
    with torch.no_grad():
        model.eval()
        for inputs, targets in iter(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            test_loss = float(loss_fn(outputs, targets))

            test_targets = torch.cat((test_targets, targets))
            test_outputs = torch.cat((test_outputs, outputs))
            
        model.train()

    test_predictions = torch.round(test_outputs)
    test_acc_best, test_prec_best, test_sens_best, test_spec_best, test_mcc_best = perf_scores(test_predictions, test_targets)
    try:
        test_auc_best = roc_auc_score(test_targets.cpu().detach().numpy(), test_outputs.cpu().detach().numpy())
    except ValueError:
        test_auc_best = 0

    save_model(state_dict_best, model_type='best',
               train_acc=train_acc_best, train_auc=train_auc_best, train_loss=train_loss_best,
               val_acc=val_acc_best, val_auc=val_auc_best, val_loss=val_loss_best,
               test_acc=test_acc_best, test_spec=test_spec_best, test_prec=test_prec_best,
               test_sens=test_sens_best, test_auc=test_auc_best, test_mcc=test_mcc_best,
               epoch=epoch_best, step=step_best)

    print('Performance of the best run:')
    print(f'  Accuracies:     Train {train_acc_best:8.5f}   Valid {val_acc_best:8.5f}   Test {test_acc_best:8.5f}')
    print(f'  Precisions:     Train {train_prec_best:8.5f}   Valid {val_prec_best:8.5f}   Test {test_prec_best:8.5f}')
    print(f'  Sensitivities:  Train {train_sens_best:8.5f}   Valid {val_sens_best:8.5f}   Test {test_sens_best:8.5f}')
    print(f'  Specificities:  Train {train_spec_best:8.5f}   Valid {val_spec_best:8.5f}   Test {test_spec_best:8.5f}')
    print(f'  AUCs:           Train {train_auc_best:8.5f}   Valid {val_auc_best:8.5f}   Test {test_auc_best:8.5f}')
    print(f'  MCCs:           Train {train_mcc_best:8.5f}   Valid {val_mcc_best:8.5f}   Test {test_mcc_best:8.5f}')