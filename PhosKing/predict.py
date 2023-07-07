import sys
import os
import os.path as path
import time as t
import argparse
from importlib import import_module
import numpy as np
import torch
from torch import nn
from torch.nn.functional import pad
import esm
sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', 'utils')))
from utils import read_fasta, phosphorylable_aas
import random

parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description='Predict phosphorylations using ESM Embeddings and a trained model')

FILE_PATH = os.path.abspath(__file__)
HERE = os.path.dirname(FILE_PATH)

parser.add_argument('-f', '--fasta', action='store',
                    dest='fasta_file_name', required=True,
                    help='Fasta file')
parser.add_argument('-p', '--params', action='store',
                    dest='params', required=True, choices=['320', '1280'],
                    help='ESM model to use, denoted by the number of output parameters by residue')
parser.add_argument('-m', '--model_file', action='store',
                    dest='model_file', required=True,
                    help='Model file (python file with PyTorch model)')
parser.add_argument('-n', '--model_name', action='store',
                    dest='model_name', required=True,
                    help='Model name (class name in the model file)')
parser.add_argument('-s', '--state_dict', action='store',
                    dest='state_dict', required=True,
                    help='File containing the state dicts for the trained model')
parser.add_argument('-o', '--output', action='store',
                    dest='output_file_name', required=True,
                    help='Output file name')
parser.add_argument('-a', '--model_args', action='store',
                    dest='model_args', default='',
                    help='Comma separated ints to pass to the model constructor (e.g. "1280,2560,1")')
parser.add_argument('-md', '--mode', action='store',
                    dest='mode', default='phospho', choices=['phospho', 'kinase'],
                    help='Prediction mode ("phospho" or "kinase")')
parser.add_argument('-c', '--force_cpu', action='store_true',
                    dest='force_cpu',
                    help='Force CPU')
parser.add_argument('-aaw', '--aa_window', action='store', type=int,
                    dest='aa_window', default=0,
                    help='Amino acid window for the tensors (concatenated tensor of the 5 amino acids)')
parser.add_argument('-fw', '--flatten_window', action='store_true',
                    dest='flatten_window',
                    help='Wether to flatten or not the amino acid window (only if -aaw > 0)')
parser.add_argument('-ad', '--add_dim', action='store_true',
                    dest='add_dim',
                    help='Wether to add an extra dimension to the tensor if aa_window is 0 (needed for CNN)')


args = parser.parse_args()



class ESM_Embeddings_test:
    '''
    PyTorch Dataset for phosphorylations. 
    '''
    def __init__(self, fasta, params, aa_window=0, mode='phospho',
                 add_dim=False, cpu=False):
        
        if mode not in ('phospho', 'kinase'):
            raise NotImplementedError(f'Mode "{mode}" not recognized')
        else:
            self.mode = mode
        
        assert isinstance(aa_window, int) and aa_window >= 0, 'window must be a positive integer'
        self.params = params
        self.window = aa_window
        self.add_dim = add_dim
        self.cpu = cpu

        self.fasta_file_name = path.abspath(fasta)
        assert path.isfile(self.fasta_file_name), f"Couldn't find FASTA file {fasta}"        
        
        self.seq_data = read_fasta(self.fasta_file_name, format=dict)
        self.IDs = list(self.seq_data.keys())
        self.idxs = {}
        self.tensors = {}
        
    
    def __len__(self):
        return len(self.IDs)


    def __getitem__(self, ID):
        try:
            return self.idxs[ID], self.tensors[ID]
        except KeyError:
            return None, None

    
    def compute_embeddings(self, IDs=None, frame_len=1024, off=50):
        models_dict = {'320':(esm.pretrained.esm2_t6_8M_UR50D, 6),
                       '1280':(esm.pretrained.esm2_t33_650M_UR50D, 33)}
        
        try:
            model_fun, layer = models_dict[self.params]
        except KeyError:
            print(f'Not implemented esm model {self.params}, models available are {[*models_dict.keys()]}')
            sys.exit(1)

        if IDs is None:
            seq_data = list(self.seq_data.items())
        elif isinstance(IDs, str):
            seq_data = [(IDs, self.seq_data[IDs])]
        else:
            try:
                seq_data = [(ID, self.seq_data[ID]) for ID in IDs]
            except:
                print(f'Incorrect IDs argument {IDs}. Must be str, iterable or None.')
                return

        model, alphabet = model_fun()
        if self.cpu or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        model.to(device)

        with torch.no_grad():
            for tup in seq_data:
                ID, seq = tup
                idxs = phosphorylable_aas(seq)
                if len(idxs) == 0:
                    print(f'Sequence {ID} has no phosphorylable aminoacids, omitting...')
                    continue


                _, _, tokens = batch_converter([tup])
                tokens = tokens.to(device)
                try:
                    x = model(tokens, repr_layers=[layer], return_contacts=False)["representations"][layer][0, 1:-1]
                except torch.cuda.OutOfMemoryError:
                    # Re-try with moving frame strategy
                    tokens_len = tokens.shape[1]

                    # First frame: from 0 to frame_len
                    x = model(tokens[:, :frame_len], repr_layers=[layer],
                            return_contacts=False)["representations"][layer][0]

                    # Normal middle frames: take x from 0 to o1 and cat new model from o1 to o2
                    #   o1 is (frame_len - off) * j and o2, as always, o1 + frame_len
                    for j in range(1, int(tokens_len / (frame_len - off))):
                        o1 = (frame_len - off) * j
                        o2 = o1 + frame_len
                        x = torch.cat((x[:o1], model(tokens[:, o1:o2], repr_layers=[layer],
                                    return_contacts=False)["representations"][layer][0]))

                    # Final frame: only if the previous iteration did't finish in the token length (perfect finish)
                    #   take x from 0 to o1 and cat new model from o1 to tokens_len (end of the tensor)
                    #   o1 is (tokens_len - frame_len)
                    if o2 != tokens_len:
                        o1 = tokens_len - frame_len
                        x = torch.cat((x[:o1], model(tokens[:, o1:], repr_layers=[layer],
                                    return_contacts=False)["representations"][layer][0]))
                    
                    x = x[1:-1]

                window = self.window
                if window == 0:
                    out_tensor = torch.index_select(x, 0, torch.tensor(idxs, dtype=torch.int64, device=device))
                    if self.add_dim:
                        out_tensor = out_tensor[None, :]
                else:
                    out_tensor = torch.empty((len(idxs), window*2+1, int(self.params)))
                    for i, pos in enumerate(idxs):
                        if (pos + window) < len(seq) and (pos - window) >= 0:  # Normal case
                            out_tensor[i] = x[pos - window : pos + window + 1]
                        elif (pos +  window) >= len(seq):
                            if (pos - window) < 0:  # Overflow over both start and end of the sequence
                                extra_pre = window - pos
                                extra_post = pos + window - len(seq) + 1
                                out_tensor[i] = pad(x, pad=(0, 0, extra_pre, extra_post), value=0)
                            else:                   # Overflow over the end of the sequence
                                extra_post = pos + window - len(seq) + 1
                                out_tensor[i] = pad(x[pos - window :], pad=(0, 0, 0, extra_post), value=0)
                        elif (pos - window) < 0:    # Overflow over the start of the sequence
                            extra_pre = window - pos
                            out_tensor[i] = pad(x[: pos + window + 1], pad=(0, 0, extra_pre, 0), value=0)
                        else:
                            raise ValueError(f'Error handling aminoacid window. Tried to get position {pos} with aminoacid window {window} in tensor of shape {x.size()}')

                self.idxs[ID] = idxs
                self.tensors[ID] = out_tensor

    
    def clear(self):
        self.idxs.clear()
        self.tensors.clear()



def parse_num(num: str):
    '''
    Convert a number from string to int or float (decides best type)
    '''
    return float(num) if '.' in num else int(num)


test_set = ESM_Embeddings_test(fasta=args.fasta_file_name,
                               params=args.params,
                               aa_window=args.aa_window,
                               mode=args.mode,
                               add_dim=args.add_dim,
                               cpu=args.force_cpu)


if args.force_cpu or not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

# Hacky thing to import the model while knowing file and class name at runtime
model_dir = os.path.dirname(args.model_file)
sys.path.append(model_dir)
model_module_name = os.path.basename(args.model_file)[:-3]
model_module = import_module(model_module_name)
model_class = getattr(model_module, args.model_name)

if args.model_args is None:
    model = model_class()
else:
    model = model_class(*[parse_num(arg) for arg in args.model_args.split(',')])

model = model.to(device)
state_dict = torch.load(args.state_dict, map_location=device)
model.load_state_dict(state_dict)
model.eval()


for ID in test_set.IDs:
    test_set.compute_embeddings(ID)
    with torch.no_grad():
        inputs = test_set.tensors[ID]
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.detach().cpu().numpy().flatten()
        
        with open(args.output_file_name, 'a') as out_file:
            for i, idx in enumerate(test_set.idxs[ID]):
                out_file.write('\t'.join([ID, str(idx+1), '', str(round(outputs[i], 3))]) + '\n')

    test_set.clear()
