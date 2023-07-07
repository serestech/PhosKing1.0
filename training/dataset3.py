import os, sys
import os.path as path
import time as t
from random import sample, shuffle
import pickle
import torch
import torch.utils.data as data
from torch.nn.functional import pad
sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', 'utils')))
from utils import read_fasta, phosphorylable_aas
from sklearn.model_selection import train_test_split
import numpy as np


class ESM_Embeddings(data.Dataset):
    '''
    PyTorch Dataset for phosphorylations. 
    '''
    def __init__(self, fasta: str, features: str, embeddings_dir: str,
                 frac_phos=0.5, train=0.8, valid_test=0.5,
                 aa_window: int=0, add_dim: bool=False, flatten_window: bool=False, 
                 verbose=True, small_data=False):
        self.start = t.perf_counter()
        self.verbose = verbose
        self._log('Initializing dataset class')
        
        assert isinstance(aa_window, int) and aa_window >= 0, 'window must be a positive integer'
        self.window = aa_window
        self.add_dim = add_dim

        self.fasta_file_name = path.abspath(fasta)
        assert path.isfile(self.fasta_file_name), f"Couldn't find FASTA file {fasta}"
        self.features_file_name = path.abspath(features)
        assert path.isfile(self.features_file_name), f"Couldn't find features file {self.features_file_name}"
        self.pickles_dir = path.abspath(embeddings_dir)
        assert path.isdir(self.pickles_dir), f"Couldn't find embeddings directory {self.pickles_dir}"
        
        self.flatten_out = flatten_window
        
        # Get embeddings from pickles
        self.embeddings_dict = self._load_pickles(self.pickles_dir, small_data)
        self.IDs = self.embeddings_dict.keys()
        self._log(f'{len(self.IDs)} unique IDs found in the pickles')
        
        self._log(f'Loading fasta file "{self.fasta_file_name}"')
        self.fasta = read_fasta(self.fasta_file_name, format=dict)

        # Keep only pickled sequences
        before = len(self.fasta)
        self.fasta = {ID : seq for ID, seq in self.fasta.items() if ID in self.IDs}
        self._log(f'Discarded {before - len(self.fasta)} sequences that were not in the pickles')
        
        # Keep only sequences in FASTA
        before = len(self.embeddings_dict)
        self.embeddings_dict = {ID: embeddings for ID, embeddings in self.embeddings_dict.items() if ID in self.fasta.keys()}
        self._log(f'Discarded {before - len(self.embeddings_dict)} sequence embeddings that were not in the FASTA file')

        assert train <= 1 and valid_test <= 1, f'Invalid train/valid/test proportions {train}, {valid_test}'

        self.true = torch.Tensor([1])
        self.false = torch.Tensor([0])
        self._load_metadata(self.features_file_name, frac_phos, train, valid_test)

        self._log('Finished initialization')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ID, pos, label = self.data[idx]
        embedding: torch.Tensor = self.embeddings_dict[ID]
        window = self.window
        bound = embedding.size()[0]
        
        if (pos + window) < bound and (pos - window) >= 0:  # Normal case
            out_tensor = embedding[pos - window : pos + window + 1]
        elif (pos +  window) >= bound:
            if (pos - window) < 0:  # Overflow over both start and end of the sequence
                extra_pre = window - pos
                extra_post = pos + window - bound + 1
                out_tensor = pad(embedding, pad=(0, 0, extra_pre, extra_post), value=0)
            else:                   # Overflow over the end of the sequence
                extra_post = pos + window - bound + 1
                out_tensor = pad(embedding[pos - window :], pad=(0, 0, 0, extra_post), value=0)
        elif (pos - window) < 0:    # Overflow over the start of the sequence
            extra_pre = window - pos
            out_tensor = pad(embedding[: pos + window + 1], pad=(0, 0, extra_pre, 0), value=0)
        else:
            raise ValueError(f'Error handling aminoacid window. Tried to get position {pos} with aminoacid window {window} in tensor of shape {embedding.size()}')
        
        if self.add_dim:
            out_tensor = out_tensor[None, :]
        
        if self.flatten_out:
            out_tensor = torch.flatten(out_tensor)
        
        out_label = self.true if label else self.false
        
        return out_tensor, out_label

    
    def _load_pickles(self, pickles_dir: str, small_dataset: bool):
        self._log(f'Loading pickles from {pickles_dir}')
        pickle_files = [file for file in os.listdir(pickles_dir) if file.endswith('.pickle')]
        self._log(f'Found {len(pickle_files)} pickle files')
        if small_dataset:
            pickle_files = pickle_files[:1]

        embeddings_dict = {}
        for i, filename in enumerate(pickle_files):
            pickle_path = path.join(self.pickles_dir, filename)
            self._log(f'Loading embeddings pickle {pickle_path} ({i + 1} of {len(pickle_files)}) {(i + 1)/len(pickle_files) * 100:5.1f}%')
            with open(pickle_path, 'rb') as pickle_file:
                embeddings_dict.update(pickle.load(pickle_file))

        self._log(f'Loaded pickles contain a total of {len(embeddings_dict)} sequences')
        return embeddings_dict
    
    def _load_metadata(self, features_file_name, frac_phos, train, valid_test):
        '''
        Load the Phosphosite metadata, filtering by sequences present in the pickles. 
        '''
        self._log(f'Loading metadata from features file "{features_file_name}"')
        
        phos_aas = {'S', 'T', 'Y'}
        self.data_true = set()
        n_missing = 0
        n_wrong = 0
        n_not_phospho = 0
        with open(features_file_name, 'r') as features_file:
            for line in features_file:
                if not line.startswith('#'):
                    ID, pos, aa, _, _, _ = line.strip().split('\t')
                    pos = int(pos) - 1
                    entry: tuple = (ID, pos)  # aa identified by seq and position (0-indexed)
                    
                    if ID not in self.IDs:
                        n_missing += 1
                        continue
                    
                    try:
                        if self.fasta[ID][pos] != aa:
                            n_wrong += 1
                            continue
                    except IndexError:
                        n_wrong += 1
                        continue

                    if aa not in phos_aas:
                        n_not_phospho += 1
                    else:
                        self.data_true.add(entry)
                        
        n_phos = len(self.data_true)
        self._log(f'Loaded {n_phos} phosphorylations. {n_missing + n_not_phospho + n_wrong} discarded ({n_missing} not in pickles, {n_not_phospho} not phosphorylable, {n_wrong} wrongly documented)')
        assert n_phos > 0, 'Loaded 0 phosphorylations'

        self._log(f'Getting all non-phosphorylated phosphorylable amino acids')
        
        self.data_false = set()
        IDs = np.unique([ID for ID,_ in self.data_true])
        for ID in IDs:
            seq = self.fasta[ID]
            for pos in phosphorylable_aas(seq, phos_aas):
                entry: tuple = (ID, pos)
                if entry not in self.data_true:
                    self.data_false.add(entry)
        
        self._log(f'Found {len(self.data_false)} non-phosphorylated phosphorylable amino acids.')

        self.data = [(*entry, True) for entry in self.data_true]
        self.data.extend([(*entry, False) for entry in self.data_false])
        
        train_IDs_list, rest_IDs = train_test_split(IDs, train_size=train, test_size=(1-train), shuffle=True)
        validation_IDs_list, test_IDs_list = train_test_split(rest_IDs, train_size=valid_test, test_size=(1-valid_test), shuffle=True)

        self.train_IDs = set(train_IDs_list)
        self.validation_IDs = set(validation_IDs_list)
        self.test_IDs = set(test_IDs_list)
        self.train_idx = []
        self.validation_idx = []
        self.test_idx = []
        self.data_true_train = []
        self.data_false_train = []
        for idx, (ID, pos, lab) in enumerate(self.data):
            if ID in self.train_IDs:
                self.train_idx.append(idx)
                if lab:
                    self.data_true_train.append((ID, pos))
                else:
                    self.data_false_train.append((ID, pos))
            elif ID in self.validation_IDs:
                self.validation_idx.append(idx)
            elif ID in self.test_IDs:
                self.test_idx.append(idx)

        self.train_dataset = data.Subset(self, self.train_idx)
        self.validation_dataset = data.Subset(self, self.validation_idx)
        self.test_dataset = data.Subset(self, self.test_idx)

        shuffle(self.data_false_train)
        self.n_false_train = len(self.data_false_train)
        self.n_subset_false_train = int(((1 - frac_phos) * len(self.data_true_train)) / frac_phos)
        

    def train_subset(self, epoch):
        i = (epoch * self.n_subset_false_train) % self.n_false_train
        j = (i + self.n_subset_false_train) % self.n_false_train
        if j <= i:
            false_train = set(self.data_false_train[i:]).union(self.data_false_train[:j])
        else:
            false_train = set(self.data_false_train[i:j])

        subset_idx = []
        true_train = set(self.data_true_train)
        for idx, (ID, pos, _) in enumerate(self.data):
            if (ID, pos) in true_train or (ID, pos) in false_train:
                subset_idx.append(idx)

        self._log(f'Subset of training dataset with {len(true_train)} positives and {len(false_train)} negatives ({len(subset_idx)} total)')
        
        return data.Subset(self, subset_idx)

        
    def _log(self, msg: str):
        if self.verbose:
            now = t.perf_counter()
            print(f'[{int((now - self.start) // 60):02d}:{(now - self.start) % 60:0>6.3f}]:', msg)

if __name__ == '__main__':
    ds = ESM_Embeddings(fasta='/zhome/52/c/174062/s220260/PhosKing1.0/data/kinase_data/merged_db_sequences_kinase.fasta',
                        features='/zhome/52/c/174062/s220260/PhosKing1.0/data/kinase_data/kinase_metadata.tsv',
                        embeddings_dir='/work3/s220260/PhosKing1.0/data/embeddings/embeddings_1280_kinase',
                        small_data=True)
