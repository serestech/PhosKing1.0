import os
import os.path as path
import sys
import time as t
from random import sample
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
import esm
sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', 'utils')))
from utils import read_fasta, phosphorylable_aas


class ESM_Embeddings(Dataset):
    '''
    PyTorch Dataset for phosphorylations. 
    '''

    def __init__(self, fasta, features, embeddings_dir, phos_fract=0.5, window=0,
                 add_dim:bool=False, mode:str='phospho', mappings_dir:str=None,
                 verbose=False, small_data=False):
        self.start = t.perf_counter()
        self.verbose = verbose
        self._log('Initializing...')
        
        if mode not in ('phospho', 'kinase'):
            raise NotImplementedError(f'Mode {mode} not recognized')
        else:
            self.mode = mode
        
        assert 0 < phos_fract < 1, 'Fraction of phosphorylated amino acids must be between 0 and 1'
        self.phos_fract = phos_fract
        
        assert isinstance(window, int) and window >= 0, 'window must be a positive integer'
        self.window = window
        self.add_dim = add_dim

        self.fasta_file_name = path.abspath(fasta)
        assert path.isfile(self.fasta_file_name), f"Couldn't find fasta file {fasta}"
        self.features_file_name = path.abspath(features)
        assert path.isfile(self.features_file_name), f"Couldn't find features file {self.features_file_name}"
        self.pickles_dir = path.abspath(embeddings_dir)
        assert path.isdir(self.pickles_dir), f"Couldn't find embeddings directory {self.pickles_dir}"
        
        # Get embeddings from pickles
        self.embeddings_dict = self._load_pickles(self.pickles_dir, small_data)
        
        self.IDs = self.embeddings_dict.keys()
        self.fasta = read_fasta(self.fasta_file_name, format=dict)

        # Keep only pickled sequences
        before = len(self.fasta)
        self.fasta = {ID : seq for ID, seq in self.fasta.items() if ID in self.IDs}
        self._log(f'Discarded {before - len(self.fasta)} sequences that were not in the pickles')
        
        if self.mode == 'kinase':
            self.mapping = {'AMPK': 0, 'ATM': 1, 'Abl': 2, 'Akt1': 3, 'AurB': 4, 'CAMK2': 5, 'CDK1': 6, 'CDK2': 7, 'CDK5': 8, 'CKI': 9, 'CKII': 10, 'DNAPK': 11, 'EGFR': 12, 'ERK1': 13, 'ERK2': 14, 'Fyn': 15, 'GSK3': 16, 'INSR': 17, 'JNK1': 18, 'MAPK': 19, 'P38MAPK': 20, 'PKA': 21, 'PKB': 22, 'PKC': 23, 'PKG': 24, 'PLK1': 25, 'RSK': 26, 'SRC': 27, 'mTOR': 28}
            self.reverse_mapping = {i : kinase for kinase, i in self.mapping.items()}
        
        self.data = self._load_metadata(self.features_file_name)
        self.true  = torch.Tensor([1])
        self.false = torch.Tensor([0])
        
        self._log('Finished initialization')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ID, pos, label = self.data[idx]
        embedding = self.embeddings_dict[ID]
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
        elif (pos - window) < 0:   # Overflow over the start of the sequence
            extra_pre = window - pos
            out_tensor = pad(embedding[: pos + window + 1], pad=(0, 0, extra_pre, 0), value=0)
        else:
            raise RuntimeError('Error cogiendo tensors, speak con Sergio')
        
        if self.add_dim:
            out_tensor = out_tensor[None, :]
        
        if self.mode == 'phospho':
            out_label = self.true if label else self.false
        elif self.mode == 'kinase':
            out_label = label
        else:
            raise NotImplementedError
        
        return out_tensor, out_label

    
    def _load_pickles(self, pickles_dir: str, small_dataset: bool):
        self._log('Loading pickles...')
        pickle_files = [file for file in os.listdir(pickles_dir) if file.endswith('.pickle')]
        if small_dataset:
            pickle_files = pickle_files[:1]

        embeddings_dict = {}
        for filename in pickle_files:
            self._log(f'Loading pickle {filename}')
            with open(path.join(self.pickles_dir, filename), 'rb') as pickle_file:
                embeddings_dict.update(pickle.load(pickle_file))

        self._log(f'Pickles contain {len(embeddings_dict)} sequences')
        return embeddings_dict
    
    def _load_metadata(self, features_file_name):
        '''
        Load the Phosphosite metadata, filtering by sequences present in the pickles. 
        '''
        self._log('Loading metadata')
        
        phos_aas = {'S', 'T', 'Y'}
        data = {}  # Metadata will be saved here and converted to list at the very end
        n_missing = 0
        n_wrong = 0
        n_not_phospho = 0
        n_unknown_kinase = 0
        with open(self.features_file_name, 'r') as features_file:
            for line in features_file:
                if not line.startswith('#'):
                    ID, pos, aa, kinases, _, _, _ = line.strip().split('\t')
                    pos = int(pos) - 1
                    entry: tuple = (ID, pos)  # aa identified by seq and position (0-indexed)
                    
                    if ID not in self.IDs:
                        n_missing += 1
                        # self._log(f'Discarding phosphorylation {entry} (seq not in pickles)')
                        continue
                    
                    try:
                        if self.fasta[ID][pos] != aa:
                            n_wrong += 1
                            # self._log(f'Discarding phosphorylation {entry} (amino acid does not correspond with FASTA)')
                            continue
                    except IndexError:
                        n_wrong += 1
                        # self._log(f'Discarding phosphorylation {entry} (amino acid does not correspond with FASTA, out of range)')
                        continue

                    if aa not in phos_aas:
                        n_not_phospho += 1
                        # self._log(f'Discarding phosphorylation {entry} (amino acid not phosphorylable)')
                        continue
                    
                    if self.mode == 'phospho':
                        data[entry] = True
                    elif self.mode == 'kinase':
                        if kinases == 'NA':
                            n_unknown_kinase += 1
                            continue
                        labels = torch.zeros(len(self.mapping))
                        for kinase in kinases.strip().split(','):
                            labels[self.mapping[kinase]] = 1
                        data[entry] = labels
        
        n_phos = len(data)
        self._log(f'Loaded {n_phos} phosphorylations. {n_missing + n_not_phospho + n_wrong + n_unknown_kinase} \
                    discarded ({n_missing} not in pickles, {n_not_phospho} not phosphorylable, {n_wrong} wrongly \
                    documented{(", " + str(n_unknown_kinase) + " unknown kinase") if self.mode == "kinase" else ""})')
        
        if self.mode == 'phospho':
            self._log(f'Getting all non-phosphorylated phosphorylable amino acids')
            
            false_aas = []
            for ID, seq in self.fasta.items():
                for pos in phosphorylable_aas(seq, phos_aas):
                    entry: tuple = (ID, pos)
                    if entry not in data.keys():
                        false_aas.append(entry)
            
            self._log(f'Found {len(false_aas)} non-phosphorylated phosphorylable amino acids. Subsetting')

            n_false_sample = int(((1 - self.phos_fract) * n_phos) / self.phos_fract)
            self._log(f'Sampling {n_false_sample} non-phosphorylated amino acids')
            false_sample = sample(false_aas, k=n_false_sample)
            total_size = n_phos + n_false_sample
            self._log(f'Total dataset: {total_size} entries, of which \
                      {(n_phos / total_size) * 100:.2f}% are phosphorylated \
                      and {(n_false_sample / total_size) * 100:.2f}% are non-phosphorylated')
            
            for entry in false_sample:
                data[entry] = False
            
        data_list = [(*entry, label) for entry, label in data.items()]
        
        self._log(f'Checking that all extracted positions actually correspond to a phosphorylable amino acid ({phos_aas}) in the FASTA ')
        phos_delete = set()
        for seq, pos, _ in data_list:
            try:
                if self.fasta[seq][pos] not in phos_aas:
                    phos_delete.add((seq, pos))
                    self._log(f'WARNING: Found non-phosphorylable amino acid "{self.fasta[seq][pos]}" in seq {seq}, position {pos})')
            except IndexError as error:
                phos_delete.add((seq, pos))
                self._log(f'WARNING: Found position {pos} in sequence of {len(self.fasta[seq])}: {error}')
                
        data_list = [phos for i, phos in enumerate(data_list) if i not in phos_delete]
        
        self._log(f'Generated data list of length {len(data_list)}. Some examples:\n{sample(data_list, 50 if self.mode == "phospho" else 5)}')

        return data_list

        
    def _log(self, msg: str):
        if self.verbose:
            now = t.perf_counter()
            print(f'[{int((now - self.start) // 60):02d}:{(now - self.start) % 60:0>6.3f}]:', msg)