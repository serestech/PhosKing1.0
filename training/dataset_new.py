import os, sys
import os.path as path
import time as t
from random import sample, shuffle, randint
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', 'utils')))
from utils import read_fasta, phosphorylable_aas


class ESM_Embeddings(Dataset):
    '''
    PyTorch Dataset for phosphorylations. 
    '''
    def __init__(self, fasta: str, features: str, embeddings_dir: str, phos_fract: float=0.5, 
                 aa_window: int=0, add_dim: bool=False, flatten_window: bool=False, mode:str='phospho', 
                 verbose=True, small_data=False):
        self.start = t.perf_counter()
        self.verbose = verbose
        self._log('Initializing dataset class')
        
        if mode not in ('phospho', 'kinase'):
            raise NotImplementedError(f'Mode "{mode}" not recognized')
        else:
            self.mode = mode
        
        assert 0 < phos_fract < 1, 'Fraction of phosphorylated amino acids must be between 0 and 1'
        self.phos_fract = phos_fract
        
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
        self._log(f'{len(self.IDs)} unique IDs')
        
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
        
        if self.mode == 'kinase':
            from kinase_mapping import kinase_mapping, kinase_mapping_reverse
            self.mapping = kinase_mapping
            self.reverse_mapping = kinase_mapping_reverse
        
        self.data = self._load_metadata(self.features_file_name)
        self.true  = torch.Tensor([1])
        self.false = torch.Tensor([0])
        
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
        
        if self.mode == 'phospho':
            out_label = self.true if label else self.false
        elif self.mode == 'kinase':
            out_label = label
        else:
            raise NotImplementedError
        
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
    
    def _load_metadata(self, features_file_name):
        '''
        Load the Phosphosite metadata, filtering by sequences present in the pickles. 
        '''
        self._log(f'Loading metadata from features file "{features_file_name}"')
        
        phos_aas = {'S', 'T', 'Y'}
        data = {}  # Metadata will be saved here and converted to list at the very end
        n_missing = 0
        n_wrong = 0
        n_not_phospho = 0
        n_unknown_kinase = 0
        with open(features_file_name, 'r') as features_file:
            for line in features_file:
                if line.startswith('#'):
                    continue
                ID, pos, aa, kinases, _ = line.strip().split('\t')
                pos = int(pos) - 1
                entry: tuple = (ID, pos)  # aa identified by seq and position (0-indexed)
                
                if ID not in self.IDs:
                    n_missing += 1
                    continue
                
                try:
                    if self.fasta[ID][pos] != aa:
                        self._log(f'Wrongly documented amino acid: position {pos} in sequence {ID} was expected to be {aa} but was {self.fasta[ID][pos]}')
                        n_wrong += 1
                        continue
                except IndexError:
                    self._log(f'Wrongly documented amino acid: position {pos} in sequence {ID} is out of range')
                    n_wrong += 1
                    continue

                if aa not in phos_aas:
                    self._log(f'Non-phosphorylable amino acid: position {pos} in sequence {ID} is amino acid {aa}')
                    n_not_phospho += 1
                    continue
                
                if self.mode == 'phospho':
                    data[entry] = True
                elif self.mode == 'kinase':
                    if kinases == 'NA':
                        self._log(f'No kinases in position {pos} in sequence {ID}')
                        n_unknown_kinase += 1
                        continue
                    labels = self._parse_kinases(kinases)
                    if torch.sum(labels) == 0:
                        self._log(f'None of these kinases matched a known category: "{kinases}"')
                        n_unknown_kinase += 1
                        continue
                    data[entry] = labels
                    
        n_phos = len(data)
        self._log(f'Loaded {n_phos} phosphorylations. {n_missing + n_not_phospho + n_wrong + n_unknown_kinase} discarded ({n_missing} not in pickles, {n_not_phospho} not phosphorylable, {n_wrong} wrongly documented{f", {n_unknown_kinase}  unknown kinase" if self.mode == "kinase" else ""})')
        assert n_phos > 0, 'Loaded 0 phosphorylations'
        
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
            self._log(f'Total dataset: {total_size} entries, of which {(n_phos / total_size) * 100:.2f}% are phosphorylated and {(n_false_sample / total_size) * 100:.2f}% are non-phosphorylated')
            
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
        
        shuffle(data_list)  # If we dont do this, all True phosphorylations are at the beginning and all False at the end, which causes accuracy measures to produce ZeroDivisionError
        
        self._log(f'Generated data list of length {len(data_list)}. Some examples:\n{sample(data_list, 50 if self.mode == "phospho" else 5)}')

        if self.mode == 'kinase':
            kinase_counts = [0 for i in range(len(self.mapping))]
            for seq_id, pos, label in data_list:
                for i in range(len(self.mapping)):
                    if label[i].item() == 1:
                        kinase_counts[i] += 1

            for i, count in enumerate(kinase_counts):
                self._log(f'{count} ----- {self.reverse_mapping[i]}')

        return data_list

    def _parse_kinases(self, kinases: str) -> torch.Tensor:
        # labels = torch.zeros(len(self.mapping))
        # for kinase in kinases.strip().split(','):
        #     categories = [cat.strip() for cat in kinase.split(' ')]
        #     for i in range(len(categories)):
        #         cumulative_cateogries = categories[:i + 1]
        #         cumulative_cateogry_string = ' '.join(cumulative_cateogries)
        #         if cumulative_cateogry_string not in self.mapping:
        #             continue
        #         labels[self.mapping[cumulative_cateogry_string]] = 1
        labels = torch.zeros(len(self.mapping))
        kinases = [kinase.strip() for kinase in kinases.strip().split(',')]
        for kinase in kinases:
            individual_categories = [category.strip() for category in kinase.split(' ')]
            cumulative_categories = []
            for category in individual_categories:
                cumulative_categories.append(category)
                cumulative_category_string = ' '.join(cumulative_categories)
                if cumulative_category_string in self.mapping.keys():
                    i = self.mapping[cumulative_category_string]
                    labels[i] = 1
        return labels
        
    def _log(self, msg: str):
        if self.verbose:
            now = t.perf_counter()
            print(f'[{int((now - self.start) // 60):02d}:{(now - self.start) % 60:0>6.3f}]:', msg)

if __name__ == '__main__':
    ds = ESM_Embeddings(fasta='/zhome/52/c/174062/s220260/PhosKing1.0/data/kinase_data/merged_db_sequences_kinase.fasta',
                        features='/zhome/52/c/174062/s220260/PhosKing1.0/data/kinase_data/kinase_metadata.tsv',
                        embeddings_dir='/work3/s220260/PhosKing1.0/data/embeddings/embeddings_1280_kinase',
                        mode='kinase',
                        small_data=True)
