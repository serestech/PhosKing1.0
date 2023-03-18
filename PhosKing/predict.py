import sys
import os
import os.path as path
import time as t
import torch
from torch.nn.functional import pad
import esm
sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', 'utils')))
from utils import read_fasta, phosphorylable_aas


class ESM_Embeddings_test:
    '''
    Dataset for finding phosphorylations using a previously trained model
    '''

    def __init__(self, fasta, params, window=0, add_dim=False, mode='phospho'):
        print('Reading fasta...')
        try:
            self.seq_data = read_fasta(fasta, format=list)
        except FileNotFoundError:
            print(f'Fasta file {fasta} not found, aborting...')
            sys.exit(1)
        
        # # In case we take the whole big fasta and only want a small sample for testing
        #import random
        #self.seq_data = random.sample(self.seq_data, k=50)

        print(f'Found {len(self.seq_data)} sequences!')

        print('Computing embeddings...')
        self._compute_embeddings(params, window, add_dim)
        print('')


    def _compute_embeddings(self, params, window, add_dim):
        models_dict = {'320':(esm.pretrained.esm2_t6_8M_UR50D, 6),
                       '1280':(esm.pretrained.esm2_t33_650M_UR50D, 33)}
        
        try:
            model_fun, layer = models_dict[params]
        except KeyError:
            print(f'Not implemented esm model {params}, models available are {[*models_dict.keys()]}')

        batch_converter = alphabet.get_batch_converter()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        esm_model = esm_model.to(device)
        esm_model.eval()

        self.idxs = {}
        self.tensors = {}
        with torch.no_grad():
            k = 0
            for ID, seq in self.seq_data:
                idxs = phosphorilable_aas(seq)
                if len(idxs) == 0:
                    print(f'Sequence {ID} has no phosphorilable aminoacids, omitting...')
                    continue

                _, _, tokens = batch_converter([(ID, seq)])
                tokens = tokens.to(device)
                try:
                    x = esm_model(tokens, repr_layers=[layer], return_contacts=False)["representations"][layer][0, 1:-1]
                except torch.cuda.OutOfMemoryError:
                    print(f'It seems sequence {ID} is too long and esm could not embeddings! (out of memory)')
                    continue

                self.idxs[ID] = idxs

                if window == 0:
                    out_tensor = torch.index_select(x, 0, torch.tensor(self.idxs[ID], dtype=torch.int64, device=device))
                    if add_dim:
                        out_tensor = out_tensor[None, :]
                else:
                    bound = len(seq)
                    out_tensor = torch.empty((len(self.idxs[ID]), window*2+1, params))
                    for i, pos in enumerate(self.idxs[ID]):
                        if (pos + window) < bound and (pos - window) >= 0:  # Normal case
                            aa_tensor = embedding[pos - window : pos + window + 1]
                        elif (pos +  window) >= bound:
                            if (pos - window) < 0:  # Overflow over both start and end of the sequence
                                extra_pre = window - pos
                                extra_post = pos + window - bound + 1
                                aa_tensor = pad(embedding, pad=(0, 0, extra_pre, extra_post), value=0)
                            else:                   # Overflow over the end of the sequence
                                extra_post = pos + window - bound + 1
                                aa_tensor = pad(embedding[pos - window :], pad=(0, 0, 0, extra_post), value=0)
                        elif (pos - window) < 0:    # Overflow over the start of the sequence
                            extra_pre = window - pos
                            out_tensor = pad(embedding[: pos + window + 1], pad=(0, 0, extra_pre, 0), value=0)
                        else:
                            raise RuntimeError('shuck')

                        out_tensor[i] = aa_tensor

                self.tensors[ID] = out_tensor
                k += 1
                print(f'{k}/{len(seq_data)} embeddings computed!', end='\r')
            
            
    def __getitem__(self, ID):
        return self.idxs[ID], self.tensors[ID]
    
    def IDs(self):
        return list(self.idxs.keys())



if __name__ == '__main__':
    ds = ESM_Embeddings(fracc_non_phospho=0.6,
                        mode='phospho',
                        window=2,
                        verbose_init=True)
    
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    
    inputs, labels = next(iter(loader))
    
    print(f'Getting first batch produced tensors of sizes {inputs.size()} (inputs) and {labels.size()} (labels)')

    ds = ESM_Embeddings(fracc_non_phospho=0.6,
                        mode='kinase',
                        window=2,
                        verbose_init=True)
    
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    
    inputs, labels = next(iter(loader))
    
    print(f'Getting first batch produced tensors of sizes {inputs.size()} (inputs) and {labels.size()} (labels)')