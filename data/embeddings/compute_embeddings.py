#!/usr/bin/env python3
import torch
import esm
import pickle
import sys
import os
import os.path as path
import argparse

def load_fasta(fasta_file_name):
    seq_data = []
    with open(fasta_file_name, 'r') as fasta_file:
        seq = ''
        ID = None
        for line in fasta_file:
            if line.startswith('>'):
                if seq:
                    seq_data.append((ID, seq))
                seq = ''
                ID = line[1:].strip()
            elif ID:
                seq += line.strip()
        if seq:
            seq_data.append((ID, seq))
    
    return seq_data


def compute_embeddings(seq_data, params, output_folder, pickle_prefix='embeddings_', max_pickle=1000):
    models_dict = {'320':(esm.pretrained.esm2_t6_8M_UR50D, 6),
                   '1280':(esm.pretrained.esm2_t33_650M_UR50D, 33)}
    model_fun, layer = models_dict.get(params)

    # Load ESM-2 model
    model, alphabet = model_fun()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # By default, embeddings are computed using cuda if available, but stored in cpu for better compatibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device} for computing embeddings...')
    device_cpu = torch.device('cpu')
    model = model.to(device)

    n_pick = 1
    lengths_skipped = []
    n_skipped = 0
    representations = {}
    with torch.no_grad():
        for i, tup in enumerate(reversed(seq_data)):
            ID, seq = tup
            # Dump dict to pickle every 1000 sequences
            if len(representations) >= max_pickle:
                pickle_name = f'{pickle_prefix}{n_pick}.pickle'
                with open(path.join(output_folder, pickle_name), 'wb') as f:
                    pickle.dump(representations, f)
                representations = {}
                n_pick += 1

            batch_labels, batch_strs, batch_tokens = batch_converter([tup])
            batch_tokens = batch_tokens.to(device)

            try:
                x = model(batch_tokens, repr_layers=[layer], return_contacts=False)["representations"][layer][0, 1:-1]
                representations[ID] = x.to(device_cpu)
                if i % 10 == 9:
                    print(f'Processed sequences: {i+1}; Written files: {n_pick-1}; Skipped sequences: {n_skipped}', end='\r')
            except torch.cuda.OutOfMemoryError:
                lengths_skipped.append(len(seq))
                print(len(seq))
                n_skipped += 1 # Skip too long sequences

        if representations:
            pickle_name = f'{pickle_prefix}{n_pick}.pickle'
            with open(path.join(output_folder, pickle_name), 'wb') as f:
                pickle.dump(representations, f)

    print('Processed sequences:', i, '; Written files:', n_pick, '; Skipped sequences:', n_skipped)
    for j in sorted(lengths_skipped):
        print(j, end='\t')
    print('')


if __name__ == '__main__':
    description = 'Computes embeddings from sequences in a fasta file and store them in pickles'
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument('-i', '--input', required=True,
                        action='store', dest='fasta_file_name', default=None,
                        help='Input file in fasta format')
    parser.add_argument('-p', '--params', required=True,
                        action='store', dest='params', choices=['320', '1280'],
                        help='Parameters of the model to use, determining the model. \
                        Available models are 320, 1280')
    parser.add_argument('-m', '--max_pickle',
                        action='store', dest='max_pickle', type=int, default=1000,
                        help='Maximum number of sequence embeddings per pickle file')
    parser.add_argument('-f', '--output_folder',
                        action='store', dest='output_folder', default=None,
                        help='Directory to place the output pickles, created if absent, \
                        default is "embeddings_<params>" in the same directory of this script')
    parser.add_argument('-o', '--output_prefix',
                        action='store', dest='pickle_prefix', default='embeddings_',
                        help='Prefix for the output pickles, default is "embeddings_"')
    parser.add_argument('-r', '--remove_files',
                        action='store_true', dest='remove_files', default=False,
                        help='Empties output folder before computing the embeddings. Not implemented yet')


    args = parser.parse_args()
    fasta_file_name = args.fasta_file_name
    params = args.params
    max_pickle = args.max_pickle
    output_folder = args.output_folder
    pickle_prefix = args.pickle_prefix
    remove_files = args.remove_files


    # Retrieve merged sequences file
    if not path.exists(fasta_file_name):
        print("Couldn't find sequences file, aborting...")
        sys.exit(1)

    # Retrieve output folder
    if output_folder is None:
        output_folder = path.abspath(path.join(path.dirname(__file__), f'embeddings_{params}'))
    else:
        output_folder = path.abspath(output_folder)

    if not path.isdir(output_folder):
        os.mkdir(output_folder)
    elif os.listdir(output_folder):
        ### TODO: include option to remove files if they exist
        print(f'Output folder "{output_folder}" not empty, but this is still a TODO')


    # Read fasta file
    print('Reading sequences file...',  end=' ')
    seq_data = load_fasta(fasta_file_name)
    print(f'Found {len(seq_data)} sequences!')

    # Extract per-sequence per-residue representations
    compute_embeddings(seq_data, params, output_folder, pickle_prefix, max_pickle)
    print('Finished computing embeddings!')