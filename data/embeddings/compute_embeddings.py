#!/usr/bin/env python3
import sys, os
print(f'Using python env {sys.executable}')
import os.path as path
import argparse
import torch
print(f'Using torch version {torch.__version__}')
import esm
import pickle

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


def compute_embeddings(seq_data, params, output_folder, pickle_prefix='embeddings_',
                       max_pickle=1000, frame_len=1024, off=50, verbose=True):
    models_dict = {'320':(esm.pretrained.esm2_t6_8M_UR50D, 6),
                   '1280':(esm.pretrained.esm2_t33_650M_UR50D, 33)}
    if params not in models_dict:
        raise ValueError(f'Not valid model params: {params}, aborting...')
    model_fun, layer = models_dict.get(params)

    # Load ESM-2 model
    model, alphabet = model_fun()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # By default, embeddings are computed using cuda if available, but stored in cpu for better compatibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using torch device of type {device.type}{": " + torch.cuda.get_device_name(device) if device.type == "cuda" else ""}')
    device_cpu = torch.device('cpu')
    model = model.to(device)

    n_pick = 1
    n_moving = 0
    representations = {}
    n_seqs = len(seq_data)
    with torch.no_grad():
        for i, tup in enumerate(seq_data):
            ID, seq = tup
            # Dump dict to pickle every 1000 sequences
            if len(representations) >= max_pickle:
                with open(path.join(output_folder, f'{pickle_prefix}{n_pick}.pickle'), 'wb') as f:
                    pickle.dump(representations, f)
                representations = {}
                n_pick += 1
            
            if verbose and i % 10 == 0:
                print(f'Processed sequences: {i} ({(i + 1)/n_seqs * 100:.2f}%); Written files: {n_pick-1}; Framed sequences: {n_moving}', end='\r')

            _, _, tokens = batch_converter([tup])
            tokens = tokens.to(device)

            try:
                x = model(tokens, repr_layers=[layer], return_contacts=False)["representations"][layer][0]
            except torch.cuda.OutOfMemoryError:
                # Re-try with moving frame strategy
                tokens_len = tokens.shape[1]
                n_moving += 1 # moving sequences

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
            
            representations[ID] = x[1:-1].to(device_cpu)

        if representations:
            with open(path.join(output_folder, f'{pickle_prefix}{n_pick}.pickle'), 'wb') as f:
                pickle.dump(representations, f)

    if verbose: print(f'Processed sequences: {i}  Written files: {n_pick}  Framed sequences: {n_moving}')


if __name__ == '__main__':
    description = 'Computes embeddings from sequences in a fasta file and store them in pickles'
    parser = argparse.ArgumentParser(description=description, add_help=True)

    # python3 compute_embeddings.py -i /zhome/52/c/174062/s220260/PhosKing1.0/data/homology_reduced/cd-hit_out_29-04.fasta -o /zhome/52/c/174062/s220260/PhosKing1.0/data/embeddings/embeddings_1280

    parser.add_argument('-i', '--input', required=True,
                        action='store', dest='fasta_file_name', default=None,
                        help='Input file in fasta format')
    parser.add_argument('-p', '--params',
                        action='store', dest='params', choices=['320', '1280'],
                        help='Embeddings size (320, 1280)',
                        default='1280')
    parser.add_argument('-m', '--max_pickle',
                        action='store', dest='max_pickle', type=int, default=1000,
                        help='Maximum number of sequence embeddings per pickle file')
    parser.add_argument('-o', '--output_folder',
                        action='store', dest='output_folder', default=None,
                        help='Directory to place the output pickles, created if absent, \
                        default is "embeddings_<params>" in the same directory of this script')
    parser.add_argument('-pf', '--output_prefix',
                        action='store', dest='pickle_prefix', default='embeddings_',
                        help='Prefix for the output pickles, default is "embeddings_"')
    parser.add_argument('-F', '--frame_len',
                        action='store', dest='frame_len', type=int, default=1024,
                        help='Select frame length from which the protein embeddings will be calculated with windows if lacking GPU memory')
    parser.add_argument('-of', '--offset',
                        action='store', dest='offset', type=int, default=150,
                        help='Select offset to concatenate embeddings for sequences biggers than frame_len')
    parser.add_argument('-r', '--remove_files',
                        action='store_true', dest='remove_files', default=False,
                        help='Empties output folder before computing the embeddings. Not implemented yet')


    args = parser.parse_args()
    fasta_file_name = args.fasta_file_name
    params = args.params
    max_pickle = args.max_pickle
    output_folder = args.output_folder
    pickle_prefix = args.pickle_prefix
    frame_len = args.frame_len
    offset = args.offset
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
        print(f'WARNING: Output folder "{output_folder}" not empty, files may be overwritten')


    # Read fasta file
    print('Reading sequences file')
    seq_data = load_fasta(fasta_file_name)
    print(f'Found {len(seq_data)} sequences')

    # Extract per-sequence per-residue representations
    compute_embeddings(seq_data, params, output_folder, pickle_prefix, max_pickle, frame_len, offset)
    print('Finished computing embeddings')