import os
import os.path as path
import sys
import random
import argparse
sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', 'utils')))
from utils import read_fasta
import math

parser = argparse.ArgumentParser(prog=path.basename(__file__),
                                 description='Split fasta IDs into subsets for cross-validation')

FILE_PATH = path.abspath(__file__)
HERE = path.dirname(FILE_PATH)


parser.add_argument('-f', '--fasta', action='store', dest='fasta_path',
                    required=True,
                    help='Fasta file')
parser.add_argument('-o', '--output', action='store', dest='output_path',
                    default=None,
                    help='Output file with the indices. If None (default), prints to the screen.')
parser.add_argument('-n', '--n_splits', action='store', dest='n_splits',
                    required=True, type=int,
                    help='Total number of splits')

args = parser.parse_args()


fasta_path = args.fasta_path
output_path = args.output_path
n_splits = args.n_splits

fasta = read_fasta(fasta_path)
random.shuffle(fasta)

n_IDs = len(fasta)
IDs_per_split = math.ceil(n_IDs / n_splits)

if output_path is not None:
    sys.stdout = open(output_path, 'w')

for n in range(n_splits):
    print(f'%{n+1}')
    i = n * IDs_per_split
    j = (n+1) * IDs_per_split
    for ID, _ in fasta[i:j]:
        print(ID)