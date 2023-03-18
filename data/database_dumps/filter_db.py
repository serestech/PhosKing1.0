import os
import os.path as path
import sys
import re
import argparse

sys.path.append(path.realpath(path.join(path.dirname(__file__), '..', 'data_utils')))
from fasta_utils import read_fasta

def filter_file(in_name, out_name, seqs):
    infile = open(in_name, 'r')
    outfile = open(out_name, 'w')
    for line in infile:
        if not line.startswith('#'):
            ID = line.strip().split('\t')[0]
            if ID in seqs:
                outfile.write(line)
            
    infile.close()
    outfile.close()


if __name__ == '__main__':
    description = 'Reads fasta and filter features/metadata files to contain only IDs present in the fasta'
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument('-f', '--fasta', required=True,
                        action='store', dest='fasta_name', default=None,
                        help='Input file in fasta format')
    parser.add_argument('-i', '--input', required=True,
                        action='store', dest='in_names', nargs='+',
                        help='Input files to be filtered, space-separated')
    parser.add_argument('-o', '--output', required=True,
                        action='store', dest='out_names', nargs='+',
                        help='Output file names, in the same order than the input files')
    
    args = parser.parse_args()
    fasta_name = args.fasta_name
    in_names = args.in_names
    out_names = args.out_names

    if len(in_names) != len(out_names):
        print('Error: different number of input and output files')
        sys.exit(1)

    print('Loading fasta file...')
    seqs = read_fasta(fasta_name, format=dict).keys()

    for in_name, out_name in zip(in_names, out_names):
        print(f'Filtering {in_name}...')
        filter_file(in_name, out_name, seqs)
    
    print('Completed!')
