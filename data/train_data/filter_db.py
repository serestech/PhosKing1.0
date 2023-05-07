import os
import os.path as path
import sys
import re
import argparse

sys.path.append(path.realpath(path.join(path.dirname(__file__), '..', 'data_utils')))
from fasta_utils import read_fasta

def filter_file(in_name, out_name, seqs, species_file=None, species_targets=[]):

    if species_file and species_targets:
        species_mapping = {}
        with open(species_file, 'r') as f:
            for line in f:
                name, pref_name, _ = line.strip().split('\t')
                species_mapping[name] = pref_name

    infile = open(in_name, 'r')
    outfile = open(out_name, 'w')
    for line in infile:
        if not line.startswith('#'):
            ID = line.strip().split('\t')[0]
            if ID in seqs:
                if species_file and out_name in species_targets:
                    line_ = line.strip().split('\t')
                    name = line_[1]
                    line_[1] = species_mapping.get(name, name)
                    line = '\t'.join(line_) + '\n'
                outfile.write(line)
        else:
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
    parser.add_argument('-spe', '--species_file', default=None,
                        action='store', dest='species_file', nargs='+',
                        help='Species mapping file to replace species names in metadata files, ' \
                        'followed by the output files in which the mapping is performed')
    
    args = parser.parse_args()
    fasta_name = args.fasta_name
    in_names = args.in_names
    out_names = args.out_names
    species_file = args.species_file[0]
    species_targets = args.species_file[1:]

    if len(in_names) != len(out_names):
        print('Error: different number of input and output files')
        sys.exit(1)

    print('Loading fasta file...')
    seqs = read_fasta(fasta_name, format=dict).keys()

    for in_name, out_name in zip(in_names, out_names):
        print(f'Filtering {in_name}...')
        filter_file(in_name, out_name, seqs, species_file, species_targets)
    
    print('Completed!')
