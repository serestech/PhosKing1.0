import os
import os.path as path
import sys
import argparse


def load_clusters(clusters_name):
    clusters = {}
    with open(clusters_name, 'r') as infile:
        for line in infile:
            if not line.startswith('#'):
                rep, ID = line.strip().split('\t')
                if not clusters.get(rep):
                    clusters[rep] = []
                clusters[rep].append(ID)

    return clusters


def load_metadata_kin(metadata_name, clusters):
    metadata = {}
    with open(metadata_name, 'r') as infile:
        for line in infile:
            if not line.startswith('#'):
                ID, _, _, n_kin, length, _ = line.strip().split('\t')
                metadata[ID] = [int(n_kin), int(length)]

    repr_seqs = set()
    for rep, cluster in clusters.items():
        max_n_kin, max_length = 0, 0
        new_rep = None
        for ID in cluster:
            n_kin, length = metadata[ID]
            if n_kin and max_n_kin <= n_kin:
                max_n_kin = n_kin
                if max_length < length:
                    max_length = length
                    new_rep = ID
        if new_rep:
            repr_seqs.add(new_rep)

    return repr_seqs


def filter_files(fasta_name, features_name, metadata_name, suff, repr_seqs):
    infile = open(fasta_name, 'r')
    outfile = open(f'sequences{suff}.fasta', 'w')
    ID = None
    for line in infile:
        if not line.startswith('#'):
            if line.startswith('>'):
                ID = line[1:].strip()
            elif ID and ID in repr_seqs:
                outfile.write(f'>{ID}\n{line}')
        else:
            outfile.write(line)

    infile.close()
    outfile.close()

    infile = open(features_name, 'r')
    outfile = open(f'features{suff}.tsv', 'w')
    for line in infile:
        if not line.startswith('#'):
            ID, _, _, kins, _, _, _ = line.strip().split('\t')
            if ID in repr_seqs and kins != 'NA':
                outfile.write(line)
        else:
            outfile.write(line)
            
    infile.close()
    outfile.close()

    infile = open(metadata_name, 'r')
    outfile = open(f'metadata{suff}.tsv', 'w')
    for line in infile:
        if not line.startswith('#'):
            ID = line.strip().split('\t')[0]
            if ID in repr_seqs:
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
                        help='Input fasta file')
    parser.add_argument('-t', '--features', required=True,
                        action='store', dest='features_name',
                        help='Input features file')
    parser.add_argument('-m', '--metadata', required=True,
                        action='store', dest='metadata_name',
                        help='Output file names, in the same order than the input files')
    parser.add_argument('-c', '--clusters', required=True,
                        action='store', dest='clusters_name',
                        help='Output file names, in the same order than the input files')
    parser.add_argument('-o', '--out-suffix', default='_kin',
                        action='store', dest='out_suffix',
                        help='Suffix for the out files: sequences<>.fasta, features<>.tsv and metadata<>.tsv')

    
    args = parser.parse_args()
    fasta_name = args.fasta_name
    features_name = args.features_name
    metadata_name = args.metadata_name
    clusters_name = args.clusters_name
    suff = args.out_suffix

    print('Loading clusters file...')
    clusters = load_clusters(clusters_name)

    print('Loading metadata file...')
    repr_seqs = load_metadata_kin(metadata_name, clusters)

    print('Filtering files for maximizing kinases...')
    filter_files(fasta_name, features_name, metadata_name, suff, repr_seqs)
    print('Completed!')
