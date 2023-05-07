'''
Filters the metadata table to keep only the entried with kinase data
Then filters the FASTA file to keep only sequences with kinase annotations
'''
from os.path import abspath
import sys
sys.path.append(abspath('../data_utils'))
from fasta_utils import read_fasta

IN_TSV = abspath('../merged_dbs/merged_db_metadata.tsv')       # TODO: Replace with cleaned up version once the sequence incoherences have been resolved
IN_FASTA = abspath('../merged_dbs/merged_db_sequences.fasta')  # TODO: Replace with cleaned up version once the sequence incoherences have been resolved
OUT_DIR = abspath('.')

metadata_out = f'{OUT_DIR}/merged_db_metadata_kinase.tsv'
sequences_out = f'{OUT_DIR}/merged_db_sequences_kinase.fasta'

print(f'Metadata table {IN_TSV} will be filtered into {metadata_out}')
print(f'FASTA sequences {IN_FASTA} will be filtered into {sequences_out}')

infile_metadata  = open(IN_TSV, 'r')
outfile_metadata = open(metadata_out, 'w')

sequences = set()
for i, line in enumerate(infile_metadata):
    if i == 0:
        outfile_metadata.write(line)
        continue
    id, pos, aa, kinases, species, kin_species, source = line.strip().split('\t')
    if kinases == 'NA':
        continue
    outfile_metadata.write(line)
    sequences.add(id)

infile_metadata.close()
outfile_metadata.close()

fasta = read_fasta(IN_FASTA, format=dict)

with open(sequences_out, 'w') as outfile_fasta:
    for id, seq in fasta.items():
        if id not in sequences:
            continue
        seq_entry = f'>{id}\n{seq}\n'
        outfile_fasta.write(seq_entry)
