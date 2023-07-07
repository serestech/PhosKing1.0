from utils import read_fasta
import sys

fasta_path = sys.argv[1]
outfile_path = sys.argv[2]

fasta = read_fasta(fasta_path, dict)

sequences_file = open(sys.argv[3]) if len(sys.argv) >= 4 else sys.stdin
                      
subset_seqs = set()
for line in sequences_file:
    seq_id = line.strip()
    subset_seqs.add(seq_id)

print(f'Generating subset of {len(subset_seqs)} proteins {outfile_path}')

with open(outfile_path, 'w') as outfile:
    for seq_id in subset_seqs:
        outfile.write(f'>{seq_id}\n{fasta[seq_id]}\n')
