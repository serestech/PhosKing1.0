import os, csv 

OUTDIR = '../tmp'
DB_DUMPS = '../database_dumps'

out_metadata = f"{OUTDIR}/db_metadata.tsv"
print(f'Saving {out_metadata}')
with open(out_metadata, "w") as outfile_metadata: 
    header = '\t'.join(('id', 'pos', 'aa', 'kinases', 'species', 'source'))
    outfile_metadata.write(header + '\n')
    # Scan TSV files in the different databases
    for entry in os.scandir(DB_DUMPS):
        if not entry.is_dir() or "db_metadata.tsv" not in os.listdir(f'{DB_DUMPS}/{entry.name}'):
            continue
        tsv_path = f'{DB_DUMPS}/{entry.name}/db_metadata.tsv'
        print('Reading', tsv_path)
        with open(tsv_path, "r") as infile:
            reader = csv.reader(infile, delimiter='\t')
            for row in reader:
                row = '\t'.join(row)
                outfile_metadata.write(row + '\n')

out_sequences = f"{OUTDIR}/db_sequences.fasta"
print(f'Saving {out_sequences}')
with open(out_sequences, "w") as outfile_sequences:
    # Scan fasta files in the different databases
    for entry in os.scandir(DB_DUMPS):
        if not entry.is_dir() or not "db_sequences.fasta" in os.listdir(f'{DB_DUMPS}/{entry.name}'):
            continue
        fasta_path = f'{DB_DUMPS}/{entry.name}/db_sequences.fasta'
        print('Reading', fasta_path)
        with open(fasta_path, "r") as infile:
            reader = csv.reader(infile, delimiter=' ')
            for row in reader:
                row = ' '.join(row)
                outfile_sequences.write(row + '\n')
