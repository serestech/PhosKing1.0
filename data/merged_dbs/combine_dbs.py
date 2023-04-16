'''
Merges all the databases into 1 TSV and 1 FASTA. Doesn't do any kind of processing a part from that.
Identical sequences with different IDs and duplicate IDs must be handleded later on.
'''
from os import listdir
from os.path import abspath, dirname

this_dir = abspath(dirname(__file__))

DB_DUMPS = abspath(f'{this_dir}/../database_dumps')
OUT_DIR = this_dir

def get_table(database):
    '''
    Returns header and list of entries for the db_metadata.tsv of a database.
    '''
    tsv_path = f'{DB_DUMPS}/{database}/db_metadata.tsv'
    print(f'Getting table for {database}: {tsv_path}')
    tsv_file = open(tsv_path, 'r')
    
    entries = []
    for i, line in enumerate(tsv_file):
        if i == 0:
            header = line.strip()
            fields = header.split('\t')
            n_fields = len(fields)
            print(f'Table has {n_fields} fields: {", ".join(fields)}')
            continue
        
        entry = line.strip().split('\t')
        assert len(entry) == n_fields
        entries.append(entry)
        
    tsv_file.close()
    
    print(f'{len(entries)} total entries in {database}')
    
    return header, entries

def get_fasta(database):
    '''
    Returns the raw FASTA file for a database (with trailing newline)
    '''
    fasta_path = f'{DB_DUMPS}/{database}/db_sequences.fasta'
    print(f'Getting FASTA for {database}: {fasta_path}')
    with open(fasta_path, 'r') as fasta_file:
        fasta = f'{fasta_file.read().strip()}\n'
    print(f'{fasta.count(">")} sequences in {database}')
    return fasta

databases = listdir(DB_DUMPS)
print(f'Found {len(databases)} databases: {", ".join(databases)}')

merged_table = []
fastas = []
for i, database in enumerate(databases):
    print(f'--- Processing database {database} ---')
    header, entries = get_table(database)
    if i == 0:
        final_header = header
        print(f'Header of output file will be "{final_header}"')
    if header != final_header:
        print(f'WARNING: Found inconsistent header in {database}.\nSelected header:{final_header}\n{database} header:\n{header}')
    merged_table.extend(entries)
    
    fasta = get_fasta(database)
    fastas.append(fasta)
    print()

out_metadata = f'{OUT_DIR}/merged_db_metadata.tsv'
print(f'Writing output metadata table {out_metadata}')
with open(out_metadata, 'w') as outfile_metadata:
    outfile_metadata.write(f'{header}\n')
    for entry in merged_table:
        line = '\t'.join(entry) + '\n'
        outfile_metadata.write(line)

out_fasta = f'{OUT_DIR}/merged_db_sequences.fasta'
print(f'Writing output FASTA file {out_fasta}')
with open(out_fasta, 'w') as outfile_fasta:
    for fasta in fastas:
        outfile_fasta.write(fasta)
