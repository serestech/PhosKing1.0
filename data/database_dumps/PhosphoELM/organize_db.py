from sys import argv, stderr
print(argv[0])

table = []
with open('phosphoELM_dump.tsv', 'r') as infile:
    for line in infile:
        table.append(line.strip().split('\t'))

assert len(set([len(entry) for entry in table])) == 1, "Some rows are missing values"

sequences = {}
metadata_entries = []

all_kinases = []

def clean_kinase(kinase: str):
    if kinase == 'kinases':
        return 'NA'
    
    if kinase.endswith('_group'):
        return kinase.replace('_group', '')
    
    if kinase.endswith('_drome'):
        return kinase.replace('_drome', '').replace('_group', '')
    
    if kinase.endswith('_Caeel'):
        return kinase.replace('_Caeel', '').replace('_group', '')

    return kinase

for entry in table[1:]:
    entry = [field.strip() for field in entry]
    acc, sequence, position, code, pmids, kinases, source, species, entry_date = entry
    
    if sequence[int(position) - 1] != code:
        print(f'Warning: Skipping phosphorylation in sequence {acc} position {position}. Reported aa ({code}) is different than sequence aa ({sequence[int(position) - 1]})', file=stderr)
    
    # Save sequence if not seen yet
    if acc not in sequences.keys():
        sequences[acc] = sequence
    
    
    if kinases.strip() != '':
        kinases = clean_kinase(kinases)
    else:
        kinases = 'NA'
    
    all_kinases.append(kinases)
    
    entry = (acc, position, code, kinases, species, 'NA', 'PhosphoELM')
    metadata_entries.append(entry)

assert len(set([entry[0] for entry in metadata_entries])) == len(sequences), "Found different number of saved IDs and saved sequences"

print("Writing metadata table...")
with open('db_metadata.tsv', 'w') as outfile_metadata:
    header = '\t'.join(('id', 'pos', 'aa', 'kinases', 'species', 'kin_species', 'source'))
    outfile_metadata.write('#' + header + '\n')
    for entry in metadata_entries:
        tsv_row = "\t".join(entry)
        outfile_metadata.write(tsv_row + '\n')

print("Writing FASTA sequences...")
with open('db_sequences.fasta', 'w') as outfile_fasta:
    for id, sequence in sequences.items():
        outfile_fasta.write(f'> {id}\n{sequence}\n')

print(f'{len(set(all_kinases))} kinases')
print(f'{len(metadata_entries)} total entries')
