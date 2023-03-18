from sys import argv, stderr
print(argv[0])

table = []
with open('dbPAF_dump.tsv', 'r') as infile:
    for line in infile:
        entry = line.strip().split('\t')
        entry = [field.strip() for field in entry]
        if len(entry) != 7:
            print(f"Skipping line '{line}'. Does not seem to have 7 fields", file=stderr)
            continue
            
        table.append(entry)

assert len(set([len(entry) for entry in table])) == 1, "Some rows are missing values"

sequences = {}
metadata_entries = []


for entry in table[1:]:
    dbPAF_id, uniprot_id, position, aa, sequence, species, pmids = entry
    
    if sequence[int(position) - 1] != aa:
        print(f'Warning: Skipping phosphorylation in sequence {uniprot_id} position {position}. Reported aa ({aa}) is different than sequence aa ({sequence[int(position) - 1]})', file=stderr)
    
    # Save sequence if not seen yet
    if uniprot_id not in sequences.keys():
        sequences[uniprot_id] = sequence
    
    kinases = 'NA'
    
    entry = (uniprot_id, position, aa, kinases, species, 'NA', 'dbPAF')
    metadata_entries.append(entry)

assert len(set([entry[0] for entry in metadata_entries])) == len(sequences), "Different number of saved sequence IDs and saved sequences"

print("Writing metadata table...")
with open('db_metadata.tsv', 'w') as outfile_metadata:
    header = '#' + '\t'.join(('id', 'pos', 'aa', 'kinases', 'species', 'kin_species', 'source'))
    outfile_metadata.write(header + '\n')
    for entry in metadata_entries:
        tsv_row = "\t".join(entry)
        outfile_metadata.write(tsv_row + '\n')

print("Writing FASTA sequences...")
with open('db_sequences.fasta', 'w') as outfile_fasta:
    for id, sequence in sequences.items():
        outfile_fasta.write(f'> {id}\n{sequence}\n')

print(f'{len(metadata_entries)} total entries')
