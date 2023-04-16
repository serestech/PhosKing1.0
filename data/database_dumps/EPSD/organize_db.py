# TODO: CROSS-REFERENCE WITH UNIPROT TO GET ORGANISMS!!!

# Read data from EPSD_dump.txt and save it in a list called table
table = []
with open('EPSD_dump.txt', 'r') as infile:
    for line in infile:
        entry = line.strip().split('\t')
        entry = [field.strip() for field in entry]
        if len(entry) != 6:
            print(f"Skipping line '{line}'. Does not seem to have 6 fields")
            continue
            
        table.append(entry)

assert len(set([len(entry) for entry in table])) == 1, "Some rows are missing values"

# Read data from the fasta to store it in a dictionary
# Ids in that fasta are from epsd not from uniprot
fasta_file = "EPSD_dump.fasta"
sequences_fasta_dump = {}

with open(fasta_file, 'r') as infile:
    # Initialize the ID and sequence variables
    seq_id = ''
    seq = ''
    
    for line in infile:
        if line.startswith('>'):
            # If the line starts with '>', it is a header line
            # Save the ID and start a new sequence
            seq_id = line[1:].strip().split('|')[0]
            seq = ''
        else:
            # If the line does not start with '>', it is a sequence line
            # Append the line to the current sequence
            seq += line.strip()
            # Save the ID-sequence pair to the dictionary
            sequences_fasta_dump[seq_id] = seq
            
metadata_entries = []
sequences = {}

for entry in table[1:]:
    epsd_id, uniprot_id, aa, position, source, reference = entry

    if uniprot_id not in sequences.keys():
        sequences[uniprot_id] = sequences_fasta_dump[epsd_id] # Sequences for fasta file

    kinases = 'NA'
    species = 'NA'

    entry = (uniprot_id, position, aa, kinases, species, 'NA', 'EPSD')
    ids = (epsd_id, uniprot_id)

    metadata_entries.append(entry) # Metadata entries for the tsv file

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
        outfile_fasta.write(f'>{id}\n{sequence}\n')

print(f'{len(metadata_entries)} total entries')