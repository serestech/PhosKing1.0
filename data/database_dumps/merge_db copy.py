import os
import os.path as path
import re


main_data_dir = path.abspath(path.dirname(__file__))

print('Looking for databases and reading fasta files...')
print('Found databases', end=': ')
# Read sequences from fasta files
# Open any directory that has db_sequences.fasta and db_metadata.tsv
# Store sequences as {source : {ID : sequence, ...}, ...}
# The directory name corresponds to the source
# Filter sequence (isoforms and sequence consistency) by placing
#   them in the seq_blacklist set, to avoid them later 
seqs = {}
seq_blacklist = set()
for source in os.listdir(main_data_dir):
    data_dir = path.join(main_data_dir, source)
    seq_filename = path.join(data_dir, 'db_sequences.fasta')
    feat_filename = path.join(data_dir, 'db_metadata.tsv')
    if path.isdir(data_dir) and path.exists(seq_filename) and path.exists(feat_filename):
        print(source, end=', ')
        seqs[source] = {}
        with open(seq_filename, 'r') as seq_file:
            ID = None
            for line in seq_file:
                if line.startswith('>'):
                    # Filter isoforms -> will be removed anyway by homology reduction,
                    #   but this way we ensure to keep canonicals only
                    ID = line[1:].strip()
                    if '-' in ID:
                        seq_blacklist.add((ID, source))
                        ID = None
                elif ID:
                    seqs[source][ID] = line.strip()
                    ID = None

print('\nFiltering sequences...')
# For sequences with the same ID but not the same sequence, we keep the entries
#   that agree with the topmost source of the following improvised trustness ranking
order = {'UniProt':0, 'Phosphosite':1, 'PhosphoELM':2, 'dbPAF':3, 'PhosPhAt':4, 'EPSD':5)
for ID in set().union(*(d.keys() for d in seqs.values())):
    temp_seq = {}
    for source in seqs.keys():
        if seqs[source].get(ID):
            temp_seq[source] = seqs[source][ID]

    if len(temp_seq) > 1 and len(set(temp_seq.values())) > 1:
        prio = min(temp_seq.keys(), key=order.get)
        for source in temp_seq.keys():
            if temp_seq[prio] != temp_seq[source]:
                seq_blacklist.add((ID, source))

print('Reading features files...')
# Read phosphorylation entries from features tsv files
# Open same directories of previous fasta files
# Store entries as {source : {ID : [[pos, aa, spec, kin_spec, source], ...] , ...}, ...}
# Repeated entries (same ID and position) will be combined
#   joining kinases and sources in a comma-separated string
# Here only apply filtering within the same source: remove duplicates and check
#   concordancy with sequence
feats = {}
for source in seqs.keys():
    feat_filename = path.join(main_data_dir, source, 'db_metadata.tsv')
    feats[source] = {}
    with open(feat_filename, 'r') as feat_file:
        for line in feat_file:
            if not line.startswith('#'):
                try:
                    ID, pos, aa, kinases, species, kin_species, _ = line.strip().split('\t') # ID, pos, aa, kinases, species, source
                except ValueError as err:
                    raise ValueError(f'Error while parsing line "{line[:-1]}" in file "{feat_filename}"')
                if (ID, source) not in seq_blacklist:
                    add = True
                    # Check that the aa is actually in the sequence (against possible remaining isoforms)
                    if seqs[source][ID][int(pos)-1] != aa:
                        add = False
                    elif feats[source].get(ID): 
                        for entry in feats[source][ID]:
                            if entry[0] == pos:
                                # If the entry is completely repeated
                                if entry == [pos, aa, kinases, species, kin_species]:
                                    add = False
                                # If the entry is repeated but the kinases are different
                                elif entry[2] != kinases:
                                    add = False
                                    if entry[2] == 'NA' and kinases != 'NA':
                                        entry[2] = kinases
                                    elif entry[2] != 'NA' and kinases != 'NA':
                                        entry[2] = ','.join(set(kinases.split(',')).union(entry[2].split(',')))
                        
                    if add:
                        if not feats[source].get(ID):
                            feats[source][ID] = []
                        feats[source][ID].append([pos, aa, kinases, species, kin_species])

print('Merging sequences...')
# Merge final sequence dict: {ID : seq, ...}
# Assert just in case that different sources agree on sequences
merged_seqs = {}
for source in seqs.keys():
    for ID, seq in seqs[source].items():
        if (ID, source) not in seq_blacklist:
            if not merged_seqs.get(ID):
                merged_seqs[ID] = seq
            else:
                assert seq == merged_seqs.get(ID)

print('Merging phosphorylation entries...')
# Merge final features dict: {ID : {pos : [aa, kins, specs, kin_spec, sources], ...}, ...}
# Filter repeated entries (combining kinases and sources)
# The species names are a mess between databases, this remains unhandled by now 
merged_feats = {}
for source in feats.keys():
    for ID, entries in feats[source].items():
        if (ID, source) not in seq_blacklist:
            if not merged_feats.get(ID):
                merged_feats[ID] = {}
            for pos, aa, kinases, species, kin_species in entries:
                # Attempt to simplify some species names so the agree:
                #   remove common name in parenthesis
                if re.search(r'\([\w\s/\-]+\)', species):
                    species = species[:re.search(r'\([\w\s/\-]+\)', species).span()[0]].strip()
                # If it is the first found, assign it
                if not merged_feats[ID].get(pos):
                    merged_feats[ID][pos] = [aa, kinases, species, kin_species, source]
                # If there is already an entry for that ID-position, combine
                else:
                    curr_entry = merged_feats[ID][pos]
                    # Not-necessary but healthy sanity checks:
                    assert curr_entry[0] == aa
                    assert curr_entry[4] != source
                    # Combine kinases
                    if curr_entry[1] != kinases:
                        if curr_entry[1] == 'NA' and kinases != 'NA':
                            curr_entry[1] = kinases
                        elif curr_entry[1] != 'NA' and kinases != 'NA':
                            curr_entry[1] = ','.join(set(kinases.split(',')).union(curr_entry[1].split(',')))
                    # The species in Phosphosite are common names, avoid if possible
                    if curr_entry[2] != species and species != 'NA' and source != 'Phosphosite':
                        curr_entry[2] = species
                    # Phosphosite is special and has specified the species of the kinase in the experiment
                    if curr_entry[3] != kin_species and curr_entry[3] == 'NA':
                        curr_entry[3] = kin_species
                    curr_entry[4] = ','.join((curr_entry[4], source))


print('Writing merged_sequences.fasta...')
# Finally, write merged files following same format than in input:
# fasta: UniProt ID (preceded by space) as header,
#   and sequence in one line (next line)
# tsv: UniProt ID, position, residue, comma-separated kinases, species,
#   species of the kinase used in experiments (Phosphosite only),
#   comma-separated sources with that entry
# Missing values in kinase and species are indicated as "NA"
with open(path.join(main_data_dir, 'merged_sequences.fasta'), 'w') as seq_outfile:
    for ID, seq in merged_seqs.items():
        seq_outfile.write(f'> {ID}\n{seq}\n')

print('Writing merged_features.tsv...')
with open(path.join(main_data_dir, 'merged_features.tsv'), 'w') as feat_outfile:
    feat_outfile.write('#UniProt-ID\tposition\tresidue\tkinases\tspecies\tkinase-species\tsources\n')
    for ID in merged_feats:
        for pos, entry in merged_feats[ID].items():
            aa, kinases, species, kin_species, source = entry
            feat_outfile.write(f'{ID}\t{pos}\t{aa}\t{kinases}\t{species}\t{kin_species}\t{source}\n')

print('Databases merged!')