import os
import os.path as path
import sys
import re

main_data_dir = path.dirname(__file__)

species_mapping = {
    'Canis familiaris':'Canis lupus familiaris',
    'Torpedo californica':'Tetronarce californica'
}
spec_set = set()
tot = 0
spec_dict = {}
for source in os.listdir(main_data_dir):
    data_dir = path.join(main_data_dir, source)
    seq_filename = path.join(data_dir, 'db_sequences.fasta')
    feat_filename = path.join(data_dir, 'db_metadata.tsv')
    if path.isdir(data_dir) and path.exists(seq_filename) and path.exists(feat_filename):
        with open(feat_filename, 'r') as feat_file:
            for line in feat_file:
                if not line.startswith('#'):
                    try:
                        ID, pos, aa, kinases, species, kin_species, _ = line.strip().split('\t') # ID, pos, aa, kinases, species, source
                    except ValueError as err:
                        raise ValueError(f'Error while parsing line "{line.strip()}" in file "{feat_filename}"')

                    if re.search(r'\([^\)]*\)', species):
                        species = re.search(r'^([^\(]*)(?:\([^\)]*\))', species).group(1).strip()

                    species = species_mapping.get(species, species)

                    spec_set.add(species)
                    tot += 1
                    if not spec_dict.get(ID):
                        spec_dict[ID] = set()
                    spec_dict[ID].add(species)

print('Total lines\t', tot)
print('Total IDs\t', len(spec_dict))
print('Unique species\t', len(spec_set))


print()
n_ID_diff = 0
n_diff = 0
rep_packs = set()
rep_set = set()
for ID, subset in spec_dict.items():
    subset.discard('NA')
    if len(subset) > 1:
        rep_packs.add(tuple(subset))
        rep_set.update(subset)
        n_diff += len(subset)
        n_ID_diff += 1

print('IDs with different names\t', n_ID_diff)
print('Total names not agreeing\t', n_diff)
print('Unique names not agreeing\t', len(rep_set))
print('Unique disagreements\t', len(rep_packs))
print(rep_set)
for pack in rep_packs: print(pack)