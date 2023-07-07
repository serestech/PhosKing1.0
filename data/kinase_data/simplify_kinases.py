'''
Takes in the messy kinase names from the databases and cleans it up to generate the kinase classes according to
the kinbase classification tree (phylogenetic) of kinbase
'''
import sys
from pprint import pformat
from os.path import abspath, dirname
from collections import defaultdict
from kinase_filtering_data import *

HERE = abspath(dirname(__file__))

table = []
metadata_file = f'{HERE}/merged_db_metadata_kinase.tsv'
print(f'Reading metadata file {metadata_file}', file=sys.stderr)
with open(metadata_file, 'r') as metadata_table:
    for i, line in enumerate(metadata_table):
        if i == 0:
            continue
        
        entry = line.strip().split('\t')
        try:
            uniprot_id, position, residue, kinases, species, sources = entry
        except ValueError:
            print(f'Could not read line {i + 1}: incorrect nummber of fields ({len(entry)}): "{line.strip()}"')
            continue
        
        if kinases == 'NA':
            continue
        
        table.append(entry)

print(f'{len(table)} phosphorylations with annotated kinase', file=sys.stderr)

table_filtered = []
for entry in table:
    kinases = [kinase.strip() for kinase in entry[3].split(',')]
    kinases_prefiltered = set()
    for kinase in kinases:
        if kinase in kinases_to_ignore:
            new_kinase = 'NA'
        elif kinase in prefiltering_mapping.keys():
            new_kinase = prefiltering_mapping[kinase]
        else:
            new_kinase = kinase
        kinases_prefiltered.add(new_kinase)
    if 'NA' in kinases_prefiltered:
        kinases_prefiltered.remove('NA')
    if len(kinases_prefiltered) == 0:
        continue
    
    kinases_tree_category = set()
    for kinase in kinases_prefiltered:
        tree_kinases = kinase_category_mapping[kinase]
        for tree_kinase in tree_kinases.split(','):
            tree_kinase = tree_kinase.strip()
            if tree_kinase not in ('NA', ''):
                kinases_tree_category.add(tree_kinase)
    
    if len(kinases_tree_category) > 0:
        entry_prefilter = (*entry[:3], ','.join(kinases_tree_category), *entry[5:])
        table_filtered.append(entry_prefilter)

table = table_filtered

print(f'{len(table)} entries after filtering', file=sys.stderr)

out_table = f'{HERE}/kinase_metadata.tsv'
with open(out_table, 'w') as outfile:
    header = '#id\tpos\taa\tkinases\tkin_species\tsource\n'
    outfile.write(header)
    for entry in table:
        outfile.write('\t'.join(entry) + '\n')
print(f'Written metadata table at {out_table}', file=sys.stderr)

# Counts of cumulative categories
kinase_counts = defaultdict(int)
for entry in table:
    if entry[3].strip() == '':
        continue
    kinases = [kinase.strip() for kinase in entry[3].split(',')]
    kinases_in_entry = set()
    for kinase in kinases:
        categories = kinase.split(' ')
        cumulative_category = ''
        for category in categories:
            cumulative_category += f' {category}'
            cumulative_category = cumulative_category.strip()
            if cumulative_category == '':
                raise RuntimeError(f'Empty kinases {entry=} {entry[3]=} {kinases=}')
            kinases_in_entry.add(cumulative_category)
    
    for kinase in kinases_in_entry:
        kinase_counts[kinase] += 1
all_kinases = list(kinase_counts.keys())
all_kinases.sort(key=lambda kin:kin.lower())

# print(*(f'{kin}\t{kinase_counts[kin]}' for kin in all_kinases), sep='\n')

kinases_to_keep = set()
with open(f'{HERE}/kinases_to_keep.txt', 'r') as kinases_to_keep_file:
    for line in kinases_to_keep_file:
        kinases_to_keep.add(line.strip())
        if line.strip() not in all_kinases:
            raise RuntimeError(f'Kinase "{line.strip()}" not found in {all_kinases}')

i = 0
kinase_mapping = {}
for kinase in all_kinases:
    keep_kinase = False
    for kinase_2 in kinases_to_keep:
        is_pkl_family = kinase_2.startswith('PKL FJ FAM')  # Exception for this family because there's only a single kinase in it
        if kinase == kinase_2 or (kinase_2.startswith(kinase) and not is_pkl_family):
            keep_kinase = True
        
    if keep_kinase:
        kinase_mapping[kinase] = i
        i += 1
        print(kinase, kinase_counts[kinase], sep='\t')
kinase_mapping_rev = {i: kinase for kinase, i in kinase_mapping.items()}    

print(f'{len(kinase_mapping)} categories', file=sys.stderr)

outfile_contents = f'kinase_mapping = {pformat(kinase_mapping)}\n\nkinase_mapping_reverse = {pformat(kinase_mapping_rev)}\n'
outfile_path = abspath(f'{HERE}/../../utils/kinase_mapping.py')
with open(outfile_path, 'w') as outfile:
    outfile.write(outfile_contents)
    
print(f'Written kinase mapping at {outfile_path}', file=sys.stderr)
