from sys import argv, stderr, path
print(argv[0])
import re
from os import system
from os.path import abspath, dirname
utils_folder = abspath(dirname(__file__) + '../../../../utils')
path.append(utils_folder)
from utils import read_fasta
from pprint import pprint
    
uniprot_fasta: dict = read_fasta('uniprot_sequences.fasta', format=dict)

table = []
with open('phosphat_dump.csv', 'r') as infile:
    print('Parsing table into rows')
    skipped_lines = []
    for line in infile:
        row = line.strip().split(',')
        if len(row) != 30:
            line = re.sub(r'".*?,.*?"', "somethingwithcomma", line)
        row = line.strip().split(',')
        if len(row) != 30:
            print(len(row))
            print(line)
            continue
        table.append(row)


assert len(set([len(entry) for entry in table])) == 1, f"Some rows are missing values"

print('Compiling metadata entries from rows')
phosphat_sequences = {}
metadata_entries = []
nulls = 0
seq_blacklist = set()
for entry in table[1:]:
    entry = [field.strip() for field in entry]
    sam_name,exp_name,db_id,info,species,region_sequence,modifiedsequence,pep_position,aa_code,modificationType,mcr,charge,No_Phosphorylations,PubMed,ppm,Treatment,CellCompartment,Instrument,Enrichment,Tissue,Cultivar,quant,sequence,Mascot_Score,XCorr,Inspect_Score,pepposprot,MaxQuant,ETD_Score,uniprot_acc = entry
    
    if modificationType != 'phos':
        continue
    
    if pepposprot == 'NULL':
        nulls += 1
        continue
    
    if sequence.endswith('*'):
        seq_blacklist.add(uniprot_acc)
        continue
        
    for aa in pepposprot.split('_'):
        code, position = aa[0], int(aa[1:])
        
        entry = (uniprot_acc, position, code, 'NA', species, 'NA', 'PhosPhAt')
        metadata_entries.append(entry)
    
    if uniprot_acc in phosphat_sequences.keys():
        if phosphat_sequences[uniprot_acc] != sequence:
            seq_blacklist.add(uniprot_acc)
    else:
        phosphat_sequences[uniprot_acc] = sequence


print(f'{len(seq_blacklist)} sequences blacklisted while reading table (incoherence in protein sequence)')
print(f'Skipped {nulls} phosphorylations where the reported amino acid was NULL')


print('Filtering and deciding sequence to use')
sequence_entries = {}
for entry in metadata_entries:
    uniprot_acc, position, code, _, species, _, _ = entry
    aa = (position, code)
    if uniprot_acc not in sequence_entries.keys():
        sequence_entries[uniprot_acc] = [aa]
    else:
        sequence_entries[uniprot_acc].append(aa)

not_in_uniprot = 0
different_coherency = 0
different_sequences = 0
states = {}
for seq_id, aa_list in sequence_entries.items():
    uniprot_coherent = True
    self_coherent = True
    
    if seq_id not in uniprot_fasta.keys():
        uniprot_coherent = False
        not_in_uniprot += 1
        
    if seq_id not in phosphat_sequences.keys():
        raise RuntimeError
    
    uniprot_seq = uniprot_fasta[seq_id] if seq_id in uniprot_fasta.keys() else ''
    phosphat_seq = phosphat_sequences[seq_id] if seq_id in phosphat_sequences.keys() else ''
    
    if '' not in (uniprot_seq, phosphat_seq) and uniprot_seq.upper() != phosphat_seq.upper():
        different_sequences += 1
    
    for position, code in aa_list:
        position = int(position)
        if position > len(phosphat_seq) or phosphat_seq[position - 1] != code:
            self_coherent = False
        
        if position > len(uniprot_seq) or uniprot_seq[position - 1] != code:
            uniprot_coherent = False
    
    if uniprot_coherent != self_coherent:
        different_coherency += 1
    
    assert seq_id not in states.keys()
    
    if seq_id in seq_blacklist:
        states[seq_id] = 'blacklist'
        continue
        
    if self_coherent:
        states[seq_id] = 'phosphat'
    elif uniprot_coherent:
        states[seq_id] = 'uniprot'
    else:
        states[seq_id] = 'blacklist'

stats = {
    'phosphat': 0,
    'uniprot': 0,
    'blacklist': 0,
}
for seq, state in states.items():
    stats[state] += 1
    
print(f'For {stats["phosphat"]} sequences, the PhosPhAt sequence will be used')
print(f'For {stats["uniprot"]} sequences, the UniProt sequence will be used')
print(f'{stats["blacklist"]} sequences blacklisted (conflicts could not be resolved)')
print(f'{different_sequences} sequences were different between UniProt and PhosPhAt')
print(f'{different_coherency} sequences had different coherency between UniProt and PhosPhAt')
print(f'(of {len(phosphat_sequences)} PhosPhAt sequences and {len(uniprot_fasta)} UniProt sequences)')

filtered_sequences = {}
filtered_entries = []
for entry in metadata_entries:
    uniprot_acc = entry[0]
    
    if states[uniprot_acc] == 'blacklist':
        continue
    
    filtered_entries.append(entry)
    
    if uniprot_acc in filtered_sequences.keys():
        continue
    
    if states[uniprot_acc] == 'phosphat':
        sequence = phosphat_sequences[uniprot_acc]
    elif states[uniprot_acc] == 'uniprot':
        sequence = uniprot_fasta[uniprot_acc]
    else:
        raise RuntimeError  # Avoid mistakenly overwrite sequence
    
    filtered_sequences[uniprot_acc] = sequence

if '--download_uniprot' in argv:
    print('Writing sequences list')
    with open('seq_lsit.txt', 'w') as seqlist_file:
        seqlist_file.write('\n'.join(phosphat_sequences.keys()))
        
    uniprot_donwloader = abspath(dirname(__file__) + '../../../../utils/download_uniprot_sequences.py')
    cmd = f'{uniprot_donwloader} '
    system(cmd)

assert len(set([entry[0] for entry in filtered_entries])) == len(filtered_sequences), "Found different number of saved IDs and saved sequences"

print("Writing metadata table...")
with open('db_metadata.tsv', 'w') as outfile_metadata:
    header = '\t'.join(('#id', 'pos', 'aa', 'kinases', 'species', 'kin_species', 'source'))
    outfile_metadata.write(header + '\n')
    for entry in filtered_entries:
        tsv_row = "\t".join([str(field) for field in entry])
        outfile_metadata.write(tsv_row + '\n')

print("Writing FASTA sequences...")
with open('db_sequences.fasta', 'w') as outfile_fasta:
    for id, sequence in phosphat_sequences.items():
        if id in seq_blacklist:
            continue
        outfile_fasta.write(f'>{id}\n{sequence}\n')

print(f'Wrote {len(filtered_entries)} entries')
