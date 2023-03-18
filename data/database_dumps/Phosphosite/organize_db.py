from sys import argv, stderr, path
from os.path import dirname, realpath
print(argv[0])

# Hacky thing to import stuff relative to file directory
path.append(realpath(dirname(__file__) + '/../../data_utils'))
from fasta_utils import read_fasta

data = {}
with open('phosphosite_dump/Kinase_Substrate_Dataset.tsv', 'r') as kinase_dataset:
    fields = None
    for line in kinase_dataset:
        line = line[:-1]
        if fields is None:
            fields = line.split('\t')
            print(f'{len(fields)} fields in kinase dataset file: {" ".join(fields)}')
            continue
        
        linesplit = line.split('\t')
        if len(linesplit) != len(fields):
            print(f'Skipping line \'{line}\'. Has different amount of fields', file=stderr)
            continue
        
        phosphorylation_data = {field: value for field, value in zip(fields, linesplit)}
        
        seq, aa = phosphorylation_data['SUB_ACC_ID'], phosphorylation_data['SUB_MOD_RSD']
        
        phosphorylation = (seq, aa)
        
        if phosphorylation in data.keys():
            kinases = data[phosphorylation]['kinase'].split(',')
            kinases = set(kinases)
            kinases.add(phosphorylation_data['KINASE'])
            data[phosphorylation]['kinase'] = ','.join(kinases)
        else:
            data[phosphorylation] = {'kinase': phosphorylation_data['KINASE'],
                                    'species': phosphorylation_data['SUB_ORGANISM'],
                                    'kin_species': phosphorylation_data['KIN_ORGANISM']}

print(f'{len(data)} phosphorylations with annotated kinase')

with open('phosphosite_dump/Phosphorylation_site_dataset.tsv', 'r') as phospho_dataset:
    already_in_kinase = 0
    fields = None
    for line in phospho_dataset:
        line = line[:-1]
        if fields is None:
            fields = line.split('\t')
            print(f'{len(fields)} fields in phosphorylation site dataset file: {" ".join(fields)}')
            continue
        
        linesplit = line.split('\t')
        if len(linesplit) != len(fields):
            print(f'Skipping line \'{line}\'. Has different amount of fields', file=stderr)
            continue
    
        phosphorylation_data = {field: value for field, value in zip(fields, linesplit)}
        
        seq, aa = phosphorylation_data['ACC_ID'], phosphorylation_data['MOD_RSD'][:-2]  # [:-2] to remove the '-p'
        phosphorylation = (seq, aa)
        
        if phosphorylation in data.keys():
            already_in_kinase += 1
            continue
        
        data[phosphorylation] = {'kinase': 'NA', 
                                 'species': phosphorylation_data['ORGANISM'], 
                                 'kin_species': 'NA'}

print(f'Skipped {already_in_kinase} phosphorylations of the phosphorylation site dataset (already in kinase dataset)')
print(f'{len(data)} phosphorylations after reading phosphorylation site dataset')

fasta = read_fasta('phosphosite_dump/Phosphosite_seq.fasta', format=dict)
sequences = set(phospho[0] for phospho in data.keys())

print(f'Read {len(fasta)} sequences from FASTA file')
print(f'Writing new FASTA file of relevant sequences')
with open('db_sequences.fasta', 'w') as out_fasta:
    skipped = []
    added = set()
    for seq_id, seq in fasta.items():
        seq_id = seq_id.split('|')[-1]
        if seq_id not in sequences:
            skipped.append(seq_id)
            continue
        out_fasta.write(f'> {seq_id}\n{seq}\n')
        added.add(seq_id)

print(f'Skipped {len(skipped)} sequences (not in metadata tables)')

missing = set()
for seq_id in sequences:
    if seq_id not in added:
        missing.add(seq_id)

if len(missing) > 0:
    print(f'{len(missing)} sequences missing in the fasta file. First 10: {" ".join(list(missing)[:10])}. Removing phosphorylations of these proteins')

phosphorylations = tuple(data.keys())
removed = 0
for phosphorylation in phosphorylations:
    seq, _ = phosphorylation
    if seq in missing:
        data.pop(phosphorylation)
        removed += 1

if removed > 0:
    print(f'Removed {removed} phosphorylations due to missing sequence in FASTA')

ORGANISM_MAPPING = {
    'human': 'Homo sapiens',
    'mouse': 'Mus musculus',
    'cow': 'Bos taurus',
    'frog': 'Xenopus laevis',
    'chicken': 'Gallus gallus',
    'rabbit': 'Oryctolagus cuniculus',
    'rat': 'Rattus norvegicus',
    'dog': 'Canis lupus familiaris',
    'hamster': 'Cricetulus griseus',
    'pig': 'Sus scrofa',
    'horse': 'Equus caballus',
    'guinea pig': 'Cavia porcellus',
    'cat': 'Felis catus',
    'water buffalo': 'Bubalus bubalis',
    'goat': 'Capra hircus',
    'fruit fly': 'Drosophila melanogaster',
    'sheep': 'Ovis aries',
    'green monkey': 'Chlorocebus aethiops',
    'quail': 'Coturnix japonica',
    'torpedo': 'Tetronarce californica'
}

for phosphorylation in data.keys():
    if data[phosphorylation]['species'] in ORGANISM_MAPPING.keys():
        data[phosphorylation]['species'] = ORGANISM_MAPPING[data[phosphorylation]['species']]
    if data[phosphorylation]['kin_species'] in ORGANISM_MAPPING.keys():
        data[phosphorylation]['kin_species'] = ORGANISM_MAPPING[data[phosphorylation]['kin_species']]

print('Writing metadata table')
with open('db_metadata.tsv', 'w') as out_tsv:
    out_tsv.write('#' + '\t'.join(('id', 'pos', 'aa', 'kinases', 'species', 'kin_species', 'source')) + '\n')
    for phosphorylation, phospho_data in data.items():
        seq, aa = phosphorylation
        aa, pos = aa[0], aa[1:]
        kinase, species, kin_species = phospho_data['kinase'], phospho_data['species'], phospho_data['kin_species']
        out_tsv.write('\t'.join((seq, pos, aa, kinase, species, kin_species, 'Phosphosite')) + '\n')
    


