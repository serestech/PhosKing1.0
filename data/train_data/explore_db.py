phos = {'Total':set()}
seqs = {'Total':set()}
with open('features.tsv') as infile:
    for line in infile:
        if not line.startswith('#'):
            ID, pos, aa, kin, kin_spec, sources_ = line.strip().split('\t')
            sources = sources_.split(',')
            for source in sources:
                if source not in phos.keys():
                    phos[source] = set()
                    seqs[source] = set()
                phos[source].add((ID, pos))
                seqs[source].add(ID)
                phos['Total'].add((ID, pos))
                seqs['Total'].add(ID)

print('Source', 'Phosphorylations', 'Sequences', sep='\t')
for source in phos.keys():
    print(source, len(phos[source]), len(seqs[source]), sep='\t')
