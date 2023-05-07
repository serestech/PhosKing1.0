
metadata = {}
species = {}
with open('../clean_data/metadata.tsv', 'r') as metadata_file:
    for line in metadata_file:
        if not line.startswith('#'):
            line_ = line.strip().split('\t')
            metadata[line_[0]] = line_[1:]
            if not species.get(line_[1]):
                species[line_[1]] = int(line_[2])
            species[line_[1]] += int(line_[2])

print(len(species))
for spe in sorted(species.keys(), key=species.get, reverse=True):
    #print(f"    '{spe}' : ('', '', ''),")
    #print('{:<35}\t{}'.format(spe, species.get(spe)))
    print(spe)
