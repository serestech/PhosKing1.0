import os
import os.path as path
import sys
import re


# Read sequences from fasta files
# Open any directory that has db_sequences.fasta and db_metadata.tsv
# Store sequences as {ID : {source : sequence, ...}, ...}
# The directory name corresponds to the source
# Filter sequence (isoforms and sequence consistency) by placing
#   them in the blacklist set, to avoid them later 
def load_fastas(main_data_dir):
    seqs = {}
    blacklist = set()
    sources = []
    for source in os.listdir(main_data_dir):
        data_dir = path.join(main_data_dir, source)
        seq_filename = path.join(data_dir, 'db_sequences.fasta')
        feat_filename = path.join(data_dir, 'db_metadata.tsv')
        if path.isdir(data_dir) and path.exists(seq_filename) and path.exists(feat_filename):
            sources.append(source)
            with open(seq_filename, 'r') as seq_file:
                ID = None
                for line in seq_file:
                    if line.startswith('>'):
                        ID = line[1:].strip()
                        # Filter isoforms -> will be removed anyway by homology reduction,
                        #   but this way we ensure to keep canonicals only
                        if '-' in ID:
                            blacklist.add((ID, source))
                            ID = None
                    elif ID:
                        if not seqs.get(ID):
                            seqs[ID] = {}
                        seqs[ID][source] = line.strip()
                        ID = None
    
    return seqs, sources, blacklist


# For sequences with the same ID but not the same sequence, we keep the entries
#   that agree with the topmost source of the following improvised trustness ranking
def merge_sequences(seqs, blacklist, seq_prio):
    merged_seqs = {}
    for ID in seqs.keys():
        if len(seqs[ID]) > 1:
            ref = seqs[ID][max(seqs[ID].keys(), key=lambda x: seq_prio.get(x, -1))]
            merged_seqs[ID] = ref
            for source in seqs[ID].keys():
                if seqs[ID][source] != ref:
                    blacklist.add((ID, source))
        else:
            seq = list(seqs[ID].values()).pop()
            merged_seqs[ID] = seq

    return merged_seqs


print('Reading features files...')
# Read phosphorylation entries from features tsv files
# Open same directories of previous fasta files
# Store entries as {source : {ID : [[pos, aa, spec, kin_spec, source], ...] , ...}, ...}
def load_metadata(main_data_dir, sources, merged_seqs, blacklist):
    feats = {}
    metadata = {}
    for source in sources:
        feat_filename = path.join(main_data_dir, source, 'db_metadata.tsv')
        with open(feat_filename, 'r') as feat_file:
            for line in feat_file:
                if not line.startswith('#'):
                    try:
                        ID, pos, aa, kinases, spec, kin_spec, _ = line.strip().split('\t') # ID, pos, aa, kinases, species, source
                    except ValueError:
                        raise ValueError(f'Error while parsing line "{line[:-1]}" in file "{feat_filename}"')

                    if (ID, source) not in blacklist and ID in merged_seqs and merged_seqs[ID][int(pos)-1] == aa:
                        if not feats.get(ID):
                            feats[ID] = {}
                        if not metadata.get(ID):
                            metadata[ID] = ['NA', 0, 0, set()]     # Species, n_entries, n_entries_w_kinase, sources

                        if kinases == 'NA':
                            kin_set = set()
                        else:
                            kin_set = set(kinases.split(','))

                        if re.search(r'\([^\)]*\)', spec):
                            spec = re.search(r'^([^\(]*)(?:\([^\)]*\))', spec).group(1).strip()
                        spec = spec_mapping.get(spec, spec)

                        if feats[ID].get(pos):
                            if not feats[ID][pos][1] and kin_set:
                                metadata[ID][2] += 1
                            feats[ID][pos][1].update(kin_set)
                            if spec != 'NA' and metadata[ID][0] != spec:
                                if spec_prio.get(source, -1) > max(spec_prio.get(s, -1) for s in feats[ID][pos][3]):
                                    metadata[ID][0] = spec
                            if source == 'Phosphosite':
                                feats[ID][pos][2] = kin_spec
                            feats[ID][pos][3].add(source)
                            
                        else:
                            feats[ID][pos] = [aa, kin_set, kin_spec, {source,}]
                            metadata[ID][1] += 1
                            if kin_set:
                                metadata[ID][2] += 1
                            metadata[ID][3].add(source)

    return feats, metadata


def write_fasta(seqs, out_name):
    with open(out_name, 'w') as fasta:
        for ID, seq in seqs.items():
            fasta.write(f'> {ID}\n')
            fasta.write(f'{seq}\n')


def write_features(feats, metadata, out_name):
    with open(out_name, 'w') as tsv:
        tsv.write('#UniProt-ID\tposition\tresidue\tkinases\tspecies\tkinase-species\tsources\n')
        for ID in feats:
            for pos, (aa, kins, kin_spec, sources) in feats[ID].items():
                if kins:
                    kins_str = ','.join(kins)
                else:
                    kins_str = 'NA'
                sources_str = ','.join(sources)

                entry = [ID, pos, aa, kins_str, metadata[ID][0], kin_spec, sources_str]
                line = '\t'.join(entry) + '\n'
                tsv.write(line)


def write_metadata(metadata, out_name):
    with open(out_name, 'w') as tsv:
        tsv.write('#UniProt-ID\tspecies\tn_entries\tn_entries_with_kinase\tsources\n')
        for ID, (spec, n, n_kin, sources) in metadata.items():
            sources_str = ','.join(sources)
            entry = [ID, spec, str(n), str(n_kin), sources_str]
            line = '\t'.join(entry) + '\n'
            tsv.write(line)




if __name__ == '__main__':
    main_data_dir = path.dirname(__file__)
    print('Looking for databases and loading fasta files...')
    seqs, sources, blacklist = load_fastas(main_data_dir)
    print('Found databases: {}'.format(', '.join(sources)))

    print('Merging database sequences...')
    seq_prio = {'UniProt':8, 'Phosphosite':6, 'PhosphoELM':4, 'dbPAF':3, 'PhosPhAt':1, 'EPSD':0}
    merged_seqs = merge_sequences(seqs, blacklist, seq_prio)
    
    print('Loading metadata files...')
    spec_mapping = {
        'Canis familiaris':'Canis lupus familiaris',
        'Torpedo californica':'Tetronarce californica'
        }
    spec_prio = {'UniProt':8, 'Phosphosite':1, 'PhosphoELM':4, 'dbPAF':3, 'PhosPhAt':2, 'EPSD':0}
    feats, metadata = load_metadata(main_data_dir, sources, merged_seqs, blacklist)

    print('Writing merged files...')
    write_fasta(merged_seqs, 'temp_seqs.fasta')
    write_features(feats, metadata, 'temp_feats.tsv')
    write_metadata(metadata, 'temp_metadata.tsv')

    print('Databases merged!')