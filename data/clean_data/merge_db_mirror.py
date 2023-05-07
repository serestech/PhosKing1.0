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
def merge_sequences(seqs, blacklist={}, seq_prio={}):
    merged_seqs = {}
    rev_merged_seqs = {}
    seq_mirror = {}
    for ID in seqs.keys():
        if len(seqs[ID]) > 1:
            ref = seqs[ID][max(seqs[ID].keys(), key=lambda x: seq_prio.get(x, -1))]
            if ref not in rev_merged_seqs.keys():
                merged_seqs[ID] = ref
                rev_merged_seqs[ref] = ID
            else:
                seq_mirror[ID] = rev_merged_seqs[ref]
            for source in seqs[ID].keys():
                if seqs[ID][source] != ref:
                    blacklist.add((ID, source))
        else:
            seq = list(seqs[ID].values()).pop()
            if seq not in rev_merged_seqs.keys():
                merged_seqs[ID] = seq
                rev_merged_seqs[seq] = ID
            else:
                seq_mirror[ID] = rev_merged_seqs[seq]

    return merged_seqs, seq_mirror


# Read phosphorylation entries from features tsv files
# Open same directories of previous fasta files
# Store entries as {source : {ID : [[pos, aa, spec, kin_spec, source], ...] , ...}, ...}
def load_metadata(main_data_dir, sources, merged_seqs, blacklist, spec_mapping={}, spec_prio={}, seq_mirror={}):
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
                        print(f'Error while parsing line "{(line.strip())}" in file "{feat_filename}"')
                        sys.exit(1)

                    m_ID = seq_mirror.get(ID, ID)
                    if (ID, source) not in blacklist and m_ID in merged_seqs and merged_seqs[m_ID][int(pos)-1] == aa:
                        if not feats.get(m_ID):
                            feats[m_ID] = {}

                        if spec != 'NA':
                            if re.search(r'\([^\)]*\)', spec):
                                spec = re.search(r'^([^\(]*)(?:\([^\)]*\))', spec).group(1).strip()
                            spec = spec_mapping.get(spec, spec)

                        if kinases == 'NA':
                            kin_set = set()
                        else:
                            kin_set = set(kinases.split(','))

                        if not metadata.get(m_ID):
                            metadata[m_ID] = [spec, 0, 0, len(merged_seqs[m_ID]), {source,}, {ID,}]     # Species, n_entries, n_entries_w_kinase, sources, mirrors
                        else:
                            metadata[m_ID][4].add(source)
                            metadata[m_ID][5].add(ID)

                        if feats[m_ID].get(pos):
                            if kin_set:
                                if not feats[m_ID][pos][1] and kin_set:
                                    metadata[m_ID][2] += 1
                                feats[m_ID][pos][1].update(kin_set)
                            if source == 'Phosphosite' and kin_spec != 'NA':
                                feats[m_ID][pos][2] = kin_spec
                            feats[m_ID][pos][3].add(source)
                            if spec != 'NA' and metadata[m_ID][0] != spec:
                                if metadata[m_ID][0] != 'NA' or spec_prio.get(source, -1) > max(spec_prio.get(s, -1) for s in feats[m_ID][pos][3]):
                                    metadata[m_ID][0] = spec
                            
                        else:
                            feats[m_ID][pos] = [aa, kin_set, kin_spec, {source,}]
                            metadata[m_ID][1] += 1
                            if kin_set:
                                metadata[m_ID][2] += 1            

    return feats, metadata


def write_fasta(seqs, feats, out_name):
    print(f'Saving FASTA file at {os.path.abspath(out_name)}')
    with open(out_name, 'w') as fasta:
        for ID, seq in seqs.items():
            if ID in feats.keys():
                fasta.write(f'>{ID}\n')
                fasta.write(f'{seq}\n')


def write_features(feats, metadata, out_name):
    print(f'Saving features file at {os.path.abspath(out_name)}')
    with open(out_name, 'w') as tsv:
        tsv.write('#UniProt-ID\tposition\tresidue\tkinases\tspecies\tkinase-species\tsources\n')
        for ID in feats:
            for pos, (aa, kins, kin_spec, sources) in feats[ID].items():
                if kins:
                    kins_str = ','.join(kins)
                else:
                    kins_str = 'NA'
                sources_str = ','.join(sources)
                entry = (ID, pos, aa, kins_str, kin_spec, sources_str)
                tsv.write('\t'.join(entry) + '\n')


def write_metadata(metadata, out_name):
    print(f'Saving metadata file at {os.path.abspath(out_name)}')
    with open(out_name, 'w') as tsv:
        tsv.write('#UniProt-ID\tspecies\tn_entries\tn_entries_with_kinase\tprot_length\tsources\tmirrors\n')
        for ID, (spec, n, n_kin, length, sources, m_IDs) in metadata.items():
            sources_str = ','.join(sources)
            mirrors_str = ','.join(m_IDs)
            entry = (ID, spec, str(n), str(n_kin), str(length), sources_str, mirrors_str)
            tsv.write('\t'.join(entry) + '\n')


if __name__ == '__main__':
    main_data_dir = path.abspath(path.dirname(__file__) + '/../database_dumps')
    print('Looking for databases and loading fasta files...')
    seqs, sources, blacklist = load_fastas(main_data_dir)
    print(f'Found {len(sources)} databases: {", ".join(sources)}')
    if len(sources) == 0:
        raise RuntimeError

    print('Merging database sequences...')
    seq_prio = {'UniProt':8, 'Phosphosite':6, 'PhosphoELM':4, 'dbPAF':3, 'PhosPhAt':1, 'EPSD':0}
    merged_seqs, seq_mirror = merge_sequences(seqs, blacklist, seq_prio)
    
    print('Loading metadata files...')
    spec_mapping = {
        'Canis familiaris':'Canis lupus familiaris',
        'Torpedo californica':'Tetronarce californica'
        }
    spec_prio = {'UniProt':8, 'Phosphosite':1, 'PhosphoELM':4, 'dbPAF':3, 'PhosPhAt':2, 'EPSD':0}
    feats, metadata = load_metadata(main_data_dir, sources, merged_seqs, blacklist, spec_mapping, spec_prio, seq_mirror)

    here = path.dirname(path.abspath(__file__))

    print('Writing merged files...')
    write_fasta(merged_seqs, feats, here + '/sequences.fasta')
    write_features(feats, metadata, here + '/features.tsv')
    write_metadata(metadata, here + '/metadata.tsv')

    print('Databases merged!')