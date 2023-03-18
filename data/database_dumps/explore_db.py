import os
import os.path as path

IDs = {}
folders = []
c = 0
seq_IDs = set()
for folder in os.listdir():
    if path.isdir(folder) and path.exists(path.join(folder, 'db_sequences.fasta')):
        folders.append(folder)
        IDs[folder] = set()
        with open(path.join(folder, 'db_sequences.fasta'), 'r') as fasta_file:
            for line in fasta_file:
                if line.startswith('>'):
                    if '-' in line:
                        c += 1
                    else:
                        seq_IDs.add(line.strip()[2:])
                        IDs[folder].add(line.strip()[2:])                

print('IDs in common (excl. isoforms)')
print('\t'.join(['        ']+folders))
for f1 in folders:
    print('{:8}'.format(f1), end='\t')
    for f2 in folders:
        print(len(IDs[f1].intersection(IDs[f2])), end='\t')
    print('')
print(c, 'isoforms ignored\n')

rep_IDs = set()
for ID in seq_IDs:
    c = 0
    for folder in folders:
        if ID in IDs[folder]:
            c += 1
    if c >= 2:
        rep_IDs.add(ID)

print('Unique IDs:', len(seq_IDs), '; IDs present in 2 or more DBs:', len(rep_IDs))

seqs = {}
with open('db_sequences.fasta','r') as fasta_file:
    for line in fasta_file:
        if line.startswith('>'):
            if line.strip()[2:] in rep_IDs:
                ID = line.strip()[2:]
            else:
                ID = None
        elif ID:
            seq = line.strip()
            if seqs.get(ID):
                seqs[ID].append(seq)
            else:
                seqs[ID] = [seq]
            ID = None

c = 0
for ID, seq_list in seqs.items():
    if len(set(seq_list)) != 1:
        c += 1

print('Repeated prots with different seqs!:', c)
print('')

c = 0
tot_tot_entries = 0
phosphos = dict((folder, {}) for folder in folders)
with open('db_metadata.tsv', 'r') as metadata_file:
    for line in metadata_file:
        if not line.startswith('id'):
            tot_tot_entries += 1
            ID, pos, _, _, _, folder = line.strip().split('\t')
            if ID in rep_IDs:
                if phosphos[folder].get(ID):
                    if pos in phosphos[folder][ID]:
                        c += 1
                    phosphos[folder][ID].add(pos)
                else:
                    phosphos[folder][ID] = {pos,}



total = 0
agree = 0
partial_agree = 0
disagree = 0
total_unique = 0
lonely = 0
for ID in rep_IDs:
    n = 0
    poss = []
    unique = set()
    for folder in folders:
        if phosphos[folder].get(ID):
            poss.append(phosphos[folder].get(ID))
            unique.update(phosphos[folder].get(ID))
            total += len(phosphos[folder].get(ID))
    total_unique += len(unique)

    for pos in unique:
        c = 0
        for pos_set in poss:
            if pos in pos_set:
                c += 1
            else:
                disagree += 1

        if c == len(poss):
            agree += 1
        elif c > 1:
            partial_agree += 1
        if c == 1:
            lonely += 1


print('Total phosph. entries:                                 ', tot_tot_entries)
print(c, 'phosphorylations are redundant in the same DBs!')
print('Total phosph. entries of IDs present in 2 or more DBs: ', total)
print('    of which', total_unique, 'are unique entries:')
print('       of which are agreed by all DBs they are present:', agree)
print('       of which are agreed by some DBs they are in:    ', partial_agree)
print('       of which are only in one of the DBs:            ', lonely)
print('  *Times that a phosph. entry from a DB is not in other with that ID:', disagree)
