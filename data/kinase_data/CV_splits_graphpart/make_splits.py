from pprint import pprint

graphpart_out = 'graphpart_cv_splits.csv'

with open(graphpart_out, 'r') as graphpart_file:
    graphpart_lines = graphpart_file.read().splitlines()

seq_partitions = {}
for line in graphpart_lines[1:]:
    AC, _, _, cluster = line.split(',')
    seq_partitions[AC] = int(float(cluster))

def looped_index(i):
    return i % 10

splits = []
n = 0
while len(splits) < 10:
    new_split = {'test': set(), 
                 'valid': set(), 
                 'train': set()}
    for i in range(10):
        if i == 0:
            new_split['test'].add(looped_index(n + i))
        elif i == 1:
            new_split['valid'].add(looped_index(n + i))
        else:
            new_split['train'].add(looped_index(n + i))
    n += 1
    assert new_split['test' ].isdisjoint(new_split['valid'])
    assert new_split['test' ].isdisjoint(new_split['train'])
    assert new_split['valid'].isdisjoint(new_split['train'])
    splits.append(new_split)

def reverse_map(split):
    reverse_map = {}
    for cv_set, partitions in split.items():
        for partition in partitions:
            reverse_map[partition] = cv_set
    return reverse_map

splits = [reverse_map(split) for split in splits]

for i, split in enumerate(splits):
    with open(f'fold_{i + 1}.csv', 'w') as fold_file:
        for seq_id, gp_partition in seq_partitions.items():
            fold_file.write(f'{seq_id}\t{split[gp_partition]}\n')

