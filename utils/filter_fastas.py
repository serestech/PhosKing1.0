from utils import read_fasta

fasta = read_fasta('/zhome/52/c/174062/s220260/PhosKing1.0/PhosKing/fastas/pre_model1_test_set.fasta', format=dict)

to_discrd = []
for id, seq in fasta.items():
    if len(seq) > 19500:
        to_discrd.append(id)
        
print(f'Discarding {len(to_discrd)}')

for id in to_discrd:
    fasta.pop(id)
    
with open('filtered.fasta', 'w') as outfile:
    for id, seq in fasta.items():
        seq_out = f'> {id}\n{seq}\n'
        outfile.write(seq_out)
