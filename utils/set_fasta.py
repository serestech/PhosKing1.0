import sys

if len(sys.argv) not in (3, 4):
    print(f'Usage: {sys.argv[0]} <.info> <.fasta> [<FLAG> (default="TEST_SET)]')
    sys.exit(1)

_, info_file_name, fasta_file_name = sys.argv[:3]
if len(sys.argv) == 3:
    flag = '%TEST_SET'
elif len(sys.argv) == 4:
    flag = '%' + sys.argv[3]


IDs = set()
store = False
with open(info_file_name, 'r') as info:
    for line in info:
        if line.startswith('%'):
            if line.strip() == flag:
                store = True
            else:
                store = False
        elif store:
            IDs.add(line.strip().split(' ')[0])

write = False
with open(fasta_file_name, 'r') as fasta:
    for line in fasta:
        if line.startswith('>'):
            if line[1:].strip() in IDs:
                write = True
            else:
                write = False
        if write:
            print(line, end='')
