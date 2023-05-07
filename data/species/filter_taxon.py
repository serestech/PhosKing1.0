# https://www.ncbi.nlm.nih.gov/Taxonomy/TaxIdentifier/tax_identifier.cgi
# https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi

import sys

IDs = {}
with open('raw_lineage.txt', 'r') as lineage_file:
    for line in lineage_file:
        _, name, _, IDs_ = [i.strip() for i in line.split('|')]
        IDs[name] = set(IDs_.split(' '))

request = sys.argv[1:]
total = set()
for query in request:
    print(f'###{query}')
    for name,ID_set in IDs.items():
        if query in ID_set:
            print(name)
            total.add(name)

if len(request) >= 2:
    print('###Union')
    for name in total:
        print(name)