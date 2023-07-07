import sys
sys.path.append("/zhome/52/c/174062/s220260/PhosKing1.0/data/database_dumps/UniProt")
from uniprot_downloader import search_uniprot
from traceback import format_exc
from pprint import pprint

def get_organism(id: str):
    try:
        searched_suggestion = False
        uniprot_result, _ = search_uniprot(query=id, fields=["organism_name", "organism_id"])
        
        result_dict = uniprot_result.json()

        if result_dict['results'][0]['entryType'] == 'Inactive':
            if 'suggestions' not in result_dict.keys():
                return None, None
            searched_suggestion = True
            new_id = result_dict['suggestions'][0]['query']
            uniprot_result, _ = search_uniprot(query=new_id, fields=["organism_name", "organism_id"])
            result_dict = uniprot_result.json()
            
        
        organism = result_dict['results'][0]['organism']['scientificName']
        phylogeny = result_dict['results'][0]['organism']['lineage']
        phylogeny = ' '.join(phylogeny)
    except Exception as err:
        print()
        print(err)
        return None, None
    
    return organism, phylogeny

print('Reading metadata file')
with open('metadata.tsv') as metadata_file:
    lines = metadata_file.read().splitlines()

print('Getting IDs with unknown organism')
unknown_organism_IDs = set()
for line in lines[1:]:
    ID, species, _, _, _, _, _ = line.split('\t')

    # if species.strip() != 'NA':
    #     continue
    
    unknown_organism_IDs.add(ID.strip())

already_searched = set()
with open('seq_organisms.csv', 'r') as already_searched_file:
    for line in already_searched_file:
        already_searched.add(line.split(',')[0])

before = len(unknown_organism_IDs)
unknown_organism_IDs -= already_searched

print(f'Removed {before - len(unknown_organism_IDs)} sequences already searched')

print(f'{len(unknown_organism_IDs)} sequences with unknown ID')

results_file = open('seq_organisms.csv', 'a')
organisms = {}
failed = 0
for i, seq_id in enumerate(unknown_organism_IDs):
    print(f'Sequence {i+1} of {len(unknown_organism_IDs)} ({((i+1)/len(unknown_organism_IDs)) * 100:.2f}%) Unkonwn organism: {failed}', end='\r')
    organism, phylogeny = get_organism(seq_id)
    if organism is not None:
        results_file.write(f'{seq_id},{organism},{phylogeny}\n')
    else:
        failed += 1
print()
print('Done')
results_file.close()
