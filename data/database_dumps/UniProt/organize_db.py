#!/usr/bin/env python3
import json
import os
import os.path as path
import sys
import re

# Specify dump folder from command line and load file names
if len(sys.argv) > 1:
    data_folder = sys.argv[1]
else:
    data_folder = path.abspath(path.join(path.dirname(__file__), 'uniprot_dump'))

json_files = os.listdir(data_folder)
json_files.sort(key=lambda x: int(x.split('_')[0]))
n_files = len(json_files)

# Open output files and start loop through input files
output_fasta = open('db_sequences.fasta', 'w')
output_tsv = open('db_metadata.tsv', 'w')
output_tsv.write('#id	pos	aa	kinases	species	kin_species	source\n')
for i, filename in enumerate(json_files):
    with open(f'{data_folder}/{filename}') as json_file:
        uniprot_dict = json.load(json_file)

    for protein_dict in uniprot_dict['results']:

        # Initialize feature information, each protein may contain more than one feature entry
        ft_aas, ft_positions, ft_kinases = [], [], []
        for ft_dict in protein_dict['features']:
            ft_description = ft_dict['description']

            # Filter PTMs that aren't phosphorilations on S, T or Y
            if re.search(r'^Phospho(serine|threonine|tyrosine)', ft_description):
                # Filter PTMs assigned to isoforms (we ignore them)
                if ft_dict['location'].get('sequence') is None and not re.search(r'in variant', ft_description):

                    # Store position and aa
                    position = ft_dict['location']['start']['value']

                    if re.search(r'^Phosphoserine', ft_description):
                        aa = 'S'
                    elif re.search(r'^Phosphothreonine', ft_description):
                        aa = 'T'
                    elif re.search(r'^Phosphotyrosine', ft_description):
                        aa = 'Y'

                    # Easy regex to get list of kinases, ignoring "autocatalysis"
                    if re.search(r';\sby\s([\w/]{2,}(,\s|\sand\s)?)+', ft_description):
                        kinase_list = re.findall(r'(?:;\sby\s|,\s|\sand\s)([\w/]{2,})', ft_description)
                        kinases = ','.join([k for k in kinase_list if (k != 'autocatalysis' and k != 'host')])
                        if not kinases:
                            kinases = 'NA'
                    else:
                        kinases = 'NA'

                    # Guard against possible remaining features from isoforms
                    # Shouldn't execute with current dumps, but just in case is left as debug
                    try:
                        if protein_dict['sequence']['value'][position-1] != aa:
                            real = protein_dict['sequence']['value'][position-1]
                            ID = protein_dict['primaryAccession']
                            print(f'In {ID}, a {ft_description} is indicated at position {position}, but the aa in sequence is {real}, skipping')
                        
                        # Store values from the valid PTM
                        else:
                            ft_positions.append(str(position))
                            ft_aas.append(aa)
                            ft_kinases.append(kinases)

                    except IndexError:
                        ID = protein_dict['primaryAccession']
                        print(f'In {ID}, a {ft_description} is indicated at position {position}, but the sequence is not that long, skipping')

        # If found any valid PTM, extract the rest of the info
        if ft_aas:
            uniprot_accession = protein_dict['primaryAccession']
            org_name = protein_dict['organism']['scientificName']
            org_taxonID = protein_dict['organism']['taxonId']
            org_domain = protein_dict['organism']['lineage'][0]
            sequence=protein_dict['sequence']['value']
            
            # Write fasta with UniProt accession number and sequence
            fasta_entry = '> {header}\n{seq}\n'.format(header=uniprot_accession, seq=sequence)
            output_fasta.write(fasta_entry)

            # Create list with feature lines belonging to one sequence
            tsv_lines = []
            for position, aa, kinases in zip(ft_positions, ft_aas, ft_kinases):
                tsv_lines.append('\t'.join((uniprot_accession, position, aa, kinases, org_name, 'NA', 'UniProt')))
            
            # Write tsv with features and metadata
            tsv_entry = '\n'.join(tsv_lines) + '\n'
            output_tsv.write(tsv_entry)

    # Brief log
    if (i + 1) % 10 == 0:
        print(f'Organizing UniProt data from dump files... {i+1}/{n_files}', end='\r')

# Finishing!
print(f'Organizing UniProt data from dump files... {i+1}/{n_files}')
output_fasta.close()
output_fasta.close()
print('Finished!')
