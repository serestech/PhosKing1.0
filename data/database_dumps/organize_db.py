import os
import sys
import csv 


with open("db_metadata.tsv", "w") as outfile_metadata: 
    header = '\t'.join(('id', 'pos', 'aa', 'kinases', 'species', 'source'))
    outfile_metadata.write(header + '\n')
    # Scan TSV files in the different databases
    for entry in os.scandir():
        if entry.is_dir():
            if "db_metadata.tsv" in os.listdir(entry.name):
                tsv_path = entry.name + "/db_metadata.tsv"
                print(tsv_path)
                with open(tsv_path, "r") as infile:
                    reader = csv.reader(infile, delimiter='\t')
                    for row in reader:
                        row = '\t'.join(row)
                        outfile_metadata.write(row + '\n')


with open("db_sequences.fasta", "w") as outfile_metadata:
    # Scan fasta files in the different databases
    for entry in os.scandir(): 
        if entry.is_dir():
            if "db_sequences.fasta" in os.listdir(entry.name):
                tsv_path = entry.name + "/db_sequences.fasta"
                print(tsv_path)
                with open(tsv_path, "r") as infile: 
                    reader = csv.reader(infile, delimiter=' ')
                    for row in reader:
                        row = ' '.join(row)
                        outfile_metadata.write(row + '\n')