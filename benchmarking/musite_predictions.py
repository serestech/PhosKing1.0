from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
import os
import math
from selenium.webdriver.common.keys import Keys
import csv
import zipfile
import re

from test_config import *


def wait_download():
    downloaded = False
    while downloaded == False:
        sleep(1)
        after_download = os.listdir(download_dir)
        if after_download != before_download:
            downloaded = True
            break

def wait_prediction(): 
    print('Waiting for the prediction ...')
    loading = True
    while (loading == True):
        try: 
            down_button = driver.find_elements(By.CLASS_NAME, "output_download__20etX")[4]
            loading = False
            print("Prediction done!")
            return down_button
        except: 
            a = 1
            
print("/// MusiteDeep Predictions ///")

# Define an empty dictionary to store the sequences
sequences = {}

# Open the FASTA file and read the sequences
with open(fasta_file, 'r') as f:
    # Initialize variables to store the ID and sequence
    sequence_id = ''
    sequence = ''

    # Iterate over each line in the file
    for line in f:
        # Strip the line of leading/trailing white space
        line = line.strip()

        # Check if the line is an ID line
        if line.startswith('>'):
            # If there is a previous sequence, store it in the dictionary
            if sequence_id != '':
                sequences[sequence_id] = sequence

            # Reset the sequence ID and sequence variables
            sequence_id = line[1:]
            sequence = ''
        else:
            # Append the line to the current sequence
            sequence += line

    # Store the last sequence in the dictionary
    if sequence_id != '':
        sequences[sequence_id] = sequence

# initialize the Chrome driver
driver = webdriver.Chrome("chromedriver")
# head to github login page
driver.get("https://www.musite.net/")

n_seqs_batched = 0
batch_sequences = ""
n_seqs = len(sequences)
current_batch = 1
seqs_in_batch = 0
n_batches = int(math.ceil(n_seqs/batch_seqs_size_musite))

# Find the element by its ID
element = driver.find_element(By.ID ,"react-select-7-input")

# Click on the element
element.click()

text_to_enter = "Phosphorylation (Y)"
element.send_keys(text_to_enter)
element.send_keys(Keys.TAB)

count_batch_aa = 0
aa_limit = False

for sequence_id, sequence in sequences.items():
    print(f'Total sequences: {n_seqs_batched+1}/{n_seqs}; Batch: {current_batch}/{n_batches}; Batch sequences: {seqs_in_batch+1}/{batch_seqs_size_musite}')
    # Create new sequence adapted to musite deep requirements
    new_sequence = ">" + sequence_id + "\n"
    for idx, char in enumerate(sequence): 
        count_batch_aa += 1
        if count_batch_aa > limit_aa_batch: 
            aa_limit = True
        new_sequence = new_sequence + char
        if idx%max_aa_line_musite == 0 and idx != 0: 
            new_sequence = new_sequence + "\n"
    
    # If limit aa number is achieved in a batch, do not continue batching sequences, send the batch
    if aa_limit == False:
        new_sequence = new_sequence + "\n"
    else:
        new_sequence = ''
    
    # Concatenate sequences to be loaded in musite 
    batch_sequences = batch_sequences + new_sequence
    n_seqs_batched += 1
    seqs_in_batch += 1

    # Send sequences to musite when 10 sequences are batched (musite sequences limit)
    if (seqs_in_batch%batch_seqs_size_musite == 0) or (aa_limit == True) or (n_seqs_batched == n_seqs and seqs_in_batch%batch_seqs_size_musite != 0): 
        # find text area element to write protein sequences
        text_area = driver.find_element(By.CSS_SELECTOR, 'textarea[spellcheck="false"][placeholder=">sp..."]')
        # fill text area with batched sequences
        text_area.send_keys(batch_sequences)
        # find prediction button and click
        button = driver.find_element(By.CLASS_NAME, "textarea_submit__15dOD")
        button.click()
        # wait until download button appears and click on it
        down_button = wait_prediction()

        before_download = os.listdir(download_dir)
        down_button.click()
        wait_download()
        print(f'Batch {current_batch} downloaded!')
        
        batch_sequences = ""
        current_batch += 1
        seqs_in_batch = 0
        count_batch_aa = 0
        aa_limit = False
        text_area.clear()
        

print("Musite test predictions finished!!")

zip_files = [file for file in os.listdir(download_dir) if 'MusiteDeep' in file and file.endswith('.zip')]

data = []
for file in zip_files: 
    input_file = download_dir + '/' + file

    # Open the ZIP file
    with zipfile.ZipFile(input_file, 'r') as zip_ref:
        # Extract all the contents to the specified directory
        zip_ref.extractall(download_dir)

    output_file = download_dir + '/Test_MusiteDeep.tsv'

    txt_files = [file for file in os.listdir(download_dir) if file.startswith('Prediction_results') and file.endswith('.txt')]

    for file in txt_files: 
        input_file = download_dir + '/' + file
        # Read the input file and extract the necessary information
        
        with open(input_file, 'r') as file:
            lines = file.readlines()[1:]  # Skip the header line


        for idx, line in enumerate(lines): 
            if '>' in line: 
                seq = True
                c = 1
                aa_info = lines[c+idx]
                elements = aa_info.split()
                while(seq == True): 
                    
                    # Extract the desired elements
                    try: 
                        sequence = elements[0]
                    except: 
                        a = 12
                    position = elements[1]
                    kinase = ''                    
                    score = number = re.findall(r'\d+\.\d+', elements[3])[0]
                    
                    result = [sequence, position, kinase, score]
                    data.append(result)

                    c += 1
                    try: 
                        aa_info = lines[c+idx]
                    except: 
                        seq = False
                    elements = aa_info.split()
                    if '>' in elements[0]: 
                        seq = False

# Write the data to the output TSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['ID', 'Position', 'Kinase', 'Score'])
    writer.writerows(data)

print("Conversion completed successfully!")
        