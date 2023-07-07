from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from time import sleep, time
import os
import math
import time
import csv

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
            down_button = driver.find_element(By.XPATH, '//b[text()="back"]')
            loading = False
            print("Prediction done!")
            return down_button
        except: 
            a = 1
            

print("/// NetPhos3.1 Predictions ///")

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

n_seqs_batched = 0
batch_sequences = ""
n_seqs = len(sequences)
current_batch = 1
seqs_in_batch = 0
n_batches = int(math.ceil(n_seqs/batch_seqs_size_netphos3))

char_batch_count = 0 # number of aa in a batch (netohos3.1 has a limit per submission)
for sequence_id, sequence in sequences.items():
    print(f'Total sequences: {n_seqs_batched+1}/{n_seqs}; Batch: {current_batch}/{n_batches}; Batch sequences: {seqs_in_batch+1}/{batch_seqs_size_netphos3}')
    # Create new sequence adapted to musite deep requirements
    new_sequence = ">" + sequence_id + "\n"
    char_count = 0
    invalid_seq = False
    for idx, char in enumerate(sequence): 
        char_count +=1
        char_batch_count += 1
        new_sequence = new_sequence + char
        if idx%max_aa_line_netphos3 == 0 and idx != 0: 
            new_sequence = new_sequence + "\n"
    new_sequence = new_sequence + "\n"

    # Netphos3.1 has a limit of 20000 seq per submission
    if char_count > 20000 or char_count < 15:
            invalid_seq = True
    
    if invalid_seq == False:
        # Concatenate sequences to be loaded in musite 
        batch_sequences = batch_sequences + new_sequence
        n_seqs_batched += 1
        seqs_in_batch += 1

    # Send sequences to musite when 10 sequences are batched (musite sequences limit)
    if n_seqs_batched%batch_seqs_size_netphos3 == 0 or (n_seqs_batched == n_seqs and n_seqs_batched%batch_seqs_size_netphos3 != 0) or char_batch_count > 190000: 
        driver.get("https://services.healthtech.dtu.dk/services/NetPhos-3.1/")
        cookies_appear = False
        start_time = time.time()
        while cookies_appear == False:
            # your code
            elapsed_time = time.time() - start_time
            if elapsed_time > 3: 
                cookies_appear = True
            try: 
                element = driver.find_element(By.ID, "cookiescript_accept")
                element.click()
                cookies_appear = True
            except: 
                a=1
            
        
        sleep(1)
        # find text area element to write protein sequences
        text_area = driver.find_element(By.NAME, "SEQPASTE")
        text_area.send_keys(batch_sequences)

        best_only = driver.find_element(By.NAME, 'best_only')
        best_only.click()


        # Find the element by its type and value
        element = driver.find_element(By.CSS_SELECTOR, 'input[type="submit"][value="Submit"]')
        driver.execute_script("arguments[0].click();", element)

        before_download = os.listdir(download_dir)
        down_button = wait_prediction()

        page_source = driver.page_source
        file_path = download_dir + "/Netphos3.1_" + str(current_batch) + '.txt'
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(page_source)

        wait_download()
        current_batch += 1
        batch_sequences = ""
        char_batch_count = 0

    

print("Netphos3.1 predictions finished!!")

output_file = download_dir + '/Test_NetPhos3.1.tsv'

txt_files = [file for file in os.listdir(download_dir) if file.startswith('Netphos3.1') and file.endswith('.txt')]

data = []
for file in txt_files: 
    input_file = download_dir + '/' + file
    # Read the input file and extract the necessary information
    
    with open(input_file, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header line


    for idx, line in enumerate(lines): 
        if 'Sequence' in line: 
            seq = True
            c = 2
            aa_info = lines[c+idx]
            elements = aa_info.split()
            while(seq == True): 
                
                # Extract the desired elements
                try: 
                    sequence = elements[1]
                except: 
                    a = 12
                position = elements[2]
                kinase = elements[6]
                score = elements[5]

                result = [sequence, position, kinase, score]
                data.append(result)

                c += 1
                aa_info = lines[c+idx]
                elements = aa_info.split()
                if len(elements) < 8: 
                    seq = False

# Write the data to the output TSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['ID', 'Position', 'Kinase', 'Score'])
    writer.writerows(data)

print("Conversion completed successfully!")