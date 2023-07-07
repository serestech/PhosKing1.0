from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
import math
import os
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
                down_button = driver.find_elements(By.ID, 'downloadall')
                loading = False
                print("Prediction done!")
                return down_button
            except: 
                a=1

def select_kinases(): 
    # Select kinases to be predicted
    AGC = driver.find_element(By.ID, "treeDemo_2_check")
    AGC.click()
    CAMK = driver.find_element(By.ID, "treeDemo_77_check")
    CAMK.click()
    CK1 = driver.find_element(By.ID, "treeDemo_155_check")
    CK1.click()
    CMGC = driver.find_element(By.ID, "treeDemo_171_check")
    CMGC.click()
    PKL = driver.find_element(By.ID, "treeDemo_258_check")
    PKL.click()
    RGC = driver.find_element(By.ID, "treeDemo_261_check")
    RGC.click()
    STE = driver.find_element(By.ID, "treeDemo_264_check")
    STE.click()
    TKL = driver.find_element(By.ID, "treeDemo_327_check")
    TKL.click()
    Atypical = driver.find_element(By.ID, "treeDemo_366_check")
    Atypical.click()
    Other = driver.find_element(By.ID, "treeDemo_401_check")
    Other.click()
    Tyrosine = driver.find_element(By.ID, "treeDemo_486_check")
    Tyrosine.click()
    TK = driver.find_element(By.ID, "treeDemo_487_check")
    TK.click()


print("/// GP6.0 Predictions ///")

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
driver.get("https://gps.biocuckoo.cn/online.php")


n_seqs_batched = 0
batch_sequences = ""
n_seqs = len(sequences)
n_batches = int(math.ceil(n_seqs/batch_seqs_size_GPS))
current_batch = 1
seqs_in_batch = 0

for sequence_id, sequence in sequences.items():
    print(f'Total sequences: {n_seqs_batched+1}/{n_seqs}; Batch: {current_batch}/{n_batches}; Batch sequences: {seqs_in_batch+1}/{batch_seqs_size_GPS}')
    

    # Create new sequence adapted to musite deep requirements
    new_sequence = ">" + sequence_id + "\n"
    for idx, char in enumerate(sequence): 
        new_sequence = new_sequence + char
        """ if idx%max_aa_line == 0 and idx != 0: 
            new_sequence = new_sequence + "\n" """
    new_sequence = new_sequence + "\n"
    
    # Concatenate sequences to be loaded in musite 
    batch_sequences = batch_sequences + new_sequence
    n_seqs_batched += 1
    seqs_in_batch += 1

    # Send sequences to musite when 10 sequences are batched (musite sequences limit)
    if n_seqs_batched%batch_seqs_size_GPS == 0: 
        # find text area element to write protein sequences
        text_area = driver.find_element(By.CSS_SELECTOR, 'textarea[id="Blast_Input"]')
        
        # fill text area with batched sequences
        text_area.send_keys(batch_sequences)

        # Select kinases for prediction
        select_kinases()
        
        

        # find prediction button and click
        button = driver.find_element(By.CSS_SELECTOR, 'input[value="Submit"]')
        button.click()
        
        # wait until download button appears and click on it
        down_button = wait_prediction()

        # Check if it has found predictions (if no predictions found no file to download)
        table = driver.find_element(By.ID, "table")
        table_text_no_pred = 'ID Position Code Kinase Peptide Score Cutoff Source Links Interaction Logo'
        if table.text ==  table_text_no_pred: 
            print('No predictions in this batch\n')
        else: 
            before_download = os.listdir(download_dir)
            try: 
                down_button[0].click()
            except: 
                a = 1

            # Wait until download is completed        
            wait_download()
            print(f'Batch {current_batch} downloaded!')

        # Come back to web server predictor
        driver.get("https://gps.biocuckoo.cn/online.php")

        current_batch += 1
        seqs_in_batch = 0
        batch_sequences = ""

    # Send last batched sequences if batch length is not exactly 10
    if n_seqs_batched == n_seqs and n_seqs_batched%batch_seqs_size_GPS != 0: 
        # find text area element to write protein sequences
        text_area = driver.find_element(By.CSS_SELECTOR, 'textarea[spellcheck="false"][placeholder=">sp..."]')
        text_area.send_keys(batch_sequences)
        # click login button
        button = driver.find_element(By.CLASS_NAME, "textarea_submit__15dOD")
        button.click()

        # wait until download button appears and click on it
        down_button = wait_prediction()
        
        before_download = os.listdir(download_dir)
        down_button[0].click()

        wait_download()
        print(f'Batch {current_batch} downloaded!')
        
        current_batch += 1
        batch_sequences = ""
        text_area.clear()

print("GPS6.0 test predictions finished!!")


print("Converting prediction files to a tsv file ...")


output_file = download_dir + '/Test_GPS6.0.tsv'

txt_files = [file for file in os.listdir(download_dir) if file.startswith('Result') and file.endswith('.txt')]

idxs = [0, 1, 3, 5] # ID, Position, Kinase, Score
data = []
for file in txt_files:
    input_file = download_dir + '/' + file
    # Read the input file and extract the necessary information
    
    with open(input_file, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header line
        for line in lines:
            if line:
                fields = line.split('\t')
                desired_fields = [fields[i] for i in idxs]
                data.append(desired_fields)

# Write the data to the output TSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['ID', 'Position', 'Kinase', 'Score'])
    writer.writerows(data)

print("Conversion completed successfully!")






