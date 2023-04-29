#!/bin/bash

orig_dir=$(pwd)
cd database_dumps

for dir in */; do
    if [ -d "$dir" ]; then
        cd "$dir"
        echo "---------- Organizing $dir ----------"
        if test -f organize_db.py; then
            python3 organize_db.py
            echo -e "\033[32m$dir finished\033[0m"
        else
            echo -e "\033[33m organize_db.py doesn't exist in $dir database\033[0m"
        fi
        cd ..
    fi
done

cd "$orig_dir"

echo "---------- Merging databases ----------"

python3 ./clean_data/merge_db_mirror.py
echo -e "\033[32mMerged databases\033[0m"


echo '---------- Clustering sequences (MMseqs2) -----------'

# Requires having mmseqs in PATH, a symlink can be found in software/bin
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
MMSEQS_TMP=tmp/mmseqs_tmp_$timestamp
mkdir -p $MMSEQS_TMP
SEQ_IDENT=0.35
SENSITIVITY=7.5
cmd="mmseqs easy-cluster ./clean_data/sequences.fasta ./homology_reduced/mmseqs_$timestamp $MMSEQS_TMP --min-seq-id $SEQ_IDENT -s $SENSITIVITY -c 0.8 --cov-mode 0"
echo "--- mmseqs command ---"
echo $cmd
echo "----------------------"
$cmd
# rm clusters_all_seqs.fasta -f
# rm -r mmseqs_tmp -f
# mv clusters_rep_seq.fasta ../train_data/sequences.fasta -f
# mv clusters_cluster.tsv clusters.tsv -f

echo "---------- Clustering sequences (CD-HIT) -----------"

echo -n "CD-HIT clustering takes a long while and should NOT be done on the login nodes. What do you want to do? (skip/run/exit): "
read usr_in
if [[ $usr_in == "skip" ]]; then
    echo "Skipping CD-HIT"
elif [[ $usr_in == "exit" ]]; then
    echo "Exiting"
    exit 1
elif [[ $usr_in == "run" ]]; then
    echo "Running CD-HIT... (this takes a long while)"
    exit 1
else
    echo "The option '$usr_in' did not match an option. Exiting"
    exit 1
fi

echo "---------- Filtering databases ----------"

cd ../train_data
python3 filter_db.py -f sequences.fasta -i ../database_dumps/temp_features.tsv ../database_dumps/temp_metadata.tsv -o features.tsv metadata.tsv
python3 filter_db_kin.py -f ../database_dumps/temp_sequences.fasta -t ../database_dumps/temp_features.tsv -m ../database_dumps/temp_metadata.tsv -c ../database_dumps/clusters.tsv

echo "---------- Computing embeddings ----------"

cd ../embeddings
# # Make sure to be logged in a CUDA session and to have an environment with torch activated!!!
# python3 compute_embeddings.py -i ../train_data/sequences.fasta -p 1280 -F 4096 -of 400
# python3 compute_embeddings.py -i ../train_data/sequences_kin.fasta -p 1280 -f embeddings_kin_1280 -F 4096 -of 400

cd ..