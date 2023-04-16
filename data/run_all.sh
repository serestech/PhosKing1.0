#!/bin/bash

cd database_dumps

for dir in */; do
    if [ -d "$dir" ]; then
        cd "$dir"
        echo "---------- Organizing $dir database ----------"
        if test -f organize_db.py; then
            python3 organize_db.py
            echo -e "\033[32m$dir database finished!\033[0m"
        else
            echo -e "\033[33m organize_db.py doesn't exist in $dir database\033[0m"
        fi
        
        cd "$OLDPWD"
    fi
done

echo "---------- Merging databases ----------"

python3 merge_db.py
echo -e "\033[32mAll databases!\033[0m"


echo "---------- Clustering sequences (MMseqs2) -----------"

# Requires having mmseqs in PATH, a symlink can be found in software/bin
mkdir mmseqs_tmp
mmseqs easy-cluster temp_sequences.fasta clusters mmseqs_tmp --min-seq-id 0.3 -c 0.8 --cov-mode 0
rm clusters_all_seqs.fasta -f
rm -r mmseqs_tmp -f
mv clusters_rep_seq.fasta ../train_data/sequences.fasta -f
mv clusters_cluster.tsv clusters.tsv -f


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