#!/bin/bash

cd database_dumps

for dir in */; do
    if [ -d "$dir" ]; then
        cd "$dir"
        echo "---------- Organizing $dir database ----------"
        if test -f organize_db.py; then
            python organize_db.py
            echo -e "\033[32m$dir database finished!\033[0m"
        else
            echo -e "\033[33m organize_db.py doesn't exist in $dir database\033[0m"
        fi
        
        cd "$OLDPWD"
    fi
done

echo "---------- Merging databases ----------"

python merge_db.py

echo -e "\033[32mAll databases!\033[0m"


echo "---------- Running CD-hit ----------"

cd-hit -i temp_seqs.fasta -o cdhit_out_0.75.fasta -c 0.75 -n 5 -s 0.7 -g 1


echo "---------- Filtering databases ----------"

cp cdhit_out_0.75.fasta sequences.fasta
python filter_db.py -f sequences.fasta -i temp_feats.tsv temp_metadata.tsv -o features.tsv metadata.tsv


echo "---------- Computing embeddings ----------"

cd ../embeddings
python compute_embeddings -i ../database_dumps/sequences.fasta -p 1280 