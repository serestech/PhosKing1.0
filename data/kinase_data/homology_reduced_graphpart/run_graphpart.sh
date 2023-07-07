#!/usr/bin/env bash

graphpart mmseqs2 -ff ../merged_db_sequences_kinase.fasta -of graphpart_out.csv -th 0.35 --test-ratio 0.1 --val-ratio 0.1
