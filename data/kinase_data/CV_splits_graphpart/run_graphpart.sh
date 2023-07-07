#!/usr/bin/env bash

graphpart mmseqs2 -ff ../merged_db_sequences_kinase.fasta -of graphpart_cv_splits.csv -th 0.35 --partitions 10
