#! /usr/bin/env bash

SOFTWARE=/work3/s220260/software
CD_HIT=$SOFTWARE/cd-hit/psi-cd-hit/psi-cd-hit.pl
BLAST=$SOFTWARE/blast/bin

SEQ_IDENT=0.35  # -c
threads=$(lscpu | grep "^CPU(s):" | awk '{print $2}') # -blp, threads for blast
threads=$(( $threads - 1 ))  # Use all cores except 1


if [[ "${*}" == *"--help"* ]]; then
    $CD_HIT --help
    exit 1
fi

echo "CD-HIT will use $threads cores (on low priority)"
cmd="$CD_HIT -P $BLAST -c $SEQ_IDENT -para $threads $@"

echo '--- CD-HIT command ---'
echo "$cmd"
echo '----------------------'

nice -n 19 $cmd
