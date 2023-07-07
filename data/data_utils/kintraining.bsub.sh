#!/bin/sh
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J k_training
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
### -- request system memory --
#BSUB -R "rusage[mem=50GB]"
### -- set the email address --
#BSUB -u danyia@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /work3/s220260/logs/
#BSUB -e /work3/s220260/logs/

user=$(whoami)

if [[ $user == 's220260' ]]; then
    python_env='/zhome/52/c/174062/torch_pyenv/bin'
elif [[ $user == 's212716' ]]; then
    python_env='/zhome/6c/d/164779/pytorch_env/bin'
elif [[ $user == 's222929' ]]; then
    python_env='/zhome/6c/d/164779/pytorch_env/bin'
fi

export PATH=$python_env:/work3/s220260/software/bin:${PATH}

echo "JOB ID: $LSB_JOBID"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Python: $(which python3)"

fold=9

echo "Doing cross-validation fold $fold"

read -r -d '' cmd << EOM
python3 /work3/s220260/PhosKing1.0/training/train_model.py
 -m /zhome/52/c/174062/s220260/PhosKing1.0/PhosKing/CNN_FFNN.py
 -n CNN_FFNN
 -a 6,1280,17888,20000,3,6,48,0.5,52
 -f /zhome/52/c/174062/s220260/PhosKing1.0/data/kinase_data/merged_db_sequences_kinase.fasta
 -ft /zhome/52/c/174062/s220260/PhosKing1.0/data/kinase_data/kinase_metadata.tsv
 -emb /work3/s220260/PhosKing1.0/data/embeddings/embeddings_1280_kinase
 -l CEL
 -aaw 6
 -lr 5e-6
 -w 1e-3
 -md kinase
 -es
 -e 20
 -s /zhome/52/c/174062/s220260/PhosKing1.0/data/kinase_data/CV_splits_graphpart/fold_${fold}.csv
EOM

echo -e "Running command:\n$cmd"

$cmd
fold=10

echo "Doing cross-validation fold $fold"

read -r -d '' cmd << EOM
python3 /work3/s220260/PhosKing1.0/training/train_model.py
 -m /zhome/52/c/174062/s220260/PhosKing1.0/PhosKing/CNN_FFNN.py
 -n CNN_FFNN
 -a 6,1280,17888,20000,3,6,48,0.5,52
 -f /zhome/52/c/174062/s220260/PhosKing1.0/data/kinase_data/merged_db_sequences_kinase.fasta
 -ft /zhome/52/c/174062/s220260/PhosKing1.0/data/kinase_data/kinase_metadata.tsv
 -emb /work3/s220260/PhosKing1.0/data/embeddings/embeddings_1280_kinase
 -l CEL
 -aaw 6
 -lr 5e-6
 -w 1e-3
 -md kinase
 -es
 -e 20
 -s /zhome/52/c/174062/s220260/PhosKing1.0/data/kinase_data/CV_splits_graphpart/fold_${fold}.csv
EOM

echo -e "Running command:\n$cmd"

$cmd
