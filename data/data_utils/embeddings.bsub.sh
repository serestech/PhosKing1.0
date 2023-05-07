#!/bin/sh
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J precompute_embeddings
### -- ask for number of cores (default: 1) --
#BSUB -n 10
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 8:00
# -- request system memory --
#BSUB -R "rusage[mem=50GB]"
# -- request gpu memory --
#BSUB -R "select[gpu80gb]"
# -- request only 1 node --
#BSUB -R "span[hosts=1]"
### -- set the email address --
#BSUB -u danyia@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /zhome/52/c/174062/logs_o/
#BSUB -e /zhome/52/c/174062/logs_e/

user=$(whoami)

if [[ $user == 's220260' ]]; then
    python_env='/zhome/52/c/174062/torch_pyenv/bin'
elif [[ $user == 's212716' ]]; then
    python_env='/zhome/6c/d/164779/pytorch_env/bin'
fi

export PATH=$python_env:/work3/s220260/software/bin:${PATH}

echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Python: $(which python3)"

python3 /zhome/52/c/174062/s220260/PhosKing1.0/data/embeddings/compute_embeddings.py -i /zhome/52/c/174062/s220260/PhosKing1.0/data/homology_reduced/cd-hit_out_29-04.fasta -o /zhome/52/c/174062/s220260/PhosKing1.0/data/embeddings/embeddings_1280
