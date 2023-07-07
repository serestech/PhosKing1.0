#!/bin/sh
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J predict
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 6:00
### -- request system memory --
#BSUB -R "rusage[mem=300GB]"
### -- set the email address --
#BSUB -u danyia@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /work3/s220260/PhosKing1.0/PhosKing/models/logs/
#BSUB -e /work3/s220260/PhosKing1.0/PhosKing/models/logs/

user=$(whoami)

if [[ $user == 's220260' ]]; then
    python_env='/zhome/52/c/174062/torch_pyenv/bin'
elif [[ $user == 's212716' ]]; then
    python_env='/zhome/6c/d/164779/pytorch_env/bin'
elif [[ $user == 's222929' ]]; then
    python_env='/zhome/42/e/181512/torch_pyenv/bin'
fi

export PATH=$python_env:/work3/s220260/software/bin:${PATH}

echo "JOB ID: $LSB_JOBID"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Python: $(which python3)"

# This way the command is logged
read -r -d '' cmd << EOM
python3 /work3/s220260/PhosKing1.0/PhosKing/predict.py 
 -f /work3/s220260/PhosKing1.0/PhosKing/fastas/test_1000.fasta
 -p 1280
 -m /work3/s220260/PhosKing1.0/PhosKing/CNN_RNN.py
 -n CNN_RNN_FFNN
 -s /work3/s220260/PhosKing1.0/PhosKing/models/pre_model1.pth
 -a 0,1280,1024,512
 -o /work3/s220260/PhosKing1.0/PhosKing/models/pre_model1_test_1000.tsv
 -aaw 15
EOM

echo -e "Running command:\n$cmd"

$cmd
