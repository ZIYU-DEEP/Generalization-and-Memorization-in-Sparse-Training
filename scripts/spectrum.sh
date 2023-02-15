#!/bin/bash

# (TO MODIFY) Slurm arguments
node=c001
partition=general
mem=24G
jobname=hessian-spec

# (TO MODIFY) Python arguments
random_state=1220
random_state_pre=1220
batch_size=128
prune_ratio=0.9

# YOU SHOULD WRITE A LOOP OVER ALL EPOCH NO FROM 0 TO 250
epoch_no=0

# CHECK THE FOLLOWING SECTION FOR THE FOLDER STRUCTURE
results_root=/net/projects/yuxinlab/sparsity/results

# Get the path for the state dicts
DENSE_PATH=${results_root}/cifar100/resnet-32-4/dense/epochs_200_batch_128_sgd_lr_0.1-stepwise-100-150-0.1_mm_0.9_w_decay_0.0002_init_kaiming_seed_${random_state_pre}

FINETUNE_PATH=${results_root}/cifar100/resnet-32-4/sparse-finetune/method_l1_ratio_${prune_ratio}_epochs_250_batch_128_sgd_lr_0.1-stepwise-100-150-200-0.1_mm_0.9_w_decay_0.0002_init_kaiming_seed_${random_state}_epoch_pre_200_batch_pre_128_lr_pre_0.1-stepwise-100-150-0.1

SCRATCH_PATH=${results_root}/cifar100/resnet-32-4/sparse-scratch/method_l1_ratio_${prune_ratio}_epochs_250_batch_128_sgd_lr_0.1-stepwise-100-150-200-0.1_mm_0.9_w_decay_0.0002_init_kaiming_seed_${random_state}_epoch_pre_200_batch_pre_128_lr_pre_0.1-stepwise-100-150-0.1

# Run the scripts
srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 exp-spectrum.py -pi 0 -pt ${DENSE_PATH} -bs ${batch_size} -no ${epoch_no}

srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 exp-spectrum.py -pt ${FINETUNE_PATH} -pi 1 -bs ${batch_size} -no ${epoch_no}

srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 exp-spectrum.py -pt ${SCRATCH_PATH} -pi 1 -bs ${batch_size} -no ${epoch_no}

# Give permissions
chmod 777 ${results_root}
