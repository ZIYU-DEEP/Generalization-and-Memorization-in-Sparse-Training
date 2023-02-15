#!/bin/bash

# Python arguments
random_state=1220
random_state_pre=1220
results_root=./results
batch_size=1024

# Slurm arguments
node=c001
partition=general
mem=24G
jobname=hessian-trace

# Get the results for the dense network
srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 exp-trace.py -pi 0 -pt ${results_root}/cifar100/resnet-32-4/dense/epochs_200_batch_128_sgd_lr_0.1-stepwise-100-150-0.1_mm_0.9_w_decay_0.0002_init_kaiming_seed_${random_state_pre} -bs ${batch_size}

# Get the results for the sparse network
for prune_ratio in 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.95 0.96 0.98
do
    FINETUNE_PATH=${results_root}/cifar100/resnet-32-4/sparse-finetune/method_l1_ratio_${prune_ratio}_epochs_250_batch_128_sgd_lr_0.1-stepwise-100-150-200-0.1_mm_0.9_w_decay_0.0002_init_kaiming_seed_${random_state}_epoch_pre_200_batch_pre_128_lr_pre_0.1-stepwise-100-150-0.1

    SCRATCH_PATH=${results_root}/cifar100/resnet-32-4/sparse-scratch/method_l1_ratio_${prune_ratio}_epochs_250_batch_128_sgd_lr_0.1-stepwise-100-150-200-0.1_mm_0.9_w_decay_0.0002_init_kaiming_seed_${random_state}_epoch_pre_200_batch_pre_128_lr_pre_0.1-stepwise-100-150-0.1

    srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 exp-trace.py -pt ${FINETUNE_PATH} -pi 1 -bs ${batch_size}

    srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 exp-trace.py -pt ${SCRATCH_PATH} -pi 1 -bs ${batch_size}
done

chmod 777 ${results_root}
